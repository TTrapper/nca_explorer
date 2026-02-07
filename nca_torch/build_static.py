"""
Build a fully static web app from an NCA dynamics checkpoint.

Produces:
    output_dir/
    ├── index.html              # Self-contained page (inline CSS/JS)
    ├── onnx/
    │   ├── decode_latent.onnx  # z → first_frame + NCA params
    │   └── nca_step.onnx       # (grid, params) → new_grid
    └── data/
        ├── manifest.json       # Config, pre-encoded latents, library
        ├── seq_000.bin         # Sequence 0 frames [T*H*W*C] uint8
        └── ...

Usage:
    python build_static.py CHECKPOINT --data DATA --output ./static_build
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder import NCAAutoencoder
from datasets import SequenceDataset


# ---------------------------------------------------------------------------
# ONNX wrapper modules
# ---------------------------------------------------------------------------

class DecodeLatentExport(nn.Module):
    """Combines FirstFrameDecoder + HyperNetwork param generation for ONNX."""

    def __init__(self, model: NCAAutoencoder):
        super().__init__()
        self.first_frame_decoder = model.first_frame_decoder
        self.hypernet = model.decoder.hypernet
        self.weight_scale = model.decoder.weight_scale

        # Store split sizes for reshaping
        gc = model.decoder.grid_channels
        self.gc = gc
        self.layer1_weight_size = gc * gc * 3 * 3
        self.layer1_bias_size = gc
        self.layer2_weight_size = gc * gc * 3 * 3
        self.layer2_bias_size = gc

    def forward(self, z):
        # z: [1, latent_dim]
        first_frame = self.first_frame_decoder(z)  # [1, C, H, W]

        # HyperNetwork
        params = self.hypernet(z) * self.weight_scale
        idx = 0
        layer1_w = params[:, idx:idx + self.layer1_weight_size]
        idx += self.layer1_weight_size
        layer1_b = params[:, idx:idx + self.layer1_bias_size]
        idx += self.layer1_bias_size
        layer2_w = params[:, idx:idx + self.layer2_weight_size]
        idx += self.layer2_weight_size
        layer2_b = params[:, idx:idx + self.layer2_bias_size]

        # Reshape weights — squeeze batch dim (batch=1 always for static export)
        gc = self.gc
        layer1_w = layer1_w.view(gc, gc, 3, 3)
        layer1_b = layer1_b.view(gc)
        layer2_w = layer2_w.view(gc, gc, 3, 3)
        layer2_b = layer2_b.view(gc)

        return first_frame, layer1_w, layer1_b, layer2_w, layer2_b


class NCAStepExport(nn.Module):
    """Single NCA step with ONNX-compatible manual circular padding."""

    def forward(self, grid, layer1_w, layer1_b, layer2_w, layer2_b):
        # grid: [1, gc, H, W]
        # weights: [gc, gc, 3, 3], bias: [gc]

        # Manual circular pad (replaces F.pad(mode='circular'))
        h = torch.cat([grid[:, :, -1:, :], grid, grid[:, :, :1, :]], dim=2)
        h = torch.cat([h[:, :, :, -1:], h, h[:, :, :, :1]], dim=3)

        h = F.conv2d(h, layer1_w, bias=layer1_b)
        h = F.relu(h)

        h = torch.cat([h[:, :, -1:, :], h, h[:, :, :1, :]], dim=2)
        h = torch.cat([h[:, :, :, -1:], h, h[:, :, :, :1]], dim=3)

        update = F.conv2d(h, layer2_w, bias=layer2_b)
        return grid + update


# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, dev: str = "cpu"):
    """Load trained dynamics model from checkpoint. Returns (model, config_dict)."""
    device = torch.device(dev)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})

    context_frames = checkpoint.get("context_frames", args.get("context_frames", 4))
    in_channels = checkpoint.get("in_channels", args.get("in_channels", 3))
    grid_size = checkpoint.get("grid_size", args.get("grid_size", (32, 32)))
    num_steps = checkpoint.get("num_steps", args.get("num_steps", 1))
    latent_dim = args.get("latent_dim", 64)
    grid_channels = args.get("grid_channels", 16)
    hidden_dim = args.get("hidden_dim", 256)
    use_vae = not args.get("no_vae", False)

    if isinstance(grid_size, list):
        grid_size = tuple(grid_size)

    model = NCAAutoencoder(
        latent_dim=latent_dim,
        grid_channels=grid_channels,
        hidden_dim=hidden_dim,
        use_vae=use_vae,
        in_channels=in_channels,
        grid_size=grid_size,
        context_frames=context_frames,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    cfg = dict(
        context_frames=context_frames,
        in_channels=in_channels,
        grid_size=grid_size,
        num_steps=num_steps,
        latent_dim=latent_dim,
        grid_channels=grid_channels,
        use_vae=use_vae,
    )

    print(f"Loaded model from {checkpoint_path}")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    return model, cfg, device


def load_data(data_path: str, context_frames: int):
    """Load sequence dataset."""
    dataset = SequenceDataset(
        data_path,
        context_frames=context_frames,
        future_frames=50,
    )
    print(f"Loaded {dataset.num_sequences} sequences")
    return dataset


def load_library(checkpoint_path: str, device):
    """Load latent library from latent_library.json in checkpoint directory."""
    lib_path = Path(checkpoint_path).parent / "latent_library.json"
    library = {}
    if lib_path.exists():
        with open(lib_path) as f:
            raw = json.load(f)
        for name, arr in raw.items():
            library[name] = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
        print(f"Loaded {len(library)} latents from {lib_path}")
    else:
        print(f"No latent_library.json found in {lib_path.parent}")
    return library


def encode_sequences(model, dataset, cfg, device, max_sequences: int = 0):
    """Pre-encode all sequences, returning list of latent vectors."""
    n = dataset.num_sequences
    if max_sequences > 0:
        n = min(n, max_sequences)

    context_frames = cfg["context_frames"]
    grid_size = cfg["grid_size"]
    latents = []

    for i in range(n):
        seq = dataset.sequences[i]  # [T, H, W, C]
        gt = torch.from_numpy(seq).float().permute(0, 3, 1, 2).to(device) / 255.0
        context = gt[:context_frames]  # [N, C, H, W]
        context_stacked = context.reshape(1, -1, *grid_size)  # [1, N*C, H, W]

        with torch.no_grad():
            z, mu, _ = model.encode(context_stacked)
            # Use mu for VAE (deterministic), z for non-VAE
            latent = mu if mu is not None else z

        latents.append(latent.squeeze(0).cpu().numpy().tolist())

        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  Encoded {i + 1}/{n} sequences")

    return latents


def export_onnx_models(model, cfg, device, output_dir: Path):
    """Export ONNX wrapper models."""
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    latent_dim = cfg["latent_dim"]
    grid_channels = cfg["grid_channels"]
    H, W = cfg["grid_size"]

    # --- DecodeLatent ---
    decode_wrapper = DecodeLatentExport(model)
    decode_wrapper.eval()
    dummy_z = torch.randn(1, latent_dim, device=device)

    decode_path = onnx_dir / "decode_latent.onnx"
    torch.onnx.export(
        decode_wrapper,
        (dummy_z,),
        str(decode_path),
        input_names=["z"],
        output_names=["first_frame", "layer1_w", "layer1_b", "layer2_w", "layer2_b"],
        dynamic_axes=None,
        opset_version=17,
    )
    print(f"Exported {decode_path} ({decode_path.stat().st_size / 1024:.0f} KB)")

    # --- NCAStep ---
    nca_wrapper = NCAStepExport()
    nca_wrapper.eval()
    dummy_grid = torch.randn(1, grid_channels, H, W, device=device)
    dummy_l1w = torch.randn(grid_channels, grid_channels, 3, 3, device=device)
    dummy_l1b = torch.randn(grid_channels, device=device)
    dummy_l2w = torch.randn(grid_channels, grid_channels, 3, 3, device=device)
    dummy_l2b = torch.randn(grid_channels, device=device)

    nca_path = onnx_dir / "nca_step.onnx"
    torch.onnx.export(
        nca_wrapper,
        (dummy_grid, dummy_l1w, dummy_l1b, dummy_l2w, dummy_l2b),
        str(nca_path),
        input_names=["grid", "layer1_w", "layer1_b", "layer2_w", "layer2_b"],
        output_names=["new_grid"],
        dynamic_axes=None,
        opset_version=17,
    )
    print(f"Exported {nca_path} ({nca_path.stat().st_size / 1024:.0f} KB)")


def export_data(dataset, latents, library, cfg, output_dir: Path, max_sequences: int = 0):
    """Write sequence .bin files and manifest.json."""
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    n = len(latents)
    H, W = cfg["grid_size"]
    C = cfg["in_channels"]

    # Build slot_assignments from auto-load order (alphabetical, up to 19 slots)
    sorted_names = sorted(library.keys())
    slot_assignments = [None] * 19
    for i, name in enumerate(sorted_names):
        if i >= 19:
            break
        slot_assignments[i] = name

    sequences_meta = []
    for i in range(n):
        seq = dataset.sequences[i]  # [T, H, W, C] uint8
        T = seq.shape[0]

        # Write raw binary
        bin_path = data_dir / f"seq_{i:03d}.bin"
        seq.astype(np.uint8).tofile(str(bin_path))

        sequences_meta.append({
            "frames": T,
            "z": latents[i],
        })

        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  Wrote {i + 1}/{n} sequence bins")

    # Library: convert tensors to lists
    lib_dict = {}
    for name, tensor in library.items():
        lib_dict[name] = tensor.squeeze(0).cpu().numpy().tolist()

    manifest = {
        "num_sequences": n,
        "width": W,
        "height": H,
        "channels": C,
        "context_frames": cfg["context_frames"],
        "num_steps": cfg["num_steps"],
        "latent_dim": cfg["latent_dim"],
        "grid_channels": cfg["grid_channels"],
        "sequences": sequences_meta,
        "library": lib_dict,
        "slot_assignments": slot_assignments,
    }

    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"Wrote manifest.json ({manifest_path.stat().st_size / 1024:.0f} KB)")


def generate_html(cfg, output_dir: Path):
    """Generate index.html with inline CSS/JS using onnxruntime-web."""
    html = _build_html(cfg)
    html_path = output_dir / "index.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"Wrote index.html ({html_path.stat().st_size / 1024:.0f} KB)")


def _build_html(cfg):
    """Construct the full HTML string."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NCA Dynamics Viewer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #1a1a2e;
            color: #eee;
            font-family: 'Segoe UI', system-ui, sans-serif;
            min-height: 100vh;
            padding: 20px;
        }}
        h1 {{ color: #4fc3f7; margin-bottom: 5px; text-align: center; }}
        .subtitle {{ color: #888; text-align: center; margin-bottom: 20px; }}

        .main-container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .viewers {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}

        .viewer {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}

        .viewer-label {{
            font-size: 1.1em;
            margin-bottom: 10px;
            color: #4fc3f7;
        }}

        canvas {{
            border: 2px solid #4fc3f7;
            background: #000;
            image-rendering: pixelated;
        }}

        .context-section {{
            background: #16213e;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}

        .context-section h3 {{
            color: #4fc3f7;
            margin-bottom: 10px;
            font-size: 0.9em;
        }}

        .context-frames {{
            display: flex;
            gap: 10px;
            justify-content: center;
        }}

        .context-frame {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}

        .context-frame span {{
            font-size: 0.75em;
            color: #888;
            margin-top: 5px;
        }}

        .controls {{
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}

        .control-row {{
            display: flex;
            gap: 15px;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
        }}

        button {{
            background: #4fc3f7;
            color: #1a1a2e;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            font-size: 0.9em;
        }}

        button:hover {{ background: #81d4fa; }}
        button:disabled {{ background: #555; cursor: not-allowed; }}
        button.active-mode {{ background: #fff; }}

        .slider-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .slider-group label {{
            color: #aaa;
            font-size: 0.9em;
            min-width: 80px;
        }}

        .slider-group input[type="range"] {{
            width: 150px;
        }}

        .slider-group .value {{
            color: #4fc3f7;
            font-family: monospace;
            min-width: 40px;
        }}

        .info {{
            display: flex;
            gap: 30px;
            justify-content: center;
            font-size: 0.9em;
            color: #aaa;
        }}

        .info span {{
            color: #4fc3f7;
        }}

        .status {{
            text-align: center;
            font-size: 0.85em;
            color: #888;
            margin-top: 10px;
        }}

        #connStatus.connected {{ color: #4caf50; }}
        #connStatus.disconnected {{ color: #f44336; }}

        .piano-keyboard {{ position: relative; height: 120px; display: flex; justify-content: center; }}
        .piano-key {{
            position: relative; border: 1px solid #333; border-radius: 0 0 4px 4px;
            cursor: pointer; display: flex; flex-direction: column;
            justify-content: flex-end; align-items: center; padding-bottom: 4px;
            font-size: 0.65em; font-weight: bold; user-select: none;
        }}
        .piano-key.white {{ width: 44px; height: 110px; background: #556; color: #999; z-index: 1; }}
        .piano-key.black {{ width: 30px; height: 70px; background: #111; color: #666; z-index: 2; margin-left: -15px; margin-right: -15px; }}
        .piano-key.filled {{ color: #4fc3f7; }}
        .piano-key.filled.white {{ background: #668; }}
        .piano-key.filled.black {{ background: #224; }}
        .piano-key.selected.white {{ background: #4fc3f7; color: #1a1a2e; box-shadow: 0 0 10px #4fc3f7; }}
        .piano-key.selected.black {{ background: #4fc3f7; color: #1a1a2e; box-shadow: 0 0 10px #4fc3f7; }}
        .piano-key .note-name {{ font-size: 1.1em; }}
        .piano-key .key-hint {{ font-size: 0.9em; opacity: 0.5; }}

        .controls-with-library {{ display: flex; gap: 15px; justify-content: center; }}
        .controls-with-library > .controls {{ flex: 1; min-width: 0; }}
        .library-panel {{ background: #111827; border: 1px solid #333; border-radius: 8px; width: 200px; display: flex; flex-direction: column; overflow: hidden; flex-shrink: 0; align-self: stretch; }}
        .library-header {{ padding: 8px 10px; font-size: 0.85em; font-weight: bold; color: #4fc3f7; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }}
        .library-header button {{ padding: 2px 8px; font-size: 0.8em; }}
        .library-save-row {{ display: flex; gap: 4px; padding: 6px 8px; border-bottom: 1px solid #333; }}
        .library-save-row input {{ flex: 1; min-width: 0; background: #1a1a2e; border: 1px solid #555; border-radius: 4px; color: #eee; padding: 4px 8px; font-size: 0.8em; outline: none; }}
        .library-save-row input:focus {{ border-color: #4fc3f7; }}
        .library-save-row button {{ padding: 4px 10px; font-size: 0.8em; flex-shrink: 0; }}
        .library-list {{ overflow-y: auto; flex: 1; min-height: 40px; }}
        .lib-item {{ display: flex; align-items: center; padding: 5px 8px; cursor: grab; border-radius: 4px; color: #eee; gap: 6px; font-size: 0.85em; }}
        .lib-item:hover {{ background: #2a2a4e; }}
        .lib-item-name {{ flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .lib-item-badge {{ background: #0e7490; color: #e0f7fa; font-size: 0.7em; padding: 1px 5px; border-radius: 8px; white-space: nowrap; }}
        .lib-delete {{ color: #f44336; background: none; border: none; cursor: pointer; font-size: 1em; padding: 0 4px; width: auto; height: auto; flex-shrink: 0; }}
        .lib-empty {{ color: #888; padding: 8px; text-align: center; font-size: 0.85em; }}
        .piano-key.drag-over {{ outline: 2px dashed #4fc3f7; outline-offset: -2px; }}
    </style>
</head>
<body>
    <div class="main-container">
        <h1>NCA Dynamics Viewer</h1>
        <p class="subtitle">Comparing ground truth simulation vs learned NCA dynamics</p>

        <div class="viewers">
            <div class="viewer" id="gtViewer">
                <div class="viewer-label">Ground Truth</div>
                <canvas id="gtCanvas" width="256" height="256"></canvas>
            </div>
            <div class="viewer">
                <div class="viewer-label">NCA Prediction</div>
                <canvas id="ncaCanvas" width="256" height="256"></canvas>
            </div>
        </div>

        <div class="context-section" id="contextSection">
            <h3>Context Frames (Input to Encoder)</h3>
            <div class="context-frames" id="contextFrames"></div>
        </div>

        <div class="controls-with-library">
            <div class="library-panel" id="libraryPanel">
                <div class="library-header"><span>Latent Library</span><button id="libExportBtn">Export</button></div>
                <div class="library-save-row">
                    <input type="text" id="libSaveInput" placeholder="Name..." />
                    <button id="libSaveBtn">Save</button>
                </div>
                <div class="library-list" id="libraryList"></div>
            </div>

            <div class="controls">
                <div class="control-row">
                    <button id="seqModeBtn" class="active-mode">Sequence</button>
                    <button id="latentModeBtn">Free Latent</button>
                </div>

                <div class="control-row">
                    <button id="playPauseBtn">Pause</button>
                    <button id="stepBtn">Step</button>
                    <button id="resetBtn">Reset</button>
                </div>

                <div class="control-row" id="seqControls">
                    <button id="prevSeqBtn">&lt; Prev</button>
                    <button id="randomSeqBtn">Random Sequence</button>
                    <button id="nextSeqBtn">Next &gt;</button>
                </div>

                <div id="latentControls" style="display:none">
                    <div class="control-row">
                        <button id="randomLatentBtn">Random Latent</button>
                        <button id="perturbLatentBtn">Perturb Latent</button>
                        <div class="slider-group">
                            <label>Perturb:</label>
                            <input type="range" id="perturbSlider" min="0.01" max="1.0" step="0.01" value="0.1">
                            <span class="value" id="perturbVal">0.10</span>
                        </div>
                    </div>
                </div>

                <div class="control-row" style="margin-top: 10px;">
                    <div class="piano-keyboard" id="pianoKeyboard"></div>
                </div>

                <div class="control-row">
                    <div class="slider-group">
                        <label>Speed:</label>
                        <input type="range" id="speedSlider" min="0.1" max="3" step="0.1" value="1">
                        <span class="value" id="speedVal">1.0x</span>
                    </div>
                </div>

                <div class="info">
                    <div id="seqInfo">Sequence: <span id="seqNum">0</span> / <span id="totalSeq">0</span></div>
                    <div>Frame: <span id="frameNum">0</span></div>
                    <div>Latent: <span id="latentSource">encoded</span></div>
                </div>
            </div>
        </div>

        <div class="status">
            <span id="connStatus" class="disconnected">Loading...</span>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    <script>
    (function() {{
    "use strict";

    // -----------------------------------------------------------------------
    // Config (filled from manifest at runtime)
    // -----------------------------------------------------------------------
    let width = 0, height = 0, channels = 0;
    let contextFramesCount = 0, numSteps = 0;
    let latentDim = 0, gridChannels = 0;
    let numSequences = 0;
    let manifest = null;

    // -----------------------------------------------------------------------
    // Runtime state
    // -----------------------------------------------------------------------
    let decodeSession = null, ncaStepSession = null;
    let currentSeqIdx = 0, currentFrameIdx = 0;
    let isPlaying = true;
    let playbackSpeed = 1.0;
    let currentMode = 'sequence';

    // Current latent & decoded params
    let currentZ = null;           // Float32Array [latentDim]
    let cachedParams = null;       // {{ layer1_w, layer1_b, layer2_w, layer2_b }} ort.Tensor
    let currentNcaFrame = null;    // Float32Array [C*H*W] in [0,1]

    // Ground truth data for current sequence
    let gtData = null;             // Uint8Array [T*H*W*C]
    let gtFrameCount = 0;

    // Latent slots (piano)
    const latentSlots = new Array(19).fill(null);   // Float32Array | null
    const slotNames = new Array(19).fill(null);
    const selectedSlots = new Set();
    const filledSlots = new Set();

    // Library: name -> Float32Array
    let latentLibrary = {{}};

    // Canvas refs
    const gtCanvas = document.getElementById('gtCanvas');
    const ncaCanvas = document.getElementById('ncaCanvas');
    const gtCtx = gtCanvas.getContext('2d');
    const ncaCtx = ncaCanvas.getContext('2d');
    let contextCanvases = [];

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------
    function randn() {{
        const u1 = Math.random(), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }}

    function randnArray(n) {{
        const arr = new Float32Array(n);
        for (let i = 0; i < n; i++) arr[i] = randn();
        return arr;
    }}

    function sigmoid(x) {{
        return 1 / (1 + Math.exp(-x));
    }}

    // -----------------------------------------------------------------------
    // Rendering
    // -----------------------------------------------------------------------
    function renderFrame(ctx, canvas, pixels, w, h, c) {{
        const displayW = canvas.width;
        const displayH = canvas.height;
        const imageData = ctx.createImageData(displayW, displayH);
        const scaleX = displayW / w;
        const scaleY = displayH / h;

        for (let y = 0; y < displayH; y++) {{
            for (let x = 0; x < displayW; x++) {{
                const srcX = Math.floor(x / scaleX);
                const srcY = Math.floor(y / scaleY);
                const srcIdx = (srcY * w + srcX) * c;
                const dstIdx = (y * displayW + x) * 4;

                if (c === 3) {{
                    imageData.data[dstIdx]     = pixels[srcIdx];
                    imageData.data[dstIdx + 1] = pixels[srcIdx + 1];
                    imageData.data[dstIdx + 2] = pixels[srcIdx + 2];
                }} else {{
                    imageData.data[dstIdx]     = pixels[srcIdx];
                    imageData.data[dstIdx + 1] = pixels[srcIdx];
                    imageData.data[dstIdx + 2] = pixels[srcIdx];
                }}
                imageData.data[dstIdx + 3] = 255;
            }}
        }}
        ctx.putImageData(imageData, 0, 0);
    }}

    function renderNcaCanvas() {{
        if (!currentNcaFrame) return;
        // currentNcaFrame: float32 [C, H, W] in CHW order, values [0,1]
        const pixels = new Uint8Array(height * width * channels);
        for (let ch = 0; ch < channels; ch++) {{
            for (let y = 0; y < height; y++) {{
                for (let x = 0; x < width; x++) {{
                    const srcIdx = ch * height * width + y * width + x;
                    const dstIdx = (y * width + x) * channels + ch;
                    pixels[dstIdx] = Math.min(255, Math.max(0, Math.round(currentNcaFrame[srcIdx] * 255)));
                }}
            }}
        }}
        renderFrame(ncaCtx, ncaCanvas, pixels, width, height, channels);
    }}

    function renderGtCanvas() {{
        if (!gtData) return;
        const offset = (contextFramesCount + currentFrameIdx) * height * width * channels;
        const end = offset + height * width * channels;
        if (end > gtData.length) {{
            // Past the end — black
            const pixels = new Uint8Array(height * width * channels);
            renderFrame(gtCtx, gtCanvas, pixels, width, height, channels);
            return;
        }}
        const pixels = gtData.subarray(offset, end);
        renderFrame(gtCtx, gtCanvas, pixels, width, height, channels);
    }}

    function renderContextFrames() {{
        if (!gtData) return;
        for (let i = 0; i < contextFramesCount && i < contextCanvases.length; i++) {{
            const offset = i * height * width * channels;
            const end = offset + height * width * channels;
            if (end > gtData.length) break;
            const pixels = gtData.subarray(offset, end);
            const canvas = contextCanvases[i];
            const cctx = canvas.getContext('2d');
            renderFrame(cctx, canvas, pixels, width, height, channels);
        }}
    }}

    function createContextCanvases() {{
        const container = document.getElementById('contextFrames');
        container.innerHTML = '';
        contextCanvases = [];

        for (let i = 0; i < contextFramesCount; i++) {{
            const div = document.createElement('div');
            div.className = 'context-frame';

            const canvas = document.createElement('canvas');
            canvas.width = 64;
            canvas.height = 64;
            canvas.style.cssText = 'border: 1px solid #4fc3f7; image-rendering: pixelated;';

            const label = document.createElement('span');
            label.textContent = 't-' + (contextFramesCount - 1 - i);

            div.appendChild(canvas);
            div.appendChild(label);
            container.appendChild(div);
            contextCanvases.push(canvas);
        }}
    }}

    // -----------------------------------------------------------------------
    // ONNX inference
    // -----------------------------------------------------------------------
    async function decodeLatent(z) {{
        const zTensor = new ort.Tensor('float32', z, [1, latentDim]);
        const results = await decodeSession.run({{ z: zTensor }});

        cachedParams = {{
            layer1_w: results.layer1_w,
            layer1_b: results.layer1_b,
            layer2_w: results.layer2_w,
            layer2_b: results.layer2_b,
        }};

        // first_frame is [1, C, H, W] already sigmoided
        const ff = results.first_frame.data;
        currentNcaFrame = new Float32Array(ff.length);
        currentNcaFrame.set(ff);
    }}

    async function stepNca() {{
        // Build grid: image channels from currentNcaFrame, noise in hidden channels
        const gridSize = gridChannels * height * width;
        const imgSize = channels * height * width;
        const grid = new Float32Array(gridSize);

        // Image channels: raw [0,1] values (matching init_grid "image" mode)
        grid.set(currentNcaFrame);
        // Hidden channels: Gaussian noise * 0.1 (matching init_grid "image" mode)
        for (let i = imgSize; i < gridSize; i++) {{
            grid[i] = randn() * 0.1;
        }}

        let gridTensor = new ort.Tensor('float32', grid, [1, gridChannels, height, width]);
        for (let s = 0; s < numSteps; s++) {{
            const out = await ncaStepSession.run({{
                grid: gridTensor,
                layer1_w: cachedParams.layer1_w,
                layer1_b: cachedParams.layer1_b,
                layer2_w: cachedParams.layer2_w,
                layer2_b: cachedParams.layer2_b,
            }});
            gridTensor = out.new_grid;
        }}

        // Sigmoid on first C channels
        const raw = gridTensor.data;
        currentNcaFrame = new Float32Array(imgSize);
        for (let i = 0; i < imgSize; i++) {{
            currentNcaFrame[i] = sigmoid(raw[i]);
        }}
    }}

    // -----------------------------------------------------------------------
    // Sequence & latent management
    // -----------------------------------------------------------------------
    async function selectSequence(idx) {{
        currentSeqIdx = ((idx % numSequences) + numSequences) % numSequences;
        currentFrameIdx = 0;

        // Load GT binary
        const padded = String(currentSeqIdx).padStart(3, '0');
        const resp = await fetch('data/seq_' + padded + '.bin');
        gtData = new Uint8Array(await resp.arrayBuffer());
        gtFrameCount = manifest.sequences[currentSeqIdx].frames;

        // Pre-encoded latent
        currentZ = new Float32Array(manifest.sequences[currentSeqIdx].z);
        await decodeLatent(currentZ);

        document.getElementById('seqNum').textContent = currentSeqIdx;
        document.getElementById('latentSource').textContent = 'encoded';

        renderContextFrames();
    }}

    function resetNca() {{
        // Re-decode from currentZ to reset first frame
        decodeLatent(currentZ);
        currentFrameIdx = 0;
    }}

    async function randomLatent() {{
        currentZ = randnArray(latentDim);
        await decodeLatent(currentZ);
        currentFrameIdx = 0;
        document.getElementById('latentSource').textContent = 'random';
    }}

    async function perturbLatent(scale) {{
        for (let i = 0; i < latentDim; i++) {{
            currentZ[i] += randn() * scale;
        }}
        await decodeLatent(currentZ);
        currentFrameIdx = 0;
        document.getElementById('latentSource').textContent = 'perturb';
    }}

    async function applySlotSelection() {{
        if (selectedSlots.size === 0) return;
        const mean = new Float32Array(latentDim);
        for (const idx of selectedSlots) {{
            const z = latentSlots[idx];
            for (let i = 0; i < latentDim; i++) mean[i] += z[i];
        }}
        const n = selectedSlots.size;
        for (let i = 0; i < latentDim; i++) mean[i] /= n;
        currentZ = mean;
        currentMode = 'latent';
        await decodeLatent(currentZ);
        currentFrameIdx = 0;
    }}

    // -----------------------------------------------------------------------
    // Library persistence (localStorage)
    // -----------------------------------------------------------------------
    function getLibraryNames() {{
        return Object.keys(latentLibrary).sort();
    }}

    function saveToLibrary(name) {{
        if (!name || !currentZ) return;
        latentLibrary[name] = new Float32Array(currentZ);
        localStorage.setItem('lib_' + name, JSON.stringify(Array.from(currentZ)));
        renderLibraryPanel();
    }}

    function deleteFromLibrary(name) {{
        delete latentLibrary[name];
        localStorage.removeItem('lib_' + name);
        // Clear any slots holding this name
        for (let i = 0; i < 19; i++) {{
            if (slotNames[i] === name) {{
                latentSlots[i] = null;
                slotNames[i] = null;
                selectedSlots.delete(i);
                filledSlots.delete(i);
            }}
        }}
        updateSlotUI();
        renderLibraryPanel();
    }}

    function loadLibraryFromStorage() {{
        // Start with baked-in library from manifest
        if (manifest.library) {{
            for (const [name, arr] of Object.entries(manifest.library)) {{
                latentLibrary[name] = new Float32Array(arr);
            }}
        }}
        // Merge localStorage overrides
        for (let i = 0; i < localStorage.length; i++) {{
            const key = localStorage.key(i);
            if (key.startsWith('lib_')) {{
                const name = key.slice(4);
                try {{
                    const arr = JSON.parse(localStorage.getItem(key));
                    latentLibrary[name] = new Float32Array(arr);
                }} catch(e) {{}}
            }}
        }}
    }}

    function loadSlotsFromManifest() {{
        if (!manifest.slot_assignments) return;
        for (let i = 0; i < 19 && i < manifest.slot_assignments.length; i++) {{
            const name = manifest.slot_assignments[i];
            if (name && latentLibrary[name]) {{
                latentSlots[i] = new Float32Array(latentLibrary[name]);
                slotNames[i] = name;
                filledSlots.add(i);
            }}
        }}
        updateSlotUI();
    }}

    // -----------------------------------------------------------------------
    // Piano keyboard
    // -----------------------------------------------------------------------
    const NOTES = [
        {{ note: 'F',  key: 'a', type: 'white', freq: 349.23 }},
        {{ note: 'F#', key: 'w', type: 'black', freq: 369.99 }},
        {{ note: 'G',  key: 's', type: 'white', freq: 392.00 }},
        {{ note: 'G#', key: 'e', type: 'black', freq: 415.30 }},
        {{ note: 'A',  key: 'd', type: 'white', freq: 440.00 }},
        {{ note: 'A#', key: 'r', type: 'black', freq: 466.16 }},
        {{ note: 'B',  key: 'f', type: 'white', freq: 493.88 }},
        {{ note: 'C',  key: 'g', type: 'white', freq: 523.25 }},
        {{ note: 'C#', key: 'y', type: 'black', freq: 554.37 }},
        {{ note: 'D',  key: 'h', type: 'white', freq: 587.33 }},
        {{ note: 'D#', key: 'u', type: 'black', freq: 622.25 }},
        {{ note: 'E',  key: 'j', type: 'white', freq: 659.25 }},
        {{ note: 'F',  key: 'k', type: 'white', freq: 698.46 }},
        {{ note: 'F#', key: 'o', type: 'black', freq: 739.99 }},
        {{ note: 'G',  key: 'l', type: 'white', freq: 783.99 }},
        {{ note: 'G#', key: 'p', type: 'black', freq: 830.61 }},
        {{ note: 'A',  key: ';', type: 'white', freq: 880.00 }},
        {{ note: 'A#', key: '[', type: 'black', freq: 932.33 }},
        {{ note: 'B',  key: "'", type: 'white', freq: 987.77 }},
    ];
    const keyToSlot = {{}};
    NOTES.forEach((n, i) => {{ keyToSlot[n.key] = i; }});

    // Audio
    let audioCtx = null;
    const activeTones = {{}};
    function ensureAudioCtx() {{
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }}
    function startTone(idx) {{
        if (activeTones[idx]) return;
        ensureAudioCtx();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sine';
        osc.frequency.value = NOTES[idx].freq;
        gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        activeTones[idx] = {{ osc, gain }};
    }}
    function stopTone(idx) {{
        const tone = activeTones[idx];
        if (!tone) return;
        const t = audioCtx.currentTime;
        tone.gain.gain.setValueAtTime(tone.gain.gain.value, t);
        tone.gain.gain.exponentialRampToValueAtTime(0.001, t + 0.08);
        tone.osc.stop(t + 0.08);
        delete activeTones[idx];
    }}
    function playTone(freq) {{
        ensureAudioCtx();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sine';
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.15);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.15);
    }}

    const pianoKeys = [];
    const pianoContainer = document.getElementById('pianoKeyboard');

    for (let i = 0; i < NOTES.length; i++) {{
        const key = document.createElement('div');
        key.className = 'piano-key ' + NOTES[i].type;
        const noteName = document.createElement('div');
        noteName.className = 'note-name';
        noteName.textContent = NOTES[i].note;
        const keyHint = document.createElement('div');
        keyHint.className = 'key-hint';
        keyHint.textContent = NOTES[i].key;
        key.appendChild(noteName);
        key.appendChild(keyHint);

        key.addEventListener('click', () => {{
            playTone(NOTES[i].freq);
            if (filledSlots.has(i)) {{
                if (selectedSlots.has(i)) {{
                    selectedSlots.delete(i);
                }} else {{
                    selectedSlots.add(i);
                }}
                applySlotSelection();
                updateSlotUI();
            }}
        }});

        key.addEventListener('contextmenu', (e) => {{
            e.preventDefault();
            if (filledSlots.has(i)) {{
                latentSlots[i] = null;
                slotNames[i] = null;
                selectedSlots.delete(i);
                filledSlots.delete(i);
                if (selectedSlots.size > 0) applySlotSelection();
                updateSlotUI();
            }}
        }});

        // Drag-and-drop
        key.addEventListener('dragover', (e) => {{ e.preventDefault(); key.classList.add('drag-over'); }});
        key.addEventListener('dragleave', () => {{ key.classList.remove('drag-over'); }});
        key.addEventListener('drop', (e) => {{
            e.preventDefault();
            key.classList.remove('drag-over');
            const name = e.dataTransfer.getData('text/plain');
            if (name && latentLibrary[name]) {{
                latentSlots[i] = new Float32Array(latentLibrary[name]);
                slotNames[i] = name;
                filledSlots.add(i);
                selectedSlots.add(i);
                applySlotSelection();
                updateSlotUI();
            }}
        }});

        pianoContainer.appendChild(key);
        pianoKeys.push(key);
    }}

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {{
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        if (e.repeat) return;
        const idx = keyToSlot[e.key];
        if (idx !== undefined && filledSlots.has(idx)) {{
            startTone(idx);
            selectedSlots.add(idx);
            applySlotSelection();
            updateSlotUI();
        }}
    }});
    document.addEventListener('keyup', (e) => {{
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        const idx = keyToSlot[e.key];
        if (idx !== undefined) {{
            stopTone(idx);
            selectedSlots.delete(idx);
            if (selectedSlots.size > 0) applySlotSelection();
            updateSlotUI();
        }}
    }});

    function updateSlotUI() {{
        for (let i = 0; i < NOTES.length; i++) {{
            pianoKeys[i].className = 'piano-key ' + NOTES[i].type
                + (filledSlots.has(i) ? ' filled' : '')
                + (selectedSlots.has(i) ? ' selected' : '');
            pianoKeys[i].title = slotNames[i] || '';
        }}
        // Update latent source display
        const sel = Array.from(selectedSlots).sort((a,b) => a - b);
        if (sel.length > 1) {{
            document.getElementById('latentSource').textContent =
                'slots ' + sel.map(i => NOTES[i].note).join('+') + ' (interp)';
        }} else if (sel.length === 1) {{
            document.getElementById('latentSource').textContent = NOTES[sel[0]].note;
        }}
        renderLibraryPanel();
    }}

    // -----------------------------------------------------------------------
    // Library panel
    // -----------------------------------------------------------------------
    function renderLibraryPanel() {{
        const list = document.getElementById('libraryList');
        list.innerHTML = '';

        const nameToNotes = {{}};
        for (let i = 0; i < slotNames.length; i++) {{
            if (slotNames[i]) {{
                if (!nameToNotes[slotNames[i]]) nameToNotes[slotNames[i]] = [];
                nameToNotes[slotNames[i]].push(NOTES[i].note);
            }}
        }}

        const names = getLibraryNames();
        if (names.length === 0) {{
            const empty = document.createElement('div');
            empty.className = 'lib-empty';
            empty.textContent = 'No saved latents yet.';
            list.appendChild(empty);
            return;
        }}

        names.forEach(name => {{
            const item = document.createElement('div');
            item.className = 'lib-item';
            item.draggable = true;
            item.addEventListener('dragstart', (e) => {{
                e.dataTransfer.setData('text/plain', name);
                e.dataTransfer.effectAllowed = 'copy';
                item.style.opacity = '0.5';
            }});
            item.addEventListener('dragend', () => {{ item.style.opacity = ''; }});

            const nameSpan = document.createElement('span');
            nameSpan.className = 'lib-item-name';
            nameSpan.textContent = name;
            item.appendChild(nameSpan);

            if (nameToNotes[name]) {{
                nameToNotes[name].forEach(note => {{
                    const badge = document.createElement('span');
                    badge.className = 'lib-item-badge';
                    badge.textContent = note;
                    item.appendChild(badge);
                }});
            }}

            const del = document.createElement('button');
            del.className = 'lib-delete';
            del.textContent = '\\u00d7';
            del.title = 'Delete from library';
            del.addEventListener('click', (e) => {{
                e.stopPropagation();
                deleteFromLibrary(name);
            }});
            item.appendChild(del);
            list.appendChild(item);
        }});
    }}

    // Save input wiring
    const libSaveInput = document.getElementById('libSaveInput');
    const libSaveBtn = document.getElementById('libSaveBtn');
    function saveFromInput() {{
        const name = libSaveInput.value.trim();
        if (name) {{
            saveToLibrary(name);
            libSaveInput.value = '';
        }}
    }}
    libSaveBtn.addEventListener('click', saveFromInput);
    libSaveInput.addEventListener('keydown', (e) => {{
        e.stopPropagation();
        if (e.key === 'Enter') saveFromInput();
    }});
    libSaveInput.addEventListener('keyup', (e) => {{ e.stopPropagation(); }});

    // Export entire library as JSON download
    function downloadLibrary() {{
        const lib = {{}};
        for (const [name, arr] of Object.entries(latentLibrary)) {{
            lib[name] = Array.from(arr);
        }}
        const blob = new Blob([JSON.stringify(lib, null, 2)], {{ type: 'application/json' }});
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'latent_library.json';
        a.click();
        URL.revokeObjectURL(a.href);
    }}
    document.getElementById('libExportBtn').addEventListener('click', downloadLibrary);

    // -----------------------------------------------------------------------
    // Mode switching
    // -----------------------------------------------------------------------
    function setMode(mode) {{
        currentMode = mode;
        const isSeq = mode === 'sequence';
        document.getElementById('gtViewer').style.display = isSeq ? '' : 'none';
        document.getElementById('contextSection').style.display = isSeq ? '' : 'none';
        document.getElementById('seqControls').style.display = isSeq ? '' : 'none';
        document.getElementById('latentControls').style.display = isSeq ? 'none' : '';
        document.getElementById('seqInfo').style.display = isSeq ? '' : 'none';
        document.getElementById('seqModeBtn').classList.toggle('active-mode', isSeq);
        document.getElementById('latentModeBtn').classList.toggle('active-mode', !isSeq);
        if (!isSeq) {{
            randomLatent();
        }} else {{
            selectSequence(currentSeqIdx);
        }}
    }}

    document.getElementById('seqModeBtn').addEventListener('click', () => setMode('sequence'));
    document.getElementById('latentModeBtn').addEventListener('click', () => setMode('latent'));

    // -----------------------------------------------------------------------
    // Controls
    // -----------------------------------------------------------------------
    document.getElementById('playPauseBtn').addEventListener('click', () => {{
        isPlaying = !isPlaying;
        document.getElementById('playPauseBtn').textContent = isPlaying ? 'Pause' : 'Play';
    }});

    document.getElementById('stepBtn').addEventListener('click', async () => {{
        await stepNca();
        currentFrameIdx++;
        document.getElementById('frameNum').textContent = currentFrameIdx;
        renderNcaCanvas();
        if (currentMode === 'sequence') renderGtCanvas();
    }});

    document.getElementById('resetBtn').addEventListener('click', () => {{
        resetNca();
        currentFrameIdx = 0;
        document.getElementById('frameNum').textContent = 0;
    }});

    document.getElementById('prevSeqBtn').addEventListener('click', () => {{
        selectSequence(currentSeqIdx - 1);
    }});

    document.getElementById('nextSeqBtn').addEventListener('click', () => {{
        selectSequence(currentSeqIdx + 1);
    }});

    document.getElementById('randomSeqBtn').addEventListener('click', () => {{
        selectSequence(Math.floor(Math.random() * numSequences));
    }});

    document.getElementById('speedSlider').addEventListener('input', (e) => {{
        playbackSpeed = parseFloat(e.target.value);
        document.getElementById('speedVal').textContent = playbackSpeed.toFixed(1) + 'x';
    }});

    document.getElementById('randomLatentBtn').addEventListener('click', () => {{
        randomLatent();
    }});

    document.getElementById('perturbLatentBtn').addEventListener('click', () => {{
        const scale = parseFloat(document.getElementById('perturbSlider').value);
        perturbLatent(scale);
    }});

    document.getElementById('perturbSlider').addEventListener('input', (e) => {{
        document.getElementById('perturbVal').textContent = parseFloat(e.target.value).toFixed(2);
    }});

    // -----------------------------------------------------------------------
    // Simulation loop
    // -----------------------------------------------------------------------
    let frameCounter = 0;

    async function simulationLoop() {{
        while (true) {{
            if (isPlaying) {{
                frameCounter += playbackSpeed;
                while (frameCounter >= 1) {{
                    await stepNca();
                    currentFrameIdx++;
                    frameCounter -= 1;
                }}
            }}

            renderNcaCanvas();
            if (currentMode === 'sequence') renderGtCanvas();
            document.getElementById('frameNum').textContent = currentFrameIdx;

            await new Promise(r => setTimeout(r, 1000 / 30));
        }}
    }}

    // -----------------------------------------------------------------------
    // Initialization
    // -----------------------------------------------------------------------
    async function init() {{
        const statusEl = document.getElementById('connStatus');
        statusEl.textContent = 'Loading manifest...';

        try {{
            manifest = await (await fetch('data/manifest.json')).json();
        }} catch (e) {{
            statusEl.textContent = 'Failed to load manifest.json';
            statusEl.className = 'disconnected';
            return;
        }}

        // Store config
        width = manifest.width;
        height = manifest.height;
        channels = manifest.channels;
        contextFramesCount = manifest.context_frames;
        numSteps = manifest.num_steps;
        latentDim = manifest.latent_dim;
        gridChannels = manifest.grid_channels;
        numSequences = manifest.num_sequences;

        document.getElementById('totalSeq').textContent = numSequences;
        createContextCanvases();

        // Load ONNX models
        statusEl.textContent = 'Loading ONNX models...';
        try {{
            decodeSession = await ort.InferenceSession.create('onnx/decode_latent.onnx');
            ncaStepSession = await ort.InferenceSession.create('onnx/nca_step.onnx');
        }} catch (e) {{
            statusEl.textContent = 'Failed to load ONNX models: ' + e.message;
            statusEl.className = 'disconnected';
            console.error(e);
            return;
        }}

        // Initialize library
        loadLibraryFromStorage();
        loadSlotsFromManifest();
        renderLibraryPanel();

        // Load first sequence
        statusEl.textContent = 'Loading first sequence...';
        await selectSequence(0);

        statusEl.textContent = 'Ready';
        statusEl.className = 'connected';

        simulationLoop();
    }}

    init();
    }})();
    </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build a static web app from an NCA dynamics checkpoint"
    )
    parser.add_argument("checkpoint", type=str, help="Path to dynamics model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to sequence data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="./static_build", help="Output directory")
    parser.add_argument(
        "--max-sequences", type=int, default=0,
        help="Max sequences to export (0 = all)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Static Build: NCA Dynamics Web App")
    print("=" * 60)

    # 1a. Load model + data
    print("\n[1/5] Loading model and data...")
    model, cfg, device = load_model(args.checkpoint, args.device)
    dataset = load_data(args.data, cfg["context_frames"])

    # Load latent library from disk
    library = load_library(args.checkpoint, device)

    # 1b. Pre-encode all sequences
    print("\n[2/5] Encoding sequences...")
    latents = encode_sequences(model, dataset, cfg, device, args.max_sequences)

    # 1c. Export ONNX models
    print("\n[3/5] Exporting ONNX models...")
    export_onnx_models(model, cfg, device, output_dir)

    # 1d. Export data
    n = len(latents)
    print(f"\n[4/5] Exporting data ({n} sequences)...")
    export_data(dataset, latents, library, cfg, output_dir, args.max_sequences)

    # 1e. Generate HTML
    print("\n[5/5] Generating index.html...")
    generate_html(cfg, output_dir)

    print("\n" + "=" * 60)
    print(f"Build complete! Output in: {output_dir}")
    print(f"  To serve: cd {output_dir} && python -m http.server 8080")
    print("=" * 60)


if __name__ == "__main__":
    main()
