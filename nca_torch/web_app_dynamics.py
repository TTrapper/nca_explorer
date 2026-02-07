"""
Real-time web visualization for NCA Dynamics model.
Shows side-by-side comparison of ground truth vs NCA rollout.
"""

import argparse
import asyncio
import base64
import torch
import numpy as np
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from autoencoder import NCAAutoencoder
from datasets import SequenceDataset


# Global state
model = None
device = None
dataset = None
grid = None
nca_params = None
current_z = None  # Current latent vector [1, latent_dim]
current_nca_frame = None  # Latest NCA output [1, C, H, W] in [0, 1]
current_seq_idx = 0
current_frame_idx = 0
context_frames_count = 4
in_channels = 3
grid_size = (32, 32)
num_steps = 1  # NCA steps per frame (loaded from checkpoint)
is_playing = True
playback_speed = 1.0
current_mode = 'sequence'  # 'sequence' or 'latent'

# Latent slots
latent_slots = [None] * 12      # List of saved latent tensors (or None)
selected_slots = set()           # Set of selected slot indices
slot_names = [None] * 12        # Name of latent loaded in each slot

# Latent library (persistent on disk)
latent_save_dir = None           # Path to latents/ directory
latent_library = {}              # name -> tensor, loaded from disk

# Current sequence data
gt_sequence = None  # Ground truth frames
context_stacked = None  # For encoding
init_frame = None  # Initial frame for NCA

app = FastAPI()


def load_model(checkpoint_path: str, dev: str = "cpu"):
    """Load trained dynamics model from checkpoint."""
    global model, device, context_frames_count, in_channels, grid_size, num_steps
    global latent_save_dir, latent_library

    device = torch.device(dev)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})

    # Get model config from checkpoint
    context_frames_count = checkpoint.get("context_frames", args.get("context_frames", 4))
    in_channels = checkpoint.get("in_channels", args.get("in_channels", 3))
    grid_size = checkpoint.get("grid_size", args.get("grid_size", (32, 32)))
    num_steps = checkpoint.get("num_steps", args.get("num_steps", 1))
    if isinstance(grid_size, list):
        grid_size = tuple(grid_size)

    model = NCAAutoencoder(
        latent_dim=args.get("latent_dim", 64),
        grid_channels=args.get("grid_channels", 16),
        hidden_dim=args.get("hidden_dim", 256),
        use_vae=not args.get("no_vae", False),
        in_channels=in_channels,
        grid_size=grid_size,
        context_frames=context_frames_count,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded dynamics model from {checkpoint_path}")
    print(f"  Context frames: {context_frames_count}")
    print(f"  Channels: {in_channels}")
    print(f"  Grid size: {grid_size}")
    print(f"  NCA steps per frame: {num_steps}")

    # Initialize latent library from disk
    latent_save_dir = Path(checkpoint_path).parent / "latents"
    latent_save_dir.mkdir(exist_ok=True)
    latent_library = {}
    for f in latent_save_dir.glob("*.pt"):
        latent_library[f.stem] = torch.load(f, map_location=device, weights_only=True)
    print(f"  Latent library: {len(latent_library)} saved latents")

    return model


def load_data(data_path: str):
    """Load sequence dataset."""
    global dataset

    dataset = SequenceDataset(
        data_path,
        context_frames=context_frames_count,
        future_frames=50,  # Load enough for visualization
    )
    print(f"Loaded {dataset.num_sequences} sequences")


def select_sequence(idx: int):
    """Select a sequence and prepare for visualization."""
    global current_seq_idx, gt_sequence, context_stacked, init_frame, grid, nca_params, current_z, current_nca_frame, current_frame_idx, current_mode
    current_mode = 'sequence'

    current_seq_idx = idx % dataset.num_sequences
    current_frame_idx = 0

    # Get raw sequence data
    seq = dataset.sequences[current_seq_idx]  # [T, H, W, C]
    T = seq.shape[0]

    # Convert to tensor [T, C, H, W]
    gt_sequence = torch.from_numpy(seq).float().permute(0, 3, 1, 2).to(device) / 255.0

    # Get context frames (first N frames)
    N = context_frames_count
    context_frames = gt_sequence[:N]  # [N, C, H, W]
    context_stacked = context_frames.reshape(1, -1, *grid_size)  # [1, N*C, H, W]
    init_frame = context_frames[-1:].clone()  # [1, C, H, W]

    # Encode and generate NCA params
    with torch.no_grad():
        current_z, _, _ = model.encode(context_stacked)
        nca_params = model.decoder.generate_params(current_z)

        _reset_grid()


def _reset_grid():
    """Reset grid and current_nca_frame.

    Sequence mode: init from last context frame (pure image).
    Free latent mode: init from pure noise.
    """
    global grid, current_nca_frame, current_frame_idx
    current_frame_idx = 0
    with torch.no_grad():
        if current_mode == 'latent':
            init_images = torch.rand(1, in_channels, *grid_size, device=device)
        else:
            init_images = init_frame

        grid = model.decoder.init_grid(
            batch_size=1,
            grid_size=grid_size,
            device=device,
            init_mode="image",
            init_images=init_images,
        )
        current_nca_frame = grid[:, :in_channels].clamp(0, 1).clone()


def step_nca():
    """Run NCA steps for one frame, matching training behavior.

    During training, each frame prediction re-initializes the grid from the previous
    sigmoid output, runs num_steps NCA steps, then extracts sigmoid output.
    Without this re-init, grid values grow unboundedly and diverge.
    """
    global grid, current_nca_frame
    with torch.no_grad():
        # Re-init grid from previous output (matches training's per-step decode)
        grid = model.decoder.init_grid(
            batch_size=1,
            grid_size=grid_size,
            device=device,
            init_mode="image",
            init_images=current_nca_frame,
        )
        layer1_w, layer1_b, layer2_w, layer2_b = nca_params
        # Run num_steps NCA steps per frame (matches training)
        for _ in range(num_steps):
            grid = model.decoder.nca_step(grid, layer1_w, layer1_b, layer2_w, layer2_b)
        current_nca_frame = torch.sigmoid(grid[:, :in_channels])


def reset_nca():
    """Reset NCA to initial state."""
    _reset_grid()


def apply_latent():
    """Regenerate NCA params from current_z and reinitialize the grid."""
    global nca_params, current_mode
    current_mode = 'latent'
    with torch.no_grad():
        nca_params = model.decoder.generate_params(current_z)
    _reset_grid()


def random_latent():
    """Replace the latent with a random sample from the prior."""
    global current_z
    with torch.no_grad():
        current_z = torch.randn(1, model.latent_dim, device=device)
    apply_latent()


def perturb_latent(scale: float = 0.1):
    """Add Gaussian noise to the current latent."""
    global current_z
    with torch.no_grad():
        current_z = current_z + torch.randn_like(current_z) * scale
    apply_latent()


def apply_slot_selection():
    """Set current_z to the mean of all selected slot latents and apply."""
    global current_z
    if not selected_slots:
        return
    tensors = [latent_slots[i] for i in selected_slots]
    with torch.no_grad():
        current_z = torch.stack(tensors).mean(dim=0)
    apply_latent()


def get_slots_state() -> dict:
    """Return JSON-serializable slot state."""
    return {
        "type": "slots_updated",
        "filled": [i for i in range(12) if latent_slots[i] is not None],
        "selected": sorted(selected_slots),
        "names": list(slot_names),
    }


def get_library_state() -> dict:
    """Return JSON-serializable library state."""
    return {
        "type": "library_updated",
        "names": sorted(latent_library.keys()),
    }


def get_frame_data() -> dict:
    """Get current frames as base64 or raw bytes."""
    global current_frame_idx

    H, W = grid_size
    C = in_channels

    # NCA prediction (current_nca_frame is already in [0, 1] from sigmoid)
    with torch.no_grad():
        nca_img = (current_nca_frame[0].permute(1, 2, 0) * 255).byte().cpu().numpy()

    # Ground truth (offset by context frames)
    gt_idx = context_frames_count + current_frame_idx
    if gt_idx < gt_sequence.shape[0]:
        gt_img = (gt_sequence[gt_idx].permute(1, 2, 0) * 255).byte().cpu().numpy()
    else:
        gt_img = np.zeros((H, W, C), dtype=np.uint8)

    # Context frames (for display)
    context_imgs = []
    for i in range(context_frames_count):
        ctx_img = (gt_sequence[i].permute(1, 2, 0) * 255).byte().cpu().numpy()
        context_imgs.append(ctx_img.tobytes())

    return {
        "nca": nca_img.tobytes(),
        "gt": gt_img.tobytes(),
        "context": context_imgs,
        "frame_idx": current_frame_idx,
        "width": W,
        "height": H,
        "channels": C,
    }


# WebSocket clients
clients: list[WebSocket] = []


async def send_context_frames(websocket: WebSocket):
    """Send context frames to client as base64 in a single JSON message."""
    H, W = grid_size
    C = in_channels

    frames_b64 = []
    for i in range(context_frames_count):
        ctx_img = (gt_sequence[i].permute(1, 2, 0) * 255).byte().cpu().numpy()
        frames_b64.append(base64.b64encode(ctx_img.tobytes()).decode('ascii'))
    await websocket.send_json({
        "type": "context_frames",
        "frames": frames_b64,
        "width": W,
        "height": H,
        "channels": C,
    })


async def send_sequence_info(websocket: WebSocket):
    """Send sequence change notification and context frames."""
    await websocket.send_json({
        "type": "sequence_changed",
        "current_seq": current_seq_idx,
    })
    await send_context_frames(websocket)


@app.get("/")
async def get_index():
    return HTMLResponse(HTML_CONTENT)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global is_playing, playback_speed, current_seq_idx

    await websocket.accept()

    # Send initial config before adding to clients list,
    # so the simulation loop doesn't send interleaved data
    await websocket.send_json({
        "type": "init",
        "width": grid_size[1],
        "height": grid_size[0],
        "channels": in_channels,
        "context_frames": context_frames_count,
        "num_sequences": dataset.num_sequences,
        "current_seq": current_seq_idx,
    })

    clients.append(websocket)

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "play":
                is_playing = True

            elif data["type"] == "pause":
                is_playing = False

            elif data["type"] == "reset":
                reset_nca()

            elif data["type"] == "step":
                # Manual single step
                step_nca()
                current_frame_idx += 1

            elif data["type"] == "set_speed":
                playback_speed = data["speed"]

            elif data["type"] == "next_sequence":
                select_sequence(current_seq_idx + 1)
                await send_sequence_info(websocket)

            elif data["type"] == "prev_sequence":
                select_sequence(current_seq_idx - 1)
                await send_sequence_info(websocket)

            elif data["type"] == "select_sequence":
                select_sequence(data["index"])
                await send_sequence_info(websocket)

            elif data["type"] == "random_sequence":
                select_sequence(np.random.randint(0, dataset.num_sequences))
                await send_sequence_info(websocket)

            elif data["type"] == "random_latent":
                random_latent()
                await websocket.send_json({"type": "latent_changed", "source": "random"})

            elif data["type"] == "perturb_latent":
                scale = data.get("scale", 0.1)
                perturb_latent(scale)
                await websocket.send_json({"type": "latent_changed", "source": "perturb"})

            elif data["type"] == "toggle_slot":
                idx = int(data["index"])
                if latent_slots[idx] is not None:
                    if idx in selected_slots:
                        selected_slots.discard(idx)
                    else:
                        selected_slots.add(idx)
                    apply_slot_selection()
                await websocket.send_json(get_slots_state())

            elif data["type"] == "clear_slot":
                idx = int(data["index"])
                latent_slots[idx] = None
                slot_names[idx] = None
                selected_slots.discard(idx)
                apply_slot_selection()
                await websocket.send_json(get_slots_state())

            elif data["type"] == "save_to_library":
                name = data["name"].strip()
                if name:
                    torch.save(current_z.clone().cpu(), latent_save_dir / f"{name}.pt")
                    latent_library[name] = current_z.clone()
                    await websocket.send_json(get_library_state())

            elif data["type"] == "load_slot":
                idx = int(data["index"])
                name = data["name"]
                if name in latent_library:
                    latent_slots[idx] = latent_library[name].clone().to(device)
                    slot_names[idx] = name
                    selected_slots.add(idx)
                    apply_slot_selection()
                    await websocket.send_json(get_slots_state())

            elif data["type"] == "delete_saved":
                name = data["name"]
                # Remove from disk and memory
                pt_file = latent_save_dir / f"{name}.pt"
                if pt_file.exists():
                    pt_file.unlink()
                latent_library.pop(name, None)
                # Clear any slots that held this latent
                for i in range(12):
                    if slot_names[i] == name:
                        latent_slots[i] = None
                        slot_names[i] = None
                        selected_slots.discard(i)
                apply_slot_selection()
                await websocket.send_json(get_library_state())
                await websocket.send_json(get_slots_state())

            elif data["type"] == "list_library":
                await websocket.send_json(get_library_state())

            elif data["type"] == "get_context":
                await send_context_frames(websocket)

    except WebSocketDisconnect:
        if websocket in clients:
            clients.remove(websocket)


async def simulation_loop():
    """Main simulation loop."""
    global current_frame_idx

    frame_counter = 0

    while True:
        if is_playing:
            frame_counter += playback_speed
            while frame_counter >= 1:
                step_nca()
                current_frame_idx += 1
                frame_counter -= 1

        # Send frame data to all clients
        frame_data = get_frame_data()
        for client in clients[:]:
            try:
                # Send as binary message with header
                header = f"{frame_data['frame_idx']},{frame_data['width']},{frame_data['height']},{frame_data['channels']}"
                await client.send_text(f"frame:{header}")
                await client.send_bytes(frame_data['nca'])
                await client.send_bytes(frame_data['gt'])
            except:
                if client in clients:
                    clients.remove(client)

        await asyncio.sleep(1/30)  # 30 FPS


@app.on_event("startup")
async def startup():
    select_sequence(0)
    asyncio.create_task(simulation_loop())


HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NCA Dynamics Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #1a1a2e;
            color: #eee;
            font-family: 'Segoe UI', system-ui, sans-serif;
            min-height: 100vh;
            padding: 20px;
        }
        h1 { color: #4fc3f7; margin-bottom: 5px; text-align: center; }
        .subtitle { color: #888; text-align: center; margin-bottom: 20px; }

        .main-container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .viewers {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .viewer {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .viewer-label {
            font-size: 1.1em;
            margin-bottom: 10px;
            color: #4fc3f7;
        }

        canvas {
            border: 2px solid #4fc3f7;
            background: #000;
            image-rendering: pixelated;
        }

        .context-section {
            background: #16213e;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        .context-section h3 {
            color: #4fc3f7;
            margin-bottom: 10px;
            font-size: 0.9em;
        }

        .context-frames {
            display: flex;
            gap: 10px;
            justify-content: center;
        }

        .context-frame {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .context-frame span {
            font-size: 0.75em;
            color: #888;
            margin-top: 5px;
        }

        .controls {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .control-row {
            display: flex;
            gap: 15px;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
        }

        button {
            background: #4fc3f7;
            color: #1a1a2e;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            font-size: 0.9em;
        }

        button:hover { background: #81d4fa; }
        button:disabled { background: #555; cursor: not-allowed; }
        button.active-mode { background: #fff; }

        .slider-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .slider-group label {
            color: #aaa;
            font-size: 0.9em;
            min-width: 80px;
        }

        .slider-group input[type="range"] {
            width: 150px;
        }

        .slider-group .value {
            color: #4fc3f7;
            font-family: monospace;
            min-width: 40px;
        }

        .info {
            display: flex;
            gap: 30px;
            justify-content: center;
            font-size: 0.9em;
            color: #aaa;
        }

        .info span {
            color: #4fc3f7;
        }

        .status {
            text-align: center;
            font-size: 0.85em;
            color: #888;
            margin-top: 10px;
        }

        #connStatus.connected { color: #4caf50; }
        #connStatus.disconnected { color: #f44336; }

        .latent-slots { display: flex; gap: 6px; flex-wrap: wrap; justify-content: center; }
        .slot-btn {
            width: 36px; height: 36px; border-radius: 5px;
            border: 2px dashed #555; background: transparent;
            color: #666; font-size: 0.8em; cursor: pointer;
            padding: 0; font-weight: bold;
        }
        .slot-btn.filled { border: 2px solid #4fc3f7; color: #4fc3f7; }
        .slot-btn.selected { background: #4fc3f7; color: #1a1a2e; border-color: #81d4fa; box-shadow: 0 0 8px #4fc3f7; }

        .latent-library-popup { position: absolute; background: #16213e; border: 2px solid #4fc3f7; border-radius: 8px; padding: 8px; z-index: 100; min-width: 180px; max-height: 250px; overflow-y: auto; }
        .lib-item { display: flex; justify-content: space-between; align-items: center; padding: 6px 8px; cursor: pointer; border-radius: 4px; color: #eee; }
        .lib-item:hover { background: #2a2a4e; }
        .lib-delete { color: #f44336; background: none; border: none; cursor: pointer; font-size: 1em; padding: 0 4px; width: auto; height: auto; }
        .lib-empty { color: #888; padding: 8px; text-align: center; font-size: 0.85em; }
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

        <div class="controls">
            <div class="control-row">
                <button id="seqModeBtn" class="active-mode">Sequence</button>
                <button id="latentModeBtn">Free Latent</button>
            </div>

            <div class="control-row">
                <button id="playPauseBtn">Pause</button>
                <button id="stepBtn">Step</button>
                <button id="resetBtn">Reset</button>
                <button id="saveLatentBtn">Save Latent</button>
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
                <div class="control-row" style="margin-top: 10px;">
                    <div class="latent-slots" id="latentSlots"></div>
                </div>
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

        <div class="status">
            <span id="connStatus" class="disconnected">Connecting...</span>
        </div>
    </div>

    <script>
        const gtCanvas = document.getElementById('gtCanvas');
        const ncaCanvas = document.getElementById('ncaCanvas');
        const gtCtx = gtCanvas.getContext('2d');
        const ncaCtx = ncaCanvas.getContext('2d');

        let ws = null;
        let isPlaying = true;
        let width = 32, height = 32, channels = 3;
        let contextFramesCount = 4;
        let contextCanvases = [];
        let currentMode = 'sequence';
        let currentSeqIdx = 0;

        function createContextCanvases() {
            const container = document.getElementById('contextFrames');
            container.innerHTML = '';
            contextCanvases = [];

            for (let i = 0; i < contextFramesCount; i++) {
                const div = document.createElement('div');
                div.className = 'context-frame';

                const canvas = document.createElement('canvas');
                canvas.width = 64;
                canvas.height = 64;
                canvas.style.cssText = 'border: 1px solid #4fc3f7; image-rendering: pixelated;';

                const label = document.createElement('span');
                label.textContent = `t-${contextFramesCount - 1 - i}`;

                div.appendChild(canvas);
                div.appendChild(label);
                container.appendChild(div);
                contextCanvases.push(canvas);
            }
        }

        function renderFrame(ctx, canvas, pixels, w, h, c) {
            const displayW = canvas.width;
            const displayH = canvas.height;
            const imageData = ctx.createImageData(displayW, displayH);

            const scaleX = displayW / w;
            const scaleY = displayH / h;

            for (let y = 0; y < displayH; y++) {
                for (let x = 0; x < displayW; x++) {
                    const srcX = Math.floor(x / scaleX);
                    const srcY = Math.floor(y / scaleY);
                    const srcIdx = (srcY * w + srcX) * c;
                    const dstIdx = (y * displayW + x) * 4;

                    if (c === 3) {
                        imageData.data[dstIdx] = pixels[srcIdx];
                        imageData.data[dstIdx + 1] = pixels[srcIdx + 1];
                        imageData.data[dstIdx + 2] = pixels[srcIdx + 2];
                    } else {
                        imageData.data[dstIdx] = pixels[srcIdx];
                        imageData.data[dstIdx + 1] = pixels[srcIdx];
                        imageData.data[dstIdx + 2] = pixels[srcIdx];
                    }
                    imageData.data[dstIdx + 3] = 255;
                }
            }
            ctx.putImageData(imageData, 0, 0);
        }

        // Mode switching
        function setMode(mode) {
            currentMode = mode;
            const isSeq = mode === 'sequence';
            document.getElementById('gtViewer').style.display = isSeq ? '' : 'none';
            document.getElementById('contextSection').style.display = isSeq ? '' : 'none';
            document.getElementById('seqControls').style.display = isSeq ? '' : 'none';
            document.getElementById('latentControls').style.display = isSeq ? 'none' : '';
            document.getElementById('seqInfo').style.display = isSeq ? '' : 'none';
            document.getElementById('seqModeBtn').classList.toggle('active-mode', isSeq);
            document.getElementById('latentModeBtn').classList.toggle('active-mode', !isSeq);
            if (!isSeq) {
                if (ws) ws.send(JSON.stringify({ type: 'random_latent' }));
            } else {
                if (ws) ws.send(JSON.stringify({ type: 'select_sequence', index: currentSeqIdx }));
            }
        }

        document.getElementById('seqModeBtn').addEventListener('click', () => setMode('sequence'));
        document.getElementById('latentModeBtn').addEventListener('click', () => setMode('latent'));

        // Controls
        document.getElementById('playPauseBtn').addEventListener('click', () => {
            isPlaying = !isPlaying;
            document.getElementById('playPauseBtn').textContent = isPlaying ? 'Pause' : 'Play';
            if (ws) ws.send(JSON.stringify({ type: isPlaying ? 'play' : 'pause' }));
        });

        document.getElementById('stepBtn').addEventListener('click', () => {
            if (ws) ws.send(JSON.stringify({ type: 'step' }));
        });

        document.getElementById('resetBtn').addEventListener('click', () => {
            if (ws) ws.send(JSON.stringify({ type: 'reset' }));
        });

        document.getElementById('prevSeqBtn').addEventListener('click', () => {
            if (ws) ws.send(JSON.stringify({ type: 'prev_sequence' }));
        });

        document.getElementById('nextSeqBtn').addEventListener('click', () => {
            if (ws) ws.send(JSON.stringify({ type: 'next_sequence' }));
        });

        document.getElementById('randomSeqBtn').addEventListener('click', () => {
            if (ws) ws.send(JSON.stringify({ type: 'random_sequence' }));
        });

        document.getElementById('speedSlider').addEventListener('input', (e) => {
            const speed = parseFloat(e.target.value);
            document.getElementById('speedVal').textContent = speed.toFixed(1) + 'x';
            if (ws) ws.send(JSON.stringify({ type: 'set_speed', speed }));
        });

        document.getElementById('randomLatentBtn').addEventListener('click', () => {
            if (ws) ws.send(JSON.stringify({ type: 'random_latent' }));
        });

        document.getElementById('perturbLatentBtn').addEventListener('click', () => {
            const scale = parseFloat(document.getElementById('perturbSlider').value);
            if (ws) ws.send(JSON.stringify({ type: 'perturb_latent', scale }));
        });

        document.getElementById('perturbSlider').addEventListener('input', (e) => {
            document.getElementById('perturbVal').textContent = parseFloat(e.target.value).toFixed(2);
        });

        // Save Latent button
        document.getElementById('saveLatentBtn').addEventListener('click', () => {
            const name = prompt('Name for this latent:');
            if (name && name.trim() && ws) {
                ws.send(JSON.stringify({ type: 'save_to_library', name: name.trim() }));
            }
        });

        // Latent slots
        const slotButtons = [];
        const slotContainer = document.getElementById('latentSlots');
        const filledSlots = new Set();
        const selectedSlots = new Set();
        let pendingSlotIndex = null;
        let libraryNames = [];
        let libraryPopup = null;

        for (let i = 0; i < 12; i++) {
            const btn = document.createElement('button');
            btn.className = 'slot-btn';
            btn.textContent = i + 1;
            btn.addEventListener('click', () => {
                if (!ws) return;
                if (filledSlots.has(i)) {
                    ws.send(JSON.stringify({ type: 'toggle_slot', index: i }));
                } else {
                    pendingSlotIndex = i;
                    ws.send(JSON.stringify({ type: 'list_library' }));
                }
            });
            btn.addEventListener('contextmenu', (e) => {
                e.preventDefault();
                if (!ws) return;
                if (filledSlots.has(i)) {
                    ws.send(JSON.stringify({ type: 'clear_slot', index: i }));
                }
            });
            slotContainer.appendChild(btn);
            slotButtons.push(btn);
        }

        function updateSlotUI(filled, selected, names) {
            filledSlots.clear();
            selectedSlots.clear();
            filled.forEach(i => filledSlots.add(i));
            selected.forEach(i => selectedSlots.add(i));
            for (let i = 0; i < 12; i++) {
                slotButtons[i].className = 'slot-btn'
                    + (filledSlots.has(i) ? ' filled' : '')
                    + (selectedSlots.has(i) ? ' selected' : '');
                slotButtons[i].title = (names && names[i]) ? names[i] : '';
            }
            // Update latent source display
            if (selected.length > 1) {
                document.getElementById('latentSource').textContent =
                    'slots ' + selected.map(i => i + 1).join(',') + ' (interp)';
            } else if (selected.length === 1) {
                document.getElementById('latentSource').textContent = 'slot ' + (selected[0] + 1);
            }
        }

        // Library popup functions
        function showLibraryPopup(names) {
            const savedSlotIndex = pendingSlotIndex;
            hideLibraryPopup();
            pendingSlotIndex = savedSlotIndex;
            const popup = document.createElement('div');
            popup.className = 'latent-library-popup';

            if (names.length === 0) {
                const empty = document.createElement('div');
                empty.className = 'lib-empty';
                empty.textContent = 'No saved latents. Use "Save Latent" first.';
                popup.appendChild(empty);
            } else {
                names.forEach(name => {
                    const item = document.createElement('div');
                    item.className = 'lib-item';

                    const label = document.createElement('span');
                    label.textContent = name;
                    label.style.flex = '1';
                    label.addEventListener('click', () => {
                        if (ws && pendingSlotIndex !== null) {
                            ws.send(JSON.stringify({ type: 'load_slot', index: pendingSlotIndex, name }));
                        }
                        hideLibraryPopup();
                    });

                    const del = document.createElement('button');
                    del.className = 'lib-delete';
                    del.textContent = '\\u00d7';
                    del.title = 'Delete from library';
                    del.addEventListener('click', (e) => {
                        e.stopPropagation();
                        if (ws) ws.send(JSON.stringify({ type: 'delete_saved', name }));
                    });

                    item.appendChild(label);
                    item.appendChild(del);
                    popup.appendChild(item);
                });
            }

            // Position near the slots area
            const slotsEl = document.getElementById('latentSlots');
            const rect = slotsEl.getBoundingClientRect();
            popup.style.left = rect.left + 'px';
            popup.style.top = (rect.bottom + 8) + 'px';

            document.body.appendChild(popup);
            libraryPopup = popup;

            // Click-outside listener (delayed so current click doesn't dismiss)
            setTimeout(() => {
                document.addEventListener('click', outsideClickHandler, true);
            }, 0);
        }

        function hideLibraryPopup() {
            if (libraryPopup) {
                libraryPopup.remove();
                libraryPopup = null;
            }
            pendingSlotIndex = null;
            document.removeEventListener('click', outsideClickHandler, true);
        }

        function outsideClickHandler(e) {
            if (libraryPopup && !libraryPopup.contains(e.target)) {
                hideLibraryPopup();
            }
        }

        // WebSocket
        let pendingFrameHeader = null;
        let pendingNcaFrame = null;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('connStatus').textContent = 'Connected';
                document.getElementById('connStatus').className = 'connected';
                // Request context frames
                ws.send(JSON.stringify({ type: 'get_context' }));
            };

            ws.onclose = () => {
                document.getElementById('connStatus').textContent = 'Reconnecting...';
                document.getElementById('connStatus').className = 'disconnected';
                setTimeout(connect, 1000);
            };

            ws.onmessage = async (event) => {
                if (typeof event.data === 'string') {
                    if (event.data.startsWith('frame:')) {
                        // Parse frame header
                        const header = event.data.slice(6).split(',');
                        pendingFrameHeader = {
                            frameIdx: parseInt(header[0]),
                            w: parseInt(header[1]),
                            h: parseInt(header[2]),
                            c: parseInt(header[3]),
                        };
                        document.getElementById('frameNum').textContent = pendingFrameHeader.frameIdx;
                    } else {
                        const data = JSON.parse(event.data);

                        if (data.type === 'init') {
                            width = data.width;
                            height = data.height;
                            channels = data.channels;
                            contextFramesCount = data.context_frames;
                            document.getElementById('totalSeq').textContent = data.num_sequences;
                            document.getElementById('seqNum').textContent = data.current_seq;
                            currentSeqIdx = data.current_seq;
                            createContextCanvases();
                        } else if (data.type === 'sequence_changed') {
                            document.getElementById('seqNum').textContent = data.current_seq;
                            document.getElementById('latentSource').textContent = 'encoded';
                            currentSeqIdx = data.current_seq;
                        } else if (data.type === 'latent_changed') {
                            document.getElementById('latentSource').textContent = data.source;
                        } else if (data.type === 'slots_updated') {
                            updateSlotUI(data.filled, data.selected, data.names);
                        } else if (data.type === 'library_updated') {
                            libraryNames = data.names;
                            if (pendingSlotIndex !== null) {
                                showLibraryPopup(data.names);
                            }
                        } else if (data.type === 'context_frames') {
                            const frames = data.frames;
                            for (let i = 0; i < frames.length && i < contextCanvases.length; i++) {
                                const binary = atob(frames[i]);
                                const pixels = new Uint8Array(binary.length);
                                for (let j = 0; j < binary.length; j++) {
                                    pixels[j] = binary.charCodeAt(j);
                                }
                                const canvas = contextCanvases[i];
                                const cctx = canvas.getContext('2d');
                                renderFrame(cctx, canvas, pixels, data.width, data.height, data.channels);
                            }
                        }
                    }
                } else if (event.data instanceof Blob) {
                    const buffer = await event.data.arrayBuffer();
                    const pixels = new Uint8Array(buffer);

                    if (pendingNcaFrame === null) {
                        // This is NCA frame
                        pendingNcaFrame = pixels;
                    } else {
                        // This is GT frame, completing the NCA+GT pair
                        if (pendingFrameHeader) {
                            const { w, h, c } = pendingFrameHeader;
                            renderFrame(ncaCtx, ncaCanvas, pendingNcaFrame, w, h, c);
                            if (currentMode === 'sequence') {
                                renderFrame(gtCtx, gtCanvas, pixels, w, h, c);
                            }
                        }
                        pendingNcaFrame = null;
                    }
                }
            };
        }

        createContextCanvases();
        connect();
    </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="NCA Dynamics web visualization")
    parser.add_argument("checkpoint", type=str, help="Path to dynamics model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to sequence data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model(args.checkpoint, args.device)
    load_data(args.data)

    print(f"\nStarting server at http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
