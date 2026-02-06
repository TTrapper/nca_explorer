import asyncio
import json
import numpy as np
from scipy.ndimage import convolve
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

# Grid configuration
GRID_SIZE = 1024
UPDATE_INTERVAL = 1/30  # 30 FPS target

# Global state - RGB grid
grid = np.random.rand(GRID_SIZE, GRID_SIZE, 3).astype(np.float32)

# Two-layer architecture
# Each layer: 8 neighbors × 3 input channels × 3 output channels = 72 weights
# Layout for neighbors:
# 0 1 2
# 3 X 4
# 5 6 7
layer1_weights = np.zeros((8, 3, 3), dtype=np.float32)
layer2_weights = np.zeros((8, 3, 3), dtype=np.float32)

# Activation functions and parameters for each layer
# Options: 'gaussian', 'relu', 'tanh', 'sigmoid', 'sin', 'abs', 'identity'
layer1_activation = {'func': 'gaussian', 'center': 0.5, 'width': 0.15}
layer2_activation = {'func': 'tanh', 'center': 0.0, 'width': 1.0}

# Blend factor (how much of new state vs old state)
blend_factor = 0.9

# Connected clients
clients: list[WebSocket] = []

# Performance tracking
generation_fps = 0.0
frame_count = 0
last_fps_time = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0


def build_kernel_for_channels(weights: np.ndarray, in_ch: int, out_ch: int) -> np.ndarray:
    """Build 3x3 convolution kernel for specific input->output channel mapping."""
    kernel = np.array([
        [weights[0, in_ch, out_ch], weights[1, in_ch, out_ch], weights[2, in_ch, out_ch]],
        [weights[3, in_ch, out_ch], 0.0,                       weights[4, in_ch, out_ch]],
        [weights[5, in_ch, out_ch], weights[6, in_ch, out_ch], weights[7, in_ch, out_ch]]
    ], dtype=np.float32)
    return kernel


def apply_activation(x: np.ndarray, act_config: dict) -> np.ndarray:
    """Apply activation function based on configuration."""
    func = act_config['func']
    center = act_config.get('center', 0.0)
    width = act_config.get('width', 1.0)

    if func == 'gaussian':
        return np.exp(-((x - center) ** 2) / (width ** 2 + 1e-6))
    elif func == 'relu':
        return np.maximum(0, x - center)
    elif func == 'leaky_relu':
        shifted = x - center
        return np.where(shifted > 0, shifted, 0.1 * shifted)
    elif func == 'tanh':
        return (np.tanh((x - center) / (width + 1e-6)) + 1) / 2
    elif func == 'sigmoid':
        return 1 / (1 + np.exp(-(x - center) / (width + 1e-6)))
    elif func == 'sin':
        return (np.sin((x - center) * np.pi / width) + 1) / 2
    elif func == 'abs':
        return np.abs(x - center)
    elif func == 'identity':
        return x
    else:
        return x


def apply_convolution(input_grid: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply convolution with given weights."""
    output = np.zeros_like(input_grid)
    for out_ch in range(3):
        for in_ch in range(3):
            kernel = build_kernel_for_channels(weights, in_ch, out_ch)
            output[:, :, out_ch] += convolve(input_grid[:, :, in_ch], kernel, mode='wrap')
    return output


def step_simulation():
    """Perform one simulation step with two-layer architecture."""
    global grid

    # Layer 1: Convolution + Activation
    hidden = apply_convolution(grid, layer1_weights)
    hidden = apply_activation(hidden, layer1_activation)

    # Layer 2: Convolution + Activation
    output = apply_convolution(hidden, layer2_weights)
    output = apply_activation(output, layer2_activation)

    # Blend old and new state for smoother evolution
    grid = np.clip((1 - blend_factor) * grid + blend_factor * output, 0, 1).astype(np.float32)


def grid_to_image_bytes() -> bytes:
    """Convert RGB grid to image bytes for efficient transfer."""
    img_data = (grid * 255).astype(np.uint8)
    return img_data.tobytes()


@app.get("/")
async def get_index():
    with open("static/index.html", "r") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global blend_factor
    await websocket.accept()
    clients.append(websocket)

    try:
        # Send initial state
        await websocket.send_json({
            "type": "full_state",
            "layer1_weights": layer1_weights.tolist(),
            "layer2_weights": layer2_weights.tolist(),
            "layer1_activation": layer1_activation,
            "layer2_activation": layer2_activation,
            "blend_factor": blend_factor
        })

        while True:
            data = await websocket.receive_json()

            if data["type"] == "update_weight":
                layer = data.get("layer", 1)
                neighbor = data["neighbor"]
                in_ch = data["in_ch"]
                out_ch = data["out_ch"]
                delta = data["delta"]

                weights = layer1_weights if layer == 1 else layer2_weights

                if 0 <= neighbor < 8 and 0 <= in_ch < 3 and 0 <= out_ch < 3:
                    weights[neighbor, in_ch, out_ch] += delta
                    # Wrap from 1 to -1 and vice versa
                    val = weights[neighbor, in_ch, out_ch]
                    if val > 1:
                        weights[neighbor, in_ch, out_ch] = -1 + (val - 1)
                    elif val < -1:
                        weights[neighbor, in_ch, out_ch] = 1 + (val + 1)

                    # Broadcast updated weights to all clients
                    await broadcast_state()

            elif data["type"] == "reset":
                global grid
                grid = np.random.rand(GRID_SIZE, GRID_SIZE, 3).astype(np.float32)

            elif data["type"] == "reset_weights":
                layer1_weights[:] = 0
                layer2_weights[:] = 0
                await broadcast_state()

            elif data["type"] == "set_weights":
                layer = data.get("layer", 1)
                new_weights = data["weights"]
                weights = layer1_weights if layer == 1 else layer2_weights
                # Expecting 8 x 3 x 3 nested array
                if len(new_weights) == 8:
                    for n in range(8):
                        for i in range(3):
                            for o in range(3):
                                weights[n, i, o] = max(-1, min(1, float(new_weights[n][i][o])))
                    await broadcast_state()

            elif data["type"] == "set_full_state":
                # Set complete state (both layers, activations, blend)
                if "layer1_weights" in data:
                    w = data["layer1_weights"]
                    for n in range(8):
                        for i in range(3):
                            for o in range(3):
                                layer1_weights[n, i, o] = max(-1, min(1, float(w[n][i][o])))
                if "layer2_weights" in data:
                    w = data["layer2_weights"]
                    for n in range(8):
                        for i in range(3):
                            for o in range(3):
                                layer2_weights[n, i, o] = max(-1, min(1, float(w[n][i][o])))
                if "layer1_activation" in data:
                    layer1_activation.update(data["layer1_activation"])
                if "layer2_activation" in data:
                    layer2_activation.update(data["layer2_activation"])
                if "blend_factor" in data:
                    blend_factor = max(0, min(1, float(data["blend_factor"])))
                await broadcast_state()

            elif data["type"] == "set_activation":
                layer = data.get("layer", 1)
                act_config = layer1_activation if layer == 1 else layer2_activation
                if "func" in data:
                    act_config["func"] = data["func"]
                if "center" in data:
                    act_config["center"] = float(data["center"])
                if "width" in data:
                    act_config["width"] = float(data["width"])
                await broadcast_state()

            elif data["type"] == "set_blend":
                blend_factor = max(0, min(1, float(data["value"])))
                await broadcast_state()

    except WebSocketDisconnect:
        if websocket in clients:
            clients.remove(websocket)


async def broadcast_state():
    """Broadcast full state to all clients."""
    state = {
        "type": "full_state",
        "layer1_weights": layer1_weights.tolist(),
        "layer2_weights": layer2_weights.tolist(),
        "layer1_activation": layer1_activation,
        "layer2_activation": layer2_activation,
        "blend_factor": blend_factor
    }
    for client in clients[:]:
        try:
            await client.send_json(state)
        except:
            if client in clients:
                clients.remove(client)


async def simulation_loop():
    """Main simulation loop that broadcasts state to all clients."""
    global generation_fps, frame_count, last_fps_time

    last_fps_time = asyncio.get_event_loop().time()

    while True:
        loop_start = asyncio.get_event_loop().time()

        step_simulation()
        sim_time = asyncio.get_event_loop().time() - loop_start

        # Send grid state to all connected clients
        img_bytes = grid_to_image_bytes()
        send_start = asyncio.get_event_loop().time()

        for client in clients[:]:  # Copy list to avoid modification during iteration
            try:
                await client.send_bytes(img_bytes)
            except:
                if client in clients:
                    clients.remove(client)

        send_time = asyncio.get_event_loop().time() - send_start

        # Track FPS
        frame_count += 1
        now = asyncio.get_event_loop().time()
        if now - last_fps_time >= 1.0:
            generation_fps = frame_count / (now - last_fps_time)
            frame_count = 0
            last_fps_time = now

            # Broadcast performance stats
            for client in clients[:]:
                try:
                    await client.send_json({
                        "type": "perf",
                        "gen_fps": round(generation_fps, 1),
                        "sim_ms": round(sim_time * 1000, 1),
                        "send_ms": round(send_time * 1000, 1)
                    })
                except:
                    pass

        await asyncio.sleep(UPDATE_INTERVAL)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(simulation_loop())


# Mount static files
import os
os.makedirs("static", exist_ok=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
