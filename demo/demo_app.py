import os
import base64
import tempfile
import math
import torch
import torch.nn.functional as F
import h5py
import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

# Import your model architecture
from statera.model import StateraModel

# ==========================================
# GLOBAL ML STATE & HELPERS
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

# Dynamically resolve the path to demo/samples/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")


def get_subpixel_coords(logits_heatmap, temperature=2.0):
    """
    Temperature-Scaled Spatial Softmax.
    Eliminates the 6-pixel quantization staircase.
    """
    B, T, H, W = logits_heatmap.shape
    probs = F.softmax((logits_heatmap.view(B * T, -1)) / temperature, dim=1)
    probs = probs.view(B, T, H, W)

    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    y_center = (probs * y_grid.float().view(1, 1, H, W)).sum(dim=(2, 3))
    x_center = (probs * x_grid.float().view(1, 1, H, W)).sum(dim=(2, 3))

    return torch.stack([x_center, y_center], dim=2)


def overlay_results(frame_bgr, pred_u, pred_v, gt_u=None, gt_v=None, raw_hm=None):
    overlay = frame_bgr.copy()
    h, w = overlay.shape[:2]

    # Draw reference grid
    for i in range(1, 4):
        x = int(w * i / 4.0)
        y = int(h * i / 4.0)
        cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.3, frame_bgr, 0.7, 0, frame_bgr)

    # Render Heatmap Mask
    if raw_hm is not None:
        hm_resized = cv2.resize(raw_hm, (frame_bgr.shape[1], frame_bgr.shape[0]))
        hm_vis = np.uint8(255 * (hm_resized / (hm_resized.max() + 1e-8)))
        hm_colored = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)
        mask = hm_vis > 100
        frame_bgr[mask] = cv2.addWeighted(frame_bgr, 0.5, hm_colored, 0.5, 0)[mask]

    # Draw Ground Truth (if provided from Demo .h5)
    if gt_u is not None and gt_v is not None:
        cv2.circle(frame_bgr, (int(gt_u), int(gt_v)), 6, (0, 220, 50), 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, (int(gt_u), int(gt_v)), 2, (255, 255, 255), -1, cv2.LINE_AA)

    # Draw Predicted CoM Target
    cx, cy = int(pred_u), int(pred_v)
    cv2.drawMarker(frame_bgr, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
    cv2.circle(frame_bgr, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)

    return frame_bgr


def frames_to_base64(frames):
    return [base64.b64encode(cv2.imencode('.jpg', f)[1]).decode('utf-8') for f in frames]


# ==========================================
# FASTAPI BACKEND
# ==========================================
app = FastAPI(title="STATERA Web Engine")


@app.post("/api/load_model")
async def load_model_api(checkpoint: UploadFile = File(...)):
    global model
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
        temp_file.write(await checkpoint.read())
        temp_file.close()

        print("Loading STATERA Model from uploaded checkpoint...")
        model = StateraModel(
            decoder_type='deconv',
            temporal_mixer='conv1d',
            single_task=False,
            backbone_type='vjepa',
            finetune_blocks=2
        ).to(device)

        model.load_state_dict(torch.load(temp_file.name, map_location=device, weights_only=True), strict=True)
        model.eval()
        print("✔ Inference Engine Ready.")
        os.remove(temp_file.name)

        return JSONResponse({"status": "success", "message": "Model loaded successfully."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/list_demos")
async def list_demos():
    """
    Scans the local 'samples/' folder for .h5 files, enumerates all sequences,
    and generates thumbnail images.
    """
    if not os.path.exists(SAMPLES_DIR):
        print(f"⚠ Warning: Samples directory not found at {SAMPLES_DIR}")
        return JSONResponse({"demos": []})

    demos = []
    for file_name in sorted(os.listdir(SAMPLES_DIR)):
        if file_name.endswith(".h5"):
            file_path = os.path.join(SAMPLES_DIR, file_name)
            try:
                with h5py.File(file_path, "r") as f:
                    for seq_name in sorted(f.keys()):
                        if seq_name.startswith("seq_"):
                            # Extract the first frame directly for the thumbnail gallery
                            first_frame_np = f[seq_name]["frames"][0]
                            _, buffer = cv2.imencode('.jpg', first_frame_np)
                            b64_thumb = base64.b64encode(buffer).decode('utf-8')

                            demos.append({
                                "file_name": file_name,
                                "seq_name": seq_name,
                                "thumbnail": f"data:image/jpeg;base64,{b64_thumb}"
                            })
            except Exception as e:
                print(f"Warning: Failed to parse {file_name} -> {e}")

    return JSONResponse({"demos": demos})


@app.get("/api/demo")
async def run_demo(file: str = Query(...), seq: str = Query(...)):
    """
    Parses embedded ground truth and extracts physics directly from .h5 extraction struct.
    """
    if model is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Please initialize the backbone first.")

    h5_path = os.path.join(SAMPLES_DIR, file)
    if not os.path.exists(h5_path):
        raise HTTPException(status_code=404, detail=f"Required demo file not found: {h5_path}")

    # Read Structure
    with h5py.File(h5_path, 'r') as f:
        if seq not in f:
            raise HTTPException(status_code=404, detail=f"Sequence {seq} not found in {file}")

        grp = f[seq]
        frames_np = grp["frames"][:]
        gt_u_seq = grp["com_u"][:]
        gt_v_seq = grp["com_v"][:]
        gt_z_seq = grp["z_depth_meters"][:]

    # Pre-Process Tensors
    frames_rgb = frames_np[..., ::-1].copy()  # Convert from BGR (saved by CV2) to RGB
    vid_tensor = torch.from_numpy(frames_rgb).float() / 255.0
    vid_tensor = vid_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(device)

    # Infer
    with torch.no_grad():
        pred_h, pred_z = model(vid_tensor)

        sub_uv_seq = get_subpixel_coords(pred_h)[0].cpu().numpy() * (384.0 / 64.0)
        final_pred_u, final_pred_v = sub_uv_seq[-1]
        final_pred_z = pred_z[0, -1, 0].item()

        raw_hm_seq = torch.sigmoid(pred_h)[0].cpu().numpy()

    final_gt_u, final_gt_v = gt_u_seq[-1], gt_v_seq[-1]
    final_gt_z = float(gt_z_seq[-1])

    out_frames = []
    for i in range(16):
        pred_u, pred_v = sub_uv_seq[i]
        out = overlay_results(
            frames_np[i].copy(),
            pred_u, pred_v,
            gt_u=gt_u_seq[i], gt_v=gt_v_seq[i],
            raw_hm=raw_hm_seq[i]
        )
        out_frames.append(out)

    return JSONResponse({
        "ep_idx": f"{file} - {seq}",
        "pred_z": f"{final_pred_z:.4f} m",
        "pred_uv": f"{final_pred_u:.1f}, {final_pred_v:.1f}",
        "gt_z": f"{final_gt_z:.4f} m",
        "gt_uv": f"{final_gt_u:.1f}, {final_gt_v:.1f}",
        "error_px": f"{np.linalg.norm([final_gt_u - final_pred_u, final_gt_v - final_pred_v]):.2f} px",
        "frames": frames_to_base64(out_frames)
    })


@app.post("/api/zero_shot")
async def zero_shot_inference(
        video: UploadFile = File(...),
        start_time: float = Form(...),
        crop_x: int = Form(...),
        crop_y: int = Form(...),
        crop_size: int = Form(...)
):
    if model is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Please initialize the backbone first.")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.write(await video.read())
    temp_file.close()

    cap = cv2.VideoCapture(temp_file.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or math.isnan(fps): fps = 24.0

    start_frame = int(start_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    crop_size = max(384, int(crop_size))
    crop_x = int(crop_x)
    crop_y = int(crop_y)

    for _ in range(16):
        ret, frame = cap.read()
        if not ret: break

        h, w = frame.shape[:2]
        x1 = max(0, crop_x)
        y1 = max(0, crop_y)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        valid_x1 = max(0, min(w, x1))
        valid_x2 = max(0, min(w, x2))
        valid_y1 = max(0, min(h, y1))
        valid_y2 = max(0, min(h, y2))

        cropped = frame[valid_y1:valid_y2, valid_x1:valid_x2]

        if cropped.shape[0] < crop_size or cropped.shape[1] < crop_size:
            pad = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            pad[:cropped.shape[0], :cropped.shape[1]] = cropped
            cropped = pad

        if cropped.shape[:2] != (384, 384):
            cropped = cv2.resize(cropped, (384, 384), interpolation=cv2.INTER_AREA)

        frames.append(cropped)

    cap.release()
    os.remove(temp_file.name)

    while len(frames) < 16:
        frames.append(frames[-1] if frames else np.zeros((384, 384, 3), dtype=np.uint8))

    frames_np = np.stack(frames)
    frames_rgb = frames_np[..., ::-1].copy()

    vid_tensor = torch.from_numpy(frames_rgb).float() / 255.0
    vid_tensor = vid_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_h, pred_z = model(vid_tensor)

        sub_uv_seq = get_subpixel_coords(pred_h)[0].cpu().numpy() * (384.0 / 64.0)
        final_pred_u, final_pred_v = sub_uv_seq[-1]
        final_pred_z = pred_z[0, -1, 0].item()

        raw_hm_seq = torch.sigmoid(pred_h)[0].cpu().numpy()

    out_frames = []
    for i in range(16):
        pred_u, pred_v = sub_uv_seq[i]
        out = overlay_results(frames[i].copy(), pred_u, pred_v, raw_hm=raw_hm_seq[i])
        out_frames.append(out)

    return JSONResponse({
        "ep_idx": "Custom Zero-Shot",
        "pred_z": f"{final_pred_z:.4f} m",
        "pred_uv": f"{final_pred_u:.1f}, {final_pred_v:.1f}",
        "frames": frames_to_base64(out_frames)
    })


# ==========================================
# MATERIAL 3 FRONTEND
# ==========================================
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>STATERA</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,1,0" rel="stylesheet" />
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: { sans:['Roboto', 'sans-serif'] },
                    colors: {
                        m3: {
                            bg: '#141218',
                            surface: '#1D1B20',
                            surfaceHigh: '#2B2930',
                            primary: '#D0BCFF',
                            onPrimary: '#381E72',
                            secondaryContainer: '#4A4458',
                            onSecondaryContainer: '#E8DEF8',
                            text: '#E6E0E9',
                            textVariant: '#CAC4D0',
                            outline: '#938F99',
                            outlineVariant: '#49454F',
                            error: '#F2B8B5'
                        }
                    }
                }
            }
        }
    </script>
    <style>
        body { background-color: #141218; color: #E6E0E9; }

        input[type=range].m3-slider { -webkit-appearance: none; width: 100%; background: transparent; height: 40px; }
        input[type=range].m3-slider:focus { outline: none; }
        input[type=range].m3-slider::-webkit-slider-runnable-track { width: 100%; height: 8px; cursor: pointer; background: #4A4458; border-radius: 4px; }
        input[type=range].m3-slider::-webkit-slider-thumb { height: 20px; width: 20px; border-radius: 50%; background: #D0BCFF; cursor: pointer; -webkit-appearance: none; margin-top: -6px; box-shadow: 0 1px 3px rgba(0,0,0,0.3); transition: transform 0.1s; }
        input[type=range].m3-slider::-webkit-slider-thumb:active { transform: scale(1.2); }

        .timeline-container { position: relative; height: 40px; display: flex; align-items: center; width: 100%; }
        .timeline-track { position: absolute; width: 100%; height: 12px; background: #49454F; border-radius: 6px; overflow: hidden; }
        .timeline-segment { position: absolute; height: 100%; background: rgba(208, 188, 255, 0.4); border-left: 2px solid #D0BCFF; border-right: 2px solid #D0BCFF; pointer-events: none; transition: left 0.1s linear; }
        input[type="range"].timeline-slider { position: absolute; width: 100%; height: 100%; opacity: 0; cursor: pointer; z-index: 10; margin: 0; }

        #strictCropBox { box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.7); border: 2px solid #D0BCFF; box-sizing: border-box; }
        #modalVideoWrapper { border-radius: 12px; overflow: hidden; position: relative; background: #000; display: inline-block; user-select: none; }
    </style>
</head>
<body class="w-full min-h-screen flex flex-col items-center p-4 md:p-8">

    <!-- Top App Bar -->
    <header class="w-full max-w-6xl flex items-center justify-between mb-8">
        <div class="flex items-center gap-3">
            <span class="material-symbols-rounded text-m3-primary text-3xl">token</span>
            <h1 class="text-2xl font-normal tracking-tight text-m3-text">STATERA</h1>
        </div>
    </header>

    <!-- Engine / Model Initialization Card -->
    <div class="w-full max-w-6xl mx-auto mb-6">
        <div class="bg-m3-surface rounded-[24px] p-6 shadow-lg border border-white/5 flex flex-col md:flex-row items-center justify-between gap-4">
            <div class="flex items-center gap-4">
                <div class="w-12 h-12 rounded-full bg-m3-surfaceHigh flex items-center justify-center transition-colors duration-300">
                    <span id="modelIcon" class="material-symbols-rounded text-m3-error text-[28px] transition-colors duration-300">model_training</span>
                </div>
                <div>
                    <h2 class="text-lg font-normal text-m3-text tracking-tight">Neural Backbone</h2>
                    <p id="modelStatusText" class="text-sm text-m3-error font-medium transition-colors duration-300">Engine Offline - Load Checkpoint</p>
                </div>
            </div>

            <div class="flex items-center gap-3 w-full md:w-auto">
                <input type="file" id="checkpointUpload" accept=".pth" class="hidden">
                <label for="checkpointUpload" class="flex-1 md:flex-none px-6 py-2.5 rounded-full border border-m3-outlineVariant text-m3-primary cursor-pointer hover:bg-m3-primary/10 transition text-center text-sm font-medium">Select .pth</label>

                <button id="btnLoadModel" disabled class="flex-1 md:flex-none px-8 py-2.5 rounded-full bg-m3-surfaceHigh text-m3-textVariant font-medium flex items-center justify-center gap-2 transition opacity-50 cursor-not-allowed text-sm">
                    <span class="material-symbols-rounded text-[18px]">memory</span> Initialize
                </button>
            </div>
        </div>
        <p id="checkpointName" class="text-xs text-m3-textVariant mt-2 ml-4 hidden"></p>
    </div>

    <main class="w-full max-w-6xl flex flex-col md:flex-row gap-6 mx-auto">

        <!-- LEFT PANEL -->
        <div class="w-full md:w-[380px] flex flex-col gap-6 shrink-0">

            <!-- CONTROLS CARD -->
            <div class="bg-m3-surface rounded-[24px] p-6 shadow-lg border border-white/5">
                <h2 class="text-xl font-normal text-m3-text mb-2 tracking-tight">Custom Inference</h2>
                <p class="text-sm text-m3-textVariant mb-6 leading-relaxed">Upload any rigid-body MP4 sequence. Select multiple files to auto-batch extract features.</p>

                <input type="file" id="videoUpload" accept="video/*" class="hidden" multiple>
                <label for="videoUpload" class="w-full py-10 rounded-2xl border-2 border-dashed border-m3-outlineVariant hover:border-m3-primary hover:bg-m3-primary/5 flex flex-col items-center justify-center cursor-pointer transition mb-6">
                    <span class="material-symbols-rounded text-4xl text-m3-primary mb-3">video_library</span>
                    <span class="text-m3-text font-medium text-sm">Select MP4 Video(s)</span>
                    <span class="text-m3-textVariant text-xs mt-1">Multi-file Batch supported</span>
                </label>

                <div id="zeroShotBatchControls" class="hidden">
                    <div class="mb-6">
                        <div class="flex justify-between items-center mb-2">
                            <label class="text-sm font-medium text-m3-textVariant">Batch Video Index</label>
                            <span class="text-sm text-m3-primary font-mono bg-m3-primary/10 px-2 py-0.5 rounded-md" id="batchLabel">1 / 1</span>
                        </div>
                        <input type="range" id="batchSlider" class="m3-slider" min="0" max="0" value="0">
                    </div>
                </div>
            </div>

            <!-- TELEMETRY CARD -->
            <div id="statsPanel" class="bg-m3-surface rounded-[24px] p-6 shadow-lg border border-white/5 hidden flex-col">
                <div class="flex items-center gap-2 mb-6">
                    <span class="material-symbols-rounded text-m3-primary text-[20px]">monitoring</span>
                    <h3 class="text-sm font-medium text-m3-primary tracking-wide">LIVE TELEMETRY</h3>
                </div>

                <div class="space-y-4">
                    <div class="bg-m3-surfaceHigh rounded-xl p-4 flex justify-between items-center">
                        <span class="text-m3-textVariant text-sm">Sequence Identity</span>
                        <span id="stat_ep" class="font-mono text-sm text-m3-text truncate max-w-[150px] text-right">--</span>
                    </div>

                    <!-- GT is populated for Demos, kept N/A for Custom Zero-Shot -->
                    <div class="bg-m3-surfaceHigh rounded-xl p-4 flex flex-col gap-2">
                        <span class="text-m3-textVariant text-xs uppercase tracking-wider font-medium mb-1">Depth Coordinates (Z)</span>
                        <div class="flex justify-between items-center">
                            <span class="text-m3-textVariant text-sm">Ground Truth</span>
                            <span id="stat_gt_z" class="font-mono text-sm text-[#A8E6CF]">N/A</span>
                        </div>
                        <div class="flex justify-between items-center">
                            <span class="text-m3-textVariant text-sm">Predicted</span>
                            <span id="stat_pred_z" class="font-mono text-sm text-m3-primary">--</span>
                        </div>
                    </div>

                    <div class="bg-m3-surfaceHigh rounded-xl p-4 flex flex-col gap-2">
                        <span class="text-m3-textVariant text-xs uppercase tracking-wider font-medium mb-1">Spatial CoM (X, Y)</span>
                        <div class="flex justify-between items-center">
                            <span class="text-m3-textVariant text-sm">Ground Truth</span>
                            <span id="stat_gt_uv" class="font-mono text-sm text-[#A8E6CF]">N/A</span>
                        </div>
                        <div class="flex justify-between items-center">
                            <span class="text-m3-textVariant text-sm">Predicted</span>
                            <span id="stat_pred_uv" class="font-mono text-sm text-m3-primary">--</span>
                        </div>
                        <div class="border-t border-m3-outlineVariant/50 mt-2 pt-3 flex justify-between items-center">
                            <span class="text-m3-textVariant text-sm">Delta Error</span>
                            <span id="stat_err" class="font-mono text-sm text-m3-error font-medium">N/A</span>
                        </div>
                    </div>

                </div>
            </div>
        </div>

        <!-- RIGHT PANEL / OUTPUT STAGE -->
        <div class="bg-m3-surface rounded-[24px] p-6 flex-1 flex flex-col items-center justify-center min-h-[500px] relative border border-white/5 shadow-lg overflow-hidden">

            <div id="loader" class="hidden absolute inset-0 bg-m3-bg/80 backdrop-blur-sm z-20 flex flex-col items-center justify-center">
                <div class="w-12 h-12 border-4 border-m3-surfaceHigh border-t-m3-primary rounded-full animate-spin mb-4"></div>
                <p id="loaderText" class="text-m3-primary text-sm font-medium tracking-wide">Processing Tensors...</p>
            </div>

            <div id="emptyState" class="flex flex-col items-center text-m3-textVariant opacity-70">
                <span class="material-symbols-rounded text-[64px] mb-4">video_library</span>
                <p class="text-sm font-medium">Awaiting Sequence Injection</p>
            </div>

            <img id="resultCanvas" class="hidden w-full max-w-[500px] rounded-2xl shadow-2xl object-contain aspect-square bg-[#000]" src="" alt="Result">

            <!-- Playback Scrubber -->
            <div id="playbackControls" class="hidden w-full max-w-[500px] mt-8 flex items-center gap-4 bg-m3-surfaceHigh py-2 px-4 rounded-full">
                <button id="btnPlayPause" class="text-m3-onSurface hover:text-m3-primary transition-colors focus:outline-none flex items-center justify-center">
                    <span id="iconPlay" class="material-symbols-rounded text-3xl hidden">play_arrow</span>
                    <span id="iconPause" class="material-symbols-rounded text-3xl">pause</span>
                </button>
                <input type="range" id="frameScrubber" class="m3-slider flex-1" min="0" max="15" value="0">
                <span id="frameCounter" class="text-xs font-mono text-m3-textVariant w-10 text-right">0/15</span>
            </div>
        </div>
    </main>

    <!-- DEMOS SECTION -->
    <section class="w-full max-w-6xl mx-auto mt-8 mb-12">
        <div class="flex items-center gap-2 mb-4 ml-2">
            <span class="material-symbols-rounded text-m3-primary">grid_view</span>
            <h2 class="text-xl font-normal text-m3-text tracking-tight">Embedded Demos (.h5)</h2>
        </div>

        <!-- Dynamic Gallery Grid -->
        <div id="demoGallery" class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            <!-- Gallery injected via JS -->
        </div>
    </section>

    <!-- M3 Full-Screen Dialog for Video Preparation -->
    <div id="prepModal" class="fixed inset-0 z-50 hidden bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
        <div class="bg-m3-surfaceHigh rounded-[28px] p-8 w-full max-w-3xl flex flex-col items-center shadow-2xl border border-white/5">

            <!-- Step 1: Sequence Selector -->
            <div id="modalStep1" class="w-full flex flex-col items-center animate-fade-in">
                <div class="w-full flex justify-between items-center mb-6">
                    <div>
                        <h2 id="step1Title" class="text-2xl font-normal text-m3-text tracking-tight">Step 1: Isolate Sequence</h2>
                        <p id="step1Subtitle" class="text-sm text-m3-textVariant mt-1">Slide the window to extract exactly 16 frames.</p>
                    </div>
                    <button id="btnCloseModal" class="w-10 h-10 rounded-full hover:bg-m3-surface transition flex items-center justify-center text-m3-textVariant">
                        <span class="material-symbols-rounded">close</span>
                    </button>
                </div>

                <video id="modalVidPlayer" class="w-full rounded-2xl max-h-[45vh] bg-black mb-6 shadow-xl object-contain"></video>

                <div class="w-full bg-m3-surface rounded-2xl p-4 mb-6 shadow-inner border border-white/5">

                    <!-- UPDATED: Frame Sync Header -->
                    <div class="flex justify-between items-center mb-4">
                        <div class="flex items-center gap-4">
                            <span id="timeLabel" class="text-sm font-medium text-m3-primary font-mono">Segment: 0.00s to 0.53s</span>

                            <!-- New FPS Selector to match exact Backend extraction logic -->
                            <div class="flex items-center gap-2">
                                <span class="text-xs font-medium text-m3-textVariant uppercase tracking-wider">Source FPS:</span>
                                <select id="videoFpsSelect" class="bg-m3-bg border border-m3-outlineVariant text-m3-text text-xs rounded px-2 py-1 outline-none focus:border-m3-primary transition-colors cursor-pointer">
                                    <option value="24">24 fps</option>
                                    <option value="25">25 fps</option>
                                    <option value="30" selected>30 fps</option>
                                    <option value="60">60 fps</option>
                                </select>
                            </div>
                        </div>

                        <button id="btnPreviewSegment" class="text-xs font-medium text-m3-onSecondaryContainer bg-m3-secondaryContainer px-3 py-1.5 rounded-full hover:bg-m3-secondaryContainer/80 transition flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed">
                            <span class="material-symbols-rounded text-[16px]">play_circle</span> Preview 16 Frames
                        </button>
                    </div>

                    <div class="timeline-container">
                        <div class="timeline-track"></div>
                        <div id="timelineSegment" class="timeline-segment"></div>
                        <input type="range" id="timelineSlider" class="timeline-slider" min="0" step="0.01" value="0">
                    </div>
                </div>
                <div class="flex justify-end w-full">
                    <button id="btnNextStep" class="px-8 py-2.5 rounded-full bg-m3-primary text-m3-onPrimary font-medium hover:bg-m3-primary/90 transition-shadow shadow-md">Next: Set Boundaries</button>
                </div>
            </div>

            <!-- Step 2: Physics Scale Crop & Scrub -->
            <div id="modalStep2" class="w-full flex flex-col items-center hidden">
                <div class="w-full flex justify-between items-center mb-6">
                    <div>
                        <h2 id="step2Title" class="text-2xl font-normal text-m3-text tracking-tight">Step 2: Track & Crop Box</h2>
                        <p class="text-sm text-m3-textVariant mt-1">Scrub the frames to ensure the object stays in bounds. <br>Minimum threshold locked to <b>384x384</b>.</p>
                    </div>
                </div>
                <div id="modalVideoWrapper" class="shadow-2xl mb-6 relative">
                    <video id="modalVidFrame" class="block"></video>
                    <div id="strictCropBox" class="absolute cursor-move z-10 flex items-center justify-center">
                        <div id="cropResizeHandle" class="absolute bottom-[-2px] right-[-2px] w-5 h-5 bg-m3-primary cursor-se-resize rounded-tl-md shadow-md hover:scale-110 transition-transform"></div>
                        <span id="cropSizeLabel" class="text-white/80 bg-black/50 backdrop-blur-md px-3 py-1 rounded-full text-[11px] font-mono tracking-widest pointer-events-none">384x384</span>
                    </div>
                </div>
                <div class="w-full bg-m3-surface rounded-2xl p-4 mb-6 shadow-inner border border-white/5">
                    <div class="flex justify-between items-center mb-2">
                        <span class="text-sm font-medium text-m3-textVariant">Scrub Segment Frames</span>
                        <span id="step2FrameLabel" class="text-xs font-mono text-m3-primary">Frame 1/16</span>
                    </div>
                    <input type="range" id="step2Scrubber" class="m3-slider" min="0" max="15" value="0">
                </div>
                <div class="flex justify-between w-full">
                    <button id="btnBackStep" class="px-6 py-2.5 rounded-full border border-m3-outlineVariant text-m3-primary font-medium hover:bg-m3-primary/10 transition-colors">Back</button>
                    <button id="btnRunZeroShot" class="px-8 py-2.5 rounded-full bg-m3-primary text-m3-onPrimary font-medium flex items-center gap-2 hover:bg-m3-primary/90 transition-shadow shadow-md">
                        <span class="material-symbols-rounded text-[20px]">psychology</span> Extract Features
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // --- System State ---
        let isModelLoaded = false;

        // --- Fetch Gallery on Load ---
        window.addEventListener('DOMContentLoaded', async () => {
            try {
                const res = await fetch('/api/list_demos');
                const data = await res.json();
                const gallery = document.getElementById('demoGallery');

                if (data.demos.length === 0) {
                    gallery.innerHTML = `<p class="text-m3-textVariant col-span-full">No .h5 sequences found in samples/ directory.</p>`;
                    return;
                }

                data.demos.forEach(demo => {
                    const card = document.createElement('div');
                    card.className = "bg-m3-surface rounded-[24px] p-4 shadow-lg border border-white/5 flex flex-col hover:bg-m3-surfaceHigh transition cursor-pointer group";
                    card.onclick = () => runDemo(demo.file_name, demo.seq_name);

                    card.innerHTML = `
                        <div class="w-full aspect-square bg-[#000] rounded-xl overflow-hidden mb-4 relative">
                            <img src="${demo.thumbnail}" class="w-full h-full object-contain group-hover:scale-105 transition-transform duration-500">
                            <div class="absolute inset-0 bg-black/10 group-hover:bg-transparent transition-colors"></div>
                        </div>
                        <h3 class="text-m3-text font-medium text-sm tracking-tight truncate w-full" title="${demo.file_name}">${demo.file_name}</h3>
                        <p class="text-xs text-m3-textVariant mt-1 mb-4 font-mono">${demo.seq_name}</p>
                        <button class="text-xs text-m3-primary font-medium flex items-center gap-1 mt-auto">
                            Inject Demo <span class="material-symbols-rounded text-[16px] group-hover:translate-x-1 transition-transform">arrow_forward</span>
                        </button>
                    `;
                    gallery.appendChild(card);
                });
            } catch (err) {
                console.error("Failed to load demo gallery", err);
            }
        });


        // --- Model Loading Logic ---
        const checkpointUpload = document.getElementById('checkpointUpload');
        const btnLoadModel = document.getElementById('btnLoadModel');
        const checkpointName = document.getElementById('checkpointName');

        checkpointUpload.onchange = (e) => {
            if (e.target.files.length > 0) {
                checkpointName.innerText = `Selected: ${e.target.files[0].name}`;
                checkpointName.classList.remove('hidden');

                btnLoadModel.disabled = false;
                btnLoadModel.classList.remove('opacity-50', 'cursor-not-allowed', 'bg-m3-surfaceHigh', 'text-m3-textVariant');
                btnLoadModel.classList.add('bg-m3-primary', 'text-m3-onPrimary', 'shadow-md', 'hover:bg-m3-primary/90');
            }
        };

        btnLoadModel.onclick = async () => {
            const file = checkpointUpload.files[0];
            if (!file) return;

            btnLoadModel.disabled = true;
            btnLoadModel.innerHTML = `<span class="material-symbols-rounded animate-spin">sync</span> Booting...`;

            const fd = new FormData();
            fd.append('checkpoint', file);

            try {
                const res = await fetch('/api/load_model', { method: 'POST', body: fd });
                if (!res.ok) throw new Error("Load failed");

                isModelLoaded = true;

                // Success State Update
                document.getElementById('modelStatusText').innerText = "Engine Online - Ready";
                document.getElementById('modelStatusText').classList.replace('text-m3-error', 'text-[#A8E6CF]');

                const icon = document.getElementById('modelIcon');
                icon.innerText = "check_circle";
                icon.classList.replace('text-m3-error', 'text-[#A8E6CF]');
                icon.parentElement.classList.replace('bg-m3-surfaceHigh', 'bg-[#A8E6CF]/10');

                btnLoadModel.innerHTML = `<span class="material-symbols-rounded">check</span> Active`;
                btnLoadModel.classList.remove('bg-m3-primary', 'text-m3-onPrimary', 'shadow-md', 'hover:bg-m3-primary/90');
                btnLoadModel.classList.add('bg-[#A8E6CF]/10', 'text-[#A8E6CF]');

            } catch (e) {
                alert("Error initializing backbone checkpoint.");
                btnLoadModel.disabled = false;
                btnLoadModel.innerHTML = `<span class="material-symbols-rounded">memory</span> Initialize`;
            }
        };

        // --- Demo Sample Logic ---
        async function runDemo(file, seq) {
            if (!isModelLoaded) {
                alert("Please initialize the backbone checkpoint first!");
                return;
            }

            resetOutput();
            document.getElementById('loaderText').innerText = `Loading ${file} -> ${seq}...`;
            document.getElementById('loader').classList.remove('hidden');
            document.getElementById('loader').classList.add('flex');

            try {
                const res = await fetch(`/api/demo?file=${encodeURIComponent(file)}&seq=${encodeURIComponent(seq)}`);
                if (!res.ok) throw new Error("Failed to reach endpoint");

                const data = await res.json();

                // Telemetry Inject 
                document.getElementById('stat_ep').innerText = data.ep_idx;
                document.getElementById('stat_pred_z').innerText = data.pred_z;
                document.getElementById('stat_gt_z').innerText = data.gt_z;
                document.getElementById('stat_pred_uv').innerText = data.pred_uv;
                document.getElementById('stat_gt_uv').innerText = data.gt_uv;
                document.getElementById('stat_err').innerText = data.error_px;

                initSequence(data.frames);

            } catch (e) {
                alert("Failed to reach demo endpoint or sequence not found.");
            } finally {
                document.getElementById('loader').classList.add('hidden');
                document.getElementById('loader').classList.remove('flex');
            }
        }


        // --- Playback Engine ---
        let currentFrames =[];
        let currentFrameIdx = 0;
        let playInterval = null;
        let isPlaying = false;

        const resultCanvas = document.getElementById('resultCanvas');
        const playbackControls = document.getElementById('playbackControls');
        const frameScrubber = document.getElementById('frameScrubber');
        const frameCounter = document.getElementById('frameCounter');
        const btnPlayPause = document.getElementById('btnPlayPause');

        function resetOutput() {
            stopSequence();
            resultCanvas.classList.add('hidden');
            playbackControls.classList.add('hidden');
            document.getElementById('statsPanel').classList.add('hidden');
            document.getElementById('statsPanel').classList.remove('flex');
            document.getElementById('emptyState').classList.remove('hidden');
        }

        function renderFrame(idx) {
            if(!currentFrames.length) return;
            currentFrameIdx = parseInt(idx);
            resultCanvas.src = 'data:image/jpeg;base64,' + currentFrames[currentFrameIdx];
            frameScrubber.value = currentFrameIdx;
            frameCounter.innerText = `${currentFrameIdx + 1}/16`;
        }

        function startSequence() {
            if(!currentFrames.length) return;
            isPlaying = true;
            document.getElementById('iconPlay').classList.add('hidden'); 
            document.getElementById('iconPause').classList.remove('hidden');
            if (currentFrameIdx >= 15) currentFrameIdx = 0;

            playInterval = setInterval(() => {
                renderFrame(currentFrameIdx);
                if (currentFrameIdx >= 15) {
                    stopSequence();
                } else {
                    currentFrameIdx++;
                }
            }, 1000 / 24);
        }

        function stopSequence() {
            isPlaying = false; clearInterval(playInterval);
            document.getElementById('iconPlay').classList.remove('hidden'); 
            document.getElementById('iconPause').classList.add('hidden');
        }

        function initSequence(frames) {
            document.getElementById('emptyState').classList.add('hidden');
            resultCanvas.classList.remove('hidden');
            playbackControls.classList.remove('hidden');
            document.getElementById('statsPanel').classList.remove('hidden');
            document.getElementById('statsPanel').classList.add('flex');
            currentFrames = frames; currentFrameIdx = 0; startSequence();
        }

        btnPlayPause.onclick = () => isPlaying ? stopSequence() : startSequence();
        frameScrubber.oninput = (e) => { stopSequence(); renderFrame(e.target.value); };


        // --- Custom Video Modal Logic ---
        const prepModal = document.getElementById('prepModal');
        const modalStep1 = document.getElementById('modalStep1');
        const modalStep2 = document.getElementById('modalStep2');
        const modalVidPlayer = document.getElementById('modalVidPlayer');
        const modalVidFrame = document.getElementById('modalVidFrame');
        const videoWrapper = document.getElementById('modalVideoWrapper');
        const strictCropBox = document.getElementById('strictCropBox');
        const cropResizeHandle = document.getElementById('cropResizeHandle');
        const cropSizeLabel = document.getElementById('cropSizeLabel');
        const step2Scrubber = document.getElementById('step2Scrubber');
        const step2FrameLabel = document.getElementById('step2FrameLabel');

        let batchFiles = [];
        let batchConfigs =[];
        let batchResults =[];
        let currentConfigIdx = 0;
        let currentBatchIdx = 0;
        let displayScale = 1.0;

        const timelineSlider = document.getElementById('timelineSlider');
        const timelineSegment = document.getElementById('timelineSegment');
        const timeLabel = document.getElementById('timeLabel');

        // --- FPS Synchronization Updates ---
        let fpsEstimate = 30.0;
        let segmentDuration = 16 / fpsEstimate;

        document.getElementById('videoFpsSelect').addEventListener('change', (e) => {
            fpsEstimate = parseFloat(e.target.value);
            segmentDuration = 16 / fpsEstimate;
            initTimeline();
        });

        function initTimeline() {
            if (!modalVidPlayer.duration || isNaN(modalVidPlayer.duration)) return;
            timelineSlider.max = Math.max(0, modalVidPlayer.duration - segmentDuration);
            timelineSlider.value = 0;
            updateTimeline();
        }

        modalVidPlayer.addEventListener('loadedmetadata', initTimeline);

        function updateTimeline() {
            const dur = modalVidPlayer.duration;
            if (!dur) return;
            const val = parseFloat(timelineSlider.value);
            const pct = (val / dur) * 100;
            const widthPct = (segmentDuration / dur) * 100;

            timelineSegment.style.left = `${pct}%`;
            timelineSegment.style.width = `${widthPct}%`;

            timeLabel.innerText = `Segment: ${val.toFixed(2)}s to ${(val + segmentDuration).toFixed(2)}s`;
            modalVidPlayer.currentTime = val;
        }

        timelineSlider.addEventListener('input', () => { updateTimeline(); modalVidPlayer.pause(); });

        // Force browser to mathematically render exactly the 16 frames the backend will extract
        document.getElementById('btnPreviewSegment').onclick = async () => {
            const btn = document.getElementById('btnPreviewSegment');
            if (btn.disabled) return;
            btn.disabled = true;
            modalVidPlayer.pause();

            const startVal = parseFloat(timelineSlider.value);

            for(let i = 0; i < 16; i++) {
                modalVidPlayer.currentTime = startVal + (i / fpsEstimate);
                // Await browser render pipeline
                await new Promise(r => {
                    const onSeeked = () => { modalVidPlayer.removeEventListener('seeked', onSeeked); r(); };
                    modalVidPlayer.addEventListener('seeked', onSeeked);
                    // Safe fallback if frame is already buffered
                    setTimeout(() => { modalVidPlayer.removeEventListener('seeked', onSeeked); r(); }, 100);
                });
                // Hold frame to simulate playback speed
                await new Promise(r => setTimeout(r, 1000 / fpsEstimate));
            }

            modalVidPlayer.currentTime = startVal;
            btn.disabled = false;
        };

        document.getElementById('videoUpload').onchange = (e) => {
            if (!isModelLoaded) {
                alert("Model Offline. Please select and initialize a checkpoint first.");
                e.target.value = ''; 
                return;
            }
            if(e.target.files.length > 0) {
                batchFiles = Array.from(e.target.files);
                batchConfigs =[];
                startConfigFlow(0);
            }
        };

        function startConfigFlow(idx) {
            currentConfigIdx = idx;
            const file = batchFiles[idx];
            modalVidPlayer.src = URL.createObjectURL(file);

            const step1Title = document.getElementById('step1Title');
            const step2Title = document.getElementById('step2Title');
            const btnRunZeroShot = document.getElementById('btnRunZeroShot');

            if (batchFiles.length > 1) {
                step1Title.innerText = `Step 1: Isolate Sequence (${idx + 1}/${batchFiles.length})`;
                step2Title.innerText = `Step 2: Track & Crop Box (${idx + 1}/${batchFiles.length})`;
                document.getElementById('step1Subtitle').innerText = `Select timespan for: ${file.name}`;
                if (idx < batchFiles.length - 1) {
                    btnRunZeroShot.innerHTML = `Configure Next Video <span class="material-symbols-rounded text-[20px]">arrow_forward</span>`;
                } else {
                    btnRunZeroShot.innerHTML = `<span class="material-symbols-rounded text-[20px]">psychology</span> Extract Physics Batch`;
                }
            } else {
                step1Title.innerText = `Step 1: Isolate Sequence`;
                step2Title.innerText = `Step 2: Track & Crop Box`;
                document.getElementById('step1Subtitle').innerText = `Slide the window to extract exactly 16 frames.`;
                btnRunZeroShot.innerHTML = `<span class="material-symbols-rounded text-[20px]">psychology</span> Extract Features`;
            }

            prepModal.classList.remove('hidden');
            modalStep1.classList.remove('hidden');
            modalStep2.classList.add('hidden');
            if (modalVidPlayer.readyState >= 1) initTimeline();
        }

        document.getElementById('btnCloseModal').onclick = () => {
            prepModal.classList.add('hidden'); modalVidPlayer.pause();
        };

        document.getElementById('btnNextStep').onclick = () => {
            modalVidPlayer.pause(); 
            step2Scrubber.value = 0; step2FrameLabel.innerText = 'Frame 1/16';
            modalVidFrame.src = modalVidPlayer.src; modalVidFrame.currentTime = timelineSlider.value;
            modalStep1.classList.add('hidden'); modalStep2.classList.remove('hidden');

            setTimeout(() => {
                const vw = modalVidFrame.videoWidth; const vh = modalVidFrame.videoHeight;
                const maxDisplayHeight = window.innerHeight * 0.45;
                displayScale = Math.min(1.0, maxDisplayHeight / vh);
                const dispW = vw * displayScale; const dispH = vh * displayScale;

                videoWrapper.style.width = dispW + 'px'; videoWrapper.style.height = dispH + 'px';
                modalVidFrame.style.width = dispW + 'px'; modalVidFrame.style.height = dispH + 'px';

                let initialBoxDispSize = 384 * displayScale;
                strictCropBox.style.width = initialBoxDispSize + 'px'; strictCropBox.style.height = initialBoxDispSize + 'px';
                strictCropBox.style.display = 'block';

                const leftPos = Math.max(0, (dispW / 2) - (initialBoxDispSize / 2));
                const topPos = Math.max(0, (dispH / 2) - (initialBoxDispSize / 2));
                strictCropBox.style.left = leftPos + 'px'; strictCropBox.style.top = topPos + 'px';
                cropSizeLabel.innerText = Math.round(initialBoxDispSize / displayScale) + 'x' + Math.round(initialBoxDispSize / displayScale);
            }, 250);
        };

        document.getElementById('btnBackStep').onclick = () => { modalStep2.classList.add('hidden'); modalStep1.classList.remove('hidden'); };

        step2Scrubber.addEventListener('input', (e) => {
            const frameOffset = parseInt(e.target.value);
            modalVidFrame.currentTime = parseFloat(timelineSlider.value) + (frameOffset / fpsEstimate);
            step2FrameLabel.innerText = `Frame ${frameOffset + 1}/16`;
        });

        let isDragging = false, isResizing = false;
        let startX, startY, startLeft, startTop, startWidth;

        strictCropBox.onmousedown = (e) => {
            if (e.target === cropResizeHandle) return; 
            isDragging = true; startX = e.clientX; startY = e.clientY;
            startLeft = parseFloat(strictCropBox.style.left) || 0; startTop = parseFloat(strictCropBox.style.top) || 0;
            e.preventDefault();
        };

        cropResizeHandle.onmousedown = (e) => {
            isResizing = true; startX = e.clientX; startWidth = strictCropBox.offsetWidth;
            e.stopPropagation(); e.preventDefault();
        };

        window.onmousemove = (e) => {
            if (isDragging) {
                const dx = e.clientX - startX; const dy = e.clientY - startY;
                strictCropBox.style.left = Math.max(0, Math.min(videoWrapper.offsetWidth - strictCropBox.offsetWidth, startLeft + dx)) + 'px';
                strictCropBox.style.top = Math.max(0, Math.min(videoWrapper.offsetHeight - strictCropBox.offsetHeight, startTop + dy)) + 'px';
            } else if (isResizing) {
                const dx = e.clientX - startX; let newSize = startWidth + dx;
                const minAllowed = 384 * displayScale;
                const currentLeft = parseFloat(strictCropBox.style.left) || 0;
                const currentTop = parseFloat(strictCropBox.style.top) || 0;
                const maxAllowedW = videoWrapper.offsetWidth - currentLeft;
                const maxAllowedH = videoWrapper.offsetHeight - currentTop;
                const absoluteMax = Math.max(minAllowed, Math.min(maxAllowedW, maxAllowedH));

                newSize = Math.max(minAllowed, Math.min(newSize, absoluteMax));
                strictCropBox.style.width = newSize + 'px'; strictCropBox.style.height = newSize + 'px';
                const nativeSize = Math.round(newSize / displayScale);
                cropSizeLabel.innerText = nativeSize + 'x' + nativeSize;
            }
        };

        window.onmouseup = () => { isDragging = false; isResizing = false; };

        function loadBatchResult(idx) {
            currentBatchIdx = parseInt(idx);
            const data = batchResults[currentBatchIdx];

            document.getElementById('stat_ep').innerText = data.ep_idx;
            document.getElementById('stat_pred_z').innerText = data.pred_z;
            document.getElementById('stat_pred_uv').innerText = data.pred_uv;

            // Mask GT data for Custom Zero-Shot Tasks
            document.getElementById('stat_gt_z').innerText = "N/A";
            document.getElementById('stat_gt_uv').innerText = "N/A";
            document.getElementById('stat_err').innerText = "N/A";

            if (batchResults.length > 1) {
                document.getElementById('batchLabel').innerText = `${currentBatchIdx + 1} / ${batchResults.length}`;
            }

            initSequence(data.frames);
        }

        document.getElementById('batchSlider').addEventListener('input', (e) => {
            stopSequence(); loadBatchResult(e.target.value);
        });

        document.getElementById('btnRunZeroShot').onclick = () => {
            const nativeX = Math.round((parseFloat(strictCropBox.style.left) || 0) / displayScale);
            const nativeY = Math.round((parseFloat(strictCropBox.style.top) || 0) / displayScale);
            const nativeSize = Math.round(strictCropBox.offsetWidth / displayScale);
            const startTime = timelineSlider.value;

            batchConfigs.push({
                file: batchFiles[currentConfigIdx], startTime: startTime, cropX: nativeX, cropY: nativeY, cropSize: nativeSize
            });

            if (currentConfigIdx < batchFiles.length - 1) {
                startConfigFlow(currentConfigIdx + 1);
            } else {
                prepModal.classList.add('hidden'); runBatchInference();
            }
        };

        async function runBatchInference() {
            resetOutput();
            document.getElementById('emptyState').classList.add('hidden');
            document.getElementById('loader').classList.remove('hidden');
            document.getElementById('loader').classList.add('flex');

            batchResults =[]; currentBatchIdx = 0;

            for(let i = 0; i < batchConfigs.length; i++) {
                const config = batchConfigs[i];
                if (batchConfigs.length > 1) document.getElementById('loaderText').innerText = `Processing Video ${i + 1} of ${batchConfigs.length}...`;

                const fd = new FormData();
                fd.append('video', config.file);
                fd.append('start_time', config.startTime);
                fd.append('crop_x', config.cropX); fd.append('crop_y', config.cropY); fd.append('crop_size', config.cropSize);

                try {
                    const res = await fetch('/api/zero_shot', { method: 'POST', body: fd });
                    const data = await res.json();
                    data.ep_idx = config.file.name;
                    batchResults.push(data);
                } catch (e) { console.error(`Inference failed on file`); }
            }

            document.getElementById('loader').classList.add('hidden');
            document.getElementById('loader').classList.remove('flex');

            if (batchResults.length > 0) {
                if (batchResults.length > 1) {
                    document.getElementById('zeroShotBatchControls').classList.remove('hidden');
                    const batchSlider = document.getElementById('batchSlider');
                    batchSlider.max = batchResults.length - 1; batchSlider.value = 0;
                }
                loadBatchResult(0);
            } else {
                document.getElementById('emptyState').classList.remove('hidden');
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_CONTENT


if __name__ == "__main__":
    print("✔ Starting STATERA Web Engine...")
    uvicorn.run(app, host="0.0.0.0", port=8000)