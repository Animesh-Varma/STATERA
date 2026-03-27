import os
import sys
import torch
import h5py
import numpy as np
import cv2

# Import your model architecture
from model import StateraModel


# ==========================================
# SUB-PIXEL EXTRACTION LOGIC
# ==========================================
def get_subpixel_coords(logits_heatmap):
    """ Extracts theoretical floating-point CoM from the 64x64 heatmap """
    B, H, W = logits_heatmap.shape
    device = logits_heatmap.device
    probs = torch.sigmoid(logits_heatmap).view(B, -1)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
    probs = probs.view(B, H, W)

    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    y_center = (probs * y_grid).float().sum(dim=(1, 2))
    x_center = (probs * x_grid).float().sum(dim=(1, 2))

    return torch.stack([x_center, y_center], dim=1)


# ==========================================
# LIQUID GLASS UI HELPERS
# ==========================================
def apply_glass_morphism(bg_image, x, y, w, h, blur_ksize=(45, 45), alpha=0.6, darken=0.4):
    """ Creates a beautiful 'Liquid Glass' frosted acrylic panel over a region. """
    roi = bg_image[y:y + h, x:x + w]
    blurred_roi = cv2.GaussianBlur(roi, blur_ksize, 0)
    darkened_roi = cv2.convertScaleAbs(blurred_roi, alpha=darken)

    overlay = np.zeros_like(roi, dtype=np.uint8)
    overlay[:] = (30, 40, 50)

    glass_panel = cv2.addWeighted(darkened_roi, 1.0, overlay, alpha, 0)
    bg_image[y:y + h, x:x + w] = glass_panel

    cv2.rectangle(bg_image, (x, y), (x + w, y + h), (255, 255, 255), 1)
    cv2.line(bg_image, (x + 1, y + 1), (x + w - 1, y + 1), (200, 200, 200), 1)
    cv2.line(bg_image, (x + 1, y + 1), (x + 1, y + h - 1), (200, 200, 200), 1)


def draw_neon_marker(img, center, color, style='cross', radius=10):
    """ Draws a glowing marker (thick blurred base + bright core) """
    x, y = int(center[0]), int(center[1])
    core_color = (255, 255, 255)

    if style == 'cross':
        cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, radius + 4, 4, cv2.LINE_AA)
        cv2.drawMarker(img, (x, y), core_color, cv2.MARKER_CROSS, radius, 1, cv2.LINE_AA)
    elif style == 'circle':
        cv2.circle(img, (x, y), radius + 2, color, 4, cv2.LINE_AA)
        cv2.circle(img, (x, y), radius, core_color, 1, cv2.LINE_AA)


# ==========================================
# MAIN VIEWER APPLICATION
# ==========================================
def run_statera_lens(hdf5_path="../sim/statera_poc.hdf5", model_path="best_statera_probe.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing STATERA Lens on {device}...")

    if not os.path.exists(model_path):
        print(f"✘ Error: Model weights '{model_path}' not found! Run train.py first.")
        sys.exit(1)

    model = StateraModel(heatmap_res=64).to(device)
    model.load_state_dict(torch.load('best_statera_probe.pth'), strict=False)
    model.eval()
    print("✓ Model weights loaded successfully.")

    print("Loading HDF5 dataset into RAM...")
    with h5py.File(hdf5_path, "r") as f:
        videos_ds = f["videos"][:]
        uv_ds = f["uv_coords"][:]
        z_ds = f["z_depths"][:]
    print("✓ Dataset loaded.")
    num_episodes = len(videos_ds)

    ep_idx = 0
    frame_idx = 0
    is_paused = False

    video_res = 512
    panel_width = 400
    canvas_w = video_res + panel_width
    canvas_h = video_res

    window_name = "STATERA Lens - Material Inference Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, canvas_w, canvas_h)

    COLOR_GT = (50, 255, 50)
    COLOR_PRED = (255, 200, 0)
    COLOR_TEXT = (240, 240, 245)

    current_pred_uv = None
    current_pred_z = None

    while True:
        # --- INFERENCE TRIGGER ---
        if frame_idx == 0 and current_pred_uv is None:
            vid_array = videos_ds[ep_idx]
            vid_tensor = torch.from_numpy(vid_array).float() / 255.0
            vid_tensor = vid_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_h, pred_z = model(vid_tensor)
                sub_uv = get_subpixel_coords(pred_h)[0].cpu().numpy()

                current_pred_uv = sub_uv * (384.0 / 64.0)
                current_pred_z = pred_z[0].item()

                # NEW: Extract the raw heatmap to visualize it!
                raw_hm = torch.sigmoid(pred_h)[0].cpu().numpy()  # Shape: (64, 64)

        # --- DATA EXTRACTION ---
        frame_rgb = np.transpose(videos_ds[ep_idx][frame_idx], (1, 2, 0))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, (video_res, video_res), interpolation=cv2.INTER_CUBIC)

        # NEW: Overlay the AI's actual Heatmap guess onto the video
        hm_resized = cv2.resize(raw_hm, (video_res, video_res))
        # Normalize heatmap for visualization (scales the brightest pixel to 255)
        hm_vis = np.uint8(255 * (hm_resized / (hm_resized.max() + 1e-8)))
        hm_colored = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)

        # Blend it where the AI is confident (e.g., top 50% of its guess)
        mask = hm_vis > 120
        frame_bgr[mask] = cv2.addWeighted(frame_bgr, 0.4, hm_colored, 0.6, 0)[mask]

        gt_uv_seq = uv_ds[ep_idx].reshape(-1, 2)
        gt_z_seq = z_ds[ep_idx].flatten()

        curr_gt_uv = gt_uv_seq[frame_idx]
        final_gt_uv = gt_uv_seq[-1]
        final_gt_z = float(gt_z_seq[-1])

        # --- CANVAS CREATION ---
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :video_res] = frame_bgr

        edge_slice = frame_bgr[:, -50:]
        stretched_edge = cv2.resize(edge_slice, (panel_width, canvas_h))
        canvas[:, video_res:] = stretched_edge

        apply_glass_morphism(canvas, video_res + 20, 20, panel_width - 40, canvas_h - 40)

        # --- DRAW MARKERS ON VIDEO ---
        scale_ratio = video_res / 384.0

        curr_gt_scaled = (curr_gt_uv[0] * scale_ratio, curr_gt_uv[1] * scale_ratio)
        draw_neon_marker(canvas, curr_gt_scaled, COLOR_GT, style='circle', radius=6)

        if current_pred_uv is not None:
            pred_scaled = (current_pred_uv[0] * scale_ratio, current_pred_uv[1] * scale_ratio)
            draw_neon_marker(canvas, pred_scaled, COLOR_PRED, style='cross', radius=12)

            if frame_idx == 15:
                final_gt_scaled = (final_gt_uv[0] * scale_ratio, final_gt_uv[1] * scale_ratio)
                cv2.line(canvas, (int(final_gt_scaled[0]), int(final_gt_scaled[1])),
                         (int(pred_scaled[0]), int(pred_scaled[1])), (255, 255, 255), 1, cv2.LINE_AA)

        # --- DRAW LIQUID GLASS DASHBOARD ---
        t_x = video_res + 40
        t_y = 60
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(canvas, "STATERA LENS", (t_x, t_y), font, 0.9, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.line(canvas, (t_x, t_y + 15), (t_x + 300, t_y + 15), (100, 100, 120), 1)

        t_y += 50
        cv2.putText(canvas, f"Episode: {ep_idx + 1} / {num_episodes}", (t_x, t_y), font, 0.6, COLOR_TEXT, 1,
                    cv2.LINE_AA)
        t_y += 30
        status_text = "PAUSED" if is_paused else "PLAYING 24FPS"
        cv2.putText(canvas, f"State:   {status_text}", (t_x, t_y), font, 0.6, (150, 150, 150), 1, cv2.LINE_AA)

        t_y += 60
        cv2.putText(canvas, "Z-DEPTH ANALYSIS", (t_x, t_y), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        t_y += 30
        cv2.putText(canvas, f"Truth:   {final_gt_z:.4f} m", (t_x, t_y), font, 0.6, COLOR_GT, 1, cv2.LINE_AA)
        t_y += 30
        if current_pred_z is not None:
            cv2.putText(canvas, f"Model:   {current_pred_z:.4f} m", (t_x, t_y), font, 0.6, COLOR_PRED, 1, cv2.LINE_AA)
            z_err = abs(final_gt_z - current_pred_z)
            t_y += 30
            cv2.putText(canvas, f"Error:   {z_err:.4f} m", (t_x, t_y), font, 0.6, (255, 100, 100), 1, cv2.LINE_AA)

        t_y += 60
        cv2.putText(canvas, "SPATIAL ANALYSIS (X, Y)", (t_x, t_y), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        t_y += 30
        cv2.putText(canvas, f"Truth:   {int(final_gt_uv[0])}, {int(final_gt_uv[1])}", (t_x, t_y), font, 0.6, COLOR_GT,
                    1, cv2.LINE_AA)
        t_y += 30
        if current_pred_uv is not None:
            cv2.putText(canvas, f"Model:   {int(current_pred_uv[0])}, {int(current_pred_uv[1])}", (t_x, t_y), font, 0.6,
                        COLOR_PRED, 1, cv2.LINE_AA)
            pixel_err = np.linalg.norm(final_gt_uv - current_pred_uv)
            t_y += 30
            cv2.putText(canvas, f"Error:   {pixel_err:.1f} pixels", (t_x, t_y), font, 0.6, (255, 100, 100), 1,
                        cv2.LINE_AA)

        t_y = canvas_h - 60
        cv2.putText(canvas, "[Space] Play/Pause  |  [Q] Quit", (t_x, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(canvas, "[Left/Right] Scrub Timeline", (t_x, t_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (150, 150, 150), 1, cv2.LINE_AA)

        bar_w = video_res - 40
        bar_x, bar_y = 20, canvas_h - 20
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 4), (50, 50, 50), -1)
        progress_w = int((frame_idx / 15.0) * bar_w)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + progress_w, bar_y + 4), COLOR_PRED, -1)
        cv2.circle(canvas, (bar_x + progress_w, bar_y + 2), 6, (255, 255, 255), -1)

        cv2.imshow(window_name, canvas)

        delay = 0 if is_paused else int(1000 / 24)
        key = cv2.waitKeyEx(delay)

        if key != -1:
            if key in (2424832, 65361, 63234):
                is_paused = True
                frame_idx = max(0, frame_idx - 1)
            elif key in (2555904, 65363, 63235):
                is_paused = True
                if frame_idx == 15:
                    ep_idx = (ep_idx + 1) % num_episodes
                    frame_idx = 0
                    current_pred_uv = None
                else:
                    frame_idx += 1
            elif key < 256:
                char = chr(key & 0xFF).lower()
                if char == 'q':
                    break
                elif char == ' ':
                    is_paused = not is_paused

        if not is_paused:
            frame_idx += 1
            if frame_idx >= 16:
                frame_idx = 0
                ep_idx = (ep_idx + 1) % num_episodes
                current_pred_uv = None

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_statera_lens()