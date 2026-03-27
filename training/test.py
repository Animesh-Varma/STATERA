import os
import h5py
import numpy as np
import cv2


def generate_2d_gaussian_heatmap(image_shape, center_u, center_v, sigma):
    """ Generates the Gaussian Heatmap mathematically """
    x_coords = np.arange(0, image_shape[1], 1, float)
    y_coords = np.arange(0, image_shape[0], 1, float)[:, np.newaxis]
    heatmap = np.exp(-((x_coords - center_u) ** 2 + (y_coords - center_v) ** 2) / (2 * max(sigma, 0.1) ** 2))
    return heatmap


# --- Keep your exact Liquid Glass functions ---
def apply_glass_morphism(bg_image, x, y, w, h, blur_ksize=(45, 45), alpha=0.6, darken=0.4):
    roi = bg_image[y:y + h, x:x + w]
    blurred_roi = cv2.GaussianBlur(roi, blur_ksize, 0)
    darkened_roi = cv2.convertScaleAbs(blurred_roi, alpha=darken)
    overlay = np.zeros_like(roi, dtype=np.uint8)
    overlay[:] = (30, 40, 50)
    glass_panel = cv2.addWeighted(darkened_roi, 1.0, overlay, alpha, 0)
    bg_image[y:y + h, x:x + w] = glass_panel
    cv2.rectangle(bg_image, (x, y), (x + w, y + h), (255, 255, 255), 1)


def run_sigma_visualizer(hdf5_path="../sim/statera_poc.hdf5"):
    print("Initializing STATERA Sigma Visualizer...")

    with h5py.File(hdf5_path, "r") as f:
        videos_ds = f["videos"][:]
        uv_ds = f["uv_coords"][:]
    print("✅ Dataset loaded.")

    ep_idx = 47  # Let's look at the cheating episode
    frame_idx = 15
    current_sigma = 3.0  # Starting Sigma

    video_res = 512
    panel_width = 400
    canvas_w = video_res + panel_width
    canvas_h = video_res

    window_name = "STATERA Lens - Sigma Tuning"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, canvas_w, canvas_h)

    # OpenCV Trackbar callback
    def on_trackbar(val):
        nonlocal current_sigma
        current_sigma = max(1.0, float(val) / 10.0)

    # Create a slider from 10 to 150 (represents Sigma 1.0 to 15.0)
    cv2.createTrackbar("Sigma x10", window_name, 30, 150, on_trackbar)

    while True:
        # --- DATA EXTRACTION ---
        frame_rgb = np.transpose(videos_ds[ep_idx][frame_idx], (1, 2, 0))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, (video_res, video_res), interpolation=cv2.INTER_CUBIC)

        final_gt_uv = uv_ds[ep_idx][-1]

        # Scale GT to video_res
        scale_ratio = video_res / 384.0
        gt_scaled = (final_gt_uv[0] * scale_ratio, final_gt_uv[1] * scale_ratio)

        # --- GENERATE AND OVERLAY HEATMAP ---
        heatmap = generate_2d_gaussian_heatmap((video_res, video_res), gt_scaled[0], gt_scaled[1],
                                               sigma=current_sigma * scale_ratio)

        # Convert heatmap to a color overlay (Viridis/Jet look cool)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

        # Blend heatmap with the video frame where heatmap > 0.05
        mask = heatmap > 0.05
        frame_bgr[mask] = cv2.addWeighted(frame_bgr, 0.4, heatmap_colored, 0.6, 0)[mask]

        # --- CANVAS CREATION ---
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :video_res] = frame_bgr

        edge_slice = frame_bgr[:, -50:]
        stretched_edge = cv2.resize(edge_slice, (panel_width, canvas_h))
        canvas[:, video_res:] = stretched_edge

        apply_glass_morphism(canvas, video_res + 20, 20, panel_width - 40, canvas_h - 40)

        # --- DRAW LIQUID GLASS DASHBOARD ---
        t_x = video_res + 40
        t_y = 60
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(canvas, "SIGMA VISUALIZER", (t_x, t_y), font, 0.9, (240, 240, 245), 1, cv2.LINE_AA)
        cv2.line(canvas, (t_x, t_y + 15), (t_x + 300, t_y + 15), (100, 100, 120), 1)

        t_y += 60
        cv2.putText(canvas, f"Current Sigma: {current_sigma:.1f}", (t_x, t_y), font, 0.7, (50, 255, 50), 1, cv2.LINE_AA)
        t_y += 30
        cv2.putText(canvas, "Adjust slider to ensure the", (t_x, t_y), font, 0.5, (200, 200, 200), 1)
        t_y += 20
        cv2.putText(canvas, "Heatmap gradient touches the", (t_x, t_y), font, 0.5, (200, 200, 200), 1)
        t_y += 20
        cv2.putText(canvas, "center of the screen (192, 192).", (t_x, t_y), font, 0.5, (200, 200, 200), 1)

        t_y = canvas_h - 40
        cv2.putText(canvas, "[Q] Quit", (t_x, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow(window_name, canvas)

        key = cv2.waitKeyEx(30)
        if key != -1 and chr(key & 0xFF).lower() == 'q':
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_sigma_visualizer()