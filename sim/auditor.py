import h5py
import numpy as np
import cv2
import sys


def generate_2d_gaussian_heatmap(image_shape, center, sigma=12):
    """
    Generates a 2D Gaussian Heatmap representing the CoM target.
    """
    x_coords = np.arange(0, image_shape[1], 1, float)
    y_coords = np.arange(0, image_shape[0], 1, float)[:, np.newaxis]
    center_u, center_v = center

    heatmap = np.exp(-((x_coords - center_u) ** 2 + (y_coords - center_v) ** 2) / (2 * sigma ** 2))
    return np.uint8(255 * heatmap)


def audit_dataset(hdf5_path="statera_poc.hdf5"):
    print("Opening STATERA PoC Dataset...")
    print("Controls:")
    print("  [Space] Play/Pause | [Left/Right Arrows] Step Frames (When Paused)")
    print("  [4] Decrease Heatmap | [6] Increase Heatmap")
    print("  [H] Toggle Overlays  | [S] Toggle 8 FPS View | [Q] Quit")

    try:
        dataset_file = h5py.File(hdf5_path, "r")
    except FileNotFoundError:
        print(f"Error: Could not find {hdf5_path}. Run physics_env.py first.")
        sys.exit(1)

    videos_ds = dataset_file["videos"]
    uv_ds = dataset_file["uv_coords"]
    z_ds = dataset_file["z_depths"]

    num_episodes = videos_ds.shape[0]
    if num_episodes == 0:
        print("Error: HDF5 dataset is empty.")
        sys.exit(1)

    # State variables
    episode_index = 0
    frame_idx = 0
    pause_video = False
    show_overlays = True
    is_slow_motion = False

    # Heatmap config
    sigma_val = 12
    alpha_pct = 60

    target_fps_normal = 24
    target_fps_model = 8

    window_name = "Project STATERA - Auditor"
    cv2.namedWindow(window_name)

    # Cross-platform Arrow Key Codes for cv2.waitKeyEx()
    # (Windows, Linux X11, macOS)
    LEFT_ARROW_KEYS = (2424832, 65361, 63234)
    RIGHT_ARROW_KEYS = (2555904, 65363, 63235)

    while True:
        frames = np.transpose(videos_ds[episode_index][:], (0, 2, 3, 1))
        uv_coords = uv_ds[episode_index][:]
        z_depths = z_ds[episode_index][:]

        current_fps = target_fps_model if is_slow_motion else target_fps_normal
        frame_delay = 50 if pause_video else int(1000 / current_fps)

        frame_bgr = cv2.cvtColor(frames[frame_idx], cv2.COLOR_RGB2BGR)

        u, v = uv_coords[frame_idx]
        z = z_depths[frame_idx][0]

        if show_overlays:
            heatmap_alpha = alpha_pct / 100.0
            frame_alpha = 1.0 - heatmap_alpha

            heatmap_gray = generate_2d_gaussian_heatmap((224, 224), center=(u, v), sigma=sigma_val)
            heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)

            mask = (heatmap_gray > 10).astype(np.uint8)[:, :, np.newaxis]
            composite = np.where(mask, cv2.addWeighted(frame_bgr, frame_alpha, heatmap_color, heatmap_alpha, 0),
                                 frame_bgr)
        else:
            composite = frame_bgr

        display_img = cv2.resize(composite, (512, 512), interpolation=cv2.INTER_NEAREST)

        if show_overlays:
            cv2.putText(display_img, f"Ep: {episode_index} | Frame: {frame_idx + 1}/16", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_img, f"Z-Depth: {z:.3f} m", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(display_img, f"CoM (U,V): {int(u)}, {int(v)} | Sigma: {sigma_val}", (15, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)

            fps_text = f"FPS: {current_fps} {'(MODEL VIEW)' if is_slow_motion else '(NORMAL)'}"
            if pause_video:
                fps_text = "PAUSED (Use Arrows to Step)"

            cv2.putText(display_img, fps_text, (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255) if is_slow_motion else (0, 255, 0), 2)

            # THE FIX: Using round() to prevent sub-pixel drift during 512x resize
            scale_factor = 512 / 224.0
            marker_x = int(round(u * scale_factor))
            marker_y = int(round(v * scale_factor))

            cv2.drawMarker(display_img, (marker_x, marker_y), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
        else:
            cv2.putText(display_img, "RAW V-JEPA INPUT MODE", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200),
                        2)
            cv2.putText(display_img, "PAUSED" if pause_video else f"FPS: {current_fps}", (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow(window_name, display_img)

        # waitKeyEx captures extended keys like Arrows natively
        key = cv2.waitKeyEx(frame_delay)

        if key != -1:
            # Handle Arrow Keys for Frame Stepping (Only when paused)
            if pause_video and key in LEFT_ARROW_KEYS:
                frame_idx -= 1
                if frame_idx < 0:
                    frame_idx = 15
                    episode_index = (episode_index - 1) % num_episodes
            elif pause_video and key in RIGHT_ARROW_KEYS:
                frame_idx += 1
                if frame_idx >= 16:
                    frame_idx = 0
                    episode_index = (episode_index + 1) % num_episodes

            # Handle Standard Keys
            elif key < 256:
                char = chr(key & 0xFF).lower()
                if char == 'q':
                    dataset_file.close()
                    cv2.destroyAllWindows()
                    return
                elif char == ' ':
                    pause_video = not pause_video
                elif char == 'h':
                    show_overlays = not show_overlays
                elif char == 's':
                    is_slow_motion = not is_slow_motion
                elif char == '4':
                    sigma_val = max(1, sigma_val - 1)
                elif char == '6':
                    sigma_val += 1

        # Standard Playback Advance
        if not pause_video:
            frame_idx += 1
            if frame_idx >= 16:
                frame_idx = 0
                episode_index = (episode_index + 1) % num_episodes


if __name__ == "__main__":
    audit_dataset()