import h5py
import numpy as np
import cv2


def generate_2d_gaussian_heatmap(shape, center, sigma=12):
    """
    Generates a 2D Gaussian Heatmap centered at (U, V).
    This mimics the exact target probability distribution your V-JEPA MLP will learn to output.
    """
    x = np.arange(0, shape[1], 1, float)
    y = np.arange(0, shape[0], 1, float)
    y = y[:, np.newaxis]  # Column vector for broadcasting

    u, v = center
    # Create the mathematical Gaussian distribution
    heatmap = np.exp(-((x - u) ** 2 + (y - v) ** 2) / (2 * sigma ** 2))

    # Normalize to 0-255 for image display
    return np.uint8(255 * heatmap)


def audit_dataset(hdf5_path="statera_poc.hdf5"):
    print("Opening STATERA PoC Dataset for Auditing...")
    print("Controls: [Spacebar] to Pause/Play | [Q] to Quit | Any other key to advance frame")

    with h5py.File(hdf5_path, "r") as f:
        # Get list of all video keys (e.g., 'video_0', 'video_1', ...)
        video_keys = [k for k in f.keys() if k.startswith("video_")]
        num_episodes = len(video_keys)
        print(f"Found {num_episodes} tumbling episodes.")

        episode_idx = 0
        pause = False

        while True:
            video_key = f"video_{episode_idx}"
            label_key = f"label_{episode_idx}"

            frames = f[video_key][:]
            labels = f[label_key][:]

            for i in range(len(frames)):
                # 1. Get raw frame (Convert RGB from MuJoCo to BGR for OpenCV)
                frame = frames[i]
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # 2. Get Ground Truth Labels
                u, v, z = labels[i]

                # 3. Generate the AI-style Heatmap Overlay
                heatmap_gray = generate_2d_gaussian_heatmap((224, 224), center=(u, v))
                # Apply a "Hot" colormap (Black background, red/yellow/white hot center)
                heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_HOT)

                # Blend the heatmap with the original frame (using additive blending)
                # Since COLORMAP_HOT uses black for 0, adding it only brightens the hot spot!
                composite = cv2.add(frame_bgr, heatmap_color)

                # 4. Upscale the 224x224 image so it's easy to see on modern monitors
                display_size = 512
                display_img = cv2.resize(composite, (display_size, display_size), interpolation=cv2.INTER_NEAREST)

                # 5. Draw the Z-Depth and Data text on the upscaled image for crisp text
                text_color = (0, 255, 0)  # Green text
                cv2.putText(display_img, f"Episode: {episode_idx} | Frame: {i}/16", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

                # Display the Z-Depth
                cv2.putText(display_img, f"Z-Depth (1D Target): {z:.3f} m", (15, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                # Display the Exact Pixel Target
                cv2.putText(display_img, f"CoM (U, V): {int(u)}, {int(v)}", (15, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)

                # Draw a tiny exact crosshair at the dead center of the scaled U,V
                scale = display_size / 224.0
                center_pt = (int(u * scale), int(v * scale))
                cv2.drawMarker(display_img, center_pt, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=10,
                               thickness=1)

                # Show Video
                cv2.imshow("Project STATERA - Ground Truth Auditor", display_img)

                # Playback Controls
                delay = 0 if pause else 150  # 150ms delay for nice slow-motion tumbling
                key = cv2.waitKey(delay) & 0xFF

                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                elif key == ord(' '):  # Spacebar toggles pause
                    pause = not pause

            # Loop to next episode, or wrap around
            if not pause:
                episode_idx = (episode_idx + 1) % num_episodes


if __name__ == "__main__":
    audit_dataset()