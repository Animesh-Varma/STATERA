import h5py
import numpy as np
import cv2


def generate_2d_gaussian_heatmap(shape, center, sigma=12):
    x = np.arange(0, shape[1], 1, float)
    y = np.arange(0, shape[0], 1, float)[:, np.newaxis]
    u, v = center
    heatmap = np.exp(-((x - u) ** 2 + (y - v) ** 2) / (2 * sigma ** 2))
    return np.uint8(255 * heatmap)


def audit_dataset(hdf5_path="statera_poc.hdf5"):
    print("Opening STATERA PoC Dataset...")
    print("Controls: [Spacebar] Pause/Play |[H] Toggle Overlays | [Q] Quit")

    with h5py.File(hdf5_path, "r") as f:
        video_keys = [k for k in f.keys() if k.startswith("video_")]
        num_episodes = len(video_keys)

        episode_idx = 0
        pause = False
        show_overlays = True  # Toggled with 'H'

        while True:
            frames = f[f"video_{episode_idx}"][:]
            labels = f[f"label_{episode_idx}"][:]

            for i in range(len(frames)):
                frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
                u, v, z = labels[i]

                if show_overlays:
                    heatmap_gray = generate_2d_gaussian_heatmap((224, 224), center=(u, v))
                    heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_HOT)
                    composite = cv2.add(frame_bgr, heatmap_color)
                else:
                    composite = frame_bgr  # Raw image, no heatmap

                display_img = cv2.resize(composite, (512, 512), interpolation=cv2.INTER_NEAREST)

                if show_overlays:
                    cv2.putText(display_img, f"Ep: {episode_idx} | Frame: {i}/16", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                    cv2.putText(display_img, f"Z-Depth: {z:.3f} m", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 200, 255), 2)
                    cv2.putText(display_img, f"CoM (U,V): {int(u)}, {int(v)}", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 100, 100), 2)

                    scale = 512 / 224.0
                    cv2.drawMarker(display_img, (int(u * scale), int(v * scale)), (255, 255, 255), cv2.MARKER_CROSS, 10,
                                   1)
                else:
                    cv2.putText(display_img, "RAW V-JEPA INPUT MODE", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (200, 200, 200), 2)

                cv2.imshow("Project STATERA - Auditor", display_img)

                key = cv2.waitKey(0 if pause else 150) & 0xFF

                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                elif key == ord(' '):
                    pause = not pause
                elif key == ord('h'):
                    show_overlays = not show_overlays  # Toggle visibility

            if not pause:
                episode_idx = (episode_idx + 1) % num_episodes


if __name__ == "__main__":
    audit_dataset()