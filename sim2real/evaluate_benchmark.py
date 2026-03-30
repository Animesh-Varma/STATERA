"""
4_evaluate_benchmark.py
STATERA Benchmarking Script. Ingests the extracted Sim-to-Real MP4s and JSON targets,
runs a forward pass to predict the physical state of the 16th frame, and calculates
the explicit mathematical error across the dataset.

Updates:
- Normalizes Depth (Z) to positive magnitude for evaluation.
- Includes Centroid Offset Delta proof for mass asymmetry tracking.
"""

import os
import sys
import glob
import json
import cv2
import math
import torch
import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
training_dir = os.path.join(project_root, 'training')

if training_dir not in sys.path:
    sys.path.append(training_dir)

# noinspection PyUnresolvedReferences
from model import StateraModel


def get_var(key, prompt_msg, cast_type, default):
    """Helper to load/save user variables."""
    config_file = 'variables.txt'
    config = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except:
            pass

    if key not in config:
        val = input(f"{prompt_msg} (Default: {default}) -> ").strip()
        if not val:
            config[key] = default
        else:
            try:
                config[key] = cast_type(val)
            except ValueError:
                print(f"[!] Invalid input. Using default: {default}")
                config[key] = default

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)

    return config[key]


def get_subpixel_coords(logits_heatmap, device):
    """Calculates the expected-value float coordinates from the raw 64x64 logits."""
    B, H, W = logits_heatmap.shape
    probs = torch.sigmoid(logits_heatmap).view(B, -1)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
    probs = probs.view(B, H, W)

    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    y_center = (probs * y_grid).float().sum(dim=(1, 2))
    x_center = (probs * x_grid).float().sum(dim=(1, 2))

    return torch.stack([x_center, y_center], dim=1)


def evaluate_statera():
    print("=" * 60)
    print(" STATERA: Physics Evaluation (Normalized Depth)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing Inference Engine on {device}...")

    try:
        mtx = np.load('camera_matrix.npy')
        dist = np.load('dist_coeffs.npy')
    except FileNotFoundError:
        print("\n[!] ERROR: camera_matrix.npy or dist_coeffs.npy missing.")
        return

    model_path = os.path.join(training_dir, "best_statera_probe.pth")
    if not os.path.exists(model_path):
        print(f"\n[!] ERROR: Missing '{model_path}'.")
        return

    model = StateraModel(heatmap_res=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model.eval()

    target_dir = get_var('output_dir', 'Path to directory containing output videos and JSONs', str, 'output')
    mp4_files = sorted(glob.glob(os.path.join(target_dir, "*_statera.mp4")))

    if not mp4_files:
        print(f"\n[!] ERROR: No videos found in '{target_dir}'.")
        return

    print(f"[INFO] Discovered {len(mp4_files)} sequences. Starting Audit...\n")

    spatial_errors = []
    depth_errors = []
    dist_true_to_centroid_list = []
    dist_pred_to_centroid_list = []

    valid_count = 0

    with torch.no_grad():
        for mp4_path in mp4_files:
            base_name = mp4_path.replace("_statera.mp4", "")
            json_path = base_name + "_benchmark.json"

            if not os.path.exists(json_path):
                continue

            # Load Video
            cap = cv2.VideoCapture(mp4_path)
            frames = []
            for _ in range(16):
                ret, frame = cap.read()
                if not ret: break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            if len(frames) != 16:
                continue

            frames_np = np.stack(frames)
            vid_tensor = torch.from_numpy(frames_np).float() / 255.0
            vid_tensor = vid_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(device)

            with open(json_path, 'r') as f:
                target_data = json.load(f)

            frame_16 = target_data["frames"][15]

            # --- DEPTH NORMALIZATION ---
            # Ensure target is a positive distance
            target_z = abs(float(frame_16["z_depth_meters"]))
            target_u = frame_16["com_u"]
            target_v = frame_16["com_v"]

            # --- CENTROID RECONSTRUCTION ---
            bx, by, bz = target_data["metadata"]["box_size_m"]
            hx, hy, hz = bx / 2, by / 2, bz / 2
            box_3d = np.array([
                [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz]
            ], dtype=np.float32)

            all_xs, all_ys = [], []
            for f_data in target_data["frames"][:16]:
                r_v = np.array(f_data["rvec"], dtype=np.float32)
                t_v = np.array(f_data["tvec"], dtype=np.float32)
                box_2d, _ = cv2.projectPoints(box_3d, r_v, t_v, mtx, dist)
                all_xs.extend(box_2d[:, 0, 0])
                all_ys.extend(box_2d[:, 0, 1])

            cx, cy = (int(np.min(all_xs)) + int(np.max(all_xs))) // 2, (int(np.min(all_ys)) + int(np.max(all_ys))) // 2
            target_size = int(max(np.max(all_xs) - np.min(all_xs), np.max(all_ys) - np.min(all_ys)) * 3.0)
            target_size += target_size % 2
            x1, y1 = cx - (target_size // 2), cy - (target_size // 2)
            scale = 384.0 / target_size

            # Dynamic centroid for frame 16
            box_2d_16, _ = cv2.projectPoints(box_3d, np.array(frame_16["rvec"]), np.array(frame_16["tvec"]), mtx, dist)
            centroid_u = ((np.min(box_2d_16[:, 0, 0]) + np.max(box_2d_16[:, 0, 0])) / 2.0 - x1) * scale
            centroid_v = ((np.min(box_2d_16[:, 0, 1]) + np.max(box_2d_16[:, 0, 1])) / 2.0 - y1) * scale

            # --- INFERENCE ---
            pred_h, pred_z = model(vid_tensor)

            sub_uv = get_subpixel_coords(pred_h, device)[0].cpu().numpy()
            pred_u, pred_v = sub_uv[0] * 6.0, sub_uv[1] * 6.0

            # --- DEPTH NORMALIZATION ---
            # Ensure prediction is treated as a positive distance
            pred_z_val = abs(float(pred_z[0].item()))

            # Metrics
            l2_dist = math.sqrt((pred_u - target_u)**2 + (pred_v - target_v)**2)
            abs_z_err = abs(pred_z_val - target_z)

            dist_true_to_centroid = math.sqrt((target_u - centroid_u)**2 + (target_v - centroid_v)**2)
            dist_pred_to_centroid = math.sqrt((pred_u - centroid_u)**2 + (pred_v - centroid_v)**2)

            spatial_errors.append(l2_dist)
            depth_errors.append(abs_z_err)
            dist_true_to_centroid_list.append(dist_true_to_centroid)
            dist_pred_to_centroid_list.append(dist_pred_to_centroid)

            valid_count += 1
            print(f"   -> [Seq {valid_count:03d}] UV-Err: {l2_dist:5.2f}px | Z-Err: {abs_z_err:6.4f}m (Pred: {pred_z_val:.4f}m)")

    # Aggregation
    if valid_count == 0: return

    print("\n" + "=" * 60)
    print(" BENCHMARK RESULTS")
    print("=" * 60)
    print(f" Total Videos Validated : {valid_count}")
    print(f" Mean Spatial Error     : {mean_spatial:.2f} Pixels")
    print(f" Max Spatial Error      : {max_spatial:.2f} Pixels")
    print(f" Mean Depth Error       : {mean_depth:.4f} Meters")
    print("-" * 60)
    print(" KINEMATICS PROOF (Centroid Offset Delta)")
    print("-" * 60)
    print(f" 1. Distance (True CoM to Geometric Center) : {mean_true_to_centroid:.2f} px")
    print(f" 2. Distance (Pred CoM to Geometric Center) : {mean_pred_to_centroid:.2f} px")
    print(f" 3. True Tracking Error (Pred to True CoM)  : {mean_spatial:.2f} px")
    print("\n CONCLUSION:")

    if mean_spatial < mean_pred_to_centroid:
        print(" [SUCCESS] The Prediction Error is smaller than the distance to the")
        print(" visual center. This proves the network is mathematically tracking")
        print(" the physical mass offset trajectory, not just guessing the middle")
        print(" of the bounding box!")
    else:
        print(" [WARNING] The prediction is closer to the visual center than it")
        print(" is to the True CoM. The network may be collapsing its predictions")
        print(" to the geometric centroid instead of learning mass asymmetry.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    evaluate_statera()