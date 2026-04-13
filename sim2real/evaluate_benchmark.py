"""
4_evaluate_benchmark.py
STATERA Benchmarking Script.
*REFINED*:
- Evaluates Kinematic Jitter (Acceleration Deviation) to prove temporal smoothness.
- Dynamically loads exact Crop Bounds from the JSON to align mathematical spaces.
"""

import os
import sys
import glob
import json
import cv2
import math
import torch
import numpy as np
import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
training_dir = os.path.join(project_root, 'training')

if training_dir not in sys.path:
    sys.path.append(training_dir)

from model import StateraModel


def get_var(key, prompt_msg, cast_type, default):
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
    """Calculates expected-value float coordinates from the raw 64x64 sequence logits."""
    B, T, H, W = logits_heatmap.shape
    probs = torch.sigmoid(logits_heatmap).view(B * T, -1)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
    probs = probs.view(B, T, H, W)

    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    y_center = (probs * y_grid.float().view(1, 1, H, W)).sum(dim=(2, 3))
    x_center = (probs * x_grid.float().view(1, 1, H, W)).sum(dim=(2, 3))

    return torch.stack([x_center, y_center], dim=2)


def evaluate_statera():
    print("=" * 60)
    print(" STATERA: Temporal Sequence Physics Evaluation")
    print("=" * 60)

    wandb.init(project="STATERA", name="Sim-to-Real-Eval", job_type="evaluation")

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

    traj_spatial_errors, traj_depth_errors, traj_mag_errors = [], [], []
    term_spatial_errors, term_depth_errors, term_mag_errors = [], [], []
    kinematic_jitter_errors = [] # NEW: Tracks Acceleration Deviation

    term_mags_true, term_mags_pred = [], []
    term_mag_deltas, term_angles = [], []

    valid_count = 0

    with torch.no_grad():
        for mp4_path in mp4_files:
            base_name = mp4_path.replace("_statera.mp4", "")
            json_path = base_name + "_benchmark.json"

            if not os.path.exists(json_path): continue

            cap = cv2.VideoCapture(mp4_path)
            frames = []
            for _ in range(16):
                ret, frame = cap.read()
                if not ret: break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            if len(frames) != 16: continue

            frames_np = np.stack(frames)
            vid_tensor = torch.from_numpy(frames_np).float() / 255.0
            vid_tensor = vid_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(device)

            with open(json_path, 'r') as f:
                target_data = json.load(f)

            crop_params = target_data["metadata"]["crop_params"]
            x1 = crop_params["x1"]
            y1 = crop_params["y1"]
            scale = crop_params["scale"]

            pred_h, pred_z = model(vid_tensor)
            sub_uv_seq = get_subpixel_coords(pred_h, device)[0].cpu().numpy() * 6.0

            seq_spatial_err = 0.0
            seq_depth_err = 0.0
            seq_mag_err_px = 0.0

            true_traj = []
            pred_traj = []

            for frame_idx in range(16):
                f_data = target_data["frames"][frame_idx]

                target_z = abs(float(f_data["z_depth_meters"]))
                target_u = f_data["com_u"]
                target_v = f_data["com_v"]

                pred_u, pred_v = sub_uv_seq[frame_idx]
                pred_z_val = abs(float(pred_z[0, frame_idx, 0].item()))

                true_traj.append([target_u, target_v])
                pred_traj.append([pred_u, pred_v])

                center_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                center_2d_f, _ = cv2.projectPoints(center_3d, np.array(f_data["rvec"]), np.array(f_data["tvec"]), mtx, dist)

                centroid_u = (center_2d_f[0][0][0] - x1) * scale
                centroid_v = (center_2d_f[0][0][1] - y1) * scale

                l2_dist = math.sqrt((pred_u - target_u) ** 2 + (pred_v - target_v) ** 2)
                abs_z_err = abs(pred_z_val - target_z)

                v_true_f = np.array([target_u - centroid_u, target_v - centroid_v])
                v_pred_f = np.array([pred_u - centroid_u, pred_v - centroid_v])
                mag_true_px = np.linalg.norm(v_true_f)
                mag_pred_px = np.linalg.norm(v_pred_f)
                abs_m_err_px = abs(mag_pred_px - mag_true_px)

                traj_spatial_errors.append(l2_dist)
                traj_depth_errors.append(abs_z_err)
                traj_mag_errors.append(abs_m_err_px)

                seq_spatial_err += l2_dist
                seq_depth_err += abs_z_err
                seq_mag_err_px += abs_m_err_px

                if frame_idx == 15:
                    term_spatial_errors.append(l2_dist)
                    term_depth_errors.append(abs_z_err)
                    term_mag_errors.append(abs_m_err_px)

                    if mag_true_px > 1e-5 and mag_pred_px > 1e-5:
                        cos_theta = np.dot(v_true_f, v_pred_f) / (mag_true_px * mag_pred_px)
                        cos_theta = np.clip(cos_theta, -1.0, 1.0)
                        angle_deg = math.degrees(math.acos(cos_theta))
                    else:
                        angle_deg = 0.0

                    term_mags_true.append(mag_true_px)
                    term_mags_pred.append(mag_pred_px)
                    term_mag_deltas.append(mag_pred_px - mag_true_px)
                    term_angles.append(angle_deg)

            # NEW: Kinematic Jitter (2nd Derivative Acceleration Deviation)
            true_pts = np.array(true_traj)
            pred_pts = np.array(pred_traj)

            # a_t = p_t - 2*p_{t-1} + p_{t-2}
            true_accel = true_pts[2:] - 2 * true_pts[1:-1] + true_pts[:-2]
            pred_accel = pred_pts[2:] - 2 * pred_pts[1:-1] + pred_pts[:-2]

            accel_errors = np.linalg.norm(pred_accel - true_accel, axis=1) # L2 norm of the difference
            seq_jitter_err = np.mean(accel_errors)
            kinematic_jitter_errors.append(seq_jitter_err)

            avg_seq_spatial = seq_spatial_err / 16.0
            avg_seq_mag_px = seq_mag_err_px / 16.0
            valid_count += 1

            print(f"   -> [Seq {valid_count:03d}] UV: {avg_seq_spatial:5.2f}px | Mag: {avg_seq_mag_px:5.1f}px | Ang: {term_angles[-1]:5.1f}° | Jitter: {seq_jitter_err:5.2f} px/f²")

        if valid_count == 0: return

        mean_traj_spatial = np.mean(traj_spatial_errors)
        mean_traj_depth = np.mean(traj_depth_errors)
        mean_traj_mag = np.mean(traj_mag_errors)
        mean_kinematic_jitter = np.mean(kinematic_jitter_errors)

        mean_term_spatial = np.mean(term_spatial_errors)
        max_term_spatial = np.max(term_spatial_errors)
        mean_term_depth = np.mean(term_depth_errors)
        mean_term_mag = np.mean(term_mag_errors)

        mean_mag_true = np.mean(term_mags_true)
        mean_mag_pred = np.mean(term_mags_pred)
        mean_mag_delta = np.mean(term_mag_deltas)
        mean_angle = np.mean(term_angles)

        print("\n" + "=" * 60)
        print(" TEMPORAL BENCHMARK RESULTS")
        print("=" * 60)
        print(f" Total Sequences Validated : {valid_count}")
        print("\n --- TRAJECTORY PERFORMANCE (All 16 Frames) ---")
        print(f" Mean Trajectory Spatial Error : {mean_traj_spatial:.2f} Pixels")
        print(f" Mean Trajectory Depth Error   : {mean_traj_depth:.4f} Meters")
        print(f" Mean Trajectory Amplitude Err : {mean_traj_mag:.2f} Pixels")
        print(f" Kinematic Jitter (Accel Err)  : {mean_kinematic_jitter:.3f} px/f²")
        print("\n --- TERMINAL PERFORMANCE (Final Frame 16) ---")
        print(f" Mean Terminal Spatial Error   : {mean_term_spatial:.2f} Pixels")
        print(f" Max Terminal Spatial Error    : {max_term_spatial:.2f} Pixels")
        print(f" Mean Terminal Amplitude Err   : {mean_term_mag:.2f} Pixels")

    print("\n" + "-" * 60)
    print(" 2D VECTOR ANALYSIS (Terminal Frame)")
    print("-" * 60)
    print(f" 1. Magnitude True (Amplitude)    : {mean_mag_true:.2f} px")
    print(f" 2. Magnitude Pred (Amplitude)    : {mean_mag_pred:.2f} px")
    print(f" 3. Magnitude Delta (Pred - True) : {mean_mag_delta:+.2f} px")
    print(f" 4. Phase Angle Error (Pred VS True): {mean_angle:.1f}°")

    wandb.log({
        "eval/traj_spatial_err_px": mean_traj_spatial,
        "eval/traj_depth_err_m": mean_traj_depth,
        "eval/traj_mag_err_cm": mean_traj_mag,
        "eval/kinematic_jitter_px_f2": mean_kinematic_jitter,
        "eval/term_spatial_err_px": mean_term_spatial,
        "eval/term_max_spatial_px": max_term_spatial,
        "eval/term_depth_err_m": mean_term_depth,
        "eval/term_mag_err_cm": mean_term_mag,
        "eval/term_mag_delta_px": mean_mag_delta,
        "eval/term_phase_angle_deg": mean_angle
    })

    print("\n DIAGNOSTICS:")
    if mean_mag_delta > 5.0:
        print(" [!] FOV ILLUSION DETECTED: The model is consistently overestimating")
        print(f"     the mass displacement by ~{mean_mag_delta:.1f}px. It correctly learned the wobble,")
        print("     but the simulated camera FOV or crop scaling is warping the magnitude.")
    elif mean_mag_delta < -5.0:
        print(" [!] DAMPENED AMPLITUDE: The model is underestimating the mass offset.")
        print("     This implies real-world physics are stiffer, or Z-depth translation failed.")
    else:
        print(" [OK] 2D Amplitude bounds are perfectly aligned with reality.")

    if mean_angle > 45.0:
        print(f" [!] MASSIVE PHASE ERROR ({mean_angle:.1f}°): The network is tracking the correct")
        print("     wobble physics, but mapping it to the WRONG FACE of the box.")
        print("     Use `6_auto_face_finder.py` to deduce the physical Axis Rotation offset.")
    elif mean_angle > 15.0:
        print(f" [!] MINOR PHASE SHIFT ({mean_angle:.1f}°): Check physical camera roll or slight")
        print("     misalignments in ArUco marker placement vs MuJoCo.")
    else:
        print(f" [OK] Phase Tracking is locked in. The neural network's angular prediction")
        print("     matches the Ground Truth vector perfectly.")

    print("\n" + "=" * 60 + "\n")

    wandb.finish()


if __name__ == "__main__":
    evaluate_statera()