import os
import glob
import torch
import torch.nn.functional as F
import h5py
import numpy as np
import cv2
import gc
from tqdm import tqdm
from model import StateraModel


def snap_to_standard_resolution(val):
    for res in [480, 720, 1080, 1440, 1920, 2160, 3840]:
        if abs(val - res) < 50: return res
    return int(val)


def get_subpixel_coords(logits_heatmap, temperature=2.0):
    B, T, H, W = logits_heatmap.shape
    probs = F.softmax((logits_heatmap.reshape(B * T, -1)) / temperature, dim=1)
    probs = probs.reshape(B, T, H, W)
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=probs.device), torch.arange(W, device=probs.device),
                                    indexing='ij')
    y_center = (probs * y_grid.float().reshape(1, 1, H, W)).sum(dim=(2, 3))
    x_center = (probs * x_grid.float().reshape(1, 1, H, W)).sum(dim=(2, 3))
    return torch.stack([x_center, y_center], dim=2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mtx = np.load('../sim2real/camera_matrix.npy')
    dist = np.load('../sim2real/dist_coeffs.npy')

    cx_guess, cy_guess = mtx[0, 2] * 2, mtx[1, 2] * 2
    frame_w = snap_to_standard_resolution(cx_guess)
    scale = 384.0 / frame_w

    checkpoints = {
        "Run-00-Anchor": "checkpoints/Run-00-Anchor-Saved_epoch_30.pth",
        "Run-04-StaticDot": "checkpoints/Run-04-Static-Dot-Saved_epoch_30.pth",
        "Run-05-StdSigma": "checkpoints/Run-05-Standard-Sigma-Saved_epoch_30.pth",
        "Run-01-DINOv2": "checkpoints/Run-01-DINOv2_epoch_23.pth",
        "Run-03-ResNet3D": "checkpoints/Run-03-ResNet3D-Fixed_epoch_30.pth",
        "SOTA-Crescent": "sota_50k_checkpoints/STATERA-50K-SOTA_epoch_7.pth",
        "SOTA-Sigma": "sota_50k_sigma_checkpoints/STATERA-50K-Sigma_epoch_10.pth"
    }

    data_dir = '../sim2real/output'
    h5_files = sorted(glob.glob(os.path.join(data_dir, '*_statera.h5')))

    final_results = {}
    true_gc_dist_avg = 0.0

    for name, path in checkpoints.items():
        if not os.path.exists(path):
            continue

        backbone = 'vjepa'
        if 'DINOv2' in name:
            backbone = 'dinov2'
        elif 'ResNet3D' in name:
            backbone = 'resnet3d'

        model = StateraModel(decoder_type='deconv', temporal_mixer='conv1d', backbone_type=backbone, scratch=False).to(
            device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        total_proj_ratio = 0.0
        total_pred_gc_dist = 0.0
        total_true_gc_dist = 0.0
        seq_count = 0

        with torch.no_grad():
            for f_path in tqdm(h5_files, desc=f"Evaluating {name}", leave=False):
                with h5py.File(f_path, 'r') as f:
                    hidden_com_m = f.attrs["hidden_com_m"]

                    for group_name in f.keys():
                        if not group_name.startswith('seq_'): continue
                        grp = f[group_name]

                        frames = torch.from_numpy(grp["frames"][:][:, :, :, ::-1].copy()).float() / 255.0
                        vids = frames.permute(3, 0, 1, 2).unsqueeze(0).to(device)
                        pred_h, _ = model(vids)
                        pred_coords = get_subpixel_coords(pred_h, temperature=2.0)[0] * (384.0 / 64.0)
                        gt_coords = torch.stack([torch.from_numpy(grp["com_u"][:]), torch.from_numpy(grp["com_v"][:])],
                                                dim=1).to(device)

                        rvecs, tvecs = grp["rvecs"][:], grp["tvecs"][:]

                        # --- THE FIX: CROP-INVARIANT VECTOR MATH ---
                        gc_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                        com_3d = np.array([hidden_com_m], dtype=np.float32)

                        v_true_list = []
                        for i in range(16):
                            proj_gc, _ = cv2.projectPoints(gc_3d, rvecs[i], tvecs[i], mtx, dist)
                            proj_com, _ = cv2.projectPoints(com_3d, rvecs[i], tvecs[i], mtx, dist)

                            # Subtracting the two absolute points gives the offset vector, rendering the crop irrelevant!
                            vec_2d = (proj_com[0][0] - proj_gc[0][0]) * scale
                            v_true_list.append(vec_2d)

                        v_true = torch.tensor(np.array(v_true_list), dtype=torch.float32).to(device)

                        # We reconstruct the exact cropped Geometric Center by subtracting the offset from the Ground Truth
                        gc_coords = gt_coords - v_true

                        # Vector from Geometric Center to Model Prediction
                        v_pred = pred_coords - gc_coords

                        true_mag = torch.norm(v_true, dim=1)
                        pred_mag = torch.norm(v_pred, dim=1)

                        dot_product = (v_pred * v_true).sum(dim=1)
                        true_mag_sq = (v_true * v_true).sum(dim=1) + 1e-8

                        # Clamp ratio between 0 and 1 (so overshooting doesn't count as "better" physics)
                        proj_ratio = torch.clamp(dot_product / true_mag_sq, 0.0, 1.0)

                        total_proj_ratio += proj_ratio.mean().item()
                        total_pred_gc_dist += pred_mag.mean().item()
                        total_true_gc_dist += true_mag.mean().item()
                        seq_count += 1

        avg_ratio = total_proj_ratio / seq_count
        avg_pred_dist = total_pred_gc_dist / seq_count
        true_gc_dist_avg = total_true_gc_dist / seq_count

        final_results[name] = {"ratio": avg_ratio, "pred_dist": avg_pred_dist}

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 85)
    print(f"{'DISENTANGLEMENT METRIC: GEOMETRIC COLLAPSE VS PHYSICS CAPTURE':^85}")
    print("=" * 85)
    print(f"Ground Truth Average Offset from Geometric Center: {true_gc_dist_avg:.2f} pixels")
    print("-" * 85)
    print(f"| {'Model Identifier':<20} | {'Physics Capture Ratio':^25} | {'Dist. from Center':^25} |")
    print("-" * 85)

    sorted_res = sorted(final_results.items(), key=lambda x: x[1]['ratio'], reverse=True)
    for name, metrics in sorted_res:
        ratio_str = f"{metrics['ratio'] * 100:.2f}%"
        dist_str = f"{metrics['pred_dist']:.2f} px"
        print(f"| {name:<20} | {ratio_str:>16}          | {dist_str:>16}          |")
    print("=" * 85)


if __name__ == "__main__":
    main()