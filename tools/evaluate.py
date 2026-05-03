import os
import sys
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import h5py
import cv2
import gc
import wandb
import copy
import argparse
import urllib.request
import urllib.error
import subprocess
from tqdm import tqdm

from statera.dataset import StateraDataset
from statera.model import StateraModel

CHECKPOINTS = {
    "Run-00-Anchor": ".checkpoints/STATERA-1K-Anchor.pth",
    "Run-04-StaticDot": ".checkpoints/Run-04-Static-Dot.pth",
    "Run-05-StdSigma": ".checkpoints/Run-05-Standard-Sigma.pth",
    "Run-01-DINOv2": ".checkpoints/Run-01-DINOv2.pth",
    "Run-03-ResNet3D": ".checkpoints/Run-03-ResNet3D.pth",
    "SOTA-Crescent": ".sota_50k_crescent_checkpoints/STATERA-50K-Crescent.pth",
    "SOTA-Sigma": ".sota_50k_sigma_checkpoints/STATERA-50K-Sigma.pth"
}


# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def get_subpixel_coords(logits_heatmap, temperature=2.0):
    B, T, H, W = logits_heatmap.shape
    probs = F.softmax((logits_heatmap.reshape(B * T, -1)) / temperature, dim=1)
    probs = probs.reshape(B, T, H, W)
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=probs.device), torch.arange(W, device=probs.device),
                                    indexing='ij')
    y_center = (probs * y_grid.float().reshape(1, 1, H, W)).sum(dim=(2, 3))
    x_center = (probs * x_grid.float().reshape(1, 1, H, W)).sum(dim=(2, 3))
    return torch.stack([x_center, y_center], dim=2)


def compute_kecs_metrics(pred, gt, diags):
    errs = torch.norm(pred - gt, dim=-1)
    n_come = (errs / diags).mean()
    v_p, v_g = pred[1:] - pred[:-1], gt[1:] - gt[:-1]
    a_p, a_g = v_p[1:] - v_p[:-1], v_g[1:] - v_g[:-1]
    j_errs = torch.norm(a_p - a_g, dim=-1)
    norm_jitter = (j_errs / diags[2:]).mean()
    kecs = torch.sqrt(n_come ** 2 + (0.1 * norm_jitter) ** 2)
    return n_come.item(), norm_jitter.item(), kecs.item()


def snap_res(val):
    for res in [480, 720, 1080, 1440, 1920, 2160, 3840]:
        if abs(val - res) < 50: return res
    return int(val)


def calculate_esd(logits, temperature=2.0):
    B, T, H, W = logits.shape
    device = logits.device
    logits_flat = logits.view(B * T, -1) / temperature
    probs = F.softmax(logits_flat, dim=-1).view(B, T, H, W)
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device).float(), torch.arange(W, device=device).float(),
                                    indexing='ij')
    grid_y = grid_y.view(1, 1, H, W)
    grid_x = grid_x.view(1, 1, H, W)
    pred_y = torch.sum(probs * grid_y, dim=(2, 3))
    pred_x = torch.sum(probs * grid_x, dim=(2, 3))
    dy_sq = (grid_y - pred_y.view(B, T, 1, 1)) ** 2
    dx_sq = (grid_x - pred_x.view(B, T, 1, 1)) ** 2
    dispersion = torch.sum(probs * (dy_sq + dx_sq), dim=(2, 3))
    return dispersion


def calculate_entropy(logits, temperature=2.0):
    B, T, H, W = logits.shape
    logits_flat = logits.view(B * T, -1) / temperature
    probs = F.softmax(logits_flat, dim=-1)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-8), dim=-1)
    return entropy.view(B, T)


# ==========================================
# EVALUATION TASKS
# ==========================================
def run_50k_sota(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Fast-Track Retrospective Eval on {device}")
    wandb.init(project="STATERA", name=f"{args.run_name}-Metrics-Update", config=vars(args))

    base_dataset = StateraDataset(args.dataset_path, target_type='crescent', start_sigma=12.5, jitter_box=False)
    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    _, val_indices = random_split(range(len(base_dataset)), [train_size, val_size], generator=generator)

    val_base_dataset = copy.deepcopy(base_dataset)
    val_base_dataset.jitter_box = False
    val_ds = torch.utils.data.Subset(val_base_dataset, val_indices)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = StateraModel(
        decoder_type='deconv', temporal_mixer='conv1d', single_task=False,
        backbone_type='vjepa', scratch=False, finetune_blocks=2
    ).to(device)

    criterion_h = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    criterion_z = nn.HuberLoss()
    metrics_history = {}
    target_epochs = [7, 8, 10]

    for epoch in target_epochs:
        ckpt_path = os.path.join(args.checkpoint_dir, f"{args.run_name}_epoch_{epoch}.pth")
        if not os.path.exists(ckpt_path):
            print(f"[!] Warning: {ckpt_path} not found. Skipping Epoch {epoch}.")
            continue

        print(f"\n[>] Loading Checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        funnel_ratio = min(1.0, (epoch - 1) / 2.0)
        current_sigma = max(3.0, 12.5 - (9.5 * funnel_ratio))
        val_ds.dataset.update_sigma(current_sigma)
        val_ds.dataset.update_phase_alpha(funnel_ratio)

        val_h, val_z, total_px_err = 0.0, 0.0, 0.0
        total_n_come = 0.0
        total_jitter_err = 0.0
        total_norm_jitter = 0.0
        all_batch_errors = []

        print(f"[*] Evaluating Epoch {epoch}/10 on {val_size} validation samples...")
        with torch.no_grad():
            for vids, gt_h, gt_z, gt_coords, bbox_diags in tqdm(val_loader, desc=f"Epoch {epoch}", leave=False):
                vids = vids.to(device)
                gt_h = gt_h.to(device)
                gt_z = gt_z.to(device)
                gt_coords = gt_coords.to(device)
                bbox_diags = bbox_diags.to(device)

                pred_h, pred_z = model(vids)

                for f in range(16):
                    w = max(0.05, (f / 15.0) ** 2)
                    val_h += criterion_h(pred_h[:, f], gt_h[:, f]).item() * w
                    val_z += criterion_z(pred_z[:, f], gt_z[:, f]).item() * w

                pred_coords = get_subpixel_coords(pred_h, temperature=args.temperature)
                scale_factor = 384 / 64
                pred_coords_real = pred_coords * scale_factor
                gt_coords_real = gt_coords * scale_factor

                batch_px_errors = torch.norm(pred_coords_real - gt_coords_real, dim=2)
                total_px_err += batch_px_errors.mean().item()
                all_batch_errors.append(batch_px_errors.detach().cpu())

                n_come_batch = batch_px_errors / bbox_diags
                total_n_come += n_come_batch.mean().item()

                pred_vel = pred_coords_real[:, 1:, :] - pred_coords_real[:, :-1, :]
                gt_vel = gt_coords_real[:, 1:, :] - gt_coords_real[:, :-1, :]
                pred_accel = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]
                gt_accel = gt_vel[:, 1:, :] - gt_vel[:, :-1, :]

                batch_accel_errors = torch.norm(pred_accel - gt_accel, dim=2)
                total_jitter_err += batch_accel_errors.mean().item()
                norm_jitter_batch = batch_accel_errors / bbox_diags[:, 2:]
                total_norm_jitter += norm_jitter_batch.mean().item()

        avg_val_h = val_h / len(val_loader)
        avg_val_z = val_z / len(val_loader)
        avg_px_err = total_px_err / len(val_loader)
        avg_jitter = total_jitter_err / len(val_loader)
        avg_n_come = total_n_come / len(val_loader)
        avg_norm_jitter = total_norm_jitter / len(val_loader)
        h_ke = (2 * avg_n_come * avg_norm_jitter) / (avg_n_come + avg_norm_jitter + 1e-8)

        cat_errors = torch.cat(all_batch_errors, dim=0)
        p95_px_err = torch.quantile(cat_errors.float(), 0.95).item()

        print(
            f"[✓] Epoch {epoch} Results | Loss: {avg_val_h:.4f} | N-CoME: {avg_n_come * 100:.2f}% | H_KE: {h_ke:.4f} | PX Err: {avg_px_err:.2f}px")

        wandb.log({
            "val/heatmap_loss": avg_val_h,
            "val/z_loss": avg_val_z,
            "metrics/pixel_error": avg_px_err,
            "metrics/p95_error": p95_px_err,
            "metrics/kinematic_jitter": avg_jitter,
            "metrics/N-CoME": avg_n_come,
            "metrics/H_KE": h_ke,
        }, step=epoch)

        metrics_history[f"Epoch_{epoch}"] = {
            "val_heatmap_loss": avg_val_h,
            "val_z_loss": avg_val_z,
            "val_pixel_error": avg_px_err,
            "val_p95_error": p95_px_err,
            "val_kinematic_jitter": avg_jitter,
            "val_n_come": avg_n_come,
            "val_h_ke": h_ke
        }

    with open(args.metrics_file, 'w') as f:
        json.dump(metrics_history, f, indent=4)
    print(f"\n[✓] Retrospective Evaluation Complete! Data saved to {args.metrics_file} and W&B.")
    wandb.finish()


def run_centroid(args):
    print("=" * 80)
    print(f"{'EVALUATING GEOMETRIC CENTROID BASELINE (STATIC 192, 192)':^80}")
    print("=" * 80)
    centroid = torch.tensor([192.0, 192.0])

    if os.path.exists(args.dataset_path):
        with h5py.File(args.dataset_path, 'r') as f:
            gt_uv_sim = torch.from_numpy(f['uv_coords'][-10000:]).reshape(-1, 16, 2)
            diags_sim = torch.ones(10000, 16) * 50.0

        res_sim = [compute_kecs_metrics(centroid.expand(16, 2), gt_uv_sim[i], diags_sim[i]) for i in
                   range(len(gt_uv_sim))]
        avg_sim = np.mean(res_sim, axis=0)
        print(
            f"Simulation (10K Set) | N-CoME: {avg_sim[0] * 100:.2f}% | NormJitter: {avg_sim[1]:.4f} | KECS: {avg_sim[2]:.4f}")
    else:
        print(f"[!] Sim path {args.dataset_path} not found.")

    if os.path.exists(args.mtx_path) and os.path.exists(args.dist_path):
        mtx = np.load(args.mtx_path)
        dist = np.load(args.dist_path)
        frame_w = snap_res(mtx[0, 2] * 2)
        scale = 384.0 / frame_w

        h5_files = glob.glob(os.path.join(args.data_dir, '*_statera.h5'))
        res_real = []
        for f_path in h5_files:
            with h5py.File(f_path, 'r') as f:
                bx, by, bz = f.attrs["box_size_m"]
                hx, hy, hz = bx / 2, by / 2, bz / 2
                box_3d = np.array(
                    [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz], [-hx, -hy, hz], [hx, -hy, hz],
                     [hx, hy, hz], [-hx, hy, hz]], dtype=np.float32)

                for grp_name in f.keys():
                    if not grp_name.startswith('seq_'): continue
                    grp = f[grp_name]
                    gt = torch.stack([torch.from_numpy(grp["com_u"][:]), torch.from_numpy(grp["com_v"][:])], dim=1)
                    rvecs, tvecs = grp["rvecs"][:], grp["tvecs"][:]

                    diags = []
                    for i in range(16):
                        p2d, _ = cv2.projectPoints(box_3d, rvecs[i], tvecs[i], mtx, dist)
                        p2d = p2d.reshape(-1, 2)
                        w, h = np.max(p2d[:, 0]) - np.min(p2d[:, 0]), np.max(p2d[:, 1]) - np.min(p2d[:, 1])
                        diags.append(np.sqrt(w ** 2 + h ** 2) * scale)

                    res_real.append(compute_kecs_metrics(centroid.expand(16, 2), gt, torch.tensor(diags)))

        if res_real:
            avg_real = np.mean(res_real, axis=0)
            print(
                f"Real World ({len(res_real)} Seq)  | N-CoME: {avg_real[0] * 100:.2f}% | NormJitter: {avg_real[1]:.4f} | KECS: {avg_real[2]:.4f}")
    print("=" * 80)


def run_dispersion(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initialized Spatial Dispersion Suite on {device}")

    h5_files = glob.glob(os.path.join(args.data_dir, '*_statera.h5'))
    final_results = {}

    for name, path in CHECKPOINTS.items():
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

        total_esd = 0.0
        total_frames = 0
        with torch.no_grad():
            for f_path in tqdm(h5_files, desc=f"Evaluating {name}", leave=False):
                with h5py.File(f_path, 'r') as f:
                    for group_name in f.keys():
                        if not group_name.startswith('seq_'): continue
                        frames = torch.from_numpy(f[group_name]["frames"][:][:, :, :, ::-1].copy()).float() / 255.0
                        vids = frames.permute(3, 0, 1, 2).unsqueeze(0).to(device)
                        pred_h, _ = model(vids)
                        batch_esd = calculate_esd(pred_h, temperature=2.0)
                        total_esd += batch_esd.sum().item()
                        total_frames += batch_esd.numel()

        if total_frames > 0:
            avg_esd = total_esd / total_frames
            final_results[name] = avg_esd

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"{'EXPECTED SPATIAL DISPERSION (ESD) - PHYSICAL SPREAD':^70}")
    print("=" * 70)
    print(f"| {'Model Identifier':<25} | {'Dispersion (px^2)':^20} | {'Std Dev (px)':^15} |")
    print("-" * 70)
    sorted_res = sorted(final_results.items(), key=lambda x: x[1])
    for name, val in sorted_res:
        std_dev = np.sqrt(val)
        print(f"| {name:<25} | {val:>18.4f} | {std_dev:>13.2f} |")
    print("=" * 70)


def run_entropy(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initialized Entropy Suite on {device}")

    h5_files = glob.glob(os.path.join(args.data_dir, '*_statera.h5'))
    final_results = {}

    for name, path in CHECKPOINTS.items():
        print(f"\n[>] Processing: {name}")
        if not os.path.exists(path):
            print(f"[!] Checkpoint missing: {path}")
            continue

        backbone = 'vjepa'
        if 'DINOv2' in name:
            backbone = 'dinov2'
        elif 'ResNet3D' in name:
            backbone = 'resnet3d'

        model = StateraModel(decoder_type='deconv', temporal_mixer='conv1d', backbone_type=backbone, scratch=False).to(
            device)
        try:
            model.load_state_dict(torch.load(path, map_location=device))
        except Exception as e:
            print(f"    [!] Error loading {name}: {e}")
            continue
        model.eval()

        total_entropy = 0.0
        total_frames = 0
        with torch.no_grad():
            for f_path in tqdm(h5_files, desc=f"Scanning Real World", leave=False):
                with h5py.File(f_path, 'r') as f:
                    for group_name in f.keys():
                        if not group_name.startswith('seq_'): continue
                        frames = torch.from_numpy(f[group_name]["frames"][:][:, :, :, ::-1].copy()).float() / 255.0
                        vids = frames.permute(3, 0, 1, 2).unsqueeze(0).to(device)
                        pred_h, _ = model(vids)
                        batch_h = calculate_entropy(pred_h, temperature=2.0)
                        total_entropy += batch_h.sum().item()
                        total_frames += batch_h.numel()

        if total_frames > 0:
            avg_h = total_entropy / total_frames
            final_results[name] = avg_h
            print(f"    [✓] {name}: {avg_h:.4f} bits")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f"{'STATERA SPATIAL PREDICTIVE ENTROPY (Lower is Better)':^60}")
    print("=" * 60)
    print(f"| {'Model Identifier':<25} | {'Entropy (Bits)':^20} |")
    print("-" * 60)
    sorted_res = sorted(final_results.items(), key=lambda x: x[1])
    for name, val in sorted_res:
        print(f"| {name:<25} | {val:>15.4f} bits |")
    print("=" * 60)


def run_euclidean(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initialized Weighted Euclidean Error Suite on {device}")

    h5_files = glob.glob(os.path.join(args.data_dir, '*_statera.h5'))
    if not h5_files:
        print("[!] No real-world HDF5 files found!")
        return

    raw_weights = torch.arange(1, 17, dtype=torch.float32, device=device)
    temporal_weights = raw_weights / raw_weights.sum()
    final_results = {}

    for name, path in CHECKPOINTS.items():
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

        total_weighted_error = 0.0
        seq_count = 0
        with torch.no_grad():
            for f_path in tqdm(h5_files, desc=f"Evaluating {name}", leave=False):
                with h5py.File(f_path, 'r') as f:
                    for group_name in f.keys():
                        if not group_name.startswith('seq_'): continue
                        grp = f[group_name]
                        frames = torch.from_numpy(grp["frames"][:][:, :, :, ::-1].copy()).float() / 255.0
                        vids = frames.permute(3, 0, 1, 2).unsqueeze(0).to(device)

                        com_u = torch.from_numpy(grp["com_u"][:]).float().to(device)
                        com_v = torch.from_numpy(grp["com_v"][:]).float().to(device)
                        gt_coords = torch.stack([com_u, com_v], dim=1)

                        pred_h, _ = model(vids)
                        pred_coords = get_subpixel_coords(pred_h, temperature=2.0)[0] * (384.0 / 64.0)
                        frame_distances = torch.norm(pred_coords - gt_coords, dim=1)
                        seq_weighted_error = torch.sum(frame_distances * temporal_weights).item()

                        total_weighted_error += seq_weighted_error
                        seq_count += 1

        if seq_count > 0:
            avg_error = total_weighted_error / seq_count
            final_results[name] = avg_error

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 75)
    print(f"{'TEMPORALLY-WEIGHTED EUCLIDEAN PIXEL ERROR (Physical Space)':^75}")
    print("=" * 75)
    print(f"| {'Model Identifier':<25} | {'Weighted Pixel Error (px)':^25} |")
    print("-" * 75)
    sorted_res = sorted(final_results.items(), key=lambda x: x[1])
    for name, val in sorted_res:
        print(f"| {name:<25} | {val:>22.2f} px |")
    print("=" * 75)


def run_geom_collapse(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtx = np.load(args.mtx_path)
    dist = np.load(args.dist_path)
    cx_guess, cy_guess = mtx[0, 2] * 2, mtx[1, 2] * 2
    frame_w = snap_res(cx_guess)
    scale = 384.0 / frame_w

    h5_files = sorted(glob.glob(os.path.join(args.data_dir, '*_statera.h5')))
    final_results = {}
    true_gc_dist_avg = 0.0
    seq_count_total = 0

    for name, path in CHECKPOINTS.items():
        if not os.path.exists(path): continue

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

                        gc_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                        com_3d = np.array([hidden_com_m], dtype=np.float32)

                        v_true_list = []
                        for i in range(16):
                            proj_gc, _ = cv2.projectPoints(gc_3d, rvecs[i], tvecs[i], mtx, dist)
                            proj_com, _ = cv2.projectPoints(com_3d, rvecs[i], tvecs[i], mtx, dist)
                            vec_2d = (proj_com[0][0] - proj_gc[0][0]) * scale
                            v_true_list.append(vec_2d)

                        v_true = torch.tensor(np.array(v_true_list), dtype=torch.float32).to(device)
                        gc_coords = gt_coords - v_true
                        v_pred = pred_coords - gc_coords
                        true_mag = torch.norm(v_true, dim=1)
                        pred_mag = torch.norm(v_pred, dim=1)

                        dot_product = (v_pred * v_true).sum(dim=1)
                        true_mag_sq = (v_true * v_true).sum(dim=1) + 1e-8
                        proj_ratio = torch.clamp(dot_product / true_mag_sq, 0.0, 1.0)

                        total_proj_ratio += proj_ratio.mean().item()
                        total_pred_gc_dist += pred_mag.mean().item()
                        total_true_gc_dist += true_mag.mean().item()
                        seq_count += 1

        if seq_count > 0:
            avg_ratio = total_proj_ratio / seq_count
            avg_pred_dist = total_pred_gc_dist / seq_count
            if seq_count_total == 0:
                true_gc_dist_avg = total_true_gc_dist / seq_count
                seq_count_total = seq_count
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


# --- Point Tracker Helpers ---
def extract_bbox_diags(rvecs, tvecs, mtx, dist, box_size_m, scale):
    bx, by, bz = box_size_m
    hx, hy, hz = bx / 2, by / 2, bz / 2
    box_3d = np.array(
        [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz], [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz],
         [-hx, hy, hz]], dtype=np.float32)
    diags = []
    for i in range(16):
        rv, tv = rvecs[i], tvecs[i]
        pts2d, _ = cv2.projectPoints(box_3d, rv, tv, mtx, dist)
        pts2d = pts2d.reshape(-1, 2)
        w = np.max(pts2d[:, 0]) - np.min(pts2d[:, 0])
        h = np.max(pts2d[:, 1]) - np.min(pts2d[:, 1])
        diag_orig = np.sqrt(w ** 2 + h ** 2)
        diags.append(diag_orig * scale)
    return np.array(diags, dtype=np.float32)


def load_real_world_data(data_dir, cam_mtx_path, dist_coeffs_path):
    print("[*] Linking Real-World Dataset & Computing BBox Diagonals...")
    mtx = np.load(cam_mtx_path)
    dist = np.load(dist_coeffs_path)
    cx_guess, cy_guess = mtx[0, 2] * 2, mtx[1, 2] * 2
    frame_w, frame_h = snap_res(cx_guess), snap_res(cy_guess)
    scale = 384.0 / min(frame_w, frame_h)
    samples = []
    h5_files = sorted(glob.glob(os.path.join(data_dir, '*_statera.h5')))
    for f_path in h5_files:
        with h5py.File(f_path, 'r') as f:
            box_size_m = f.attrs["box_size_m"]
            for group_name in f.keys():
                if not group_name.startswith('seq_'): continue
                grp = f[group_name]
                frames_bgr = grp["frames"][:]
                frames_rgb = frames_bgr[:, :, :, ::-1].copy()
                gt_coords = np.stack([grp["com_u"][:], grp["com_v"][:]], axis=1)
                diags = extract_bbox_diags(grp["rvecs"][:], grp["tvecs"][:], mtx, dist, box_size_m, scale)
                samples.append({
                    'frames': frames_rgb,
                    'gt_coords': gt_coords,
                    'diags': diags
                })
    print(f"[✓] Loaded {len(samples)} Physical Sequences.")
    return samples


def calculate_sequence_metrics(pred_coords, gt_coords, bbox_diags):
    px_errors = torch.norm(pred_coords - gt_coords, dim=1)
    n_come = (px_errors / bbox_diags).mean().item()
    pred_vel = pred_coords[1:] - pred_coords[:-1]
    gt_vel = gt_coords[1:] - gt_coords[:-1]
    pred_accel = pred_vel[1:] - pred_vel[:-1]
    gt_accel = gt_vel[1:] - gt_vel[:-1]
    accel_errors = torch.norm(pred_accel - gt_accel, dim=1)
    norm_jitter = (accel_errors / bbox_diags[2:]).mean().item()
    raw_jitter = accel_errors.mean().item()
    return n_come, norm_jitter, raw_jitter


def run_cotracker(samples, device):
    print("\n[*] Initializing Meta's CoTracker2...")
    try:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device)
        model.eval()
    except Exception as e:
        print(f"[!] CoTracker failed to load.\n{e}")
        return None

    queries = torch.tensor([[[0.0, 192.0, 192.0]]]).to(device)
    tot_ncome, tot_njitter, tot_rawjitter = 0.0, 0.0, 0.0
    with torch.no_grad():
        for seq in tqdm(samples, desc="Evaluating CoTracker"):
            video = torch.from_numpy(seq['frames']).float().to(device)
            video = video.permute(0, 3, 1, 2).unsqueeze(0)
            pred_tracks, _ = model(video, queries=queries)
            pred_coords = pred_tracks[0, :, 0, :]
            gt_coords = torch.from_numpy(seq['gt_coords']).float().to(device)
            diags = torch.from_numpy(seq['diags']).float().to(device)
            n_come, norm_jitt, raw_jitt = calculate_sequence_metrics(pred_coords, gt_coords, diags)
            tot_ncome += n_come
            tot_njitter += norm_jitt
            tot_rawjitter += raw_jitt

    n = len(samples)
    avg_ncome = tot_ncome / n
    avg_njitter = tot_njitter / n
    avg_rawjitter = tot_rawjitter / n
    h_ke = (2 * avg_ncome * avg_njitter) / (avg_ncome + avg_njitter + 1e-8)

    wandb.init(project="STATERA", name="Baseline-CoTracker-RealWorld")
    wandb.log({"real_world/N-CoME": avg_ncome, "real_world/Raw_Jitter": avg_rawjitter, "real_world/H_KE": h_ke})
    wandb.finish()
    return {'n_come': avg_ncome, 'raw_jitter': avg_rawjitter, 'h_ke': h_ke}


def run_tapir(samples, device):
    print("\n[*] Initializing Google's TAPIR...")
    repo_dir = os.path.abspath("tapnet_repo")
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", "https://github.com/deepmind/tapnet.git", repo_dir])
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        from tapnet.torch import tapir_model
    except Exception as e:
        print(f"[!] TAPIR module import failed: {e}")
        return None
    ckpt_path = "bootstapir_checkpoint_v2.pt"
    if not os.path.exists(ckpt_path):
        url = "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt"
        urllib.request.urlretrieve(url, ckpt_path)
    try:
        model = tapir_model.TAPIR(pyramid_level=1)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[!] Failed to load TAPIR weights: {e}")
        return None

    queries = torch.tensor([[[0.0, 192.0, 192.0]]]).to(device)
    tot_ncome, tot_njitter, tot_rawjitter = 0.0, 0.0, 0.0
    with torch.no_grad():
        for seq in tqdm(samples, desc="Evaluating TAPIR"):
            video = torch.from_numpy(seq['frames']).float().to(device) / 255.0
            video = video.unsqueeze(0)
            video_tapir = (video * 2.0) - 1.0
            outputs = model(video_tapir, query_points=queries, is_training=False)
            tracks = outputs['tracks']
            pred_coords = tracks[0, 0, :, :]
            gt_coords = torch.from_numpy(seq['gt_coords']).float().to(device)
            diags = torch.from_numpy(seq['diags']).float().to(device)
            n_come, norm_jitt, raw_jitt = calculate_sequence_metrics(pred_coords, gt_coords, diags)
            tot_ncome += n_come
            tot_njitter += norm_jitt
            tot_rawjitter += raw_jitt
    n = len(samples)
    avg_ncome = tot_ncome / n
    avg_njitter = tot_njitter / n
    avg_rawjitter = tot_rawjitter / n
    h_ke = (2 * avg_ncome * avg_njitter) / (avg_ncome + avg_njitter + 1e-8)
    wandb.init(project="STATERA", name="Baseline-TAPIR-RealWorld")
    wandb.log({"real_world/N-CoME": avg_ncome, "real_world/Raw_Jitter": avg_rawjitter, "real_world/H_KE": h_ke})
    wandb.finish()
    return {'n_come': avg_ncome, 'raw_jitter': avg_rawjitter, 'h_ke': h_ke}


def print_summary_table(results):
    print("\n" + "=" * 90)
    print(f"{'POINT TRACKER BASELINES (Sim2Real Physical Verification)':^90}")
    print("=" * 90)
    print(f"| {'Model Name':^26} | {'N-CoME (%)':^16} | {'Raw Jitter':^16} | {'H_KE Metric':^16} |")
    print("-" * 90)
    for name, metrics in results.items():
        if metrics is None:
            print(f"| {name:<26} | {'FAILED':^16} | {'FAILED':^16} | {'FAILED':^16} |")
        else:
            print(
                f"| {name:<26} | {metrics['n_come'] * 100:^15.2f}% | {metrics['raw_jitter']:^16.3f} | {metrics['h_ke']:^16.5f} |")
    print("=" * 90 + "\n")


def run_point_trackers(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_real_world_data(args.data_dir, args.mtx_path, args.dist_path)
    if not samples: return
    results = {}
    results["Meta CoTracker2"] = run_cotracker(samples, device)
    results["Google TAPIR"] = run_tapir(samples, device)
    print_summary_table(results)


def run_real_world(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtx = np.load(args.mtx_path)
    dist = np.load(args.dist_path)
    frame_w = snap_res(mtx[0, 2] * 2)
    scale = 384.0 / frame_w

    print(f"\n{'TABLE 2: ZERO-SHOT REAL-WORLD TRANSFER':^75}")
    print("-" * 75)
    print(f"| {'Model Name':<20} | {'N-CoME (%)':^14} | {'NormJitter':^14} | {'KECS':^12} |")
    print("-" * 75)

    h5_files = glob.glob(os.path.join(args.data_dir, '*_statera.h5'))

    for name, path in CHECKPOINTS.items():
        if not os.path.exists(path):
            print(f"| {name:<20} | {'MISSING':^14} | {'MISSING':^14} | {'MISSING':^12} |")
            continue
        backbone = 'dinov2' if 'DINO' in name else ('resnet3d' if 'ResNet' in name else 'vjepa')
        model = StateraModel(decoder_type='deconv', temporal_mixer='conv1d', backbone_type=backbone).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        results = []
        with torch.no_grad():
            for f_path in h5_files:
                with h5py.File(f_path, 'r') as f:
                    bx, by, bz = f.attrs["box_size_m"]
                    hx, hy, hz = bx / 2, by / 2, bz / 2
                    box_3d = np.array(
                        [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz], [-hx, -hy, hz], [hx, -hy, hz],
                         [hx, hy, hz], [-hx, hy, hz]], dtype=np.float32)

                    for grp_name in f.keys():
                        if not grp_name.startswith('seq_'): continue
                        grp = f[grp_name]
                        vids = torch.from_numpy(grp["frames"][:][:, :, :, ::-1].copy()).float() / 255.0
                        vids = vids.permute(3, 0, 1, 2).unsqueeze(0).to(device)
                        pred_h, _ = model(vids)
                        pred_px = get_subpixel_coords(pred_h, 2.0)[0] * 6.0
                        gt_px = torch.stack([torch.from_numpy(grp["com_u"][:]), torch.from_numpy(grp["com_v"][:])],
                                            dim=1).to(device)
                        rvecs, tvecs = grp["rvecs"][:], grp["tvecs"][:]
                        diags = []
                        for i in range(16):
                            p2d, _ = cv2.projectPoints(box_3d, rvecs[i], tvecs[i], mtx, dist)
                            p2d = p2d.reshape(-1, 2)
                            w, h = np.max(p2d[:, 0]) - np.min(p2d[:, 0]), np.max(p2d[:, 1]) - np.min(p2d[:, 1])
                            diags.append(np.sqrt(w ** 2 + h ** 2) * scale)
                        results.append(compute_kecs_metrics(pred_px, gt_px, torch.tensor(diags).to(device)))

        avg = np.mean(results, axis=0)
        print(f"| {name:<20} | {avg[0] * 100:>12.2f}% | {avg[1]:>14.4f} | {avg[2]:>12.4f} |")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("-" * 75)


def run_sim_50k(args):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initialized Inference Engine on {device}")

    ds = StateraDataset(args.dataset_path, target_type='dot')
    val_indices = list(range(max(0, len(ds) - 10000), len(ds)))
    val_ds = Subset(ds, val_indices)
    loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    print(f"\n{'TABLE 1: IN-DOMAIN SIMULATION PERFORMANCE':^75}")
    print("-" * 75)
    print(f"| {'Model Name':<20} | {'N-CoME (%)':^14} | {'NormJitter':^14} | {'KECS':^12} |")
    print("-" * 75)

    for name, path in CHECKPOINTS.items():
        if not os.path.exists(path):
            print(f"| {name:<20} | {'MISSING':^14} | {'MISSING':^14} | {'MISSING':^12} |")
            continue
        backbone = 'vjepa'
        if 'DINOv2' in name:
            backbone = 'dinov2'
        elif 'ResNet3D' in name:
            backbone = 'resnet3d'

        model = StateraModel(decoder_type='deconv', temporal_mixer='conv1d', backbone_type=backbone).to(device)
        try:
            state_dict = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"| {name:<20} | LOAD ERROR    | LOAD ERROR    | LOAD ERROR   |")
            continue
        model.eval()

        all_results = []
        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Evaluating {name}", leave=False)
            for vids, _, _, gt_uv, _ in pbar:
                vids = vids.to(device)
                pred_h, _ = model(vids)
                pred_px = get_subpixel_coords(pred_h, temperature=2.0) * 6.0
                gt_px = gt_uv.to(device) * 6.0
                diags = torch.ones(vids.size(0), 16).to(device) * 50.0
                for i in range(vids.size(0)):
                    all_results.append(compute_kecs_metrics(pred_px[i], gt_px[i], diags[i]))

        avg = np.mean(all_results, axis=0)
        print(f"| {name:<20} | {avg[0] * 100:>12.2f}% | {avg[1]:>14.4f} | {avg[2]:>12.4f} |")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("-" * 75)


# ==========================================
# MAIN ROUTINE
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description="STATERA Evaluation Suite")
    parser.add_argument('--task', type=str, required=True, choices=[
        '50k_sota', 'centroid', 'dispersion', 'entropy', 'euclidean',
        'geom_collapse', 'point_trackers', 'real_world', 'sim_50k', 'all'],
                        help='Task to run or "all" to run sequentially')
    parser.add_argument('--dataset_path', type=str, default='sim/HiddenMass-50K.hdf5')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='scripts/sota_50k_crescent_checkpoints')
    parser.add_argument('--run_name', type=str, default='STATERA-50K-Crescent-SOTA')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--metrics_file', type=str, default='sota_50k_retro_metrics.json')
    parser.add_argument('--data_dir', type=str, default='sim2real/output')
    parser.add_argument('--mtx_path', type=str, default='sim2real/camera_matrix.npy')
    parser.add_argument('--dist_path', type=str, default='sim2real/dist_coeffs.npy')
    return parser.parse_args()


def main():
    args = get_args()
    tasks = {
        '50k_sota': run_50k_sota,
        'centroid': run_centroid,
        'dispersion': run_dispersion,
        'entropy': run_entropy,
        'euclidean': run_euclidean,
        'geom_collapse': run_geom_collapse,
        'point_trackers': run_point_trackers,
        'real_world': run_real_world,
        'sim_50k': run_sim_50k
    }

    if args.task == 'all':
        for task_name, task_func in tasks.items():
            print(f"\n[{'*' * 10} RUNNING TASK: {task_name.upper()} {'*' * 10}]\n")
            task_func(args)
    else:
        tasks[args.task](args)


if __name__ == '__main__':
    main()