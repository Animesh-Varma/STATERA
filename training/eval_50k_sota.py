import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import wandb
import copy
import argparse
from tqdm import tqdm

from dataset import StateraDataset
from model import StateraModel


def parse_args():
    parser = argparse.ArgumentParser(description="STATERA 50K Fast-Track Metric Evaluation")
    parser.add_argument('--dataset_path', type=str, default='../sim/HiddenMass-50K.hdf5')
    parser.add_argument('--checkpoint_dir', type=str, default='sota_50k_checkpoints')
    parser.add_argument('--run_name', type=str, default='STATERA-50K-SOTA')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--metrics_file', type=str, default='sota_50k_retro_metrics.json')
    return parser.parse_args()


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
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Fast-Track Retrospective Eval on {device}")

    # Launch W&B Run (Appending "-Metrics-Update" so it doesn't overwrite original system logs)
    wandb.init(project="STATERA", name=f"{args.run_name}-Metrics-Update", config=vars(args))

    # 1. Dataset & Split Setup
    base_dataset = StateraDataset(args.dataset_path, target_type='crescent', start_sigma=12.5, jitter_box=False)

    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    _, val_indices = random_split(range(len(base_dataset)), [train_size, val_size], generator=generator)

    val_base_dataset = copy.deepcopy(base_dataset)
    val_base_dataset.jitter_box = False
    val_ds = torch.utils.data.Subset(val_base_dataset, val_indices)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Model Initialization (Based on run_50k_sota.sh config)
    model = StateraModel(
        decoder_type='deconv',
        temporal_mixer='conv1d',
        single_task=False,
        backbone_type='vjepa',
        scratch=False,
        finetune_blocks=2
    ).to(device)

    criterion_h = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    criterion_z = nn.HuberLoss()

    metrics_history = {}

    # 3. FAST TRACK: Only test the Best Metrics (7), Best Loss (8), and Final State (10)
    target_epochs = [7, 8, 10]

    for epoch in target_epochs:
        ckpt_path = os.path.join(args.checkpoint_dir, f"{args.run_name}_epoch_{epoch}.pth")

        if not os.path.exists(ckpt_path):
            print(f"[!] Warning: {ckpt_path} not found. Skipping Epoch {epoch}.")
            continue

        print(f"\n[>] Loading Checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # Replicate the curriculum target state for exact Heatmap Loss comparison
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

                # Pure Pixel Calculation
                batch_px_errors = torch.norm(pred_coords_real - gt_coords_real, dim=2)
                total_px_err += batch_px_errors.mean().item()
                all_batch_errors.append(batch_px_errors.detach().cpu())

                # N-CoME Calculation
                n_come_batch = batch_px_errors / bbox_diags
                total_n_come += n_come_batch.mean().item()

                # Kinematics
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

        # H_KE Mathematical Combine
        h_ke = (2 * avg_n_come * avg_norm_jitter) / (avg_n_come + avg_norm_jitter + 1e-8)

        cat_errors = torch.cat(all_batch_errors, dim=0)
        p95_px_err = torch.quantile(cat_errors.float(), 0.95).item()

        print(
            f"[✓] Epoch {epoch} Results | Loss: {avg_val_h:.4f} | N-CoME: {avg_n_come * 100:.2f}% | H_KE: {h_ke:.4f} | PX Err: {avg_px_err:.2f}px")

        # By explicitly mapping step=epoch, W&B will map these correctly to step 7, 8, and 10 on the graph
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


if __name__ == "__main__":
    main()