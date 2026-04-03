import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import wandb
import argparse
import traceback

from dataset import StateraDataset
from model import StateraModel


def parse_args():
    parser = argparse.ArgumentParser(description="STATERA Ablation Training Pipeline")
    parser.add_argument('--decoder_type', type=str, default='mlp', choices=['mlp', 'deconv'])
    parser.add_argument('--temporal_mixer', type=str, default='conv1d', choices=['conv1d', 'transformer'])
    parser.add_argument('--target_type', type=str, default='dot', choices=['dot', 'blend', 'crescent'])
    parser.add_argument('--backbone', type=str, default='vjepa', choices=['vjepa', 'dinov2'])
    parser.add_argument('--temporal_weighting', type=str, default='exponential', choices=['exponential', 'uniform'])
    parser.add_argument('--single_task', action='store_true', help="Disable Z-Depth tracking")
    parser.add_argument('--static_sigma', action='store_true', help="Keep Sigma 3.0 instead of annealing")
    parser.add_argument('--scratch', action='store_true', help="Train backbone without pre-trained weights")
    parser.add_argument('--wandb_name', type=str, required=True)
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'])
    return parser.parse_args()


def get_subpixel_coords(logits_heatmap):
    B, T, H, W = logits_heatmap.shape
    probs = torch.sigmoid(logits_heatmap).view(B * T, -1)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
    probs = probs.view(B, T, H, W)
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=probs.device), torch.arange(W, device=probs.device),
                                    indexing='ij')
    y_center = (probs * y_grid.float().view(1, 1, H, W)).sum(dim=(2, 3))
    x_center = (probs * x_grid.float().view(1, 1, H, W)).sum(dim=(2, 3))
    return torch.stack([x_center, y_center], dim=2)


def save_metrics(run_name, h_loss, z_loss, pixel_err):
    metrics_file = "run_metrics.json"
    data = {}
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
    data[run_name] = {"val_heatmap_loss": h_loss, "val_z_loss": z_loss, "val_pixel_error": pixel_err}
    with open(metrics_file, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    args = parse_args()
    log_file = f"{args.wandb_name}_train.log"

    def log(msg):
        print(msg)
        with open(log_file, "a") as f: f.write(msg + "\n")

    wandb.init(project="STATERA-Ablations", name=args.wandb_name, config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Initialized {args.wandb_name} on {device}")

    try:
        start_sigma = 3.0 if args.static_sigma else 12.5
        dataset = StateraDataset('../sim/statera_poc.hdf5', target_type=args.target_type, start_sigma=start_sigma)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_ds, batch_size=24, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=24, shuffle=False, num_workers=4, pin_memory=True)

        model = StateraModel(
            decoder_type=args.decoder_type, temporal_mixer=args.temporal_mixer,
            single_task=args.single_task, backbone_type=args.backbone, scratch=args.scratch
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion_h = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
        criterion_z = nn.HuberLoss()

        epochs = 35  # Strictly exactly 35 epochs per prompt request
        train_h_losses, val_h_losses = [], []
        best_val_err = float('inf')

        for epoch in range(epochs):
            if not args.static_sigma:
                current_sigma = max(3.0, 12.5 - (9.5) * (epoch / 25.0))
                train_ds.dataset.update_sigma(current_sigma)

            if args.target_type in ['blend', 'crescent']:
                train_ds.dataset.update_phase_alpha(min(1.0, epoch / 25.0))

            model.train()
            run_h, run_z = 0.0, 0.0

            for vids, gt_h, gt_z in train_loader:
                vids, gt_h, gt_z = vids.to(device), gt_h.to(device), gt_z.to(device)
                optimizer.zero_grad()
                pred_h, pred_z = model(vids)

                w_h, w_z = 0.0, 0.0
                for f in range(16):
                    w = 1.0 if args.temporal_weighting == 'uniform' else (f / 15.0) ** 2
                    w_h += criterion_h(pred_h[:, f], gt_h[:, f]) * w
                    if not args.single_task:
                        w_z += criterion_z(pred_z[:, f], gt_z[:, f]) * w

                loss = w_h + (0.1 * w_z if not args.single_task else 0.0)
                loss.backward()
                optimizer.step()

                run_h += w_h.item()
                run_z += w_z.item()

            # Validation
            model.eval()
            val_h, val_z, total_px_err = 0.0, 0.0, 0.0
            with torch.no_grad():
                for vids, gt_h, gt_z in val_loader:
                    vids, gt_h, gt_z = vids.to(device), gt_h.to(device), gt_z.to(device)
                    pred_h, pred_z = model(vids)

                    for f in range(16):
                        w = 1.0 if args.temporal_weighting == 'uniform' else (f / 15.0) ** 2
                        val_h += criterion_h(pred_h[:, f], gt_h[:, f]).item() * w
                        if not args.single_task:
                            val_z += criterion_z(pred_z[:, f], gt_z[:, f]).item() * w

                    # Calculate Pixel Error
                    pred_coords = get_subpixel_coords(pred_h)
                    B, T, H_res, W_res = gt_h.shape
                    idx = gt_h.view(B * T, -1).argmax(dim=1)
                    gt_coords = torch.stack([idx % W_res, idx // W_res], dim=1).float().view(B, T, 2)
                    total_px_err += torch.norm(pred_coords - gt_coords, dim=2).mean().item()

            avg_val_h = val_h / len(val_loader)
            avg_val_z = val_z / len(val_loader)
            avg_px_err = total_px_err / len(val_loader)

            train_h_losses.append(run_h / len(train_loader))
            val_h_losses.append(avg_val_h)

            log(f"Epoch [{epoch + 1}/{epochs}] | Val Heatmap Loss: {avg_val_h:.4f} | Val Pixel Error: {avg_px_err * (384 / 64):.2f} px")
            wandb.log({"val/heatmap_loss": avg_val_h, "val/z_loss": avg_val_z,
                       "metrics/pixel_error": avg_px_err * (384 / 64)})

        # Save individual run graph locally
        plt.figure()
        plt.plot(train_h_losses, label='Train Heatmap Loss')
        plt.plot(val_h_losses, label='Val Heatmap Loss')
        plt.title(f"{args.wandb_name} Loss Curve")
        plt.legend()
        plt.savefig(f"{args.wandb_name}_loss.png")
        plt.close()

        save_metrics(args.wandb_name, avg_val_h, avg_val_z, avg_px_err * (384 / 64))
        log("Run completed successfully.")

    except Exception as e:
        log(f"CRITICAL ERROR ENCOUNTERED: {e}")
        log(traceback.format_exc())
        save_metrics(args.wandb_name, 999.0, 999.0, 999.0)  # Mark as failed in JSON but allow bash to continue

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()