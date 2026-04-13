import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    parser.add_argument('--decoder_type', type=str, default='mlp', choices=['mlp', 'deconv', 'regression'])
    parser.add_argument('--temporal_mixer', type=str, default='conv1d', choices=['conv1d', 'transformer', 'none'])
    parser.add_argument('--target_type', type=str, default='dot', choices=['dot', 'blend', 'crescent', 'twostage'])
    parser.add_argument('--backbone', type=str, default='vjepa', choices=['vjepa', 'dinov2'])
    parser.add_argument('--temporal_weighting', type=str, default='exponential', choices=['exponential', 'uniform'])
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'])
    parser.add_argument('--single_task', action='store_true')
    parser.add_argument('--static_sigma', action='store_true')
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--temporal_dropout', type=float, default=0.0)
    parser.add_argument('--jitter_box', action='store_true')
    parser.add_argument('--finetune_blocks', type=int, default=0)
    parser.add_argument('--wandb_name', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--metrics_file', type=str, default='run_metrics.json')

    # CHANGE 1: Add accumulation argument
    parser.add_argument('--accumulate_steps', type=int, default=1)

    return parser.parse_args()


def get_subpixel_coords(logits_heatmap):
    B, T, H, W = logits_heatmap.shape
    probs = torch.sigmoid(logits_heatmap).reshape(B * T, -1)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
    probs = probs.reshape(B, T, H, W)
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=probs.device), torch.arange(W, device=probs.device),
                                    indexing='ij')
    y_center = (probs * y_grid.float().reshape(1, 1, H, W)).sum(dim=(2, 3))
    x_center = (probs * x_grid.float().reshape(1, 1, H, W)).sum(dim=(2, 3))
    return torch.stack([x_center, y_center], dim=2)


def focal_loss_bce(logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = torch.exp(-bce)
    focal_loss = alpha * (1 - p_t) ** gamma * bce
    return focal_loss.mean()


# CHANGE 2: Update definition to accept all 5 metrics
def save_metrics(run_name, h_loss, z_loss, pixel_err, p95_err, jitter_err, metrics_file):
    data = {}
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
    data[run_name] = {
        "val_heatmap_loss": h_loss,
        "val_z_loss": z_loss,
        "val_pixel_error": pixel_err,
        "val_p95_error": p95_err,
        "val_kinematic_jitter": jitter_err
    }
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
        init_target = 'crescent' if args.target_type == 'twostage' else args.target_type
        start_sigma = 3.0 if args.static_sigma else 12.5
        dataset = StateraDataset('../sim/statera_poc.hdf5', target_type=init_target, start_sigma=start_sigma,
                                 jitter_box=args.jitter_box)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        if args.scratch:
            batch_size = 2
        elif args.finetune_blocks > 0 or args.temporal_mixer == 'transformer':
            batch_size = 8
        else:
            batch_size = 24

        log(f"Using dynamic batch size: {batch_size} | Accumulation steps: {args.accumulate_steps}")
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = StateraModel(decoder_type=args.decoder_type, temporal_mixer=args.temporal_mixer,
                             single_task=args.single_task, backbone_type=args.backbone, scratch=args.scratch,
                             finetune_blocks=args.finetune_blocks).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion_h = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
        criterion_z = nn.HuberLoss()

        train_h_losses, val_h_losses = [], []

        for epoch in range(args.epochs):
            if not args.static_sigma:
                train_ds.dataset.update_sigma(max(3.0, 12.5 - (9.5) * (epoch / 25.0)))

            if args.target_type == 'twostage':
                train_ds.dataset.target_type = 'crescent' if epoch < 25 else 'dot'
                if epoch < 25: train_ds.dataset.update_phase_alpha(min(1.0, epoch / 25.0))
            elif args.target_type in ['blend', 'crescent']:
                train_ds.dataset.update_phase_alpha(min(1.0, epoch / 25.0))

            model.train()
            optimizer.zero_grad(set_to_none=True)  # CHANGE 3: zero_grad outside inner loop
            run_h, run_z = 0.0, 0.0

            for i, (vids, gt_h, gt_z, gt_coords) in enumerate(train_loader):
                vids, gt_h, gt_z, gt_coords = vids.to(device), gt_h.to(device), gt_z.to(device), gt_coords.to(device)
                if args.temporal_dropout > 0.0:
                    vids = vids * (
                                torch.rand(vids.shape[0], 1, 16, 1, 1, device=device) > args.temporal_dropout).float()

                pred_h, pred_z = model(vids)
                w_h, w_z = 0.0, 0.0
                for f in range(16):
                    w = (f / 15.0) ** 2 if args.temporal_weighting == 'exponential' else 1.0
                    if args.decoder_type == 'regression':
                        w_h += F.mse_loss(pred_h[:, f], gt_coords[:, f]) * w
                    else:
                        w_h += (focal_loss_bce(pred_h[:, f], gt_h[:, f]) if args.loss_type == 'focal' else criterion_h(
                            pred_h[:, f], gt_h[:, f])) * w
                    if not args.single_task: w_z += criterion_z(pred_z[:, f], gt_z[:, f]) * w

                # CHANGE 4: Scale loss by accumulation steps
                loss = (w_h + (0.1 * w_z if not args.single_task else 0.0)) / args.accumulate_steps
                loss.backward()

                # CHANGE 5: Step only after N accumulation steps
                if (i + 1) % args.accumulate_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                run_h += w_h.item() if isinstance(w_h, torch.Tensor) else float(w_h)
                run_z += w_z.item() if isinstance(w_z, torch.Tensor) else float(w_z)

            model.eval()
            val_h, val_z, total_px_err, total_jitter_err = 0.0, 0.0, 0.0, 0.0
            all_batch_errors = []
            with torch.no_grad():
                for vids, gt_h, gt_z, gt_coords in val_loader:
                    vids, gt_h, gt_z, gt_coords = vids.to(device), gt_h.to(device), gt_z.to(device), gt_coords.to(
                        device)
                    pred_h, pred_z = model(vids)
                    for f in range(16):
                        w = (f / 15.0) ** 2 if args.temporal_weighting == 'exponential' else 1.0
                        val_h += (F.mse_loss(pred_h[:, f],
                                             gt_coords[:, f]) if args.decoder_type == 'regression' else criterion_h(
                            pred_h[:, f], gt_h[:, f])).item() * w
                        if not args.single_task: val_z += criterion_z(pred_z[:, f], gt_z[:, f]).item() * w

                    pred_coords = pred_h if args.decoder_type == 'regression' else get_subpixel_coords(pred_h)
                    batch_px_errors = torch.norm(pred_coords - gt_coords, dim=2)
                    total_px_err += batch_px_errors.mean().item()
                    all_batch_errors.append(batch_px_errors.detach().cpu())

                    pred_vel = pred_coords[:, 1:, :] - pred_coords[:, :-1, :]
                    gt_vel = gt_coords[:, 1:, :] - gt_coords[:, :-1, :]
                    total_jitter_err += torch.norm(
                        (pred_vel[:, 1:, :] - pred_vel[:, :-1, :]) - (gt_vel[:, 1:, :] - gt_vel[:, :-1, :]),
                        dim=2).mean().item()

            scale = 384 / 64
            avg_val_h, avg_val_z = val_h / len(val_loader), val_z / len(val_loader)
            avg_px_err, avg_jitter = (total_px_err / len(val_loader)) * scale, (
                        total_jitter_err / len(val_loader)) * scale
            p95_px_err = torch.quantile(torch.cat(all_batch_errors, dim=0).float(), 0.95).item() * scale

            log(f"Epoch [{epoch + 1}/{args.epochs}] | Val Loss: {avg_val_h:.4f} | PX Err: {avg_px_err:.2f} px | P95 Err: {p95_px_err:.2f} px | Jitter: {avg_jitter:.4f}")
            wandb.log({"val/heatmap_loss": avg_val_h, "val/z_loss": avg_val_z, "metrics/pixel_error": avg_px_err,
                       "metrics/p95_error": p95_px_err, "metrics/kinematic_jitter": avg_jitter})

        plt.figure();
        plt.plot(train_h_losses, label='Train');
        plt.plot(val_h_losses, label='Val');
        plt.legend();
        plt.savefig(f"{args.wandb_name}_loss.png");
        plt.close()
        save_metrics(args.wandb_name, avg_val_h, avg_val_z, avg_px_err, p95_px_err, avg_jitter, args.metrics_file)
        wandb.finish()

    except Exception as e:
        log(traceback.format_exc())
        save_metrics(args.wandb_name, 999, 999, 999, 999, 999, args.metrics_file)
        wandb.finish()


if __name__ == "__main__":
    main()