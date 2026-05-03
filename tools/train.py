import os
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
import copy
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statera.dataset import StateraDataset
from statera.model import StateraModel


def parse_args():
    parser = argparse.ArgumentParser(description="STATERA Training Pipeline (1K Ablations)")
    parser.add_argument('--dataset_path', type=str, default='../sim/1K-ablation.hdf5')
    parser.add_argument('--decoder_type', type=str, default='mlp', choices=['mlp', 'deconv', 'regression'])
    parser.add_argument('--temporal_mixer', type=str, default='conv1d', choices=['conv1d', 'transformer', 'none'])
    parser.add_argument('--target_type', type=str, default='dot',
                        choices=['dot', 'blend', 'crescent', 'twostage', 'dynamic_twostage'])
    parser.add_argument('--backbone', type=str, default='vjepa', choices=['vjepa', 'dinov2', 'resnet3d'])
    parser.add_argument('--temporal_weighting', type=str, default='exponential', choices=['exponential', 'uniform'])
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'])

    parser.add_argument('--single_task', action='store_true')
    parser.add_argument('--static_sigma', action='store_true')
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--temporal_dropout', type=float, default=0.0)
    parser.add_argument('--jitter_box', action='store_true')
    parser.add_argument('--finetune_blocks', type=int, default=0)

    parser.add_argument('--wandb_name', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--funnel_epochs', type=int, default=10)
    parser.add_argument('--accumulate_steps', type=int, default=1)
    parser.add_argument('--batch_size_override', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--metrics_file', type=str, default='run_metrics.json')

    parser.add_argument('--save_checkpoint_every_epoch', action='store_true')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    # Curriculum Scheduler Args
    parser.add_argument('--curriculum_patience', type=int, default=2)
    parser.add_argument('--curriculum_delta', type=float, default=0.01)

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


def focal_loss_bce(logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = torch.exp(-bce)
    focal_loss = alpha * (1 - p_t) ** gamma * bce
    return focal_loss.mean()


def save_metrics(run_name, h_loss, z_loss, pixel_err, p95_err, jitter_err, n_come, h_ke, metrics_file):
    data = {}
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
    data[run_name] = {
        "val_heatmap_loss": h_loss,
        "val_z_loss": z_loss,
        "val_pixel_error": pixel_err,
        "val_p95_error": p95_err,
        "val_kinematic_jitter": jitter_err,
        "val_n_come": n_come,
        "val_h_ke": h_ke
    }
    with open(metrics_file, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    args = parse_args()
    log_file = f"{args.wandb_name}_train.log"

    def log(msg):
        print(msg)
        with open(log_file, "a") as f: f.write(msg + "\n")

    wandb.init(project="STATERA", name=args.wandb_name, config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[*] Initialized {args.wandb_name} on {device}")

    try:
        init_target = 'crescent' if args.target_type in ['twostage', 'dynamic_twostage'] else args.target_type
        start_sigma = 3.0 if args.static_sigma else 12.5

        base_dataset = StateraDataset(args.dataset_path, target_type=init_target, start_sigma=start_sigma,
                                      jitter_box=args.jitter_box)
        train_size = int(0.8 * len(base_dataset))
        val_size = len(base_dataset) - train_size

        generator = torch.Generator().manual_seed(42)
        train_indices, val_indices = random_split(range(len(base_dataset)), [train_size, val_size], generator=generator)

        train_ds = torch.utils.data.Subset(base_dataset, train_indices)
        val_base_dataset = copy.deepcopy(base_dataset)
        val_base_dataset.jitter_box = False
        val_ds = torch.utils.data.Subset(val_base_dataset, val_indices)

        if args.batch_size_override > 0:
            batch_size = args.batch_size_override
        elif args.scratch:
            batch_size = 2
        elif args.finetune_blocks > 0 or args.temporal_mixer == 'transformer':
            batch_size = 8
        else:
            batch_size = 24

        log(f"[*] Using dynamic batch size: {batch_size} | Accumulation steps: {args.accumulate_steps}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = StateraModel(
            decoder_type=args.decoder_type, temporal_mixer=args.temporal_mixer,
            single_task=args.single_task, backbone_type=args.backbone, scratch=args.scratch,
            finetune_blocks=args.finetune_blocks
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        criterion_h = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
        criterion_z = nn.HuberLoss()

        train_h_losses, val_h_losses = [], []

        if args.save_checkpoint_every_epoch:
            os.makedirs(args.checkpoint_dir, exist_ok=True)

        curriculum_stage = 'crescent'
        best_curriculum_loss = float('inf')
        curriculum_patience_counter = 0

        for epoch in range(args.epochs):
            funnel_ratio = min(1.0, epoch / args.funnel_epochs)

            if not args.static_sigma:
                current_sigma = max(3.0, 12.5 - (9.5 * funnel_ratio))
                train_ds.dataset.update_sigma(current_sigma)
                val_ds.dataset.update_sigma(current_sigma)

            if train_ds.dataset.target_type in ['crescent', 'blend']:
                train_ds.dataset.update_phase_alpha(funnel_ratio)
                val_ds.dataset.update_phase_alpha(funnel_ratio)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            run_h, run_z = 0.0, 0.0

            for i, (vids, gt_h, gt_z, gt_coords, bbox_diags) in enumerate(train_loader):
                vids = vids.to(device)
                gt_h = gt_h.to(device)
                gt_z = gt_z.to(device)
                gt_coords = gt_coords.to(device)

                if args.temporal_dropout > 0.0:
                    drop_mask = (torch.rand(vids.shape[0], 1, 16, 1, 1, device=device) > args.temporal_dropout).float()
                    vids = vids * drop_mask

                pred_h, pred_z = model(vids)

                w_h, w_z = 0.0, 0.0

                for f in range(16):
                    w = 1.0 if args.temporal_weighting == 'uniform' else max(0.05, (f / 15.0) ** 2)

                    if args.decoder_type == 'regression':
                        w_h += F.mse_loss(pred_h[:, f], gt_coords[:, f]) * w
                    else:
                        if args.loss_type == 'focal':
                            w_h += focal_loss_bce(pred_h[:, f], gt_h[:, f]) * w
                        else:
                            w_h += criterion_h(pred_h[:, f], gt_h[:, f]) * w

                    if not args.single_task:
                        w_z += criterion_z(pred_z[:, f], gt_z[:, f]) * w

                loss = (w_h + (0.1 * w_z if not args.single_task else 0.0)) / args.accumulate_steps
                loss.backward()

                if (i + 1) % args.accumulate_steps == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                run_h += w_h.item() if isinstance(w_h, torch.Tensor) else float(w_h)
                run_z += w_z.item() if isinstance(w_z, torch.Tensor) else float(w_z)

            # Validation Loop
            model.eval()
            val_h, val_z, total_px_err = 0.0, 0.0, 0.0
            total_n_come = 0.0
            total_jitter_err = 0.0
            total_norm_jitter = 0.0
            all_batch_errors = []

            with torch.no_grad():
                for vids, gt_h, gt_z, gt_coords, bbox_diags in val_loader:
                    vids = vids.to(device)
                    gt_h = gt_h.to(device)
                    gt_z = gt_z.to(device)
                    gt_coords = gt_coords.to(device)
                    bbox_diags = bbox_diags.to(device)

                    pred_h, pred_z = model(vids)

                    for f in range(16):
                        w = 1.0 if args.temporal_weighting == 'uniform' else max(0.05, (f / 15.0) ** 2)

                        if args.decoder_type == 'regression':
                            val_h += F.mse_loss(pred_h[:, f], gt_coords[:, f]).item() * w
                        else:
                            val_h += criterion_h(pred_h[:, f], gt_h[:, f]).item() * w

                        if not args.single_task:
                            val_z += criterion_z(pred_z[:, f], gt_z[:, f]).item() * w

                    if args.decoder_type == 'regression':
                        pred_coords = pred_h
                    else:
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

            train_h_losses.append(run_h / len(train_loader))
            val_h_losses.append(avg_val_h)

            log(f"Epoch[{epoch + 1}/{args.epochs}] | Target: {train_ds.dataset.target_type.upper()} | Val Loss: {avg_val_h:.4f} | N-CoME: {avg_n_come * 100:.2f}% | H_KE: {h_ke:.4f} | PX Err: {avg_px_err:.2f} px | Jitter: {avg_jitter:.4f}")
            wandb.log({
                "val/heatmap_loss": avg_val_h,
                "val/z_loss": avg_val_z,
                "metrics/pixel_error": avg_px_err,
                "metrics/p95_error": p95_px_err,
                "metrics/kinematic_jitter": avg_jitter,
                "metrics/N-CoME": avg_n_come,
                "metrics/H_KE": h_ke,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            if args.target_type in ['twostage', 'dynamic_twostage'] and curriculum_stage == 'crescent':
                safe_alpha = min(1.0, epoch / args.funnel_epochs)

                if safe_alpha >= 1.0:
                    if avg_val_h < (best_curriculum_loss - args.curriculum_delta):
                        best_curriculum_loss = avg_val_h
                        curriculum_patience_counter = 0
                    else:
                        curriculum_patience_counter += 1
                        log(f"[*] Curriculum Patience: {curriculum_patience_counter}/{args.curriculum_patience}")

                    if curriculum_patience_counter >= args.curriculum_patience:
                        log("\n[>>] CURRICULUM PHASE SHIFT TRIGGERED: Plateau detected. Switching target to DOT.\n")
                        curriculum_stage = 'dot'

                        train_ds.dataset.target_type = 'dot'
                        train_ds.dataset.update_phase_alpha(1.0)

                        val_ds.dataset.target_type = 'dot'
                        val_ds.dataset.update_phase_alpha(1.0)

                        best_curriculum_loss = float('inf')

            if args.save_checkpoint_every_epoch:
                ckpt_path = os.path.join(args.checkpoint_dir, f"{args.wandb_name}_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), ckpt_path)
                log(f"[✓] Saved checkpoint to: {ckpt_path}")

            scheduler.step()

        plt.figure()
        plt.plot(train_h_losses, label='Train Loss')
        plt.plot(val_h_losses, label='Val Loss')
        plt.title(f"{args.wandb_name} Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{args.wandb_name}_loss.png")
        plt.close()

        save_metrics(args.wandb_name, avg_val_h, avg_val_z, avg_px_err, p95_px_err, avg_jitter, avg_n_come, h_ke,
                     args.metrics_file)
        log("[✓] Run completed successfully.")

    except Exception as e:
        log(f"[!] CRITICAL ERROR ENCOUNTERED: {e}")
        log(traceback.format_exc())
        save_metrics(args.wandb_name, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, args.metrics_file)

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()