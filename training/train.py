import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import wandb

from dataset import StateraDataset
from model import StateraModel

LOG_FILE = "statera_training.log"


def log_print(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def get_subpixel_coords(logits_heatmap):
    B, T, H, W = logits_heatmap.shape
    device = logits_heatmap.device
    probs = torch.sigmoid(logits_heatmap).view(B * T, -1)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
    probs = probs.view(B, T, H, W)
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    y_center = (probs * y_grid.float().view(1, 1, H, W)).sum(dim=(2, 3))
    x_center = (probs * x_grid.float().view(1, 1, H, W)).sum(dim=(2, 3))
    return torch.stack([x_center, y_center], dim=2)


def run_training():
    # Change 5A: Initialize Weights & Biases
    wandb.init(project="STATERA", name="10K-Temporal-Polar")

    if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"Hardware allocated: {device}")

    full_dataset = StateraDataset('../sim/statera_poc.hdf5')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    BATCH_SIZE = 24
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = StateraModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Change 4: Setup Loss Metrics
    criterion_h = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    criterion_z = nn.HuberLoss()
    criterion_m = nn.HuberLoss()

    log_print("STATERA PoC Temporal Tracking Training Started...")

    epochs = 50
    best_val_loss = float('inf')
    patience = 6
    patience_counter = 0

    train_h_losses, train_z_losses, train_m_losses = [], [], []
    val_h_losses, val_z_losses, val_m_losses = [], [], []
    global_start_time = time.time()

    start_sigma = 12.5
    end_sigma = 3.0
    decay_epochs = 30

    for epoch in range(epochs):
        current_sigma = start_sigma - (start_sigma - end_sigma) * (
                    epoch / decay_epochs) if epoch < decay_epochs else end_sigma
        train_dataset.dataset.update_sigma(current_sigma)

        epoch_start_time = time.time()
        model.train()
        running_h_loss, running_z_loss, running_m_loss = 0.0, 0.0, 0.0
        interval_start = time.time()

        for i, (vids, gt_h, gt_z, gt_m) in enumerate(train_loader):
            vids, gt_h = vids.to(device, non_blocking=True), gt_h.to(device, non_blocking=True)
            gt_z, gt_m = gt_z.to(device, non_blocking=True), gt_m.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_h, pred_z, pred_m = model(vids)

            # Change 4: EXPONENTIAL TEMPORAL PROGRESSIVE WEIGHTING
            weighted_loss_h, weighted_loss_z, weighted_loss_m = 0.0, 0.0, 0.0

            for f in range(16):
                temporal_weight = (f / 15.0) ** 2
                weighted_loss_h += criterion_h(pred_h[:, f], gt_h[:, f]) * temporal_weight
                weighted_loss_z += criterion_z(pred_z[:, f], gt_z[:, f]) * temporal_weight
                weighted_loss_m += criterion_m(pred_m[:, f] * 100.0, gt_m[:, f] * 100.0) * temporal_weight

            total_loss = weighted_loss_h + (0.1 * weighted_loss_z) + (0.1 * weighted_loss_m)
            total_loss.backward()
            optimizer.step()

            running_h_loss += weighted_loss_h.item()
            running_z_loss += weighted_loss_z.item()
            running_m_loss += weighted_loss_m.item()

            if i % 10 == 0:
                elapsed_time = time.time() - interval_start
                interval_start = time.time()
                time_label = "1st batch" if i == 0 else "10 batches"
                log_print(f"Epoch[{epoch + 1}/{epochs}] | Train Batch[{i}/{len(train_loader)}] "
                          f"| Time ({time_label}): {elapsed_time:.2f}s "
                          f"| Loss: {total_loss.item():.4f} (wH: {weighted_loss_h.item():.4f}, wZ: {weighted_loss_z.item():.4f}, wM: {weighted_loss_m.item():.4f})")

        # ==========================================
        #              VALIDATION PHASE
        # ==========================================
        model.eval()
        val_running_h, val_running_z, val_running_m = 0.0, 0.0, 0.0
        total_pixel_error = 0.0

        with torch.no_grad():
            for vids, gt_h, gt_z, gt_m in val_loader:
                vids, gt_h = vids.to(device, non_blocking=True), gt_h.to(device, non_blocking=True)
                gt_z, gt_m = gt_z.to(device, non_blocking=True), gt_m.to(device, non_blocking=True)

                pred_h, pred_z, pred_m = model(vids)

                weighted_val_h, weighted_val_z, weighted_val_m = 0.0, 0.0, 0.0

                for f in range(16):
                    w = (f / 15.0) ** 2
                    weighted_val_h += criterion_h(pred_h[:, f], gt_h[:, f]) * w
                    weighted_val_z += criterion_z(pred_z[:, f], gt_z[:, f]) * w
                    weighted_val_m += criterion_m(pred_m[:, f] * 100.0, gt_m[:, f] * 100.0) * w
                val_running_h += weighted_val_h.item()
                val_running_z += weighted_val_z.item()
                val_running_m += weighted_val_m.item()

                pred_coords = get_subpixel_coords(pred_h)

                # GT coords extraction across sequence
                B, T, H_res, W_res = gt_h.shape
                flat_gt = gt_h.view(B * T, -1)
                idx = flat_gt.argmax(dim=1)
                gt_coords = torch.stack([idx % W_res, idx // W_res], dim=1).float().view(B, T, 2)

                pixel_dist = torch.norm(pred_coords - gt_coords, dim=2).mean().item()
                total_pixel_error += pixel_dist

        avg_train_h = running_h_loss / len(train_loader)
        avg_train_z = running_z_loss / len(train_loader)
        avg_train_m = running_m_loss / len(train_loader)

        avg_val_h = val_running_h / len(val_loader)
        avg_val_z = val_running_z / len(val_loader)
        avg_val_m = val_running_m / len(val_loader)

        avg_pixel_err = total_pixel_error / len(val_loader)

        train_total_loss = avg_train_h + (0.1 * avg_train_z) + (0.1 * avg_train_m)
        val_total_loss = avg_val_h + (0.1 * avg_val_z) + (0.1 * avg_val_m)

        train_h_losses.append(avg_train_h)
        train_z_losses.append(avg_train_z)
        train_m_losses.append(avg_train_m)
        val_h_losses.append(avg_val_h)
        val_z_losses.append(avg_val_z)
        val_m_losses.append(avg_val_m)

        # Change 5A: Log all losses and dynamic metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]['lr'],
            "sigma": current_sigma,
            "train/heatmap_loss": avg_train_h,
            "train/z_loss": avg_train_z,
            "train/mag_loss": avg_train_m,
            "train/total_loss": train_total_loss,
            "val/heatmap_loss": avg_val_h,
            "val/z_loss": avg_val_z,
            "val/mag_loss": avg_val_m,
            "val/total_loss": val_total_loss,
            "metrics/pixel_error_internal": avg_pixel_err,
            "metrics/pixel_error_true": avg_pixel_err * (384.0 / 64.0)
        })

        saved_flag = ""
        # -------------------------------------------------------------
        # SIGMA-AWARE EARLY STOPPING LOGIC
        # -------------------------------------------------------------
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            patience_counter = 0
            probe_state = {k: v for k, v in model.state_dict().items() if model.get_parameter(k).requires_grad}
            torch.save(probe_state, 'best_statera_probe.pth')
            saved_flag = "✓[Checkpoint Saved]"
        else:
            # Change 5B: Patience ONLY ticks if Sigma Annealing has hit floor
            if epoch >= decay_epochs:
                patience_counter += 1
                saved_flag = f"![No Improvement: Patience {patience_counter}/{patience}]"
            else:
                saved_flag = "![Ignored: Annealing Sigma]"

        scale_factor = 384.0 / 64.0
        true_video_error = avg_pixel_err * scale_factor
        epoch_time = time.time() - epoch_start_time

        log_print(f"=== Epoch {epoch + 1} Completed in {epoch_time:.1f}s | Sigma: {current_sigma:.2f} ===")
        log_print(f"Train - wHeatmap: {avg_train_h:.4f} | wZ-Depth: {avg_train_z:.4f} | wMag: {avg_train_m:.4f}")
        log_print(f"Val   - wHeatmap: {avg_val_h:.4f} | wZ-Depth: {avg_val_z:.4f} | wMag: {avg_val_m:.4f} {saved_flag}")
        log_print(
            f"➔ PHYSICAL SCRUTINY: Missed CoM by {avg_pixel_err:.2f} internal pixels -> ~{true_video_error:.1f} pixels on ACTUAL 384x384 video.\n")

        if patience_counter >= patience:
            log_print(f"!! Early stopping triggered! Validation loss plateaued for {patience} epochs post-annealing.")
            break

    total_time = (time.time() - global_start_time) / 60
    log_print(f"✓ Training completed in {total_time:.2f} minutes.")

    # ==========================================
    #           GENERATE CONVERGENCE PLOTS
    # ==========================================
    log_print("Generating training plots...")
    plt.figure(figsize=(18, 5))

    actual_epochs = len(train_h_losses)
    plot_epochs = range(2, actual_epochs + 1) if actual_epochs > 1 else [1]
    slice_idx = 1 if actual_epochs > 1 else 0

    plt.subplot(1, 3, 1)
    plt.plot(plot_epochs, train_h_losses[slice_idx:], marker='o', color='b', label='Train')
    plt.plot(plot_epochs, val_h_losses[slice_idx:], marker='o', color='c', linestyle='dashed', label='Val')
    plt.title('Weighted BCE Heatmap Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(plot_epochs, train_z_losses[slice_idx:], marker='s', color='r', label='Train')
    plt.plot(plot_epochs, val_z_losses[slice_idx:], marker='s', color='m', linestyle='dashed', label='Val')
    plt.title('Weighted Z-Depth Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(plot_epochs, train_m_losses[slice_idx:], marker='^', color='g', label='Train')
    plt.plot(plot_epochs, val_m_losses[slice_idx:], marker='^', color='orange', linestyle='dashed', label='Val')
    plt.title('Weighted Magnitude Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plot_filename = 'statera_temporal_loss.png'
    plt.savefig(plot_filename)
    plt.close()
    log_print(f"✓ Training plots saved to '{plot_filename}'")
    log_print(f"✓ Best model weights saved to 'best_statera_probe.pth'")

    wandb.finish()


if __name__ == "__main__":
    if not os.path.exists('../sim/statera_poc.hdf5'):
        print("✘ Error: statera_poc.hdf5 not found in ../sim/")
    else:
        run_training()