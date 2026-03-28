import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from dataset import StateraDataset
from model import StateraModel


def get_subpixel_coords(logits_heatmap):
    """
    Solves the 64x64 quantization limit.
    Instead of finding an integer pixel, calculates the true theoretical
    floating-point Center of Mass of the probability distribution.
    """
    B, H, W = logits_heatmap.shape
    device = logits_heatmap.device

    # Convert raw logits back to 0-1 probabilities for coordinate extraction
    probs = torch.sigmoid(logits_heatmap).view(B, -1)

    # Normalize so all probabilities sum to 1.0 (creating a true distribution)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
    probs = probs.view(B, H, W)

    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    y_grid = y_grid.float()
    x_grid = x_grid.float()

    # Expected Value (Sum of Value * Probability)
    y_center = (probs * y_grid).sum(dim=(1, 2))
    x_center = (probs * x_grid).sum(dim=(1, 2))

    return torch.stack([x_center, y_center], dim=1)


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware allocated: {device}")

    full_dataset = StateraDataset('../sim/statera_poc.hdf5')
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    BATCH_SIZE = 24
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,
                              prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
                            prefetch_factor=2)

    model = StateraModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # THE ULTIMATE LOSS FUNCTION
    # Computes loss directly on raw tensors.
    # pos_weight=10.0 forces the model to prioritize finding the actual physical box 10x harder than background.
    criterion_h = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    criterion_z = torch.nn.SmoothL1Loss()

    print("STATERA PoC Training Started...")

    epochs = 55
    best_val_loss = float('inf')

    train_h_losses, train_z_losses = [],[]
    val_h_losses, val_z_losses = [],[]
    global_start_time = time.time()

    # Scheduler Settings
    start_sigma = 12.5
    end_sigma = 3.0
    decay_epochs = 30

    for epoch in range(epochs):
        # Calculate Target Sigma
        if epoch < decay_epochs:
            current_sigma = start_sigma - (start_sigma - end_sigma) * (epoch / decay_epochs)
        else:
            current_sigma = end_sigma

        # Inject into dataset via PyTorch Subset parent (.dataset property)
        # Note: Both train and val subsets share the same underlying base dataset instance.
        train_dataset.dataset.update_sigma(current_sigma)

        epoch_start_time = time.time()
        model.train()
        running_h_loss, running_z_loss = 0.0, 0.0
        interval_start = time.time()
        weight_z = 1.0

        for i, (vids, gt_h, gt_z) in enumerate(train_loader):
            vids, gt_h, gt_z = vids.to(device, non_blocking=True), gt_h.to(device, non_blocking=True), gt_z.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_h, pred_z = model(vids)

            loss_h = criterion_h(pred_h, gt_h)
            loss_z = criterion_z(pred_z, gt_z)

            total_loss = loss_h + (weight_z * loss_z)

            total_loss.backward()
            optimizer.step()

            running_h_loss += loss_h.item()
            running_z_loss += loss_z.item()

            if i % 10 == 0:
                elapsed_time = time.time() - interval_start
                interval_start = time.time()
                time_label = "1st batch" if i == 0 else "10 batches"
                print(f"Epoch[{epoch + 1}/{epochs}] | Train Batch[{i}/{len(train_loader)}] "
                      f"| Time ({time_label}): {elapsed_time:.2f}s "
                      f"| Loss: {total_loss.item():.4f} (H: {loss_h.item():.4f}, Z: {loss_z.item():.4f})")

        # ==========================================
        #              VALIDATION PHASE
        # ==========================================
        model.eval()
        val_running_h, val_running_z = 0.0, 0.0
        total_pixel_error = 0.0

        with torch.no_grad():
            for vids, gt_h, gt_z in val_loader:
                vids, gt_h, gt_z = vids.to(device, non_blocking=True), gt_h.to(device, non_blocking=True), gt_z.to(
                    device, non_blocking=True)

                pred_h, pred_z = model(vids)

                loss_h = criterion_h(pred_h, gt_h)
                loss_z = criterion_z(pred_z, gt_z)

                val_running_h += loss_h.item()
                val_running_z += loss_z.item()

                # Extract sub-pixel floats instead of integer grid points
                pred_coords = get_subpixel_coords(pred_h)

                # We apply an arbitrary transform just to extract ground truth coords accurately
                # Note: Ground truth is already 0-1, but our function expects raw logits.
                # For GT, standard integer extraction is safer since it's a perfect Gaussian.
                B, H_res, W_res = gt_h.shape
                flat_gt = gt_h.view(B, -1)
                idx = flat_gt.argmax(dim=1)
                gt_coords = torch.stack([idx % W_res, idx // W_res], dim=1).float()

                pixel_dist = torch.norm(pred_coords - gt_coords, dim=1).mean().item()
                total_pixel_error += pixel_dist

        avg_train_h = running_h_loss / len(train_loader)
        avg_train_z = running_z_loss / len(train_loader)
        avg_val_h = val_running_h / len(val_loader)
        avg_val_z = val_running_z / len(val_loader)
        avg_pixel_err = total_pixel_error / len(val_loader)

        val_total_loss = avg_val_h + avg_val_z

        train_h_losses.append(avg_train_h)
        train_z_losses.append(avg_train_z)
        val_h_losses.append(avg_val_h)
        val_z_losses.append(avg_val_z)

        saved_flag = ""
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            probe_state = {k: v for k, v in model.state_dict().items() if model.get_parameter(k).requires_grad}
            torch.save(probe_state, 'best_statera_probe.pth')
            saved_flag = "✓[Checkpoint Saved (Probe Only)]"

        scale_factor = 384.0 / 64.0
        true_video_error = avg_pixel_err * scale_factor

        epoch_time = time.time() - epoch_start_time
        # Added tracking Sigma log so you can monitor the actual decay behavior on the console
        print(f"=== Epoch {epoch + 1} Completed in {epoch_time:.1f}s | Sigma: {current_sigma:.2f} ===")
        print(f"Train - Heatmap: {avg_train_h:.4f} | Z-Depth: {avg_train_z:.4f}")
        print(f"Val   - Heatmap: {avg_val_h:.4f} | Z-Depth: {avg_val_z:.4f} {saved_flag}")
        print(
            f"➔ PHYSICAL SCRUTINY: Missed CoM by {avg_pixel_err:.2f} internal pixels -> ~{true_video_error:.1f} pixels on ACTUAL 384x384 video.\n")

    total_time = (time.time() - global_start_time) / 60
    print(f"✓ Training completed in {total_time:.2f} minutes.")

    # ==========================================
    #           GENERATE CONVERGENCE PLOTS
    # ==========================================
    print("Generating training plots (Skipping Epoch 1 for visual scale)...")
    plt.figure(figsize=(12, 5))

    plot_epochs = range(2, epochs + 1) if epochs > 1 else [1]
    slice_idx = 1 if epochs > 1 else 0

    plt.subplot(1, 2, 1)
    plt.plot(plot_epochs, train_h_losses[slice_idx:], marker='o', color='b', label='Train')
    plt.plot(plot_epochs, val_h_losses[slice_idx:], marker='o', color='c', linestyle='dashed', label='Val')
    plt.title('BCE-Logits Heatmap Loss (64x64)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(plot_epochs, train_z_losses[slice_idx:], marker='s', color='r', label='Train')
    plt.plot(plot_epochs, val_z_losses[slice_idx:], marker='s', color='m', linestyle='dashed', label='Val')
    plt.title('Z-Depth Loss in Meters (SmoothL1)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plot_filename = 'statera_training_loss.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"✓ Training plots saved to '{plot_filename}'")
    print(f"✓ Best model weights saved to 'best_statera_probe.pth'")


if __name__ == "__main__":
    if not os.path.exists('../sim/statera_poc.hdf5'):
        print("✘ Error: statera_poc.hdf5 not found in ../sim/")
    else:
        run_training()