import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from dataset import StateraDataset
from model import StateraModel


def get_subpixel_coords(logits_heatmap):
    """Replicated from your utils for deterministic subpixel extraction"""
    B, H, W = logits_heatmap.shape
    device = logits_heatmap.device
    probs = torch.sigmoid(logits_heatmap).view(B, -1)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
    probs = probs.view(B, H, W)
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    y_center = (probs * y_grid.float()).sum(dim=(1, 2))
    x_center = (probs * x_grid.float()).sum(dim=(1, 2))
    return torch.stack([x_center, y_center], dim=1)


def ruthless_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AUDIT] Firing up on {device}...")

    # Load Model
    model = StateraModel(heatmap_res=64).to(device)
    model.load_state_dict(torch.load('best_statera_probe.pth', map_location=device, weights_only=True), strict=False)
    model.eval()

    # Load Dataset (Using a fixed Sigma of 3.0 to test maximum precision)
    full_dataset = StateraDataset('../sim/statera_poc.hdf5', sigma=3.0)

    # ==========================================
    # AUDIT SIZE CONFIGURATION
    # ==========================================
    TARGET_AUDIT_SIZE = 200

    # If your dataset is exactly 200, it tests all of them.
    # If it's larger, it randomly samples 200.
    num_audit_samples = min(TARGET_AUDIT_SIZE, len(full_dataset))

    # Grab a random subset of the data
    np.random.seed(42)
    audit_indices = np.random.choice(len(full_dataset), num_audit_samples, replace=False)

    audit_subset = Subset(full_dataset, audit_indices)
    audit_loader = DataLoader(audit_subset, batch_size=16, shuffle=False)

    total_pixel_error = 0.0
    total_z_error = 0.0
    max_pixel_error = 0.0

    print(f"[AUDIT] Running deterministic forward passes on a random set of {num_audit_samples} samples...")
    with torch.no_grad():
        for vids, gt_h, gt_z in audit_loader:
            vids, gt_h, gt_z = vids.to(device), gt_h.to(device), gt_z.to(device)

            pred_h, pred_z = model(vids)

            # Spatial Metrics
            pred_coords = get_subpixel_coords(pred_h)

            # Extract GT coords
            B, H_res, W_res = gt_h.shape
            flat_gt = gt_h.view(B, -1)
            idx = flat_gt.argmax(dim=1)
            gt_coords = torch.stack([idx % W_res, idx // W_res], dim=1).float()

            # L2 Euclidean Distance in 64x64 space
            batch_pixel_dists = torch.norm(pred_coords - gt_coords, dim=1)
            total_pixel_error += batch_pixel_dists.sum().item()

            if batch_pixel_dists.nelement() > 0:
                max_pixel_error = max(max_pixel_error, batch_pixel_dists.max().item())

            # Z-Depth Absolute Error
            total_z_error += torch.abs(pred_z - gt_z).sum().item()

    # Scale metrics
    avg_internal_pixel_err = total_pixel_error / num_audit_samples
    avg_true_pixel_err = avg_internal_pixel_err * (384.0 / 64.0)
    max_true_pixel_err = max_pixel_error * (384.0 / 64.0)
    avg_z_err = total_z_error / num_audit_samples

    print("\n" + "=" * 40)
    print("      STATERA FINAL PROBE METRICS")
    print("=" * 40)
    print(f"Total Samples Audited : {num_audit_samples}")
    print(f"Avg Z-Depth Error     : {avg_z_err:.4f} meters")
    print(f"Avg CoM Spatial Error : {avg_true_pixel_err:.2f} pixels (on 384x384)")
    print(f"Max CoM Spatial Error : {max_true_pixel_err:.2f} pixels (Worst case)")
    print("=" * 40)

    if avg_true_pixel_err < 15.0:
        print("RESULT: 🟢 The network has achieved high-fidelity physics grounding.")
    else:
        print("RESULT: 🟡 Check your coordinate scaling or bounds.")


if __name__ == "__main__":
    ruthless_audit()