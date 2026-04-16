import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import argparse

from dataset import StateraDataset
from model import StateraModel


def parse_args():
    parser = argparse.ArgumentParser(description="STATERA Temperature Validation")
    parser.add_argument('--dataset_path', type=str, default='../sim/statera_poc.hdf5')
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to Vanguard-5K-Hero-V3_epoch_18.pth")
    parser.add_argument('--temperature', type=float, default=2.0, help="Softmax Temperature (Try 2.0 to 4.0)")
    parser.add_argument('--batch_size', type=int, default=8)
    return parser.parse_args()


def get_subpixel_coords_temperature(logits_heatmap, temperature=2.0):
    """
    THE FIX: Temperature-Scaled Spatial Softmax.
    Eliminates the 6-pixel quantization staircase by softly bleeding
    the one-hot peak into neighboring pixels, restoring true sub-pixel interpolation.
    """
    B, T, H, W = logits_heatmap.shape
    # Divide logits by temperature BEFORE Softmax
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

    print("\n" + "=" * 50)
    print(f"[>] STATERA Zero-Retrain Validation Check")
    print(f"[>] Target: Vanguard-5K-Hero-V3")
    print(f"[>] Applying Softmax Temperature: {args.temperature}")
    print("=" * 50 + "\n")

    # 1. Reconstruct the EXACT validation set from train.py
    # Since we are evaluating Epoch 18 (post-funnel), the target is a pure DOT with sigma 3.0
    print("[*] Reconstructing strict Validation Split...")
    base_dataset = StateraDataset(args.dataset_path, target_type='dot', start_sigma=3.0, jitter_box=False)

    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size
    generator = torch.Generator().manual_seed(42)  # MUST match train.py seed
    _, val_indices = random_split(range(len(base_dataset)), [train_size, val_size], generator=generator)

    val_ds = torch.utils.data.Subset(base_dataset, val_indices)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Initialize Model (Matching Vanguard V3 Params)
    print(f"[*] Initializing Model and loading weights from:\n    {args.checkpoint}")
    model = StateraModel(
        decoder_type='deconv',
        temporal_mixer='conv1d',
        single_task=False,
        backbone_type='vjepa',
        finetune_blocks=2
    ).to(device)

    # Load weights
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 3. Evaluation Loop
    total_px_err = 0.0
    total_jitter_err = 0.0
    all_batch_errors = []

    print(f"[*] Running forward pass on {val_size} videos. Please wait...")

    with torch.no_grad():
        for i, (vids, gt_h, gt_z, gt_coords) in enumerate(val_loader):
            vids = vids.to(device)
            gt_coords = gt_coords.to(device)

            pred_h, _ = model(vids)

            # Extract coordinates using the NEW Temperature Logic
            pred_coords = get_subpixel_coords_temperature(pred_h, temperature=args.temperature)

            # 1. Pixel Error
            batch_px_errors = torch.norm(pred_coords - gt_coords, dim=2)
            total_px_err += batch_px_errors.mean().item()
            all_batch_errors.append(batch_px_errors.detach().cpu())

            # 2. Kinematic Jitter (Acceleration Variance)
            pred_vel = pred_coords[:, 1:, :] - pred_coords[:, :-1, :]
            gt_vel = gt_coords[:, 1:, :] - gt_coords[:, :-1, :]
            pred_accel = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]
            gt_accel = gt_vel[:, 1:, :] - gt_vel[:, :-1, :]

            batch_jitter = torch.norm(pred_accel - gt_accel, dim=2).mean().item()
            total_jitter_err += batch_jitter

    # 4. Final Math
    scale_factor = 384 / 64
    avg_px_err = (total_px_err / len(val_loader)) * scale_factor
    avg_jitter = (total_jitter_err / len(val_loader)) * scale_factor

    cat_errors = torch.cat(all_batch_errors, dim=0)
    p95_px_err = torch.quantile(cat_errors.float(), 0.95).item() * scale_factor

    print("\n" + "=" * 50)
    print("TEMPERATURE EVALUATION RESULTS")
    print("=" * 50)
    print(f"   Mean Pixel Error : {avg_px_err:.2f} px")
    print(f"   P95 Pixel Error  : {p95_px_err:.2f} px")
    print(f"   Kinematic Jitter : {avg_jitter:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()