import os
import glob
import torch
import torch.nn.functional as F
import h5py
import numpy as np
import gc
from tqdm import tqdm
from model import StateraModel


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
    print(f"[*] Initialized Weighted Euclidean Error Suite on {device}")

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
    h5_files = glob.glob(os.path.join(data_dir, '*_statera.h5'))

    if not h5_files:
        print("[!] No real-world HDF5 files found!")
        return

    # Create Linearly Increasing Weights for 16 frames
    # Frame 0 weight = 1/136, Frame 15 weight = 16/136
    # This heavily penalizes drift at the end of the sequence!
    raw_weights = torch.arange(1, 17, dtype=torch.float32, device=device)
    temporal_weights = raw_weights / raw_weights.sum()

    final_results = {}

    for name, path in checkpoints.items():
        if not os.path.exists(path):
            print(f"[!] Skipping {name}: Path not found.")
            continue

        # Load appropriate backbone
        backbone = 'vjepa'
        if 'DINOv2' in name:
            backbone = 'dinov2'
        elif 'ResNet3D' in name:
            backbone = 'resnet3d'

        model = StateraModel(
            decoder_type='deconv', temporal_mixer='conv1d',
            backbone_type=backbone, scratch=False
        ).to(device)
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
                        # Load and flip BGR->RGB
                        frames = torch.from_numpy(grp["frames"][:][:, :, :, ::-1].copy()).float() / 255.0
                        vids = frames.permute(3, 0, 1, 2).unsqueeze(0).to(device)

                        # Ground Truth
                        com_u = torch.from_numpy(grp["com_u"][:]).float().to(device)
                        com_v = torch.from_numpy(grp["com_v"][:]).float().to(device)
                        gt_coords = torch.stack([com_u, com_v], dim=1)  # [16, 2]

                        # Inference
                        pred_h, _ = model(vids)

                        # Extract coordinates & scale to 384x384 space
                        pred_coords = get_subpixel_coords(pred_h, temperature=2.0)[0] * (384.0 / 64.0)

                        # 1. Calculate raw Euclidean distance per frame [16]
                        frame_distances = torch.norm(pred_coords - gt_coords, dim=1)

                        # 2. Apply Temporal Weighting & Sum for the sequence
                        seq_weighted_error = torch.sum(frame_distances * temporal_weights).item()

                        total_weighted_error += seq_weighted_error
                        seq_count += 1

        avg_error = total_weighted_error / seq_count
        final_results[name] = avg_error

        # Cleanup memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Output Table
    print("\n" + "=" * 75)
    print(f"{'TEMPORALLY-WEIGHTED EUCLIDEAN PIXEL ERROR (Physical Space)':^75}")
    print("=" * 75)
    print(f"| {'Model Identifier':<25} | {'Weighted Pixel Error (px)':^25} |")
    print("-" * 75)

    # Sort by lowest error first
    sorted_res = sorted(final_results.items(), key=lambda x: x[1])
    for name, val in sorted_res:
        print(f"| {name:<25} | {val:>22.2f} px |")
    print("=" * 75)


if __name__ == "__main__":
    main()