import os
import glob
import torch
import torch.nn.functional as F
import h5py
import numpy as np
import gc
from tqdm import tqdm
from model import StateraModel


def calculate_esd(logits, temperature=2.0):
    """
    Calculates Expected Spatial Dispersion (ESD).
    ESD = E[(X - E[X])^2 + (Y - E[Y])^2]
    Units: Feature-map pixels squared (on a 64x64 grid).
    """
    B, T, H, W = logits.shape
    device = logits.device

    # 1. Softmax Probabilities
    logits_flat = logits.view(B * T, -1) / temperature
    probs = F.softmax(logits_flat, dim=-1).view(B, T, H, W)

    # 2. Create Coordinate Grids
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device).float(),
        torch.arange(W, device=device).float(),
        indexing='ij'
    )
    grid_y = grid_y.view(1, 1, H, W)
    grid_x = grid_x.view(1, 1, H, W)

    # 3. Soft-Argmax (Expected Coordinates)
    pred_y = torch.sum(probs * grid_y, dim=(2, 3))  # [B, T]
    pred_x = torch.sum(probs * grid_x, dim=(2, 3))  # [B, T]

    # 4. Calculate Distance from Center for every pixel
    # (grid - center)^2
    dy_sq = (grid_y - pred_y.view(B, T, 1, 1)) ** 2
    dx_sq = (grid_x - pred_x.view(B, T, 1, 1)) ** 2

    # 5. Dispersion = Sum(P(i) * Distance^2)
    dispersion = torch.sum(probs * (dy_sq + dx_sq), dim=(2, 3))

    return dispersion  # [B, T]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initialized Spatial Dispersion Suite on {device}")

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

        total_esd = 0.0
        total_frames = 0

        with torch.no_grad():
            for f_path in tqdm(h5_files, desc=f"Evaluating {name}", leave=False):
                with h5py.File(f_path, 'r') as f:
                    for group_name in f.keys():
                        if not group_name.startswith('seq_'): continue

                        # Load and flip BGR->RGB
                        frames = torch.from_numpy(f[group_name]["frames"][:][:, :, :, ::-1].copy()).float() / 255.0
                        vids = frames.permute(3, 0, 1, 2).unsqueeze(0).to(device)

                        pred_h, _ = model(vids)

                        # Math
                        batch_esd = calculate_esd(pred_h, temperature=2.0)
                        total_esd += batch_esd.sum().item()
                        total_frames += batch_esd.numel()

        avg_esd = total_esd / total_frames
        final_results[name] = avg_esd

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Output Table
    print("\n" + "=" * 70)
    print(f"{'EXPECTED SPATIAL DISPERSION (ESD) - PHYSICAL SPREAD':^70}")
    print("=" * 70)
    print(f"| {'Model Identifier':<25} | {'Dispersion (px^2)':^20} | {'Std Dev (px)':^15} |")
    print("-" * 70)

    # Sort by dispersion ascending (tighter predictions first)
    sorted_res = sorted(final_results.items(), key=lambda x: x[1])
    for name, val in sorted_res:
        # Standard Deviation is the sqrt of variance (dispersion)
        # It gives a more "human readable" spread in pixels
        std_dev = np.sqrt(val)
        print(f"| {name:<25} | {val:>18.4f} | {std_dev:>13.2f} |")
    print("=" * 70)


if __name__ == "__main__":
    main()