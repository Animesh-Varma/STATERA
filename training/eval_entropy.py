import os
import glob
import torch
import torch.nn.functional as F
import h5py
import numpy as np
import gc
from tqdm import tqdm
from model import StateraModel


def calculate_entropy(logits, temperature=2.0):
    """
    Calculates Shannon Entropy in bits for a 2D heatmap.
    H = -sum(p * log2(p))
    """
    B, T, H, W = logits.shape
    # Flatten spatial dimensions for softmax [B*T, 4096]
    logits_flat = logits.view(B * T, -1) / temperature
    probs = F.softmax(logits_flat, dim=-1)

    # Calculate Shannon Entropy per frame
    entropy = -torch.sum(probs * torch.log2(probs + 1e-8), dim=-1)
    return entropy.view(B, T)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initialized Entropy Suite on {device}")

    # Checkpoint Manifest
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
        print(f"\n[>] Processing: {name}")
        if not os.path.exists(path):
            print(f"    [!] Checkpoint missing: {path}")
            continue

        # Detect backbone type
        backbone = 'vjepa'
        if 'DINOv2' in name:
            backbone = 'dinov2'
        elif 'ResNet3D' in name:
            backbone = 'resnet3d'

        # Load Model
        model = StateraModel(
            decoder_type='deconv', temporal_mixer='conv1d',
            backbone_type=backbone, scratch=False
        ).to(device)

        try:
            model.load_state_dict(torch.load(path, map_location=device))
        except Exception as e:
            print(f"    [!] Error loading {name}: {e}")
            continue

        model.eval()

        total_entropy = 0.0
        total_frames = 0

        with torch.no_grad():
            for f_path in tqdm(h5_files, desc=f"Scanning Real World", leave=False):
                with h5py.File(f_path, 'r') as f:
                    for group_name in f.keys():
                        if not group_name.startswith('seq_'): continue

                        # Data Prep
                        frames = torch.from_numpy(f[group_name]["frames"][:][:, :, :, ::-1].copy()).float() / 255.0
                        vids = frames.permute(3, 0, 1, 2).unsqueeze(0).to(device)

                        # Inference
                        pred_h, _ = model(vids)

                        # Math
                        batch_h = calculate_entropy(pred_h, temperature=2.0)
                        total_entropy += batch_h.sum().item()
                        total_frames += batch_h.numel()

        avg_h = total_entropy / total_frames
        final_results[name] = avg_h
        print(f"    [✓] {name}: {avg_h:.4f} bits")

        # Memory Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Final Output Table
    print("\n" + "=" * 60)
    print(f"{'STATERA SPATIAL PREDICTIVE ENTROPY (Lower is Better)':^60}")
    print("=" * 60)
    print(f"| {'Model Identifier':<25} | {'Entropy (Bits)':^20} |")
    print("-" * 60)

    # Sort by entropy ascending (best models at top)
    sorted_res = sorted(final_results.items(), key=lambda x: x[1])
    for name, val in sorted_res:
        print(f"| {name:<25} | {val:>15.4f} bits |")
    print("=" * 60)


if __name__ == "__main__":
    main()