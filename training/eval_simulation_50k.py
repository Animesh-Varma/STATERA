import os
import torch
import h5py
import numpy as np
import gc
from torch.utils.data import DataLoader, Subset
from dataset import StateraDataset
from model import StateraModel
from train import get_subpixel_coords
from tqdm import tqdm


def compute_kecs_metrics(pred, gt, diags):
    # Standard KECS: sqrt(N-CoME^2 + (0.1 * NormJitter)^2)
    errs = torch.norm(pred - gt, dim=-1)
    n_come = (errs / diags).mean()

    v_p, v_g = pred[1:] - pred[:-1], gt[1:] - gt[:-1]
    a_p, a_g = v_p[1:] - v_p[:-1], v_g[1:] - v_g[:-1]

    j_errs = torch.norm(a_p - a_g, dim=-1)
    norm_jitter = (j_errs / diags[2:]).mean()

    kecs = torch.sqrt(n_come ** 2 + (0.1 * norm_jitter) ** 2)
    return n_come.item(), norm_jitter.item(), kecs.item()


def main():
    # Performance hack for memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initialized Inference Engine on {device}")

    # Load Dataset (Using 10k validation split)
    ds = StateraDataset('../sim/HiddenMass-50K.hdf5', target_type='dot')
    val_indices = list(range(len(ds) - 10000, len(ds)))
    val_ds = Subset(ds, val_indices)

    # SAFER BATCH SIZE: 8 is standard for ViT-L on 16GB VRAM
    loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    checkpoints = {
        "Run-00-Anchor": "checkpoints/Run-00-Anchor-Saved_epoch_30.pth",
        "Run-04-StaticDot": "checkpoints/Run-04-Static-Dot-Saved_epoch_30.pth",
        "Run-05-StdSigma": "checkpoints/Run-05-Standard-Sigma-Saved_epoch_30.pth",
        "Run-01-DINOv2": "checkpoints/Run-01-DINOv2_epoch_23.pth",
        "Run-03-ResNet3D": "checkpoints/Run-03-ResNet3D-Fixed_epoch_30.pth",
        "SOTA-Crescent": "sota_50k_checkpoints/STATERA-50K-SOTA_epoch_7.pth",
        "SOTA-Sigma": "sota_50k_sigma_checkpoints/STATERA-50K-Sigma_epoch_10.pth"
    }

    print(f"\n{'TABLE 1: IN-DOMAIN SIMULATION PERFORMANCE':^75}")
    print("-" * 75)
    print(f"| {'Model Name':<20} | {'N-CoME (%)':^14} | {'NormJitter':^14} | {'KECS':^12} |")
    print("-" * 75)

    for name, path in checkpoints.items():
        if not os.path.exists(path):
            print(f"| {name:<20} | {'MISSING':^14} | {'MISSING':^14} | {'MISSING':^12} |")
            continue

        # Load appropriate backbone
        backbone = 'vjepa'
        if 'DINOv2' in name:
            backbone = 'dinov2'
        elif 'ResNet3D' in name:
            backbone = 'resnet3d'

        model = StateraModel(decoder_type='deconv', temporal_mixer='conv1d', backbone_type=backbone).to(device)

        # Load weights safely
        try:
            state_dict = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"| {name:<20} | LOAD ERROR    | LOAD ERROR    | LOAD ERROR   |")
            continue

        model.eval()

        all_results = []

        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Evaluating {name}", leave=False)
            for vids, _, _, gt_uv, _ in pbar:
                vids = vids.to(device)

                # Forward Pass
                pred_h, _ = model(vids)

                # Coordinate Conversion (64x64 Heatmap -> 384x384 Reality Space)
                pred_px = get_subpixel_coords(pred_h, temperature=2.0) * 6.0
                gt_px = gt_uv.to(device) * 6.0

                # Sim Diagonals (Constants for Table 1 fairness)
                diags = torch.ones(vids.size(0), 16).to(device) * 50.0

                for i in range(vids.size(0)):
                    all_results.append(compute_kecs_metrics(pred_px[i], gt_px[i], diags[i]))

        avg = np.mean(all_results, axis=0)
        print(f"| {name:<20} | {avg[0] * 100:>12.2f}% | {avg[1]:>14.4f} | {avg[2]:>12.4f} |")

        # AGGRESSIVE MEMORY CLEANUP
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("-" * 75)


if __name__ == "__main__":
    main()