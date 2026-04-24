import os
import glob
import re
import json
import torch
import torch.nn.functional as F
import h5py
import numpy as np
import cv2
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


def get_target_epochs():
    """HARDCODED TARGETS: Bypasses the auto-parser to load fully converged checkpoints."""
    return {
        'STATERA-50K-SOTA': 7,
        'Run-00-Anchor-Saved': 30,
        'Run-04-Static-Dot-Saved': 30,
        'Run-05-Standard-Sigma-Saved': 30,
        'Run-01-DINOv2-Fixed': 24,  # Recovered from corruption
        'Run-03-ResNet3D-Fixed': 30
    }


def load_champion_models(target_ckpts, device):
    """Instantiates the specific models and loads the target weights."""
    print("\n[*] Loading Target Epoch Models to VRAM...")
    loaded_models = {}

    for run_name, epoch in target_ckpts.items():
        if 'DINOv2' in run_name:
            backbone = 'dinov2'
        elif 'ResNet3D' in run_name:
            backbone = 'resnet3d'
        else:
            backbone = 'vjepa'

        # Determine checkpoint path
        if '50K-SOTA' in run_name:
            ckpt_path = f"sota_50k_checkpoints/{run_name}_epoch_{epoch}.pth"
        else:
            ckpt_path = f"checkpoints/{run_name}_epoch_{epoch}.pth"

        if not os.path.exists(ckpt_path):
            print(f"  [!] Missing weight file: {ckpt_path}")
            continue

        model = StateraModel(
            decoder_type='deconv', temporal_mixer='conv1d',
            single_task=False, backbone_type=backbone,
            scratch=False, finetune_blocks=0
        ).to(device)

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        loaded_models[run_name] = model
        print(f"  [✓] Loaded {run_name} (Epoch {epoch})")

    return loaded_models


def draw_legend(img, color_map):
    y_offset = 20
    for name, color in color_map.items():
        display_name = name.replace('Run-', '').replace('-Saved', '').replace('-Fixed', '')
        cv2.putText(img, display_name, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        y_offset += 15


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Hardcoded Epoch Targets
    target_ckpts = get_target_epochs()
    models = load_champion_models(target_ckpts, device)

    # 2. Assign specific colors to the 1K ablation runs (BGR format for OpenCV)
    colors = [
        (0, 0, 255),  # Red (Anchor)
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0)  # Cyan
    ]
    color_map = {}
    color_idx = 0
    sota_key = None

    for name in models.keys():
        if '50K' in name:
            sota_key = name
        else:
            color_map[name] = colors[color_idx % len(colors)]
            color_idx += 1

    # 3. Setup output directory
    out_dir = "visualization_output"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[*] Rendering Videos to '{out_dir}/'...")

    # 4. Iterate over physical dataset
    data_dir = '../sim2real/output'
    h5_files = sorted(glob.glob(os.path.join(data_dir, '*_statera.h5')))

    if not h5_files:
        print(f"[!] No real-world h5 files found in {data_dir}. Check paths!")
        return

    for f_path in tqdm(h5_files, desc="Processing Sequences"):
        with h5py.File(f_path, 'r') as f:
            for group_name in f.keys():
                if not group_name.startswith('seq_'): continue

                # Load sequence
                grp = f[group_name]
                frames_bgr = grp["frames"][:]
                frames_rgb = frames_bgr[:, :, :, ::-1].copy()

                com_u = grp["com_u"][:]
                com_v = grp["com_v"][:]
                gt_coords = np.stack([com_u, com_v], axis=1)  # [16, 2]

                # Format for network
                vids_tensor = torch.from_numpy(frames_rgb).float() / 255.0
                vids_tensor = vids_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(device)

                # Run Inference on all models
                predictions = {}
                heatmaps_50k = None

                with torch.no_grad():
                    for name, model in models.items():
                        pred_h, _ = model(vids_tensor)
                        if name == sota_key:
                            heatmaps_50k = torch.sigmoid(pred_h[0]).cpu().numpy()
                        else:
                            pred_coords = get_subpixel_coords(pred_h, temperature=2.0)
                            pred_coords_real = (pred_coords[0] * (384.0 / 64.0)).cpu().numpy()
                            predictions[name] = pred_coords_real

                seq_id = group_name.split('_')[1]
                vid_name = os.path.basename(f_path).replace('.h5', '')

                out_path_1k = os.path.join(out_dir, f"{vid_name}_{seq_id}_1K_Traces.mp4")
                out_path_50k = os.path.join(out_dir, f"{vid_name}_{seq_id}_50K_Heatmap.mp4")

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer_1k = cv2.VideoWriter(out_path_1k, fourcc, 5.0, (384, 384))
                writer_50k = cv2.VideoWriter(out_path_50k, fourcc, 5.0, (384, 384))

                for frame_idx in range(16):
                    # ----------------------------------------------------
                    # VIDEO 1: The 1K Ablation Trajectories
                    # ----------------------------------------------------
                    img_1k = frames_bgr[frame_idx].copy()

                    gt_x, gt_y = int(gt_coords[frame_idx][0]), int(gt_coords[frame_idx][1])
                    cv2.drawMarker(img_1k, (gt_x, gt_y), (0, 0, 0), cv2.MARKER_CROSS, 14, 3)
                    cv2.drawMarker(img_1k, (gt_x, gt_y), (255, 255, 255), cv2.MARKER_CROSS, 14, 1)

                    for name, coords in predictions.items():
                        col = color_map[name]
                        for i in range(1, frame_idx + 1):
                            pt1 = (int(coords[i - 1][0]), int(coords[i - 1][1]))
                            pt2 = (int(coords[i][0]), int(coords[i][1]))
                            cv2.line(img_1k, pt1, pt2, col, 2, cv2.LINE_AA)

                        curr_pt = (int(coords[frame_idx][0]), int(coords[frame_idx][1]))
                        cv2.circle(img_1k, curr_pt, 4, col, -1, cv2.LINE_AA)

                    draw_legend(img_1k, color_map)
                    writer_1k.write(img_1k)

                    # ----------------------------------------------------
                    # VIDEO 2: The 50K SOTA Heatmap
                    # ----------------------------------------------------
                    if heatmaps_50k is not None:
                        img_50k = frames_bgr[frame_idx].copy()

                        hm_raw = heatmaps_50k[frame_idx]
                        hm_resized = cv2.resize(hm_raw, (384, 384), interpolation=cv2.INTER_CUBIC)

                        hm_norm = np.clip(hm_resized * 255.0, 0, 255).astype(np.uint8)
                        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)

                        alpha_mask = (hm_resized > 0.1).astype(np.float32)[..., np.newaxis]
                        blended = (hm_color * alpha_mask * 0.6) + (img_50k * (1.0 - (alpha_mask * 0.6)))
                        img_50k = blended.astype(np.uint8)

                        cv2.drawMarker(img_50k, (gt_x, gt_y), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
                        cv2.putText(img_50k, "50K SOTA Prediction", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1, cv2.LINE_AA)

                        writer_50k.write(img_50k)

                writer_1k.release()
                if heatmaps_50k is not None: writer_50k.release()

    print(f"\n[SUCCESS] Rendered all requested evaluation videos to '{out_dir}/'!")


if __name__ == "__main__":
    main()