import os
import glob
import h5py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from statera.model import StateraModel


def get_subpixel_coords(logits_heatmap, temperature=2.0):
    B, T, H, W = logits_heatmap.shape
    probs = F.softmax((logits_heatmap.reshape(B * T, -1)) / temperature, dim=1)
    probs = probs.reshape(B, T, H, W)
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=probs.device), torch.arange(W, device=probs.device),
                                    indexing='ij')
    y_center = (probs * y_grid.float().reshape(1, 1, H, W)).sum(dim=(2, 3))
    x_center = (probs * x_grid.float().reshape(1, 1, H, W)).sum(dim=(2, 3))
    return torch.stack([x_center, y_center], dim=2)


def load_sota_model(device):
    ckpt_path = 'sota_50k_sigma_checkpoints/STATERA-50K-Sigma_epoch_10.pth'
    print(f"[*] Loading SOTA Champion Weights: {ckpt_path}")
    model = StateraModel(
        decoder_type='deconv', temporal_mixer='conv1d',
        single_task=False, backbone_type='vjepa',
        scratch=False, finetune_blocks=0
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def process_heatmap_overlay(logits, frame_bgr):
    # Normalize logits with sigmoid (0 to 1 range)
    hm = torch.sigmoid(logits).cpu().numpy()
    hm_resized = cv2.resize(hm, (384, 384), interpolation=cv2.INTER_CUBIC)

    # Create Colormap (Boost intensity for visual pop)
    hm_norm = np.clip(hm_resized * 255.0 * 1.5, 0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)

    # Alpha Blend (50% Original, 50% Heatmap)
    return cv2.addWeighted(frame_bgr, 0.5, hm_color, 0.5, 0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize SOTA Model
    model = load_sota_model(device)

    # 2. Path Setup
    data_dir = '../sim2real/output'
    out_dir = "SOTA_50k_sigma_all_heatmaps"
    os.makedirs(out_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(data_dir, '*_statera.h5')))
    if not h5_files:
        print(f"[!] No .h5 files found in {data_dir}!")
        return

    print(f"[*] Found {len(h5_files)} video databases. Commencing Render...")

    # 3. Process every sequence in every file
    for f_path in h5_files:
        vid_base = os.path.basename(f_path).replace('_statera.h5', '')

        with h5py.File(f_path, 'r') as f:
            sequences = [k for k in f.keys() if k.startswith('seq_')]

            for seq_name in tqdm(sequences, desc=f"Rendering {vid_base}", leave=False):
                grp = f[seq_name]
                frames_bgr = grp["frames"][:]
                gt_u, gt_v = grp["com_u"][:], grp["com_v"][:]

                # Tensor Prep (BGR -> RGB)
                vids_tensor = torch.from_numpy(frames_bgr[:, :, :, ::-1].copy()).float() / 255.0
                vids_tensor = vids_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(device)

                # Inference
                with torch.no_grad():
                    pred_h, _ = model(vids_tensor)
                    pred_h = pred_h[0]  # [16, 64, 64]

                    # Coordinate Extraction
                    coords = get_subpixel_coords(pred_h.unsqueeze(0), 2.0)[0] * (384.0 / 64.0)
                    coords = coords.cpu().numpy()

                # Save Settings
                out_path = os.path.join(out_dir, f"{vid_base}_{seq_name}_SOTA_Heatmap.mp4")
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (384, 384))

                for i in range(16):
                    # Step A: Overlay Heatmap
                    img = process_heatmap_overlay(pred_h[i], frames_bgr[i])

                    # Step B: Draw Ground Truth (White Crosshair)
                    gt_pt = (int(gt_u[i]), int(gt_v[i]))
                    cv2.drawMarker(img, gt_pt, (255, 255, 255), cv2.MARKER_CROSS, 16, 2)

                    # Step C: Draw Model Prediction (Black Solid Circle)
                    pr_pt = (int(coords[i][0]), int(coords[i][1]))
                    cv2.circle(img, pr_pt, 5, (0, 0, 0), -1)

                    # Step D: Labels
                    cv2.putText(img, "STATERA 50K SOTA", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(img, f"Seq: {seq_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    writer.write(img)

                writer.release()

    print(f"\n[✓] DONE. All heatmaps rendered to '{out_dir}/'")


if __name__ == "__main__":
    main()