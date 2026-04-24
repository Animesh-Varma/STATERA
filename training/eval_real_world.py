import os
import torch
import json
import h5py
import glob
import cv2
import numpy as np
from tqdm import tqdm
from model import StateraModel
from train import get_subpixel_coords

def compute_kecs_metrics(pred, gt, diags):
    errs = torch.norm(pred - gt, dim=-1)
    n_come = (errs / diags).mean()
    v_p, v_g = pred[1:] - pred[:-1], gt[1:] - gt[:-1]
    a_p, a_g = v_p[1:] - v_p[:-1], v_g[1:] - v_g[:-1]
    j_errs = torch.norm(a_p - a_g, dim=-1)
    norm_jitter = (j_errs / diags[2:]).mean()
    kecs = torch.sqrt(n_come ** 2 + (0.1 * norm_jitter) ** 2)
    return n_come.item(), norm_jitter.item(), kecs.item()


def snap_res(val):
    for res in [480, 720, 1080, 1920, 3840]:
        if abs(val - res) < 50: return res
    return int(val)


def main():
    device = torch.device("cuda")
    data_dir = '../sim2real/output'
    mtx = np.load('../sim2real/camera_matrix.npy')
    dist = np.load('../sim2real/dist_coeffs.npy')
    frame_w = snap_res(mtx[0, 2] * 2)
    scale = 384.0 / frame_w

    checkpoints = {
        "Run-00-Anchor": "checkpoints/Run-00-Anchor-Saved_epoch_30.pth",
        "Run-04-StaticDot": "checkpoints/Run-04-Static-Dot-Saved_epoch_30.pth",
        "Run-05-StdSigma": "checkpoints/Run-05-Standard-Sigma-Saved_epoch_30.pth",
        "Run-01-DINOv2": "checkpoints/Run-01-DINOv2_epoch_23.pth",
        "Run-03-ResNet3D": "checkpoints/Run-03-ResNet3D-Fixed_epoch_30.pth",
        "SOTA-Crescent": "sota_50k_checkpoints/STATERA-50K-SOTA_epoch_7.pth",
        "SOTA-Sigma": "sota_50k_sigma_checkpoints/STATERA-50K-Sigma_epoch_10.pth"
    }

    print(f"\n{'TABLE 2: ZERO-SHOT REAL-WORLD TRANSFER':^75}")
    print("-" * 75)
    print(f"| {'Model Name':<20} | {'N-CoME (%)':^14} | {'NormJitter':^14} | {'KECS':^12} |")
    print("-" * 75)

    h5_files = glob.glob(os.path.join(data_dir, '*_statera.h5'))

    for name, path in checkpoints.items():
        backbone = 'dinov2' if 'DINO' in name else ('resnet3d' if 'ResNet' in name else 'vjepa')
        model = StateraModel(decoder_type='deconv', temporal_mixer='conv1d', backbone_type=backbone).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        results = []
        with torch.no_grad():
            for f_path in h5_files:
                with h5py.File(f_path, 'r') as f:
                    bx, by, bz = f.attrs["box_size_m"]
                    hx, hy, hz = bx / 2, by / 2, bz / 2
                    box_3d = np.array(
                        [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz], [-hx, -hy, hz], [hx, -hy, hz],
                         [hx, hy, hz], [-hx, hy, hz]], dtype=np.float32)

                    for grp_name in f.keys():
                        if not grp_name.startswith('seq_'): continue
                        grp = f[grp_name]

                        # Prepare Video Tensor
                        vids = torch.from_numpy(grp["frames"][:][:, :, :, ::-1].copy()).float() / 255.0
                        vids = vids.permute(3, 0, 1, 2).unsqueeze(0).to(device)

                        # Inference
                        pred_h, _ = model(vids)
                        pred_px = get_subpixel_coords(pred_h, 2.0)[0] * 6.0
                        gt_px = torch.stack([torch.from_numpy(grp["com_u"][:]), torch.from_numpy(grp["com_v"][:])],
                                            dim=1).to(device)

                        # BBox Diagonals
                        rvecs, tvecs = grp["rvecs"][:], grp["tvecs"][:]
                        diags = []
                        for i in range(16):
                            p2d, _ = cv2.projectPoints(box_3d, rvecs[i], tvecs[i], mtx, dist)
                            p2d = p2d.reshape(-1, 2)
                            w, h = np.max(p2d[:, 0]) - np.min(p2d[:, 0]), np.max(p2d[:, 1]) - np.min(p2d[:, 1])
                            diags.append(np.sqrt(w ** 2 + h ** 2) * scale)

                        results.append(compute_kecs_metrics(pred_px, gt_px, torch.tensor(diags).to(device)))

        avg = np.mean(results, axis=0)
        print(f"| {name:<20} | {avg[0] * 100:>12.2f}% | {avg[1]:>14.4f} | {avg[2]:>12.4f} |")

    # [MANUAL ADDITION]: Add TAPIR and CoTracker results from point tracker eval here
    print("-" * 75)


if __name__ == "__main__":
    main()