import os
import glob
import sys
import torch
import h5py
import numpy as np
import cv2
import wandb
import urllib.request
import urllib.error
import subprocess
from tqdm import tqdm


def snap_to_standard_resolution(val):
    for res in [480, 720, 1080, 1440, 1920, 2160, 3840]:
        if abs(val - res) < 50: return res
    return int(val)


def extract_bbox_diags(rvecs, tvecs, mtx, dist, box_size_m, scale):
    bx, by, bz = box_size_m
    hx, hy, hz = bx / 2, by / 2, bz / 2
    box_3d = np.array(
        [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz], [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz],
         [-hx, hy, hz]], dtype=np.float32)

    diags = []
    for i in range(16):
        rv, tv = rvecs[i], tvecs[i]
        pts2d, _ = cv2.projectPoints(box_3d, rv, tv, mtx, dist)
        pts2d = pts2d.reshape(-1, 2)

        w = np.max(pts2d[:, 0]) - np.min(pts2d[:, 0])
        h = np.max(pts2d[:, 1]) - np.min(pts2d[:, 1])
        diag_orig = np.sqrt(w ** 2 + h ** 2)
        diags.append(diag_orig * scale)
    return np.array(diags, dtype=np.float32)


def load_real_world_data(data_dir, cam_mtx_path, dist_coeffs_path):
    print("[*] Linking Real-World Dataset & Computing BBox Diagonals...")
    mtx = np.load(cam_mtx_path)
    dist = np.load(dist_coeffs_path)

    cx_guess, cy_guess = mtx[0, 2] * 2, mtx[1, 2] * 2
    frame_w, frame_h = snap_to_standard_resolution(cx_guess), snap_to_standard_resolution(cy_guess)
    scale = 384.0 / min(frame_w, frame_h)

    samples = []
    h5_files = sorted(glob.glob(os.path.join(data_dir, '*_statera.h5')))

    for f_path in h5_files:
        with h5py.File(f_path, 'r') as f:
            box_size_m = f.attrs["box_size_m"]
            for group_name in f.keys():
                if not group_name.startswith('seq_'): continue
                grp = f[group_name]

                frames_bgr = grp["frames"][:]
                frames_rgb = frames_bgr[:, :, :, ::-1].copy()

                gt_coords = np.stack([grp["com_u"][:], grp["com_v"][:]], axis=1)
                diags = extract_bbox_diags(grp["rvecs"][:], grp["tvecs"][:], mtx, dist, box_size_m, scale)

                samples.append({
                    'frames': frames_rgb,
                    'gt_coords': gt_coords,
                    'diags': diags
                })

    print(f"[✓] Loaded {len(samples)} Physical Sequences.")
    return samples


def calculate_sequence_metrics(pred_coords, gt_coords, bbox_diags):
    # px_errors: [16]
    px_errors = torch.norm(pred_coords - gt_coords, dim=1)
    n_come = (px_errors / bbox_diags).mean().item()

    pred_vel = pred_coords[1:] - pred_coords[:-1]
    gt_vel = gt_coords[1:] - gt_coords[:-1]
    pred_accel = pred_vel[1:] - pred_vel[:-1]
    gt_accel = gt_vel[1:] - gt_vel[:-1]

    accel_errors = torch.norm(pred_accel - gt_accel, dim=1)
    norm_jitter = (accel_errors / bbox_diags[2:]).mean().item()
    raw_jitter = accel_errors.mean().item()

    return n_come, norm_jitter, raw_jitter


def run_cotracker(samples, device):
    print("\n[*] Initializing Meta's CoTracker2...")
    try:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device)
        model.eval()
    except Exception as e:
        print(f"[!] CoTracker failed to load.\n{e}")
        return None

    queries = torch.tensor([[[0.0, 192.0, 192.0]]]).to(device)
    tot_ncome, tot_njitter, tot_rawjitter = 0.0, 0.0, 0.0

    with torch.no_grad():
        for seq in tqdm(samples, desc="Evaluating CoTracker"):
            video = torch.from_numpy(seq['frames']).float().to(device)
            video = video.permute(0, 3, 1, 2).unsqueeze(0)  # [1, 16, 3, 384, 384]

            pred_tracks, _ = model(video, queries=queries)
            pred_coords = pred_tracks[0, :, 0, :]  # [16, 2]

            gt_coords = torch.from_numpy(seq['gt_coords']).float().to(device)
            diags = torch.from_numpy(seq['diags']).float().to(device)

            n_come, norm_jitt, raw_jitt = calculate_sequence_metrics(pred_coords, gt_coords, diags)
            tot_ncome += n_come
            tot_njitter += norm_jitt
            tot_rawjitter += raw_jitt

    n = len(samples)
    avg_ncome = tot_ncome / n
    avg_njitter = tot_njitter / n
    avg_rawjitter = tot_rawjitter / n
    h_ke = (2 * avg_ncome * avg_njitter) / (avg_ncome + avg_njitter + 1e-8)

    wandb.init(project="STATERA", name="Baseline-CoTracker-RealWorld")
    wandb.log({"real_world/N-CoME": avg_ncome, "real_world/Raw_Jitter": avg_rawjitter, "real_world/H_KE": h_ke})
    wandb.finish()

    return {'n_come': avg_ncome, 'raw_jitter': avg_rawjitter, 'h_ke': h_ke}


def run_tapir(samples, device):
    print("\n[*] Initializing Google's TAPIR...")

    repo_dir = os.path.abspath("tapnet_repo")
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", "https://github.com/deepmind/tapnet.git", repo_dir])

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    try:
        from tapnet.torch import tapir_model
    except Exception as e:
        print(f"[!] TAPIR module import failed: {e}")
        return None

    ckpt_path = "bootstapir_checkpoint_v2.pt"
    if not os.path.exists(ckpt_path):
        url = "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt"
        urllib.request.urlretrieve(url, ckpt_path)

    try:
        model = tapir_model.TAPIR(pyramid_level=1)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[!] Failed to load TAPIR weights: {e}")
        return None

    queries = torch.tensor([[[0.0, 192.0, 192.0]]]).to(device)
    tot_ncome, tot_njitter, tot_rawjitter = 0.0, 0.0, 0.0

    with torch.no_grad():
        for seq in tqdm(samples, desc="Evaluating TAPIR"):
            video = torch.from_numpy(seq['frames']).float().to(device) / 255.0
            video = video.unsqueeze(0)  # [1, 16, 384, 384, 3]
            video_tapir = (video * 2.0) - 1.0

            # TAPIR output is a dictionary in the DeepMind PyTorch port
            outputs = model(video_tapir, query_points=queries, is_training=False)
            tracks = outputs['tracks']  # [B, N, T, 2]

            # DeepMind TAPIR usually outputs in pixel coordinates relative to input res
            pred_coords = tracks[0, 0, :, :]  # [16, 2]

            gt_coords = torch.from_numpy(seq['gt_coords']).float().to(device)
            diags = torch.from_numpy(seq['diags']).float().to(device)

            n_come, norm_jitt, raw_jitt = calculate_sequence_metrics(pred_coords, gt_coords, diags)
            tot_ncome += n_come
            tot_njitter += norm_jitt
            tot_rawjitter += raw_jitt

    n = len(samples)
    avg_ncome = tot_ncome / n
    avg_njitter = tot_njitter / n
    avg_rawjitter = tot_rawjitter / n
    h_ke = (2 * avg_ncome * avg_njitter) / (avg_ncome + avg_njitter + 1e-8)

    wandb.init(project="STATERA", name="Baseline-TAPIR-RealWorld")
    wandb.log({"real_world/N-CoME": avg_ncome, "real_world/Raw_Jitter": avg_rawjitter, "real_world/H_KE": h_ke})
    wandb.finish()

    return {'n_come': avg_ncome, 'raw_jitter': avg_rawjitter, 'h_ke': h_ke}


def print_summary_table(results):
    print("\n" + "=" * 90)
    print(f"{'POINT TRACKER BASELINES (Sim2Real Physical Verification)':^90}")
    print("=" * 90)
    print(f"| {'Model Name':^26} | {'N-CoME (%)':^16} | {'Raw Jitter':^16} | {'H_KE Metric':^16} |")
    print("-" * 90)
    for name, metrics in results.items():
        if metrics is None:
            print(f"| {name:<26} | {'FAILED':^16} | {'FAILED':^16} | {'FAILED':^16} |")
        else:
            print(
                f"| {name:<26} | {metrics['n_come'] * 100:^15.2f}% | {metrics['raw_jitter']:^16.3f} | {metrics['h_ke']:^16.5f} |")
    print("=" * 90 + "\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = '../sim2real/output'
    mtx_path = '../sim2real/camera_matrix.npy'
    dist_path = '../sim2real/dist_coeffs.npy'

    samples = load_real_world_data(data_dir, mtx_path, dist_path)
    if not samples: return

    results = {}
    results["Meta CoTracker2"] = run_cotracker(samples, device)
    results["Google TAPIR"] = run_tapir(samples, device)

    print_summary_table(results)


if __name__ == "__main__":
    main()