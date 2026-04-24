import os
import torch
import numpy as np
import h5py
import glob
import cv2
from tqdm import tqdm


def compute_kecs_metrics(pred, gt, diags):
    # pred, gt: [T, 2], diags: [T]
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
    print("=" * 80)
    print(f"{'EVALUATING GEOMETRIC CENTROID BASELINE (STATIC 192, 192)':^80}")
    print("=" * 80)
    centroid = torch.tensor([192.0, 192.0])

    # --- SIMULATION (10K Validation Set) ---
    sim_path = '../sim/HiddenMass-50K.hdf5'
    with h5py.File(sim_path, 'r') as f:
        # Load last 10,000 samples
        gt_uv_sim = torch.from_numpy(f['uv_coords'][-10000:]).reshape(-1, 16, 2)
        diags_sim = torch.ones(10000, 16) * 50.0  # Baseline sim diagonal (5cm box @ 384px)

    res_sim = [compute_kecs_metrics(centroid.expand(16, 2), gt_uv_sim[i], diags_sim[i]) for i in range(10000)]
    avg_sim = np.mean(res_sim, axis=0)
    print(
        f"Simulation (10K Set) | N-CoME: {avg_sim[0] * 100:.2f}% | NormJitter: {avg_sim[1]:.4f} | KECS: {avg_sim[2]:.4f}")

    # --- REAL WORLD (76 Sequences) ---
    data_dir = '../sim2real/output'
    mtx = np.load('../sim2real/camera_matrix.npy')
    dist = np.load('../sim2real/dist_coeffs.npy')
    frame_w = snap_res(mtx[0, 2] * 2)
    scale = 384.0 / frame_w

    h5_files = glob.glob(os.path.join(data_dir, '*_statera.h5'))
    res_real = []

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
                gt = torch.stack([torch.from_numpy(grp["com_u"][:]), torch.from_numpy(grp["com_v"][:])], dim=1)
                rvecs, tvecs = grp["rvecs"][:], grp["tvecs"][:]

                diags = []
                for i in range(16):
                    p2d, _ = cv2.projectPoints(box_3d, rvecs[i], tvecs[i], mtx, dist)
                    p2d = p2d.reshape(-1, 2)
                    w, h = np.max(p2d[:, 0]) - np.min(p2d[:, 0]), np.max(p2d[:, 1]) - np.min(p2d[:, 1])
                    diags.append(np.sqrt(w ** 2 + h ** 2) * scale)

                res_real.append(compute_kecs_metrics(centroid.expand(16, 2), gt, torch.tensor(diags)))

    avg_real = np.mean(res_real, axis=0)
    print(
        f"Real World (76 Seq)  | N-CoME: {avg_real[0] * 100:.2f}% | NormJitter: {avg_real[1]:.4f} | KECS: {avg_real[2]:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()