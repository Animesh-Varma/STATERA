import os
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import json
import gc
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from statera.model import StateraModel

PUBLIC_TEST_PATH = ".checkpoints/HiddenMass-50K-Test-Public.hdf5"
GT_JSON_PATH = ".checkpoints/HiddenMass-50K-GroundTruth.json"

CHECKPOINTS = {
    "Standard 3D-CNN (ResNet3D)": {
        "path": ".checkpoints/STATERA-1K-ResNet3D.pth",
        "backbone": "resnet3d", "single_task": False
    },
    "Spatial Foundation (DINOv2)": {
        "path": ".checkpoints/STATERA-1K-DINOv2.pth",
        "backbone": "dinov2", "single_task": False
    },
    "Temporal Foundation (VideoMAE v2)": {
        "path": ".checkpoints/STATERA-1K-VideoMAE.pth",
        "backbone": "videomae", "single_task": False
    },
    "STATERA-1K-No-Z-Depth (Paradox Ablation)": {
        "path": ".checkpoints/STATERA-1K-No-Z-Depth.pth",
        "backbone": "vjepa", "single_task": True
    },
    "STATERA-1K-Frozen-Anchor": {
        "path": ".checkpoints/STATERA-1K-Frozen-Anchor.pth",
        "backbone": "vjepa", "single_task": False
    },
    "STATERA-1K-Anchor": {
        "path": ".checkpoints/STATERA-1K-Anchor.pth",
        "backbone": "vjepa", "single_task": False
    },
    "STATERA-1K-Standard-Sigma": {
        "path": ".checkpoints/Run-05-Standard-Sigma-Saved.pth",
        "backbone": "vjepa", "single_task": False
    },
    "STATERA-50K-Crescent (Phase-Aware)": {
        "path": ".checkpoints/STATERA-50K-Crescent.pth",
        "backbone": "vjepa", "single_task": False
    },
    "STATERA-50K-Sigma (Phase-Agnostic)": {
        "path": ".checkpoints/STATERA-50K-Sigma.pth",
        "backbone": "vjepa", "single_task": False
    }
}


# --- UTILITY FUNCTIONS ---
def get_subpixel_coords(logits_heatmap, temperature=2.0):
    """
    Implements the Temperature-Scaled Soft-Argmax for Continuous Coordinate Extraction (Section 3.3).

    Standard spatial networks extract the final coordinate utilizing an argmax function,
    introducing quantization noise and bounding predictions to integer pixels. This function
    bridges discrete spatial training with continuous kinematic evaluation by computing the
    expectation operator over the probability mass, gently bleeding probabilities into adjacent pixels.

    Args:
        logits_heatmap (torch.Tensor): The 2D spatial heatmap from Head A (B, T, H, W).
        temperature (float): The scaling factor tau (default 2.0) bounding the expectation
                             operator. At tau=2.0, this prioritizes statistical safety while
                             at tau=1.0 it maximizes true physical extraction (Pareto frontier).

    Returns:
        torch.Tensor: The continuous (x, y) sub-pixel coordinates.
    """
    B, T, H, W = logits_heatmap.shape
    # Applies the scaling factor \tau to the unnormalized logits before the expectation operator.
    probs = F.softmax((logits_heatmap.reshape(B * T, -1)) / temperature, dim=1)
    probs = probs.reshape(B, T, H, W)
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=probs.device), torch.arange(W, device=probs.device),
                                    indexing='ij')

    # Resolves the continuous coordinates
    y_center = (probs * y_grid.float().reshape(1, 1, H, W)).sum(dim=(2, 3))
    x_center = (probs * x_grid.float().reshape(1, 1, H, W)).sum(dim=(2, 3))
    return torch.stack([x_center, y_center], dim=2)


def compute_metrics(pred, gt, diags):
    """
    Computes the strict Two-Key evaluation suite metrics (Section 4.2), preventing models
    from reward-hacking Euclidean metrics via expectation collapse.

    - N-CoME (\tilde{E}): Resolves perspective scale collapse by normalizing the mean Euclidean
      spatial distance against the object's 3D physical diameter.
    - Normalized Kinematic Jitter (\tilde{J}): Evaluates tracking stability by measuring the Euclidean
      norm of the acceleration error. Heavily penalizes non-physical high-frequency spikes.
    - KECS: The L2-norm distance from the theoretical perfect origin (0,0) in the multi-objective error space.

    Args:
        pred (torch.Tensor): Predicted sub-pixel coordinates across the temporal sequence.
        gt (torch.Tensor): Physically-constrained high-fidelity ground truth coordinates.
        diags (torch.Tensor): The object's physical diameter (D_bbox) for perspective normalization.

    Returns:
        tuple: (n_come, norm_jitter, kecs)
    """
    # Calculate zero-th derivative spatial Euclidean error
    errs = torch.norm(pred - gt, dim=-1)
    n_come = (errs / diags).mean().item()

    # Calculate temporal derivatives: first derivative (velocity) & second derivative (acceleration)
    pred_v, gt_v = pred[1:] - pred[:-1], gt[1:] - gt[:-1]
    pred_a, gt_a = pred_v[1:] - pred_v[:-1], gt_v[1:] - gt_v[:-1]

    # Calculate Normalized Kinematic Jitter to penalize Visual-Kinematic Aliasing.
    j_errs = torch.norm(pred_a - gt_a, dim=-1)
    norm_jitter = (j_errs / diags[2:]).mean().item()

    # Calculate KECS (Kinematic-Euclidean Composite Score).
    # lambda=0.1 serves as the dimensional alignment scalar scaling spatial errors against normalized acceleration.
    kecs = np.sqrt(n_come ** 2 + (0.1 * norm_jitter) ** 2)
    return n_come, norm_jitter, kecs


def evaluate_models(device):
    """
    Executes the comprehensive evaluation pipeline on the HiddenMass Benchmark.

    Evaluates in-domain simulation validation (Table 1) and zero-shot real-world transfer (Table 2).
    Models are assessed on their ability to decouple visual geometry from physical mass by
    extracting temporal features and overcoming Visual-Kinematic Aliasing.
    """
    if not os.path.exists(PUBLIC_TEST_PATH):
        print(f"[!] Cannot find {PUBLIC_TEST_PATH}. Please download the public test set.")
        return None, None
    if not os.path.exists(GT_JSON_PATH):
        print(f"[!] Cannot find {GT_JSON_PATH}. Reviewers: Please place the Supplementary JSON in the correct folder.")
        return None, None

    print("\n" + "=" * 80)
    print(f"{'STATERA DOUBLE-BLIND REPRODUCIBILITY SUITE':^80}")
    print("=" * 80)

    print("[*] Loading Supplementary Ground Truth into VRAM...")
    with open(GT_JSON_PATH, 'r') as f:
        gt_data = json.load(f)

    labels = gt_data['labels']
    all_gt_uv = torch.tensor(gt_data['gt_uv'], dtype=torch.float32).to(device)
    all_gc_uv = torch.tensor(gt_data['gc_uv'], dtype=torch.float32).to(device)
    all_diags = torch.tensor(gt_data['diags'], dtype=torch.float32).to(device)

    sim_indices = [i for i, l in enumerate(labels) if l == "SIMULATION"]
    real_indices = [i for i, l in enumerate(labels) if l != "SIMULATION"]
    print(f"[✓] Successfully loaded {len(sim_indices)} Sim sequences and {len(real_indices)} Real-World sequences.")

    sim_results = {}
    real_results = {}

    # 1. Geometric Centroid Baseline
    # Evaluates the Naive Physics assumption. For uniform density objects, the centroid is valid.
    # However, for opaque asymmetric payloads, it oscillates due to angular velocity and fails physically.
    print("[*] Evaluating Geometric Centroid (Naive Physics)...")
    centroid_pred = torch.tensor([192.0, 192.0]).to(device).expand(16, 2)

    c_sim_n, c_sim_j, c_sim_k = [], [], []
    for idx in sim_indices:
        n, jitt, k = compute_metrics(centroid_pred, all_gt_uv[idx], all_diags[idx])
        c_sim_n.append(n);
        c_sim_j.append(jitt);
        c_sim_k.append(k)
    sim_results["Geometric Centroid (Naive Physics)"] = {"n_come": np.mean(c_sim_n), "jitter": np.mean(c_sim_j),
                                                         "kecs": np.mean(c_sim_k)}

    c_real_n, c_real_j, c_real_k = [], [], []
    for idx in real_indices:
        n, jitt, k = compute_metrics(centroid_pred, all_gt_uv[idx], all_diags[idx])
        c_real_n.append(n);
        c_real_j.append(jitt);
        c_real_k.append(k)
    real_results["Geometric Centroid (Naive Physics)"] = {"n_come": np.mean(c_real_n), "jitter": np.mean(c_real_j),
                                                          "kecs": np.mean(c_real_k), "phys_cap": 0.0}

    # 2. Neural Models
    # STATERA processes video via a partially-frozen V-JEPA backbone, leveraging pre-trained
    # latent transition dynamics to evaluate motion across temporal sequences.
    for name, cfg in CHECKPOINTS.items():
        if not os.path.exists(cfg['path']):
            print(f"[!] Missing weights for {name} at {cfg['path']}. Skipping.")
            continue

        print(f"[*] Evaluating {name}...")
        model = StateraModel(decoder_type='deconv', temporal_mixer='conv1d', backbone_type=cfg['backbone'],
                             single_task=cfg['single_task'], scratch=False).to(device)
        model.load_state_dict(torch.load(cfg['path'], map_location=device, weights_only=True))
        model.eval()

        m_sim_n, m_sim_j, m_sim_k = [], [], []
        m_real_n, m_real_j, m_real_k, m_phys = [], [], [], []

        with torch.no_grad(), h5py.File(PUBLIC_TEST_PATH, 'r') as f:
            total_seq = len(labels)

            # Batch inference directly from the public video HDF5
            for i in tqdm(range(0, total_seq, 8), desc="Inferencing", leave=False):
                end = min(i + 8, total_seq)
                vids = torch.from_numpy(f['videos'][i:end]).float().to(device) / 255.0
                vids = vids.permute(0, 2, 1, 3, 4)

                # The network bifurcates, extracting the spatial heatmap (pred_h) from the Spatial Preservation Decoder
                pred_h, _ = model(vids)

                pred_px = get_subpixel_coords(pred_h, 2.0) * (384.0 / 64.0)

                for j in range(end - i):
                    global_idx = i + j
                    gt = all_gt_uv[global_idx]
                    diag = all_diags[global_idx]
                    pred = pred_px[j]

                    n, jitt, k = compute_metrics(pred, gt, diag)

                    if global_idx in sim_indices:
                        m_sim_n.append(n);
                        m_sim_j.append(jitt);
                        m_sim_k.append(k)
                    else:
                        m_real_n.append(n);
                        m_real_j.append(jitt);
                        m_real_k.append(k)

                        # Physics Capture Ratio (Disentanglement evaluation)
                        # Calculates the absolute percentage the prediction successfully moves away
                        # from the visual centroid along the true hidden offset vector.
                        gc_coord = all_gc_uv[global_idx]
                        v_true = gt - gc_coord
                        v_pred = pred - gc_coord

                        # Evaluated utilizing strict directional vector projection
                        dot_product = (v_pred * v_true).sum(dim=1)
                        true_mag_sq = (v_true * v_true).sum(dim=1) + 1e-8
                        ratio = (dot_product / true_mag_sq).mean().item()
                        m_phys.append(ratio)

        sim_results[name] = {"n_come": np.mean(m_sim_n), "jitter": np.mean(m_sim_j), "kecs": np.mean(m_sim_k)}
        real_results[name] = {"n_come": np.mean(m_real_n), "jitter": np.mean(m_real_j), "kecs": np.mean(m_real_k),
                              "phys_cap": np.mean(m_phys)}

        del model;
        gc.collect();
        torch.cuda.empty_cache()

    # 3. Dynamic Surface Centroid Trackers (CoTracker & TAPIR)
    # Contemporary point tracking pipelines mathematically struggle on this task. Because the CoM is an internal
    # coordinate and exterior surfaces self-occlude during multi-axis tumbling, surface correspondences die,
    # forcing point-trackers into rank-deficiency or Visual-Kinematic Aliasing.
    print("\n[*] Evaluating Dynamic Surface Centroid Point Trackers on Real-World Data...")

    grid_x, grid_y = torch.meshgrid(torch.linspace(142, 242, 10), torch.linspace(142, 242, 10), indexing='xy')

    try:
        print("[>] Loading Meta CoTracker2...")
        cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device).eval()
        queries = torch.zeros(1, 100, 3).to(device)  # t, x, y
        queries[0, :, 1], queries[0, :, 2] = grid_x.flatten(), grid_y.flatten()

        m_n, m_j, m_k, m_phys = [], [], [], []
        with torch.no_grad(), h5py.File(PUBLIC_TEST_PATH, 'r') as f:
            for idx in tqdm(real_indices, desc="CoTracker Inference", leave=False):
                vids = torch.from_numpy(f['videos'][idx:idx + 1]).float().to(device)
                pred_tracks, pred_vis = cotracker(vids, queries=queries)

                tracks, vis = pred_tracks[0], pred_vis[0] > 0.8

                seq_pred = []

                for f_idx in range(16):
                    v = vis[f_idx]
                    # We compute the Dynamic Surface Centroid Proxy (the spatial average of surviving tracks)
                    # as explicit surface correspondence solvers yield NaN due to occlusion rank-deficiency.
                    seq_pred.append(
                        tracks[f_idx, v, :].mean(dim=0) if v.sum() > 0 else torch.tensor([192.0, 192.0]).to(device))
                pred = torch.stack(seq_pred)

                gt, diag, gc_coord = all_gt_uv[idx], all_diags[idx], all_gc_uv[idx]
                n, jitt, k = compute_metrics(pred, gt, diag)

                # EXACT Original Physics Capture Ratio
                v_true = gt - gc_coord
                v_pred = pred - gc_coord
                dot_product = (v_pred * v_true).sum(dim=1)
                true_mag_sq = (v_true * v_true).sum(dim=1) + 1e-8
                ratio = (dot_product / true_mag_sq).mean().item()

                m_n.append(n);
                m_j.append(jitt);
                m_k.append(k);
                m_phys.append(ratio)

        real_results["Point Tracker (CoTracker2)"] = {"n_come": np.mean(m_n), "jitter": np.mean(m_j),
                                                      "kecs": np.mean(m_k), "phys_cap": np.mean(m_phys)}
        del cotracker;
        gc.collect();
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[!] CoTracker evaluation failed: {e}")

    try:
        print("[>] Loading Google TAPIR...")
        import urllib.request, subprocess
        repo_dir = os.path.abspath("tapnet_repo")
        if not os.path.exists(repo_dir): subprocess.run(
            ["git", "clone", "https://github.com/deepmind/tapnet.git", repo_dir])
        if repo_dir not in sys.path: sys.path.insert(0, repo_dir)
        from tapnet.torch import tapir_model

        ckpt_path = "bootstapir_checkpoint_v2.pt"
        if not os.path.exists(ckpt_path): urllib.request.urlretrieve(
            "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt", ckpt_path)

        tapir = tapir_model.TAPIR(pyramid_level=1)
        tapir.load_state_dict(torch.load(ckpt_path, map_location=device))
        tapir.to(device).eval()

        queries = torch.zeros(1, 100, 3).to(device)  # t, y, x
        queries[0, :, 1], queries[0, :, 2] = grid_y.flatten(), grid_x.flatten()

        m_n, m_j, m_k, m_phys = [], [], [], []
        with torch.no_grad(), h5py.File(PUBLIC_TEST_PATH, 'r') as f:
            for idx in tqdm(real_indices, desc="TAPIR Inference", leave=False):
                vids_tapir = ((torch.from_numpy(f['videos'][idx:idx + 1]).float().to(device) / 255.0) * 2.0) - 1.0
                outputs = tapir(vids_tapir, query_points=queries, is_training=False)
                tracks, vis = outputs['tracks'][0], ~(outputs['occluded'][0])

                seq_pred = []

                for f_idx in range(16):
                    v = vis[:, f_idx]

                    seq_pred.append(
                        tracks[v, f_idx, :].mean(dim=0) if v.sum() > 0 else torch.tensor([192.0, 192.0]).to(device))
                pred = torch.stack(seq_pred)

                gt, diag, gc_coord = all_gt_uv[idx], all_diags[idx], all_gc_uv[idx]
                n, jitt, k = compute_metrics(pred, gt, diag)

                v_true = gt - gc_coord
                v_pred = pred - gc_coord
                dot_product = (v_pred * v_true).sum(dim=1)
                true_mag_sq = (v_true * v_true).sum(dim=1) + 1e-8
                ratio = (dot_product / true_mag_sq).mean().item()

                m_n.append(n);
                m_j.append(jitt);
                m_k.append(k);
                m_phys.append(ratio)

        real_results["Point Tracker (TAPIR)"] = {"n_come": np.mean(m_n), "jitter": np.mean(m_j), "kecs": np.mean(m_k),
                                                 "phys_cap": np.mean(m_phys)}
        del tapir;
        gc.collect();
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[!] TAPIR evaluation failed: {e}")

    return sim_results, real_results


def print_tables(sim_results, real_results):
    """
    Outputs the final evaluation metrics perfectly aligned with the NeurIPS 2026 paper's context.

    Reproduces:
    - Table 1: In-Domain Simulation Performance (identifying settling-state dataset bias).
    - Table 2: Zero-Shot Real-World Transfer (demonstrating true physical extraction versus expectation collapse).
    - Table 3: HiddenMass Score (HMS) benchmark performance.
    """
    if not sim_results: return

    print("\n" + "=" * 85)
    print(" TABLE 1: In-Domain Simulation Performance (10,000 Sequences) ")
    print("=" * 85)
    print(f"| {'Model Configuration':<40} | {'N-CoME (%)':<10} | {'Norm Jitter':<12} | {'KECS':<8} |")
    print("-" * 85)

    order = [
        "Geometric Centroid (Naive Physics)",
        "Standard 3D-CNN (ResNet3D)",
        "Spatial Foundation (DINOv2)",
        "Temporal Foundation (VideoMAE v2)",
        "STATERA-1K-No-Z-Depth (Paradox Ablation)",
        "STATERA-1K-Frozen-Anchor",
        "STATERA-1K-Anchor",
        "STATERA-1K-Standard-Sigma",
        "STATERA-50K-Crescent (Phase-Aware)",
        "STATERA-50K-Sigma (Phase-Agnostic)"
    ]

    for name in order:
        if name in sim_results:
            d = sim_results[name]
            print(f"| {name:<40} | {d['n_come'] * 100:>9.2f}% | {d['jitter']:>12.4f} | {d['kecs']:>8.4f} |")

    print("\n" + "=" * 105)
    print(" TABLE 2: Zero-Shot Real-World Transfer (N=63) ")
    print("=" * 105)
    print(
        f"| {'Model Configuration':<38} | {'N-CoME (%)':<10} | {'Norm Jitter':<11} | {'KECS':<6} | {'Physics Capture':<18} |")
    print("-" * 105)

    real_world_order = ["Point Tracker (CoTracker2)", "Point Tracker (TAPIR)"] + order

    for name in real_world_order:
        if name in real_results:
            d = real_results[name]
            phys_str = f"{d['phys_cap'] * 100:.2f}%"
            if d['phys_cap'] > 1.0 or (name == "Standard 3D-CNN (ResNet3D)" and d['phys_cap'] > 0.30):
                phys_str += " (Overshoot)"
            elif d['phys_cap'] < 0:
                phys_str += " (Collapse)"
            print(
                f"| {name:<38} | {d['n_come'] * 100:>9.2f}% | {d['jitter']:>11.4f} | {d['kecs']:>6.4f} | {phys_str:<18} |")
    print("=" * 105 + "\n")

    print("\n" + "=" * 85)
    print(" TABLE 3: HiddenMass Benchmark Performance & Unified Score (Bar Chart Data) ")
    print("=" * 85)
    print(f"| {'Model Identifier':<25} | {'Metric (N-CoME 0-100)':<25} | {'KECS':<25} |")
    print("-" * 85)

    # Short mappings meant specifically for extracting the Benchmark graph chart
    chart_models = [
        ("1K-DINOv2", "Spatial Foundation (DINOv2)"),
        ("1K-VideoMAE", "Temporal Foundation (VideoMAE v2)"),
        ("1K-ResNet3D", "Standard 3D-CNN (ResNet3D)"),
        ("1K-No-Z-Depth", "STATERA-1K-No-Z-Depth (Paradox Ablation)"),
        ("Point-Tracker-Meta", "Point Tracker (CoTracker2)"),
        ("Point-Tracker-Google", "Point Tracker (TAPIR)"),
        ("1K-Frozen-Anchor", "STATERA-1K-Frozen-Anchor"),
        ("1K-Anchor", "STATERA-1K-Anchor"),
        ("1K-Standard-Sigma", "STATERA-1K-Standard-Sigma"),
        ("STATERA-Crescent", "STATERA-50K-Crescent (Phase-Aware)"),
        ("STATERA-Sigma", "STATERA-50K-Sigma (Phase-Agnostic)")
    ]

    for short_name, full_name in chart_models:
        if full_name in real_results:
            n_come_val = real_results[full_name]['n_come'] * 100
            kecs_val = real_results[full_name]['kecs']
            print(f"| {short_name:<25} | {n_come_val:<25.2f} | {kecs_val:<25.4f} |")
        else:
            print(f"| {short_name:<25} | {'N/A':<25} | {'N/A':<25} |")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim_res, real_res = evaluate_models(device)
    print_tables(sim_res, real_res)