import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class StateraDataset(Dataset):
    """
    StateraDataset: HiddenMass-50K Streaming & Target Generation Pipeline.

    This dataset class is engineered to stream the 50,000 MuJoCo sequences and generate
    the necessary Curriculum Label Smoothing targets (Crescent/Sigma) for the STATERA
    architecture. It extracts the raw 16-frame temporal trajectories, computes the
    sub-pixel ground truth continuous coordinates, and isolates absolute Z-Depth for
    the Head B spatial bounds regularizer.

    Args:
        h5_path (str): Path to the procedural MuJoCo HDF5 dataset.
        target_type (str): Dictates the phase-aware ('crescent') or phase-agnostic ('dot', 'blend') target.
        heatmap_res (int): Spatial dimensions of the Spatial Preservation Decoder (default: 64x64).
        start_sigma (float): Initial variance for Curriculum Label Smoothing (default: 12.5).
        jitter_box (bool): Applies Stochastic Camera Perturbation via bounding box translation.
    """

    def __init__(self, h5_path, target_type='crescent', heatmap_res=64, start_sigma=12.5, jitter_box=False):
        self.h5_path = h5_path
        self.heatmap_res = heatmap_res

        # Initializes the Curriculum Label Smoothing Variance (sigma). As described in Section 3.4,
        # sigma shrinks from a broad Gaussian dot (12.5) to a high-frequency impulse (3.0).
        self.sigma = start_sigma

        # Curriculum concentration parameter alpha (used for Von Mises kappa calculation)
        self.phase_alpha = 0.0
        self.target_type = target_type
        self.jitter_box = jitter_box

        # Initialize as None. The PyTorch DataLoader workers will open this dynamically.
        # This prevents the multi-processing memory leak standard in large HDF5 video datasets.
        self.h5_file = None

        print(f"[*] Probing dataset length from disk (Lazy Loading)...")
        # Open temporarily just to get the length, then immediately close it.
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['videos'])
        print(f"[✓] Dataset linked. Total sequences streaming from SSD: {self.length}")

        # Native 64x64 index grid for calculating sub-pixel accurate spatial targets
        y, x = torch.meshgrid(
            torch.arange(self.heatmap_res, dtype=torch.float32),
            torch.arange(self.heatmap_res, dtype=torch.float32),
            indexing='ij'
        )
        self.grid_x = x
        self.grid_y = y

    def update_sigma(self, new_sigma):
        """
        Dynamically updates the spatial regularizer decay per epoch (Variance Decay).
        Shrinks the probability mass bounded by the object's physical rotation arc.
        """
        self.sigma = new_sigma

    def update_phase_alpha(self, new_alpha):
        """
        Dynamically updates the temporal exponential weighting mask alpha.
        Enforces heavily penalized late-stage tracking where multi-axis tumbling
        unveils the true mass offset (Section 3.4).
        """
        self.phase_alpha = new_alpha

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # LAZY LOAD: Open the HDF5 file per-worker to prevent Multiprocessing crashes
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        # Stream ONLY the specific index needed for this batch
        # T=16 sequence frames at 24 FPS (representing the 0.66s temporal window capturing parabolic apex dynamics)
        video_np = self.h5_file['videos'][idx]
        uv_np = self.h5_file['uv_coords'][idx].reshape(16, 2)
        box_np = self.h5_file['box_center_uv'][idx].reshape(16, 2)
        z_depth_np = self.h5_file['z_depths'][idx]

        # EXTRACT BOUNDING BOX DIAGONAL (For N-CoME normalization)
        # N-CoME heavily depends on a strictly accurate constant 3D physical diameter
        # projected into the 2D plane (D_bbox) to guarantee scale-invariant Euclidean evaluation.
        try:
            # If dataset directly stores box width and height
            box_dims_np = self.h5_file['box_dims'][idx].reshape(16, 2)
        except KeyError:
            try:
                # If dataset stores bounding boxes as[x1, y1, x2, y2]
                bboxes = self.h5_file['bboxes'][idx].reshape(16, 4)
                box_dims_np = np.stack([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]], axis=1)
            except KeyError:
                # Safe geometric fallback if neither exists (assumes standard 50px box)
                box_dims_np = np.ones((16, 2), dtype=np.float32) * 50.0

        # Calculate standard hypotenuse: sqrt(W^2 + H^2) -> Represents D_bbox denominator for metrics
        bbox_diag = torch.sqrt(
            torch.tensor(box_dims_np[:, 0] ** 2 + box_dims_np[:, 1] ** 2, dtype=torch.float32) + 1e-8)

        # Video tensor conversion and permutation to shape (T, C, H, W)
        # This matches the temporal tubelet expectation for the V-JEPA 2.1 ViT-L backbone
        video = torch.from_numpy(video_np).float() / 255.0
        video = video.permute(1, 0, 2, 3)

        if self.jitter_box:
            # Applies Stochastic Camera Perturbation to prevent perspective scale collapse
            noise = np.random.uniform(-5.0, 5.0, size=box_np.shape)
            box_np = box_np + noise

        # Ground-truth Absolute Z-Depth for Head B (Multi-Task Regularization)
        # Binds 2D perspective space to 3D continuous Newtonian coordinate space
        # to prevent the Vector Projection Overshooting Artifact (Appendix E).
        z_depth = torch.tensor(z_depth_np.flatten(), dtype=torch.float32).view(16, 1)

        # Map from V-JEPA's 384x384 input resolution to the Spatial Preservation Decoder's 64x64 output
        scale = self.heatmap_res / 384.0

        true_uv = torch.stack([
            torch.tensor(uv_np[:, 0] * scale, dtype=torch.float32),
            torch.tensor(uv_np[:, 1] * scale, dtype=torch.float32)
        ], dim=1)  # Shape: [16, 2]

        u_t = true_uv[:, 0].view(16, 1, 1)
        v_t = true_uv[:, 1].view(16, 1, 1)

        u_c = torch.tensor(box_np[:, 0] * scale, dtype=torch.float32).view(16, 1, 1)
        v_c = torch.tensor(box_np[:, 1] * scale, dtype=torch.float32).view(16, 1, 1)

        # Calculate Euclidean distances on the 64x64 grid to the visual geometric centroid
        dx_c = self.grid_x.unsqueeze(0) - u_c
        dy_c = self.grid_y.unsqueeze(0) - v_c
        grid_dist_c = torch.sqrt(dx_c ** 2 + dy_c ** 2 + 1e-8)

        # Calculate Euclidean distances to the true, unobservable CoM offset
        dx_t = self.grid_x.unsqueeze(0) - u_t
        dy_t = self.grid_y.unsqueeze(0) - v_t
        grid_dist_t = torch.sqrt(dx_t ** 2 + dy_t ** 2 + 1e-8)

        # R: Distance magnitude between geometric centroid and true hidden mass offset
        R_true = torch.sqrt((u_t - u_c) ** 2 + (v_t - v_c) ** 2 + 1e-8)

        # 'dot' generates the phase-agnostic target. During Curriculum Smooth Labeling,
        # this decays into an isotropic Gaussian impulse (Sigma baseline).
        dot = torch.exp(-(grid_dist_t ** 2) / (2 * self.sigma ** 2))

        # 'ring' establishes the spatial radial prior defined in Appendix A.
        ring = torch.exp(-((grid_dist_c - R_true) ** 2) / (2 * self.sigma ** 2))

        if self.target_type == 'dot':
            heatmap = dot
        elif self.target_type == 'blend':
            heatmap = (1.0 - self.phase_alpha) * ring + (self.phase_alpha) * dot
        elif self.target_type == 'crescent':
            # -------------------------------------------------------------------------
            # CRESCENT TARGET GENERATION (Section 3.4 & Appendix A)
            # -------------------------------------------------------------------------
            # Formed by intersecting the spatial radial prior (ring) with a
            # Von Mises angular phase mask. This allows the network to continuously
            # optimize for rotational torque direction while preventing static visual
            # overfitting to the dataset's settling state.

            vec_t_x = u_t - u_c
            vec_t_y = v_t - v_c
            norm_t = torch.sqrt(vec_t_x ** 2 + vec_t_y ** 2)

            # Prevents division-by-zero anomalies when evaluating geometrically uniform
            # bodies where the visual centroid perfectly equals the Center of Mass.
            is_overlapping = (norm_t < 1e-3)
            safe_norm_t = torch.where(is_overlapping, torch.ones_like(norm_t), norm_t)

            vec_t_x_norm = vec_t_x / safe_norm_t
            vec_t_y_norm = vec_t_y / safe_norm_t
            vec_g_x_norm = dx_c / grid_dist_c
            vec_g_y_norm = dy_c / grid_dist_c

            # Angular difference computed via dot product: cos(\theta_p)
            cos_sim = (vec_g_x_norm * vec_t_x_norm) + (vec_g_y_norm * vec_t_y_norm)

            # Applies the Von Mises angular mask: exp(kappa * (cos(\theta_p) - 1))
            # where kappa = 50.0 * alpha (self.phase_alpha) acts as the curriculum parameter.
            mask = torch.exp((self.phase_alpha * 50.0) * (cos_sim - 1.0))
            mask = torch.where(is_overlapping, torch.ones_like(mask), mask)

            heatmap = ring * mask
        else:
            raise ValueError(f"Unknown target type: {self.target_type}")

        return video, heatmap, z_depth, true_uv, bbox_diag