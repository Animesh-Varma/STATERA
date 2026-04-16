import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class StateraDataset(Dataset):
    def __init__(self, h5_path, target_type='crescent', heatmap_res=64, start_sigma=12.5, jitter_box=False):
        self.h5_path = h5_path
        self.heatmap_res = heatmap_res
        self.sigma = start_sigma
        self.phase_alpha = 0.0
        self.target_type = target_type
        self.jitter_box = jitter_box

        # Initialize as None. The PyTorch DataLoader workers will open this dynamically.
        self.h5_file = None

        print(f"[*] Probing dataset length from disk (Lazy Loading)...")
        # Open temporarily just to get the length, then immediately close it.
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['videos'])
        print(f"[✓] Dataset linked. Total sequences streaming from SSD: {self.length}")

        y, x = torch.meshgrid(
            torch.arange(self.heatmap_res, dtype=torch.float32),
            torch.arange(self.heatmap_res, dtype=torch.float32),
            indexing='ij'
        )
        self.grid_x = x
        self.grid_y = y

    def update_sigma(self, new_sigma):
        self.sigma = new_sigma

    def update_phase_alpha(self, new_alpha):
        self.phase_alpha = new_alpha

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # LAZY LOAD: Open the HDF5 file per-worker to prevent Multiprocessing crashes
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        # Stream ONLY the specific index needed for this batch
        video_np = self.h5_file['videos'][idx]
        uv_np = self.h5_file['uv_coords'][idx].reshape(16, 2)
        box_np = self.h5_file['box_center_uv'][idx].reshape(16, 2)
        z_depth_np = self.h5_file['z_depths'][idx]

        video = torch.from_numpy(video_np).float() / 255.0
        video = video.permute(1, 0, 2, 3)

        if self.jitter_box:
            noise = np.random.uniform(-5.0, 5.0, size=box_np.shape)
            box_np = box_np + noise

        z_depth = torch.tensor(z_depth_np.flatten(), dtype=torch.float32).view(16, 1)

        scale = self.heatmap_res / 384.0

        true_uv = torch.stack([
            torch.tensor(uv_np[:, 0] * scale, dtype=torch.float32),
            torch.tensor(uv_np[:, 1] * scale, dtype=torch.float32)
        ], dim=1)  # Shape: [16, 2]

        u_t = true_uv[:, 0].view(16, 1, 1)
        v_t = true_uv[:, 1].view(16, 1, 1)

        u_c = torch.tensor(box_np[:, 0] * scale, dtype=torch.float32).view(16, 1, 1)
        v_c = torch.tensor(box_np[:, 1] * scale, dtype=torch.float32).view(16, 1, 1)

        dx_c = self.grid_x.unsqueeze(0) - u_c
        dy_c = self.grid_y.unsqueeze(0) - v_c
        grid_dist_c = torch.sqrt(dx_c ** 2 + dy_c ** 2 + 1e-8)

        dx_t = self.grid_x.unsqueeze(0) - u_t
        dy_t = self.grid_y.unsqueeze(0) - v_t
        grid_dist_t = torch.sqrt(dx_t ** 2 + dy_t ** 2 + 1e-8)

        R_true = torch.sqrt((u_t - u_c) ** 2 + (v_t - v_c) ** 2 + 1e-8)

        dot = torch.exp(-(grid_dist_t ** 2) / (2 * self.sigma ** 2))
        ring = torch.exp(-((grid_dist_c - R_true) ** 2) / (2 * self.sigma ** 2))

        if self.target_type == 'dot':
            heatmap = dot
        elif self.target_type == 'blend':
            heatmap = (1.0 - self.phase_alpha) * ring + (self.phase_alpha) * dot
        elif self.target_type == 'crescent':
            vec_t_x = u_t - u_c
            vec_t_y = v_t - v_c
            norm_t = torch.sqrt(vec_t_x ** 2 + vec_t_y ** 2)
            is_overlapping = (norm_t < 1e-3)
            safe_norm_t = torch.where(is_overlapping, torch.ones_like(norm_t), norm_t)

            vec_t_x_norm = vec_t_x / safe_norm_t
            vec_t_y_norm = vec_t_y / safe_norm_t
            vec_g_x_norm = dx_c / grid_dist_c
            vec_g_y_norm = dy_c / grid_dist_c

            cos_sim = (vec_g_x_norm * vec_t_x_norm) + (vec_g_y_norm * vec_t_y_norm)
            mask = torch.exp((self.phase_alpha * 50.0) * (cos_sim - 1.0))
            mask = torch.where(is_overlapping, torch.ones_like(mask), mask)
            heatmap = ring * mask
        else:
            raise ValueError(f"Unknown target type: {self.target_type}")

        return video, heatmap, z_depth, true_uv