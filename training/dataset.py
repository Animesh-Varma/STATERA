import torch
from torch.utils.data import Dataset
import h5py


class StateraDataset(Dataset):
    def __init__(self, h5_path, heatmap_res=64, sigma=12.5):
        self.h5_path = h5_path
        self.heatmap_res = heatmap_res
        self.sigma = sigma
        self.phase_alpha = 0.0  # Controls the Von Mises Crescent concentration

        print(f"Loading dataset into RAM to bypass HDF5 bottleneck...")
        with h5py.File(self.h5_path, 'r') as f:
            self.videos = f['videos'][:]
            self.uv_coords = f['uv_coords'][:]
            self.z_depths = f['z_depths'][:]
            # Change: Swap magnitude cache for 2D bounding box center cache
            self.box_center_uv = f['box_center_uv'][:]
        print("✓ Dataset successfully cached in RAM!")

        self.length = len(self.videos)

        y, x = torch.meshgrid(
            torch.arange(self.heatmap_res, dtype=torch.float32),
            torch.arange(self.heatmap_res, dtype=torch.float32),
            indexing='ij'
        )
        self.grid_x = x
        self.grid_y = y

    def update_sigma(self, new_sigma):
        """Setter to dynamically shrink the heatmap Gaussian width"""
        self.sigma = new_sigma

    def update_phase_alpha(self, new_alpha):
        """Setter to dynamically concentrate the Ring into a Crescent"""
        self.phase_alpha = new_alpha

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # [16, C, H, W] expected for 16-frame video
        video_np = self.videos[idx]
        video = torch.from_numpy(video_np).float() / 255.0
        video = video.permute(1, 0, 2, 3)

        # Extract 16 frames of Target CoM coordinates
        uv_np = self.uv_coords[idx].reshape(16, 2)
        u_val = uv_np[:, 0]
        v_val = uv_np[:, 1]

        # Extract 16 frames of Box Geometry Centers
        box_np = self.box_center_uv[idx].reshape(16, 2)
        box_u = box_np[:, 0]
        box_v = box_np[:, 1]

        z_np = self.z_depths[idx].flatten()  # [16]
        z_depth = torch.tensor(z_np, dtype=torch.float32).view(16, 1)

        scale = self.heatmap_res / 384.0
        u_t = torch.tensor(u_val * scale, dtype=torch.float32).view(16, 1, 1)
        v_t = torch.tensor(v_val * scale, dtype=torch.float32).view(16, 1, 1)

        u_c = torch.tensor(box_u * scale, dtype=torch.float32).view(16, 1, 1)
        v_c = torch.tensor(box_v * scale, dtype=torch.float32).view(16, 1, 1)

        # 1. Distances & Radii
        dx = self.grid_x.unsqueeze(0) - u_c
        dy = self.grid_y.unsqueeze(0) - v_c
        grid_dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

        # True radius of the CoM from the geometry center
        R = torch.sqrt((u_t - u_c) ** 2 + (v_t - v_c) ** 2 + 1e-8)

        # 2. Draw the Gaussian Ring
        ring = torch.exp(-((grid_dist - R) ** 2) / (2 * self.sigma ** 2))

        # 3. Calculate Cosine Similarity for the Sweep
        vec_t_x = u_t - u_c
        vec_t_y = v_t - v_c
        norm_t = torch.sqrt(vec_t_x ** 2 + vec_t_y ** 2)

        # Check if 2D projections perfectly overlap (Radius < 0.001 pixels)
        is_overlapping = (norm_t < 1e-3)

        # Prevent division by zero safely
        safe_norm_t = torch.where(is_overlapping, torch.ones_like(norm_t), norm_t)

        vec_t_x_norm = vec_t_x / safe_norm_t
        vec_t_y_norm = vec_t_y / safe_norm_t

        vec_g_x_norm = dx / grid_dist
        vec_g_y_norm = dy / grid_dist

        cos_sim = (vec_g_x_norm * vec_t_x_norm) + (vec_g_y_norm * vec_t_y_norm)

        # 4. Apply the Von Mises Concentration
        concentration = self.phase_alpha * 50.0
        mask = torch.exp(concentration * (cos_sim - 1.0))

        mask = torch.where(is_overlapping, torch.ones_like(mask), mask)

        # 5. Multiply to form the Crescent Heatmap
        heatmap = ring * mask

        return video, heatmap, z_depth