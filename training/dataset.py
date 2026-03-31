import torch
from torch.utils.data import Dataset
import h5py

class StateraDataset(Dataset):
    def __init__(self, h5_path, heatmap_res=64, sigma=12.5):
        self.h5_path = h5_path
        self.heatmap_res = heatmap_res
        self.sigma = sigma

        print(f"Loading dataset into RAM to bypass HDF5 bottleneck...")
        with h5py.File(self.h5_path, 'r') as f:
            self.videos = f['videos'][:]
            self.uv_coords = f['uv_coords'][:]
            self.z_depths = f['z_depths'][:]
            # Change 3: Extract magnitude cache
            self.com_magnitudes = f['com_magnitudes'][:]
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

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # [16, C, H, W] expected for 16-frame video
        video_np = self.videos[idx]
        video = torch.from_numpy(video_np).float() / 255.0
        video = video.permute(1, 0, 2, 3)

        # Extract 16 frames of coordinates
        uv_np = self.uv_coords[idx].reshape(16, 2)
        u_val = uv_np[:, 0]
        v_val = uv_np[:, 1]

        z_np = self.z_depths[idx].flatten() # [16]
        z_depth = torch.tensor(z_np, dtype=torch.float32).view(16, 1)

        # Change 3: Safely broadcast the static scalar magnitude across 16 timesteps
        mag_val = float(self.com_magnitudes[idx].item())
        com_mag = torch.full((16, 1), mag_val, dtype=torch.float32)

        scale = self.heatmap_res / 384.0
        u_s = torch.tensor(u_val * scale, dtype=torch.float32).view(16, 1, 1)
        v_s = torch.tensor(v_val * scale, dtype=torch.float32).view(16, 1, 1)

        # Broadcast grid subtraction across the 16 time steps: [16, 64, 64]
        dist_sq = (self.grid_x.unsqueeze(0) - u_s) ** 2 + (self.grid_y.unsqueeze(0) - v_s) ** 2
        heatmap = torch.exp(-dist_sq / (2 * self.sigma ** 2))

        return video, heatmap, z_depth, com_mag