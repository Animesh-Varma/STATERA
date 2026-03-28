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
        video_np = self.videos[idx]
        video = torch.from_numpy(video_np).float() / 255.0
        video = video.permute(1, 0, 2, 3)

        uv_np = self.uv_coords[idx].reshape(-1, 2)
        u_val = float(uv_np[-1, 0])
        v_val = float(uv_np[-1, 1])

        z_np = self.z_depths[idx].flatten()
        z_val = float(z_np[-1])
        z_depth = torch.tensor([z_val], dtype=torch.float32)

        scale = self.heatmap_res / 384.0
        u_s, v_s = u_val * scale, v_val * scale

        dist_sq = (self.grid_x - u_s) ** 2 + (self.grid_y - v_s) ** 2
        heatmap = torch.exp(-dist_sq / (2 * self.sigma ** 2))

        return video, heatmap, z_depth