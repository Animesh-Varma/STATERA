import os
import urllib.request
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def ensure_vjepa_weights_exist():
    checkpoint_dir = Path(os.path.expanduser("~/.cache/torch/hub/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "vjepa2_1_vitl_dist_vitG_384.pt"
    if not checkpoint_path.exists():
        print("⚠️ Missing V-JEPA 2.1 weights. Downloading...")
        urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
                                   checkpoint_path)


ensure_vjepa_weights_exist()


class StateraModel(nn.Module):
    def __init__(self, heatmap_res=64):
        super().__init__()
        self.heatmap_res = heatmap_res

        hub_output = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_large_384')
        self.backbone = hub_output[0] if isinstance(hub_output, tuple) else hub_output

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.embed_dim = 1024

        self.pooler = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=2)
        )

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=3, padding=1),
            nn.GELU()
        )

        self.funnel = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU()
        )

        # Removed Magnitude Head, keeping only 2D Spatial and Z-Depth
        self.heatmap_head = nn.Sequential(nn.Linear(256, heatmap_res * heatmap_res))
        self.z_head = nn.Sequential(nn.Linear(256, 1))

    def forward(self, x):
        with torch.no_grad():
            tokens = self.backbone(x)
            if isinstance(tokens, tuple): tokens = tokens[0]

        B, L, C = tokens.shape
        T_vjepa = 8
        num_patches = L // T_vjepa

        # Explicit 4D segregation mapping [Batch, 8, 576, 1024]
        tokens = tokens.view(B, T_vjepa, num_patches, C)

        # Pool spatially across the 576 patches (dim=2)
        weights = self.pooler(tokens) # [B, 8, num_patches, 1]
        pooled = torch.sum(tokens * weights, dim=2)  # Output: [B, 8, 1024]

        # Temporal Velocity Mapping
        feat = pooled.transpose(1, 2)    # [B, 1024, 8]
        feat = self.temporal_conv(feat)  # [B, 1024, 8]

        # Temporal Upscale 8 -> 16 to sync with GT frames
        feat = F.interpolate(feat, size=16, mode='linear', align_corners=False)  # [B, 1024, 16]
        feat = feat.transpose(1, 2)  # [B, 16, 1024]

        # Condense channel-space down
        feat = self.funnel(feat)  # Output: [B, 16, 256]

        # Execute Dual-Head routing
        h_2d = self.heatmap_head(feat).view(B, 16, self.heatmap_res, self.heatmap_res)
        z = self.z_head(feat)

        return h_2d, z