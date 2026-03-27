import os
import urllib.request
from pathlib import Path
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def ensure_vjepa_weights_exist():
    checkpoint_dir = Path(os.path.expanduser("~/.cache/torch/hub/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "vjepa2_1_vitl_dist_vitG_384.pt"
    if not checkpoint_path.exists():
        print("⚠️ Missing V-JEPA 2.1 weights. Downloading...")
        urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt", checkpoint_path)


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
            nn.Softmax(dim=1)
        )

        self.funnel = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU()
        )

        # Head A: Outputs raw mathematical logits
        self.heatmap_head = nn.Sequential(
            nn.Linear(256, heatmap_res * heatmap_res)
        )

        # Head B: Uncapped Meters
        self.z_head = nn.Sequential(
            nn.Linear(256, 1)
        )

    def forward(self, x):
        with torch.no_grad():
            tokens = self.backbone(x)
            if isinstance(tokens, tuple): tokens = tokens[0]

        weights = self.pooler(tokens)
        pooled = torch.sum(tokens * weights, dim=1)

        feat = self.funnel(pooled)
        h_2d = self.heatmap_head(feat).view(-1, self.heatmap_res, self.heatmap_res)
        z = self.z_head(feat)

        return h_2d, z