import torch
import torch.nn as nn
import torch.nn.functional as F


class StateraModel(nn.Module):
    """
    STATERA (Spatio-Temporal Analysis of Tensor Embeddings for Rigid-body Asymmetry)

    A 2.5D multi-task architecture that extracts the hidden Center of Mass (CoM) of opaque,
    asymmetric rigid bodies from raw video. By processing video through a partially-frozen
    video backbone (e.g., V-JEPA), the model tracks spatial acceleration across time to isolate
    inertial mechanics from visual geometry, effectively curing Visual-Kinematic Aliasing.

    Reference: Section 3.2 (STATERA Architecture)
    """

    def __init__(self, decoder_type='mlp', temporal_mixer='conv1d',
                 single_task=False, backbone_type='vjepa', scratch=False,
                 finetune_blocks=0, heatmap_res=64):
        super().__init__()
        # Target spatial resolution for continuous sub-pixel extraction (Head A).
        # Standard configuration maps 24x24 latent patches to a dense 64x64 grid.
        self.heatmap_res = heatmap_res
        self.decoder_type = decoder_type
        self.temporal_mixer = temporal_mixer
        self.single_task = single_task
        self.backbone_type = backbone_type
        self.scratch = scratch
        self.finetune_blocks = finetune_blocks

        # ---------------------------------------------------------------------
        # VISUAL FOUNDATION MODEL BACKBONES
        # ---------------------------------------------------------------------
        if self.backbone_type == 'vjepa':
            # Meta V-JEPA 2.1: The primary temporally-aware vision backbone.
            # Learns transition dynamics in an abstract latent space, enabling the network
            # to track continuous momentum and rotational inertia rather than static heuristics.
            hub_out = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_large_384')
            self.backbone = hub_out[0] if isinstance(hub_out, tuple) else hub_out
            self.embed_dim = 1024
        elif self.backbone_type == 'dinov2':
            # Spatial Foundation Model Baseline (DINOv2).
            # Ablations show this model fails at kinematics and suffers from Visual-Kinematic
            # Aliasing because it processes isolated spatial frames without temporal derivatives.
            hub_out = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.backbone = hub_out[0] if isinstance(hub_out, tuple) else hub_out
            self.embed_dim = 1024
        elif self.backbone_type == 'videomae':
            # Temporal Foundation Baseline (VideoMAE v2).
            # Fails to achieve positive Physics Capture because its pixel-level reconstruction
            # objective forces the memorization of surface textures over latent physical priors.
            from transformers import VideoMAEModel
            self.backbone = VideoMAEModel.from_pretrained('MCG-NJU/videomae-large')
            self.embed_dim = 1024
        elif self.backbone_type == 'resnet3d':
            # Standard 3D-CNN Baseline.
            from torchvision.models.video import r3d_18, R3D_18_Weights
            self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            self.embed_dim = 512

        # ---------------------------------------------------------------------
        # PARTIALLY-FROZEN ADAPTATION LOGIC
        # ---------------------------------------------------------------------
        if self.scratch:
            self.backbone.apply(self._init_weights)
        else:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

            # STATERA utilizes parameter-efficient training (~2.5M trainable params).
            # "During training, the final two transformer adaptation blocks are unfrozen"
            if self.finetune_blocks > 0:
                if hasattr(self.backbone, 'blocks'):
                    for block in self.backbone.blocks[-self.finetune_blocks:]:
                        for param in block.parameters():
                            param.requires_grad = True
                    if hasattr(self.backbone, 'norm'):
                        for param in self.backbone.norm.parameters():
                            param.requires_grad = True
                elif self.backbone_type == 'resnet3d':
                    for i in range(1, self.finetune_blocks + 1):
                        layer_idx = 5 - i
                        if layer_idx >= 1:
                            for param in self.backbone[layer_idx].parameters():
                                param.requires_grad = True

        # ---------------------------------------------------------------------
        # TEMPORAL TUBELET MIXING
        # ---------------------------------------------------------------------
        if self.temporal_mixer == 'conv1d':
            # "We apply a 1D Convolution (kernel=3) across the temporal dimension
            # of the latent tubelets... to compute temporal velocity derivatives."
            self.temp_net = nn.Sequential(nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
                                          nn.GELU())
        elif self.temporal_mixer == 'transformer':
            layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=8, batch_first=True)
            self.temp_net = nn.TransformerEncoder(layer, num_layers=2)
            self.pos_embed = nn.Parameter(torch.zeros(1, 16, self.embed_dim))
            nn.init.normal_(self.pos_embed, std=0.02)
        elif self.temporal_mixer == 'none':
            # Ablation configuration: "STATERA-1K-Frozen-Anchor (Ablation)"
            # Removing the temporal mixer degrades performance to match static spatial baselines.
            self.temp_net = nn.Identity()

        # ---------------------------------------------------------------------
        # SPATIAL PRESERVATION DECODER
        # ---------------------------------------------------------------------
        if self.decoder_type in ['mlp', 'regression']:
            # Naive global average pooling (destroys localized geometry/torque representations)
            self.pooler = nn.Sequential(nn.Linear(self.embed_dim, 256), nn.Tanh(), nn.Linear(256, 1), nn.Softmax(dim=2))
            self.funnel = nn.Sequential(nn.Linear(self.embed_dim, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU())

            if self.decoder_type == 'mlp':
                self.heatmap_head = nn.Linear(256, heatmap_res * heatmap_res)
            elif self.decoder_type == 'regression':
                self.coord_head = nn.Linear(256, 2)

        elif self.decoder_type == 'deconv':
            # "STATERA reshapes the 576 patches into their native 24 x 24 spatial grid.
            # Stacked ConvTranspose2d layers upscale this feature map to 64 x 64,
            # preserving local spatial equivariance and enabling sub-pixel acuity."
            self.deconv_net = nn.Sequential(
                nn.ConvTranspose2d(self.embed_dim, 256, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 1, kernel_size=3, padding=1),
                nn.AdaptiveAvgPool2d((heatmap_res, heatmap_res))
            )
            # Dedicated pathway for Head B (Z-Depth Regularization)
            self.z_pooler = nn.Sequential(nn.Linear(self.embed_dim, 1), nn.Softmax(dim=2))
            self.z_funnel = nn.Sequential(nn.Linear(self.embed_dim, 256), nn.GELU())

        # MULTI-TASK REGULARIZATION (Head B)
        # "1D absolute Z-Depth regularizer. Empirically forces the latent space to decouple
        # 2D pixel-velocity from true 3D momentum, ensuring perspective-invariant physics."
        if not self.single_task:
            self.z_head = nn.Linear(256, 1)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        """
        Forward pass for non-causal (offline) sequence inference over T=16 frames.
        Tracks the unobservable CoM by evaluating off-axis rotational torques.
        """
        B = x.shape[0]
        # Maintains VRAM stability (e.g., training on single RTX 5070 Ti) via conditional gradients
        enable_grad = (self.scratch or self.finetune_blocks > 0) and self.training

        # ---------------------------------------------------------------------
        # BACKBONE FEATURE EXTRACTION (LATENT COMPRESSION)
        # ---------------------------------------------------------------------
        if self.backbone_type == 'vjepa':
            # "The network processes the 16-frame sequence by compressing it into
            # T=8 temporal tubelets, each containing 576 spatial patches."
            with torch.set_grad_enabled(enable_grad):
                tokens = self.backbone(x)
                if isinstance(tokens, tuple): tokens = tokens[0]
            tokens = tokens.reshape(B, 8, 576, self.embed_dim)
            T_current = 8


        elif self.backbone_type == 'videomae':
            B, C, T_in, H, W = x.shape
            x_res = x.permute(0, 2, 1, 3, 4).reshape(B * T_in, C, H, W)
            x_res = F.interpolate(x_res, size=(224, 224), mode='bilinear')

            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            x_res = (x_res - mean) / std

            x_res = x_res.reshape(B, T_in, C, 224, 224)

            with torch.set_grad_enabled(enable_grad):
                tokens = self.backbone(pixel_values=x_res).last_hidden_state

            tokens = tokens.reshape(B * 8, 196, self.embed_dim).permute(0, 2, 1).reshape(B * 8, self.embed_dim, 14, 14)

            tokens = F.interpolate(tokens, size=(24, 24), mode='bilinear')
            tokens = tokens.reshape(B, 8, self.embed_dim, 576).permute(0, 1, 3, 2)
            T_current = 8

        elif self.backbone_type == 'dinov2':
            # Processes isolated spatial frames. Lacks temporal derivatives, leading
            # to metric collapse/gaming Euclidean metrics.
            if len(x.shape) == 5 and x.shape[1] == 3 and x.shape[2] == 16:
                x_inter = x.permute(0, 2, 1, 3, 4).reshape(B * 16, 3, 384, 384)
            else:
                x_inter = x.reshape(B * 16, 3, 384, 384)

            x_inter = F.interpolate(x_inter, size=392, mode='bilinear')
            tokens_list = []
            chunk_size = 16

            with torch.set_grad_enabled(enable_grad):
                for i in range(0, B * 16, chunk_size):
                    chunk = x_inter[i:i + chunk_size]
                    out = self.backbone.forward_features(chunk)['x_norm_patchtokens']
                    tokens_list.append(out)

            tokens = torch.cat(tokens_list, dim=0)
            tokens = tokens.permute(0, 2, 1).reshape(B * 16, self.embed_dim, 28, 28)
            tokens = F.interpolate(tokens, size=24, mode='bilinear')
            tokens = tokens.reshape(B, 16, self.embed_dim, 576).permute(0, 1, 3, 2)
            T_current = 16

        elif self.backbone_type == 'resnet3d':
            x_inter = x
            with torch.set_grad_enabled(enable_grad):
                out = self.backbone(x_inter)
            out = F.interpolate(out, size=(16, 24, 24), mode='trilinear', align_corners=False)
            tokens = out.permute(0, 2, 1, 3, 4).reshape(B, 16, self.embed_dim, 576).permute(0, 1, 3, 2)
            T_current = 16

        # ---------------------------------------------------------------------
        # TEMPORAL MIXING & LATENT DERIVATIVES
        # ---------------------------------------------------------------------
        if self.decoder_type in ['mlp', 'regression']:
            weights = self.pooler(tokens)
            feat = torch.sum(tokens * weights, dim=2)
        else:
            feat = tokens

        if self.temporal_mixer in ['conv1d', 'transformer', 'none']:
            if self.decoder_type in ['mlp', 'regression']:
                if self.temporal_mixer == 'conv1d':
                    feat = self.temp_net(feat.transpose(1, 2)).transpose(1, 2)
                elif self.temporal_mixer == 'transformer':
                    feat = feat + self.pos_embed[:, :feat.size(1), :]
                    feat = self.temp_net(feat)
            else:
                B_curr, T_curr, P_curr, C_curr = feat.shape
                # "Applies a 1D Convolution across the temporal dimension of the latent tubelets"
                if self.temporal_mixer == 'conv1d':
                    feat_flat = feat.permute(0, 2, 1, 3).reshape(B_curr * P_curr, T_curr, C_curr).permute(0, 2, 1)
                    feat_flat = self.temp_net(feat_flat)
                    feat = feat_flat.permute(0, 2, 1).reshape(B_curr, P_curr, T_curr, C_curr).permute(0, 2, 1, 3)
                elif self.temporal_mixer == 'transformer':
                    feat_flat = feat.permute(0, 2, 1, 3).reshape(B_curr * P_curr, T_curr, C_curr)
                    feat_flat = feat_flat + self.pos_embed[:, :T_curr, :]
                    feat_flat = self.temp_net(feat_flat)
                    feat = feat_flat.reshape(B_curr, P_curr, T_curr, C_curr).permute(0, 2, 1, 3)

        # ---------------------------------------------------------------------
        # CONTINUOUS LATENT UPSAMPLING
        # ---------------------------------------------------------------------
        if T_current == 8:
            if self.decoder_type in ['mlp', 'regression']:
                feat = F.interpolate(feat.transpose(1, 2), size=16, mode='linear').transpose(1, 2)
            else:
                B_curr, T_curr, P_curr, C_curr = feat.shape
                feat_flat = feat.permute(0, 2, 1, 3).reshape(B_curr * P_curr, T_curr, C_curr).permute(0, 2, 1)
                # "Crucially, this F.interpolate operation occurs within the high-dimensional
                # continuous latent space, rather than discrete coordinate space. This preserves
                # smooth, differentiable transition dynamics of the object's momentum."
                feat_flat = F.interpolate(feat_flat, size=16, mode='linear')
                feat = feat_flat.permute(0, 2, 1).reshape(B_curr, P_curr, 16, C_curr).permute(0, 2, 1, 3)

        # ---------------------------------------------------------------------
        # MULTI-TASK HEAD BIFURCATION
        # ---------------------------------------------------------------------
        out = None
        if self.decoder_type == 'mlp':
            feat_condensed = self.funnel(feat)
            out = self.heatmap_head(feat_condensed).reshape(B, 16, self.heatmap_res, self.heatmap_res)
            z_feat = feat_condensed
        elif self.decoder_type == 'regression':
            # Direct coordinate regression is notoriously unstable and injects high-frequency noise.
            feat_condensed = self.funnel(feat)
            out = self.coord_head(feat_condensed)
            z_feat = feat_condensed
        elif self.decoder_type == 'deconv':
            # Head A: Outputs a continuous spatial heatmap evaluated via KL Divergence.
            # In inference, a Temperature-Scaled Soft-Argmax (tau=2.0) is applied to extract
            # sub-pixel continuous coordinates.
            P_curr = feat.shape[2]
            spatial_dim = int(P_curr ** 0.5)
            grid_feat = feat.reshape(B * 16, spatial_dim, spatial_dim, self.embed_dim).permute(0, 3, 1, 2)
            out = self.deconv_net(grid_feat).reshape(B, 16, self.heatmap_res, self.heatmap_res)
            if not self.single_task:
                z_w = self.z_pooler(feat)
                z_pooled = torch.sum(feat * z_w, dim=2)
                z_feat = self.z_funnel(z_pooled)

        # Head B: 1D Absolute Z-Depth scalar. Solves the Vector Projection artifact by grounding
        # the 2D offset coordinate against the unconstrained 3D simulated depth.
        z = self.z_head(z_feat) if not self.single_task else torch.zeros((B, 16, 1)).to(x.device)
        return out, z