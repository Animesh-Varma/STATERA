import torch
import torch.nn as nn
import torch.nn.functional as F


class StateraModel(nn.Module):
    def __init__(self, decoder_type='mlp', temporal_mixer='conv1d',
                 single_task=False, backbone_type='vjepa', scratch=False, heatmap_res=64):
        super().__init__()
        self.heatmap_res = heatmap_res
        self.decoder_type = decoder_type
        self.temporal_mixer = temporal_mixer
        self.single_task = single_task
        self.backbone_type = backbone_type
        self.scratch = scratch

        if self.backbone_type == 'vjepa':
            hub_out = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_large_384')
            self.backbone = hub_out[0] if isinstance(hub_out, tuple) else hub_out
        elif self.backbone_type == 'dinov2':
            hub_out = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.backbone = hub_out[0] if isinstance(hub_out, tuple) else hub_out

        self.embed_dim = 1024

        if self.scratch:
            self.backbone.apply(self._init_weights)
        else:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Temporal Mixer Toggle
        if self.temporal_mixer == 'conv1d':
            self.temp_net = nn.Sequential(nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
                                          nn.GELU())
        elif self.temporal_mixer == 'transformer':
            layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=8, batch_first=True)
            self.temp_net = nn.TransformerEncoder(layer, num_layers=2)
        elif self.temporal_mixer == 'none':
            self.temp_net = nn.Identity()

            # Spatial Decoder Toggle
        if self.decoder_type in ['mlp', 'regression']:
            self.pooler = nn.Sequential(nn.Linear(self.embed_dim, 256), nn.Tanh(), nn.Linear(256, 1), nn.Softmax(dim=2))
            self.funnel = nn.Sequential(nn.Linear(self.embed_dim, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU())

            if self.decoder_type == 'mlp':
                self.heatmap_head = nn.Linear(256, heatmap_res * heatmap_res)
            elif self.decoder_type == 'regression':
                self.coord_head = nn.Linear(256, 2)

        elif self.decoder_type == 'deconv':
            self.deconv_net = nn.Sequential(
                nn.ConvTranspose2d(self.embed_dim, 256, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 1, kernel_size=3, padding=1),
                nn.AdaptiveAvgPool2d((heatmap_res, heatmap_res))
            )
            self.z_pooler = nn.Sequential(nn.Linear(self.embed_dim, 1), nn.Softmax(dim=2))
            self.z_funnel = nn.Sequential(nn.Linear(self.embed_dim, 256), nn.GELU())

        if not self.single_task:
            self.z_head = nn.Linear(256, 1)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        B = x.shape[0]
        enable_grad = self.scratch and self.training

        if self.backbone_type == 'vjepa':
            with torch.set_grad_enabled(enable_grad):
                tokens = self.backbone(x)
                if isinstance(tokens, tuple): tokens = tokens[0]
            tokens = tokens.view(B, 8, 576, self.embed_dim)
            T_current = 8

        elif self.backbone_type == 'dinov2':
            x = F.interpolate(x.reshape(B * 16, 3, 384, 384), size=392, mode='bilinear')
            tokens_list = []
            chunk_size = 16

            with torch.set_grad_enabled(enable_grad):
                for i in range(0, B * 16, chunk_size):
                    chunk = x[i:i + chunk_size]
                    out = self.backbone.forward_features(chunk)['x_norm_patchtokens']
                    tokens_list.append(out)

            tokens = torch.cat(tokens_list, dim=0)
            tokens = tokens.permute(0, 2, 1).reshape(B * 16, self.embed_dim, 28, 28)
            tokens = F.interpolate(tokens, size=24, mode='bilinear')
            tokens = tokens.reshape(B, 16, self.embed_dim, 576).permute(0, 1, 3, 2)
            T_current = 16

        # Spatial Pool
        if self.decoder_type in ['mlp', 'regression']:
            weights = self.pooler(tokens)
            feat = torch.sum(tokens * weights, dim=2)
        else:
            feat = tokens

        # -------------------------------------------------------------------
        # CORRECTED: Temporal Mixing (Treating Patches as independent Batches)
        # -------------------------------------------------------------------
        if self.temporal_mixer in ['conv1d', 'transformer', 'none']:
            if self.decoder_type in ['mlp', 'regression']:
                # feat is [B, T, C]
                if self.temporal_mixer == 'conv1d':
                    feat = self.temp_net(feat.transpose(1, 2)).transpose(1, 2)
                elif self.temporal_mixer == 'transformer':
                    feat = self.temp_net(feat)
            else:
                # feat is [B, T, P, C]. We want to temporally mix over T.
                B, T, P, C = feat.shape

                if self.temporal_mixer == 'conv1d':
                    # [B, T, P, C] -> [B, P, T, C] -> [B*P, T, C] -> [B*P, C, T]
                    feat_flat = feat.permute(0, 2, 1, 3).reshape(B * P, T, C).permute(0, 2, 1)
                    feat_flat = self.temp_net(feat_flat)
                    # [B*P, C, T] -> [B*P, T, C] -> [B, P, T, C] -> [B, T, P, C]
                    feat = feat_flat.permute(0, 2, 1).reshape(B, P, T, C).permute(0, 2, 1, 3)

                elif self.temporal_mixer == 'transformer':
                    # [B, T, P, C] -> [B, P, T, C] -> [B*P, T, C]
                    feat_flat = feat.permute(0, 2, 1, 3).reshape(B * P, T, C)
                    feat_flat = self.temp_net(feat_flat)
                    # [B*P, T, C] -> [B, P, T, C] -> [B, T, P, C]
                    feat = feat_flat.reshape(B, P, T, C).permute(0, 2, 1, 3)

        # -------------------------------------------------------------------
        # CORRECTED: Temporal Upscale 8 -> 16
        # -------------------------------------------------------------------
        if T_current == 8:
            if self.decoder_type in ['mlp', 'regression']:
                feat = F.interpolate(feat.transpose(1, 2), size=16, mode='linear').transpose(1, 2)
            else:
                B, T, P, C = feat.shape
                feat_flat = feat.permute(0, 2, 1, 3).reshape(B * P, T, C).permute(0, 2, 1)  # [B*P, C, T=8]
                feat_flat = F.interpolate(feat_flat, size=16, mode='linear')  # [B*P, C, T=16]
                feat = feat_flat.permute(0, 2, 1).reshape(B, P, 16, C).permute(0, 2, 1, 3)  # [B, 16, P, C]

        # Decode
        out = None
        if self.decoder_type == 'mlp':
            feat_condensed = self.funnel(feat)
            out = self.heatmap_head(feat_condensed).view(B, 16, self.heatmap_res, self.heatmap_res)
            z_feat = feat_condensed
        elif self.decoder_type == 'regression':
            feat_condensed = self.funnel(feat)
            out = self.coord_head(feat_condensed)
            z_feat = feat_condensed
        elif self.decoder_type == 'deconv':
            grid_feat = feat.view(B * 16, 24, 24, self.embed_dim).permute(0, 3, 1, 2)
            out = self.deconv_net(grid_feat).view(B, 16, self.heatmap_res, self.heatmap_res)
            if not self.single_task:
                z_w = self.z_pooler(feat)
                z_pooled = torch.sum(feat * z_w, dim=2)
                z_feat = self.z_funnel(z_pooled)

        z = self.z_head(z_feat) if not self.single_task else torch.zeros((B, 16, 1)).to(x.device)
        return out, z