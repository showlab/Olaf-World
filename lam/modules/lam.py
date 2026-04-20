from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from lam.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer
from torch import Tensor
from torch.cuda.amp import autocast

class LatentActionModel(nn.Module):
    """Latent action VAE.

    Encodes a video clip of T frames into T-1 latent actions, each of which
    summarizes the transition from frame t to frame t+1.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
            causal_temporal: bool = False
    ) -> None:
        super(LatentActionModel, self).__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.action_prompt = nn.Parameter(torch.empty(1, 1, 1, patch_token_dim))
        nn.init.uniform_(self.action_prompt, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim,
            model_dim=model_dim,
            out_dim=model_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=causal_temporal
        )
        self.fc = nn.Linear(model_dim, latent_dim * 2)
        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=patch_token_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout
        )

        self.mu_record = None

    def encode(self, videos: Tensor) -> Dict:
        B, T = videos.shape[:2]
        patches = patchify(videos, self.patch_size)
        action_pad = self.action_prompt.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, patches], dim=2)

        z = self.encoder(padded_patches)  # (B, T, 1+N, E)
        z = z[:, 1:, 0]                   # take the action prompt slot for frames 1..T-1

        z = z.reshape(B * (T - 1), self.model_dim)
        moments = self.fc(z)
        moments = torch.nan_to_num(moments, nan=0.0, posinf=0.0, neginf=0.0)

        z_mu, z_logvar = torch.chunk(moments, 2, dim=1)
        z_mu = z_mu.float()
        z_logvar = z_logvar.float()
        z_logvar = torch.nan_to_num(z_logvar, nan=0.0, posinf=0.0, neginf=0.0)
        z_logvar = z_logvar.clamp(-10.0, 10.0)

        if not self.training:
            z_rep = z_mu
            z_var_out = z_logvar
        else:
            with autocast(False):
                std = torch.exp(0.5 * z_logvar)
                eps = torch.randn_like(std)
                z_rep_f32 = z_mu + eps * std
            z_rep = z_rep_f32.to(torch.bfloat16 if z_mu.dtype == torch.bfloat16 else z_mu.dtype)
            z_var_out = z_logvar

        z_rep = z_rep.reshape(B, T - 1, 1, self.latent_dim)

        if not self.training:
            if self.mu_record is None:
                self.mu_record = z_mu
            else:
                self.mu_record = torch.cat([self.mu_record, z_mu], dim=0)

        return {
            "patches": patches,
            "z_rep": z_rep,
            "z_mu": z_mu,
            "z_var": z_var_out,
        }

    def forward(self, batch: Dict) -> Dict:
        H, W = batch["videos"].shape[2:4]
        outputs = self.encode(batch["videos"])
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_rep"])
        video_action_patches = video_patches + action_patches

        del outputs["patches"]

        video_recon = self.decoder(video_action_patches)
        video_recon = torch.sigmoid(video_recon)
        outputs["recon"] = unpatchify(video_recon, self.patch_size, H, W)
        return outputs
