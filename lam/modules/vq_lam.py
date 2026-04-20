from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from lam.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer, VectorQuantizer
from torch import Tensor


class VQLatentActionModel(nn.Module):
    """
    Vector Quantized Latent Action Model.
    Replaces the continuous VAE with discrete vector quantization.
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
            num_embeddings: int = 512,  # VQ codebook size
            commitment_cost: float = 0.25,  # Beta for VQ loss
            dropout: float = 0.0,
            causal_temporal: bool = False  # NEW: control temporal causality
    ) -> None:
        super(VQLatentActionModel, self).__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        patch_token_dim = in_dim * patch_size ** 2

        # Action prompt token (same as original)
        self.action_prompt = nn.Parameter(torch.empty(1, 1, 1, patch_token_dim))
        nn.init.uniform_(self.action_prompt, a=-1, b=1)
        
        # Encoder (same as original)
        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim,
            model_dim=model_dim,
            out_dim=model_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=causal_temporal
        )
        
        # Replace VAE head with projection to latent_dim for VQ
        # with normalized projection to bound z_e scale
        self.pre_vq = nn.Sequential(
            nn.Linear(model_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        
        # Vector Quantizer
        self.vq = VectorQuantizer(
            num_latents=num_embeddings,
            latent_dim=latent_dim,
            code_restart=True  # Enable dead code restart
        )
        
        # Projection layers (same as original)
        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        
        # Decoder (same as original)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=patch_token_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout
        )

        # For tracking codebook usage during evaluation
        self.indices_record = None

    def encode(self, videos: Tensor) -> Dict:
        # Preprocess videos (same as original)
        B, T = videos.shape[:2]
        patches = patchify(videos, self.patch_size)
        action_pad = self.action_prompt.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, patches], dim=2)

        # Encode (same as original)
        z = self.encoder(padded_patches)  # (B, T, 1+N, E)
        # Get latent action for all future frames
        z = z[:, 1:, 0]  # (B, T-1, E)

        # Project to latent dimension for VQ
        z = z.reshape(B * (T - 1), self.model_dim)
        z_e = self.pre_vq(z)  # (B*(T-1), latent_dim)
        
        # Vector Quantization
        z_q, _, _, indices = self.vq(z_e)
        # Reshape back
        z_q = z_q.reshape(B, T - 1, 1, self.latent_dim)
        indices = indices.reshape(B, T - 1)

        # Track indices during evaluation for analysis
        if not self.training:
            if self.indices_record is None:
                self.indices_record = indices
            else:
                self.indices_record = torch.cat([self.indices_record, indices], dim=0)

        return {
            "patches": patches,
            "z_rep": z_q,
            "z_e": z_e.reshape(B, T - 1, self.latent_dim),  # Pre-quantization embeddings
            "indices": indices
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VQ
        H, W = batch["videos"].shape[2:4]
        outputs = self.encode(batch["videos"])
        
        # Decoder pathway (same as original)
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_rep"])
        video_action_patches = video_patches + action_patches

        # Remove patches to save memory
        del outputs["patches"]

        # Decode
        video_recon = self.decoder(video_action_patches)
        video_recon = torch.sigmoid(video_recon)
        
        # Compute VQ loss
        z_e = outputs["z_e"]  # (B, T-1, latent_dim)
        z_q = outputs["z_rep"].squeeze(2)  # (B, T-1, latent_dim)
        
        # VQ loss: commitment loss + codebook loss
        commitment_loss = F.mse_loss(z_q.detach(), z_e)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        outputs.update({
            "recon": unpatchify(video_recon, self.patch_size, H, W),
            "vq_loss": vq_loss,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss
        })
        
        return outputs

    def get_codebook_usage(self) -> Dict:
        """Get codebook usage statistics."""
        if hasattr(self.vq, 'usage'):
            usage = self.vq.usage
            total_usage = usage.sum().item()
            active_codes = (usage > 0).sum().item()
            perplexity = torch.exp(-torch.sum(usage / total_usage * torch.log(usage / total_usage + 1e-10)))
            return {
                "active_codes": active_codes,
                "total_codes": self.num_embeddings,
                "utilization": active_codes / self.num_embeddings,
                "perplexity": perplexity.item() if total_usage > 0 else 0.0
            }
        return {}

    def reset_codebook_usage(self):
        """Reset codebook usage statistics."""
        if hasattr(self.vq, 'reset_usage'):
            self.vq.reset_usage()