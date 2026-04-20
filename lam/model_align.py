"""τ-aligned Latent Action Model (SeqΔ-REPA).

Extends the baseline VAE LAM with a regression loss that aligns latent actions
to the temporal difference features of a frozen visual teacher
(V-JEPA 2 or VideoMAEv2).
"""
from os import makedirs, path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import piq
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer

from lam.modules import LatentActionModel

OptimizerCallable = Callable[[Iterable], Optimizer]


class LAMAlign(LightningModule):
    def __init__(
        self,
        image_channels: int = 3,
        lam_model_dim: int = 512,
        lam_latent_dim: int = 32,
        lam_patch_size: int = 16,
        lam_enc_blocks: int = 8,
        lam_dec_blocks: int = 8,
        lam_num_heads: int = 8,
        lam_dropout: float = 0.0,
        lam_causal_temporal: bool = False,
        beta: float = 0.0002,
        align_weight: float = 0.01,
        teacher_type: str = "vjepa",
        vjepa_weights_path: str = "checkpoints/ref_encoders/vjepa2/vitg-384.pt",
        vjepa_img_size: int = 384,
        videomae_weights_path: str = "checkpoints/ref_encoders/VideoMAEv2-giant",
        videomae_img_size: int = 224,
        align_dim: int = 1408,
        log_interval: int = 1000,
        log_path: str = "log_imgs",
        optimizer: OptimizerCallable = AdamW,
    ):
        super().__init__()

        self.lam = LatentActionModel(
            in_dim=image_channels,
            model_dim=lam_model_dim,
            latent_dim=lam_latent_dim,
            patch_size=lam_patch_size,
            enc_blocks=lam_enc_blocks,
            dec_blocks=lam_dec_blocks,
            num_heads=lam_num_heads,
            dropout=lam_dropout,
            causal_temporal=lam_causal_temporal,
        )

        self.align_dim = align_dim
        self.traj_head = nn.Sequential(
            nn.LayerNorm(lam_latent_dim),
            nn.Linear(lam_latent_dim, lam_latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(lam_latent_dim, lam_latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(lam_latent_dim, align_dim),
        )
        self.traj_head.apply(self._init_linear)

        _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if teacher_type == "videomae":
            from lam.align_utils.videomae_aligner import VideoMAEv2Aligner
            self.teacher = VideoMAEv2Aligner(
                model_path=videomae_weights_path,
                img_size=videomae_img_size,
                device=_dev,
            )
        elif teacher_type == "vjepa":
            from lam.align_utils.vjepa_aligner import VJEPAAligner
            self.teacher = VJEPAAligner(
                vjepa_weights_path=vjepa_weights_path,
                img_size=vjepa_img_size,
                device=_dev,
            )
        else:
            raise ValueError(
                f"Unsupported teacher_type '{teacher_type}'. "
                "Supported: {'vjepa', 'videomae'}"
            )

        self.beta = beta
        self.align_weight = align_weight
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.save_hyperparameters()

    def _init_linear(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _make_traj_embed(self, z_rep: Tensor) -> Tensor:
        z_seq = z_rep.squeeze(2)
        z_pool = z_seq.mean(dim=1)
        z_emb = self.traj_head(z_pool)
        return z_emb

    def _align_loss(self, z_emb: Tensor, tau: Tensor) -> Tensor:
        device = z_emb.device
        tau = tau.to(device=device, dtype=torch.float32)
        assert tau.shape[-1] == self.align_dim, "D_teacher must match align_dim"

        z_emb = torch.nan_to_num(z_emb, nan=0.0, posinf=0.0, neginf=0.0)
        tau = torch.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)

        student = F.normalize(z_emb.float(), dim=-1, eps=1e-6)
        teacher = F.normalize(tau, dim=-1, eps=1e-6)
        loss = 1.0 - (student * teacher).sum(dim=-1)

        return loss.mean()

    def _shared_compute(self, batch: Dict) -> Tuple[Dict, Tensor, dict]:
        outputs = self.lam(batch)
        gt_future = batch["videos"][:, 1:]

        gt_safe = torch.nan_to_num(gt_future, nan=0.0, posinf=1.0, neginf=0.0)
        rc_safe = torch.nan_to_num(outputs["recon"], nan=0.0, posinf=1.0, neginf=0.0)

        gt_f32 = gt_safe.float()
        rc_f32 = rc_safe.float()
        mse = ((gt_f32 - rc_f32) ** 2).mean()

        z_mu = outputs["z_mu"].float()
        z_var = outputs["z_var"].float()
        kl = -0.5 * torch.sum(1 + z_var - z_mu**2 - z_var.exp(), dim=1).mean()

        z_emb = self._make_traj_embed(outputs["z_rep"])
        with torch.no_grad():
            taus = self.teacher.compute_tau_batch(batch["videos"])
        align = self._align_loss(z_emb, taus)

        total = mse + self.beta * kl + self.align_weight * align

        gt = torch.nan_to_num(gt_future, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        rc = torch.nan_to_num(outputs["recon"], nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        gt = rearrange(gt, "b t h w c -> (b t) c h w").float()
        rc = rearrange(rc, "b t h w c -> (b t) c h w").float()
        psnr = piq.psnr(gt, rc, data_range=1.0).mean()
        ssim = piq.ssim(gt, rc, data_range=1.0).mean()

        logs = {
            "mse_loss": mse,
            "kl_loss": kl,
            "align_loss": align,
            "psnr": psnr,
            "ssim": ssim,
        }
        return outputs, total, logs

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        outputs, loss, logs = self._shared_compute(batch)
        self.log_dict(
            {"train_loss": loss, **{f"train/{k}": v for k, v in logs.items()}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        if batch_idx % self.log_interval == 0:
            self._log_images(batch, outputs, split="train")
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        outputs, loss, logs = self._shared_compute(batch)
        self.log_dict(
            {"val_loss": loss, **{f"val/{k}": v for k, v in logs.items()}},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        if batch_idx % self.log_interval == 0:
            self._log_images(batch, outputs, split="val")
        return loss

    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        outputs, loss, logs = self._shared_compute(batch)
        self.log_dict(
            {"test_loss": loss, **{f"test/{k}": v for k, v in logs.items()}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self._log_images(batch, outputs, split="test")
        return loss

    def _log_images(self, batch: Dict, outputs: Dict, split: str) -> None:
        gt_seq = torch.nan_to_num(batch["videos"][0], nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1).cpu()
        rc_seq = torch.nan_to_num(outputs["recon"][0], nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1).cpu()
        rc_seq = torch.cat([gt_seq[:1], rc_seq], dim=0)
        compare = torch.cat([gt_seq, rc_seq], dim=1)
        compare = rearrange((compare * 255).byte(), "t h w c -> h (t w) c").numpy()
        img_path = path.join(self.hparams.log_path, f"{split}_step{self.global_step:06}.png")
        makedirs(path.dirname(img_path), exist_ok=True)
        Image.fromarray(compare).save(img_path)

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer(self.parameters())
