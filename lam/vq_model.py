"""Vector-Quantized Latent Action Model — MSE + VQ bottleneck."""
from os import makedirs, path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer

try:
    import piq
    _HAS_PIQ = True
except ImportError:
    _HAS_PIQ = False

try:
    from kmeans_pytorch import kmeans
    _HAS_KMEANS = True
except ImportError:
    _HAS_KMEANS = False

OptimizerCallable = Callable[[Iterable], Optimizer]

from lam.modules.vq_lam import VQLatentActionModel


class VQLAM(LightningModule):
    """Vector-quantized Latent Action Model (Lightning module)."""
    
    def __init__(
            self,
            image_channels: int = 3,
            # VQ Latent action autoencoder
            lam_model_dim: int = 512,
            lam_latent_dim: int = 32,
            lam_patch_size: int = 16,
            lam_enc_blocks: int = 8,
            lam_dec_blocks: int = 8,
            lam_num_heads: int = 8,
            lam_dropout: float = 0.0,
            lam_causal_temporal: bool = False,  # control temporal causality
            # VQ specific parameters
            num_embeddings: int = 512,
            commitment_cost: float = 0.25,
            vq_beta: float = 1.0,  # Weight for VQ loss in total loss
            log_interval: int = 1000,
            log_path: str = "log_imgs",
            optimizer: OptimizerCallable = AdamW
    ) -> None:
        super(VQLAM, self).__init__()
        self.vq_lam = VQLatentActionModel(
            in_dim=image_channels,
            model_dim=lam_model_dim,
            latent_dim=lam_latent_dim,
            patch_size=lam_patch_size,
            enc_blocks=lam_enc_blocks,
            dec_blocks=lam_dec_blocks,
            num_heads=lam_num_heads,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
            dropout=lam_dropout,
            causal_temporal=lam_causal_temporal
        )
        self.vq_beta = vq_beta
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer

        self.save_hyperparameters()

    def shared_step(self, batch: Dict) -> Tuple:
        outputs = self.vq_lam(batch)
        gt_future_frames = batch["videos"][:, 1:]

        # Compute reconstruction loss
        mse_loss = ((gt_future_frames - outputs["recon"]) ** 2).mean()
        
        # VQ loss (replaces KL loss from VAE)
        vq_loss = outputs["vq_loss"]
        
        # Total loss
        loss = mse_loss + self.vq_beta * vq_loss

        # Compute monitoring measurements
        metrics = [
            ("mse_loss", mse_loss),
            ("vq_loss", vq_loss),
            ("commitment_loss", outputs["commitment_loss"]),
            ("codebook_loss", outputs["codebook_loss"])
        ]
        
        if _HAS_PIQ:
            gt = gt_future_frames.clamp(0, 1).reshape(-1, *gt_future_frames.shape[2:]).permute(0, 3, 1, 2)
            recon = outputs["recon"].clamp(0, 1).reshape(-1, *outputs["recon"].shape[2:]).permute(0, 3, 1, 2)
            with torch.no_grad():
                if torch.isfinite(gt).all() and torch.isfinite(recon).all():
                    psnr = piq.psnr(gt, recon, data_range=1.0).mean()
                    ssim = piq.ssim(gt, recon, data_range=1.0).mean()
                else:
                    # Fallback if anything non-finite slipped through
                    psnr = torch.tensor(0.0, device=loss.device)
                    ssim = torch.tensor(0.0, device=loss.device)
            metrics.extend([("psnr", psnr), ("ssim", ssim)])
        
        return outputs, loss, metrics

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the training loss
        log_dict = {**{"train/loss": loss}, **{f"train/{k}": v for k, v in aux_losses}}
        
        # Add codebook usage statistics
        codebook_stats = self.vq_lam.get_codebook_usage()
        for k, v in codebook_stats.items():
            log_dict[f"train/codebook_{k}"] = v

        self.log_dict(
            log_dict,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False
        )

        if batch_idx % self.log_interval == 0:  # Log images periodically
            self.log_images(batch, outputs, "train")
            
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the validation loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the validation loss
        log_dict = {**{"val/loss": loss}, **{f"val/{k}": v for k, v in aux_losses}}
        
        # Add codebook usage statistics for validation
        codebook_stats = self.vq_lam.get_codebook_usage()
        for k, v in codebook_stats.items():
            log_dict[f"val/codebook_{k}"] = v

        self.log_dict(
            log_dict,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        if batch_idx == 0:  # Log images for first validation batch
            self.log_images(batch, outputs, "val")
            
        return loss

    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the test loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the test loss
        log_dict = {**{"test/loss": loss}, **{f"test/{k}": v for k, v in aux_losses}}
        
        # Add codebook usage statistics for test
        codebook_stats = self.vq_lam.get_codebook_usage()
        for k, v in codebook_stats.items():
            log_dict[f"test/codebook_{k}"] = v

        self.log_dict(
            log_dict,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        self.log_images(batch, outputs, "test")
        return loss

    def log_images(self, batch: Dict, outputs: Dict, split: str) -> None:
        """Log reconstructed images for visual inspection."""
        gt_seq = batch["videos"][0]
        recon_seq = outputs["recon"][0]
        # Sanitize, clamp to [0,1]
        gt_seq = torch.nan_to_num(gt_seq, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1).cpu()
        recon_seq = torch.nan_to_num(recon_seq, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1).cpu()

        recon_seq = torch.cat([gt_seq[:1], recon_seq], dim=0)
        compare_seq = torch.cat([gt_seq, recon_seq], dim=1)
        compare_seq = rearrange(compare_seq * 255, "t h w c -> h (t w) c")
        compare_seq = compare_seq.detach().numpy().astype(np.uint8)
        img_path = path.join(self.log_path, f"{split}_step{self.global_step:06}.png")
        makedirs(path.dirname(img_path), exist_ok=True)
        img = Image.fromarray(compare_seq)
        try:
            img.save(img_path)
        except Exception as e:
            print(f"Failed to save image: {e}")

    def on_train_epoch_end(self) -> None:
        """Reset codebook usage statistics at the end of each epoch."""
        self.vq_lam.reset_codebook_usage()

    def on_test_epoch_end(self) -> None:
        """Perform codebook analysis at the end of testing."""
        # Save indices for analysis
        if hasattr(self.vq_lam, 'indices_record') and self.vq_lam.indices_record is not None:
            torch.save(self.vq_lam.indices_record, "vq_indices_record.pt")
            
        # Perform clustering of discrete codes if kmeans is available
        if _HAS_KMEANS and hasattr(self.vq_lam, 'indices_record') and self.vq_lam.indices_record is not None:
            indices = self.vq_lam.indices_record
            unique_indices = torch.unique(indices)
            if len(unique_indices) > 1:
                # Get embeddings for used codes
                embeddings = self.vq_lam.vq.codebook.weight[unique_indices]
                
                # Cluster the used embeddings
                k = min(8, len(unique_indices))
                try:
                    cluster_ids, cluster_centers = kmeans(
                        X=embeddings,
                        num_clusters=k,
                        distance="euclidean",
                        device=embeddings.device
                    )
                    torch.save(cluster_centers, "vq_cluster_centers.pt")
                    print(f"Saved {k} cluster centers for VQ analysis")
                except Exception as e:
                    print(f"Failed to perform clustering: {e}")

    def on_after_backward(self) -> None:
        # Print occasionally on rank 0
        if (self.global_step % 200 != 0) or (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() != 0):
            return
        unused = [n for n, p in self.named_parameters() if p.requires_grad and p.grad is None]
        if unused:
            print(f"[DDP][unused] step={int(self.global_step)}: {len(unused)} params without grad (showing first 20): {unused[:20]}")
            
    # def configure_optimizers(self) -> Optimizer:
    #     optim = self.optimizer(self.parameters())
    #     return optim

    def configure_optimizers(self) -> Optimizer:
        codebook_params = []
        other_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "vq.codebook" in name or "vq.embedding" in name:
                codebook_params.append(p)
            else:
                other_params.append(p)

        optimizer = AdamW(
            [
                {"params": other_params, "lr": 2.5e-5, "weight_decay": 1e-2},
                {"params": codebook_params, "lr": 1e-4, "weight_decay": 0.0},  # 4x LR, often 0 WD
            ]
        )
        return optimizer