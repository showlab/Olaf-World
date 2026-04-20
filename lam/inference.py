"""Inference interface for the Latent Action Model.

Exposes a frozen encoder that maps videos to latent actions.

Contract:
    Input:  videos tensor of shape (B, T, H, W, C), float32 in [0, 1].
    Output: latent actions of shape (B, T-1, D).

Usage:
    from lam.inference import FrozenLAMEncoder, load_lam_encoder
    enc = load_lam_encoder("checkpoints/lam/lam_vjepa_align.ckpt", variant="align")
    z = enc(videos)  # (B, T-1, D)
"""
from typing import Literal, Optional, Dict, Any

import torch
from torch import Tensor, nn

from lam.modules import LatentActionModel


# Mapping from LightningModule hparam name -> inner LatentActionModel kwarg name.
_HPARAM_TO_MODULE = {
    "image_channels":      "in_dim",
    "lam_model_dim":       "model_dim",
    "lam_latent_dim":      "latent_dim",
    "lam_patch_size":      "patch_size",
    "lam_enc_blocks":      "enc_blocks",
    "lam_dec_blocks":      "dec_blocks",
    "lam_num_heads":       "num_heads",
    "lam_dropout":         "dropout",
    "lam_causal_temporal": "causal_temporal",
}

# Sensible defaults if certain hparams are missing from the checkpoint.
_MODULE_DEFAULTS = {
    "in_dim": 3,
    "dropout": 0.0,
    "causal_temporal": False,
}


def _module_kwargs_from_hparams(hparams: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the LightningModule's saved hparams into LatentActionModel kwargs."""
    out = dict(_MODULE_DEFAULTS)
    for hp_key, mod_key in _HPARAM_TO_MODULE.items():
        if hp_key in hparams:
            out[mod_key] = hparams[hp_key]
    return out


def _extract_submodule_state(state: Dict[str, Tensor], prefix: str) -> Dict[str, Tensor]:
    """Return `state` entries that start with `prefix + '.'`, stripped of the prefix."""
    plen = len(prefix) + 1
    return {k[plen:]: v for k, v in state.items() if k.startswith(prefix + ".")}


class FrozenLAMEncoder(nn.Module):
    """Load a LAM / LAMAlign checkpoint and expose a frozen encoder.

    Args:
        ckpt_path: Path to a Lightning `.ckpt` from a trained LAM variant.
        variant:   "vae"   — baseline VAE LAM checkpoint (from `LAM`).
                   "align" — τ-aligned LAM checkpoint (from `LAMAlign`).
                   Both variants share the same `LatentActionModel` encoder
                   architecture and only differ in the training objective.
        device:    Target device. Defaults to CUDA if available.
    """

    def __init__(
        self,
        ckpt_path: str,
        variant: Literal["vae", "align"] = "align",
        device: Optional[str] = None,
    ):
        super().__init__()
        if variant not in ("vae", "align"):
            raise ValueError(f"Unknown LAM variant: {variant!r}. Expected 'vae' or 'align'.")

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hparams = ckpt.get("hyper_parameters") or ckpt.get("hparams") or {}
        state = ckpt.get("state_dict", ckpt)

        base = LatentActionModel(**_module_kwargs_from_hparams(hparams))
        sub_state = _extract_submodule_state(state, "lam")
        if not sub_state:
            raise RuntimeError(
                f"No weights found under the 'lam.' prefix in {ckpt_path}. "
                f"State-dict keys start with: {sorted({k.split('.')[0] for k in state})[:6]}"
            )
        missing, _ = base.load_state_dict(sub_state, strict=False)

        # LAMAlign checkpoints carry extra teacher/alignment keys that are not
        # part of LatentActionModel. Only warn if core encoder keys are missing.
        core_missing = [k for k in missing if not k.startswith(("align_", "teacher_"))]
        if core_missing:
            preview = core_missing[:8]
            suffix = " ..." if len(core_missing) > 8 else ""
            print(f"[FrozenLAMEncoder] WARNING: missing core keys: {preview}{suffix}")

        base.eval()
        for p in base.parameters():
            p.requires_grad = False
        self.base = base.to(device)
        self.variant = variant
        self.device = device

    @torch.no_grad()
    def forward(self, videos: Tensor) -> Tensor:
        """Encode videos to latent actions.

        Args:
            videos: (B, T, H, W, C) float in [0, 1].

        Returns:
            (B, T-1, D) latent actions.
        """
        x = videos if videos.dtype == torch.float32 else videos.float()
        x = x.to(self.device, non_blocking=True)
        outs = self.base.encode(x)
        z = outs["z_rep"]       # (B, T-1, 1, D)
        return z.squeeze(2)     # (B, T-1, D)


def load_lam_encoder(
    ckpt_path: str,
    variant: Literal["vae", "align"] = "align",
    device: Optional[str] = None,
) -> FrozenLAMEncoder:
    """Convenience constructor used by `world_model/`."""
    return FrozenLAMEncoder(ckpt_path=ckpt_path, variant=variant, device=device)


__all__ = ["FrozenLAMEncoder", "load_lam_encoder"]
