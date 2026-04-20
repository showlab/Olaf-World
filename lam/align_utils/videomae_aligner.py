"""
Frozen VideoMAEv2-Giant teacher for τ-alignment regression.

Requires the ``transformers`` library with the VideoMAEv2 model.
Only needed for LAM *training* — inference with FrozenLAMEncoder does not
use this module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _load_videomae_backbone(model_path: str):
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    wrapper = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
    return wrapper.model


def _preprocess_clip(clip_thwc_uint8: torch.Tensor, img_size: int, device: torch.device):
    clip = clip_thwc_uint8.permute(0, 3, 1, 2).float() / 255.0

    _, _, H, W = clip.shape
    short_side = int(256.0 / 224 * img_size)
    if H < W:
        new_h, new_w = short_side, int(W * short_side / H)
    else:
        new_h, new_w = int(H * short_side / W), short_side
    clip = F.interpolate(clip, size=(new_h, new_w), mode="bilinear", align_corners=False)

    top = (new_h - img_size) // 2
    left = (new_w - img_size) // 2
    clip = clip[:, :, top:top + img_size, left:left + img_size]

    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=clip.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=clip.device).view(1, 3, 1, 1)
    clip = (clip - mean) / std

    x = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)
    return x


def _extract_all_tokens(vit: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = vit.patch_embed(x)
    P = x.shape[1]

    pos_embed = vit.pos_embed[:, :P]
    x = x + pos_embed.expand(x.shape[0], -1, -1).type_as(x).to(x.device).clone().detach()
    x = vit.pos_drop(x)

    for block in vit.blocks:
        x = block(x)

    x = vit.norm(x)
    return x


@torch.no_grad()
def _tau_from_clip_videomae(
    vit: nn.Module,
    clip_thwc_uint8: torch.Tensor,
    device: torch.device,
    img_size: int = 224,
    patch_size: int = 14,
):
    """Compute τ vector from a video clip using deltas + L2-normalization."""
    T = clip_thwc_uint8.shape[0]
    assert T >= 2 and T % 2 == 0, f"need T>=2 and even, got {T}"

    x = _preprocess_clip(clip_thwc_uint8, img_size, device)

    with autocast(False):
        feats = _extract_all_tokens(vit, x)
        _, P, D = feats.shape

        grid = (img_size // patch_size) ** 2
        assert P % grid == 0, f"P={P} not divisible by grid={grid}"
        t_tokens = P // grid
        feats_tsd = feats[0].view(t_tokens, grid, D)

        # Spatial mean → per-frame summary
        s_t = feats_tsd.mean(dim=1)

        # Temporal deltas → action-encoding vector
        vecs = s_t[1:] - s_t[:-1]
        vec = vecs.mean(dim=0)

        # L2-normalize
        tau = F.normalize(vec, dim=-1)

    return tau.cpu().float()


class VideoMAEv2Aligner(nn.Module):
    def __init__(
        self,
        model_path: str,
        img_size: int = 224,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size

        self.backbone = _load_videomae_backbone(model_path)
        self.backbone.to(self.device).eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.patch_size = self.backbone.patch_embed.patch_size[0]
        print(
            f"[VideoMAEv2Aligner] loaded {model_path} "
            f"img={img_size} patch={self.patch_size} dim={self.backbone.embed_dim}"
        )

    @torch.no_grad()
    def compute_tau_batch(self, vids_bthwc: torch.Tensor) -> torch.Tensor:
        B, T, _, _, _ = vids_bthwc.shape
        assert T >= 2 and T % 2 == 0, f"need T>=2 and even, got {T}"
        taus = []
        for i in range(B):
            clip = vids_bthwc[i]
            if clip.dtype != torch.uint8:
                clip = (clip.clamp(0, 1) * 255).to(torch.uint8).cpu()
            else:
                clip = clip.cpu()
            tau_i = _tau_from_clip_videomae(
                self.backbone,
                clip,
                self.device,
                img_size=self.img_size,
                patch_size=self.patch_size,
            )
            taus.append(tau_i)
        return torch.stack(taus, dim=0)
