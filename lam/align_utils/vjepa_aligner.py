"""
Frozen V-JEPA 2 teacher for τ-alignment regression.

Requires an external clone of the V-JEPA 2 repository:

    git clone https://github.com/facebookresearch/vjepa2.git

Then add its ``src/`` directory to your Python path, e.g.:

    export PYTHONPATH=/path/to/vjepa2:$PYTHONPATH

Only needed for LAM *training* — inference with FrozenLAMEncoder does not
use this module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

try:
    from src.models.vision_transformer import vit_giant_xformers_rope
    import src.datasets.utils.video.transforms as video_transforms
    import src.datasets.utils.video.volume_transforms as volume_transforms
except ImportError as exc:
    raise ImportError(
        "VJEPAAligner requires the V-JEPA 2 repository on your PYTHONPATH.\n"
        "  1. git clone https://github.com/facebookresearch/vjepa2.git\n"
        "  2. export PYTHONPATH=/path/to/vjepa2:$PYTHONPATH\n"
        "Only needed for LAM training; inference does not require it."
    ) from exc

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _build_pt_video_transform(img_size: int):
    short_side = int(256.0 / 224 * img_size)
    return video_transforms.Compose([
        video_transforms.Resize(short_side, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(img_size, img_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


def _load_pretrained_vjepa_pt_weights(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")["encoder"]
    ckpt = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ckpt.items()}
    msg = model.load_state_dict(ckpt, strict=False)
    print(f"[VJEPAAligner] loaded {ckpt_path} with msg: {msg}")


@torch.no_grad()
def _tau_from_clip(
    model_pt: nn.Module,
    pt_transform,
    clip_thwc_uint8: torch.Tensor,
    device: torch.device,
):
    """Compute τ vector from a video clip using deltas + L2-normalization."""
    T = clip_thwc_uint8.shape[0]
    assert T >= 2 and T % 2 == 0, f"need T>=2 and even, got {T}"

    clip_tchw = (clip_thwc_uint8.permute(0, 3, 1, 2).float() / 255.0)
    x = pt_transform(clip_tchw).unsqueeze(0).to(device)

    with autocast(False):
        feats = model_pt(x)
        _, P, D = feats.shape

        H = x.shape[-1]
        grid = (H // 16) * (H // 16)
        assert P % grid == 0
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


class VJEPAAligner(nn.Module):
    def __init__(
        self,
        vjepa_weights_path: str,
        img_size: int = 384,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=64).to(self.device).eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        _load_pretrained_vjepa_pt_weights(self.backbone, vjepa_weights_path)
        self.pt_transform = _build_pt_video_transform(img_size)

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
            tau_i = _tau_from_clip(
                self.backbone,
                self.pt_transform,
                clip,
                self.device,
            )
            taus.append(tau_i)
        return torch.stack(taus, dim=0)
