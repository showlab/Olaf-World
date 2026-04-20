# LAM: Latent Action Model

`lam/` is a self-contained module implementing the Latent Action Model used
for unsupervised action discovery from video.

## Variants

| Model | File | Objective |
|-------|------|-----------|
| `LAM` | `lam/model.py` | MSE + β·KL (baseline VAE) |
| `VQLAM` | `lam/vq_model.py` | MSE + VQ bottleneck |
| `LAMAlign` | `lam/model_align.py` | VAE + SeqΔ-REPA (τ-alignment regression) to a frozen visual teacher |

## Training

### Prerequisites (w SeqΔ-REPA only)

`LAMAlign` uses a frozen visual teacher to compute alignment targets.
Two teachers are supported: V-JEPA 2 and VideoMAEv2.

To train with V-JEPA 2 as the teacher:

```bash
cd Olaf-World

# Clone V-JEPA 2 and point the training script at it. The script adds it to
# PYTHONPATH automatically when VJEPA2_ROOT is set.
git clone https://github.com/facebookresearch/vjepa2.git
export VJEPA2_ROOT=vjepa2

# Download the V-JEPA 2 checkpoint.
wget https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt \
    -P checkpoints/ref_encoders/vjepa2

bash scripts/lam/train_lam_align.sh
```

> [!NOTE]
> The SeqΔ-REPA latent-action autoencoder aligned with V-JEPA 2 is available at
> [Hugging Face](https://huggingface.co/YuxinJ/Olaf-World/blob/main/lam/lam_vjepa_align.ckpt).
> A baseline VAE checkpoint is also provided at
> [Hugging Face](https://huggingface.co/YuxinJ/Olaf-World/blob/main/lam/lam.ckpt)
> for representation probing and latent-action analysis.


## Inference interface

`world_model/` (and any downstream consumer) depends on a single module:

```python
from lam.inference import FrozenLAMEncoder, load_lam_encoder

enc = load_lam_encoder(
    "checkpoints/lam/lam_vjepa_align.ckpt",
    variant="align",    # {"vae", "align"}
    device="cuda",
)

# videos: (B, T, H, W, C), float32 in [0, 1]
z = enc(videos)  # -> (B, T-1, D)
```

