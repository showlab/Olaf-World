# Quickstart — Zero-Shot Action Transfer

No training required: download the checkpoints and run.

## 1. Download checkpoints

Run all commands from the repo root.

```bash
# LAM + world-model pretrained weights
hf download YuxinJ/Olaf-World \
    --local-dir checkpoints

# SkyReels-V2 I2V backbone
hf download Skywork/SkyReels-V2-I2V-1.3B-540P \
    --local-dir checkpoints/SkyReels-V2-I2V-1.3B-540P

# The diffusers loader expects this exact filename
mv checkpoints/SkyReels-V2-I2V-1.3B-540P/model.safetensors \
   checkpoints/SkyReels-V2-I2V-1.3B-540P/diffusion_pytorch_model.safetensors
```

You should end up with:

```
checkpoints/
  lam/lam_vjepa_align.ckpt                    # frozen LAM encoder
  world_model/pretrain/model.pt               # Stage-1 bidirectional world model
  SkyReels-V2-I2V-1.3B-540P/                  # backbone (config.json, VAE, CLIP, T5, ...)
```

## 2. Run zero-shot action transfer

Transfer latent action sequences from a *reference* video onto a *target* first frame.
Single-GPU inference fits on a 24 GB card.

**Single (reference, target) pair:**

```bash
CUDA_VISIBLE_DEVICES=0 python world_model/inference/action_transfer.py \
    --checkpoint_path   checkpoints/world_model/pretrain/model.pt \
    --lam_ckpt          checkpoints/lam/lam_vjepa_align.ckpt \
    --lam_variant       align \
    --reference_video   assets/ref_videos/0.mp4 \
    --first_frame_image assets/images/0.png \
    --output_folder     outputs/action_transfer \
    --use_ema \
    --save_side_by_side
```

**Batch mode — every reference video against every target image:**

```bash
CUDA_VISIBLE_DEVICES=0 python world_model/inference/action_transfer.py \
    --checkpoint_path   checkpoints/world_model/pretrain/model.pt \
    --lam_ckpt          checkpoints/lam/lam_vjepa_align.ckpt \
    --lam_variant       align \
    --ref_video_dir     assets/ref_videos \
    --target_image_dir  assets/images \
    --output_folder     outputs/action_transfer \
    --use_ema \
    --save_side_by_side
```

Outputs are written as MP4s to `outputs/action_transfer/`. With
`--save_side_by_side`, reference↔generated side-by-side videos are also
saved under `outputs/action_transfer/side_by_side/`.

For the full argument list, see
[world_model/inference/action_transfer.py](../world_model/inference/action_transfer.py).