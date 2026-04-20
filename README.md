# Orienting Latent Actions for Video World Modeling

[Yuxin Jiang](https://yuxinn-j.github.io/), [Yuchao Gu](https://ycgu.site/), [Ivor W. Tsang](https://www.a-star.edu.sg/cfar/about-cfar/management/prof-ivor-tsang), and [Mike Zheng Shou](https://cde.nus.edu.sg/ece/staff/shou-zheng-mike/)

[![arXiv](https://img.shields.io/badge/arXiv-2602.10104-b31b1b.svg)](https://arxiv.org/abs/2602.10104)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://showlab.github.io/Olaf-World)
[![Code](https://img.shields.io/badge/Code-Repo-blue)](https://github.com/showlab/Olaf-World)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Model-orange.svg)](https://huggingface.co/YuxinJ/Olaf-World)

<p align="left">
  <img src="https://github.com/showlab/Olaf-World/blob/main/assets/teaser.gif" alt="Teaser" width="100%">
</p>

---

## 📢 Updates
- [04/2026] LAM training and inference code, plus pretrained weights, are released.
- [02/2026] Repo and project page initialized.

---

## 🛠️ Setup

```bash
conda create -n olaf-world python=3.10 -y
conda activate olaf-world

# Install PyTorch matching your CUDA version, then:
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## 🕹️ Quick Start

Zero-shot action transfer: a pretrained LAM and world model transfer motion
(as latent action sequences) from a *reference* video onto a *target* first
frame — no finetuning required. See [docs/quickstart.md](docs/quickstart.md)
for the full guide.

```bash
# 1. Download checkpoints
hf download YuxinJ/Olaf-World --local-dir checkpoints
hf download Skywork/SkyReels-V2-I2V-1.3B-540P \
    --local-dir checkpoints/SkyReels-V2-I2V-1.3B-540P

# 2. Run
python world_model/inference/action_transfer.py \
    --checkpoint_path   checkpoints/world_model/pretrain/model.pt \
    --lam_ckpt          checkpoints/lam/lam_vjepa_align.ckpt \
    --lam_variant       align \
    --reference_video   assets/ref_videos/0.mp4 \
    --first_frame_image assets/images/0.png \
    --output_folder     outputs/action_transfer \
    --use_ema --save_side_by_side
```


## 🏋️ Training

### 0️⃣ Dataset

The current release is trained on [MiraData](https://github.com/mira-space/MiraData)
(3D Rendering and City Walking categories) for both
LAM and world-model pretraining, and on [MIND](https://github.com/CSU-JPG/MIND)
for world-model finetuning and evaluation.

### 1️⃣ [LAM Training w SeqΔ-REPA](docs/lam.md)

### 2️⃣ Olaf-World Pretraining

### 3️⃣ Olaf-World Finetuning

### 4️⃣ Self-Forcing Distillation

## 📌 TODO

- [ ] Release world model pretraining and finetuning pipeline.
- [ ] Release evaluation code.
- [ ] Release distillation pipeline.

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{jiang2026olaf,
  title={Olaf-World: Orienting Latent Actions for Video World Modeling},
  author={Jiang, Yuxin and Gu, Yuchao and Tsang, Ivor W and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2602.10104},
  year={2026}
}
```

## ⭐ Acknowledgements

Olaf-World builds on several excellent open-source projects. Many thanks to
[AdaWorld](https://github.com/Little-Podi/AdaWorld) and
[Self-Forcing](https://github.com/guandeh17/Self-Forcing), and to the
pretrained visual foundation models whose checkpoints and code we rely on:
[V-JEPA 2](https://github.com/facebookresearch/vjepa2),
[VideoMAEv2](https://github.com/opengvlab/videomaev2), and
[SkyReels-V2](https://github.com/skyworkai/skyreels-v2).
