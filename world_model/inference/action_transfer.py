"""Zero-shot action transfer inference.

Transfers motion from a reference video onto a target first-frame image using
a frozen LAM encoder together with an action-conditioned SkyReels-V2 I2V world
model. No finetuning required.

Example:
    python world_model/inference/action_transfer.py \\
        --checkpoint_path  checkpoints/world_model/pretrain/model.pt \\
        --lam_ckpt         checkpoints/lam/lam_vjepa_align.ckpt \\
        --lam_variant      align \\
        --ref_video_dir    assets/ref_videos \\
        --target_image_dir assets/images \\
        --output_folder    outputs/action_transfer \\
        --use_ema --save_side_by_side
"""
import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

try:
    from torchvision.io import read_video, write_video
except ImportError:
    # torchvision >= 0.22 removed read_video / write_video. Fall back to PyAV
    # (the same ffmpeg backend torchvision previously wrapped) so decoded
    # frames match the original torchvision.io path bit-for-bit.
    import av
    import imageio

    def read_video(path, pts_unit="sec"):
        """Return (video, audio, info) with video as uint8 (T, H, W, 3) RGB."""
        container = av.open(path)
        stream = container.streams.video[0]
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
        container.close()
        if not frames:
            return torch.empty(0, 0, 0, 3, dtype=torch.uint8), torch.empty(0), {}
        return torch.from_numpy(np.stack(frames)), torch.empty(0), {}

    def write_video(path, video, fps):
        """Write a uint8 (T, H, W, 3) RGB tensor as H.264 mp4."""
        if isinstance(video, torch.Tensor):
            video = video.cpu().numpy()
        writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                    quality=8, macro_block_size=1)
        for f in video:
            writer.append_data(f)
        writer.close()

if __package__ is None or __package__ == "":
    _WORLD_MODEL_DIR = Path(__file__).resolve().parents[1]
    _REPO_ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_WORLD_MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(_WORLD_MODEL_DIR))
    if str(_REPO_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT_DIR))

from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb, gpu
from lam.inference import FrozenLAMEncoder
from pipeline import ActionSkyVReelsI2VInferencePipeline
from wan.modules.clip import CLIPModel


def set_seed(seed: int):
    """Seed `random`, `numpy`, `torch`, and `torch.cuda` for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    return _repo_root() / "configs" / "world_model" / "default_config.yaml"


def _collect_files(single_path: Optional[str], dir_path: Optional[str], suffixes: set) -> List[str]:
    """Collect files from an optional single path and/or a directory."""
    out = []
    if single_path:
        out.append(single_path)
    if dir_path:
        p = Path(dir_path)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.is_file() and f.suffix.lower() in suffixes:
                    out.append(str(f))
    return out


def _load_image_bchw(path: str, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Load an image as a [1, 3, H, W] tensor in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    H, W = size_hw
    tfm = transforms.Compose([
        transforms.Resize((H, W),
                          interpolation=transforms.InterpolationMode.BICUBIC,
                          antialias=True),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])
    return tfm(img).unsqueeze(0)


def _prepare_lam_video(video_path: str, num_frames: int, rand_start: bool
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode and resize a reference clip for the LAM encoder.

    Returns:
        vids_btHWC_01: [1, T, 272, 480, 3] float in [0, 1] — LAM input.
        vis_tHWC_u8:   [T, 272, 480, 3] uint8              — for side-by-side viz.
    """
    video, _, _ = read_video(video_path, pts_unit="sec")
    if video.dim() != 4 or video.shape[-1] != 3:
        raise ValueError(f"Unexpected video shape for {video_path}: {tuple(video.shape)}")

    total = video.shape[0]
    if total == 0:
        raise ValueError(f"No frames decoded from {video_path}")

    if total >= num_frames:
        start = int(np.random.randint(0, total - num_frames + 1)) if rand_start else 0
        clip = video[start:start + num_frames]
    else:
        pad = video[-1:].expand(num_frames - total, -1, -1, -1)
        clip = torch.cat([video, pad], dim=0)

    clip_f = clip.float() / 255.0
    clip_nchw = clip_f.permute(0, 3, 1, 2).contiguous()
    clip_res = F.interpolate(clip_nchw, size=(272, 480), mode="bicubic", align_corners=False)
    clip_hwc = clip_res.permute(0, 2, 3, 1).contiguous()
    vids_btHWC_01 = clip_hwc.unsqueeze(0)
    vis_tHWC_u8 = (clip_hwc.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return vids_btHWC_01, vis_tHWC_u8


def process_actions_for_latent_space(
    actions: torch.Tensor, num_latent_frames: int, compression_ratio: int = 4,
) -> torch.Tensor:
    """Group per-pixel-frame actions into per-latent-frame blocks.

    The WAN VAE has temporal compression `r` (= 4). Latent frame 0 broadcasts
    the first action across all r sub-steps; latent frame i (i >= 1) pulls
    actions [1 + (i-1)*r, 1 + i*r).
    """
    assert actions.dim() == 3, f"actions must be [B,T,D], got {actions.shape}"
    B, T, D = actions.shape
    r = int(compression_ratio)
    F_lat = int(num_latent_frames)
    assert F_lat >= 1 and r >= 1
    out = torch.zeros(B, F_lat, r, D, device=actions.device, dtype=actions.dtype)

    out[:, 0, :, :] = actions[:, 0:1, :].expand(B, r, D)

    for i in range(1, F_lat):
        s = 1 + (i - 1) * r
        if s >= T:
            break
        chunk = actions[:, s:min(s + r, T), :]
        n = chunk.shape[1]
        if n < r:
            pad = torch.zeros(B, r - n, D, device=actions.device, dtype=actions.dtype)
            chunk = torch.cat([chunk, pad], dim=1)
        out[:, i, :, :] = chunk
    return out


def _load_generator_checkpoint(pipeline, checkpoint_path: str, use_ema: bool):
    """Load generator (or EMA) weights into `pipeline.generator`.

    Handles checkpoints that wrap weights under `generator` / `generator_ema`,
    and strips FSDP's `._fsdp_wrapped_module` prefix if present.
    """
    print(f"[Load] checkpoint: {checkpoint_path}")
    if checkpoint_path.endswith(".safetensors"):
        raw_sd = load_file(checkpoint_path, device="cpu")
    else:
        raw_sd = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(raw_sd, dict) and ("generator" in raw_sd or "generator_ema" in raw_sd):
        key = "generator_ema" if use_ema and "generator_ema" in raw_sd else "generator"
        ckpt_sd = raw_sd[key]
        print(f"[Load] using sub-key '{key}'")
    else:
        ckpt_sd = raw_sd
        print("[Load] using root state dict")

    cleaned = {}
    skipped = 0
    for k, v in ckpt_sd.items():
        if "flat_param" in k:
            skipped += 1
            continue
        cleaned[k.replace("._fsdp_wrapped_module", "")] = v
    if skipped:
        print(f"[Load] skipped {skipped} flat_param entries")

    missing, unexpected = pipeline.generator.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[Load] missing {len(missing)} params (first 8): {missing[:8]}")
    if unexpected:
        print(f"[Load] unexpected {len(unexpected)} params (first 8): {unexpected[:8]}")
    print("[Load] done (strict=False).")


def main():
    parser = argparse.ArgumentParser("Zero-shot action transfer inference")
    parser.add_argument("--config_path", type=str,
                        default="configs/world_model/action_bi_latent_vjepa_align_config.yml")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="World-model generator checkpoint (.pt or .safetensors).")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--lam_ckpt", type=str, required=True)
    parser.add_argument("--lam_variant", type=str, default="align",
                        choices=["vae", "align"])

    parser.add_argument("--reference_video", type=str, default=None)
    parser.add_argument("--ref_video_dir", type=str, default=None)
    parser.add_argument("--ref_num_frames", type=int, default=97)
    parser.add_argument("--ref_rand_start", action="store_true")
    parser.add_argument("--ref_fps", type=int, default=16)

    parser.add_argument("--first_frame_image", type=str, default=None)
    parser.add_argument("--target_image_dir", type=str, default=None)

    parser.add_argument("--num_output_frames", type=int, default=25)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--save_side_by_side", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Free VRAM {get_cuda_free_memory_gb(gpu)} GB")
    low_memory = get_cuda_free_memory_gb(gpu) < 40

    torch.set_grad_enabled(False)

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load(str(_default_config_path()))
    config = OmegaConf.merge(default_config, config)
    config.causal = False
    if hasattr(config, "model_kwargs"):
        config.model_kwargs.is_causal = False

    pipeline = ActionSkyVReelsI2VInferencePipeline(config, device=device)
    _load_generator_checkpoint(pipeline, args.checkpoint_path, args.use_ema)

    pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)

    clip_wrapper = CLIPModel(
        dtype=torch.bfloat16,
        device=gpu,
        checkpoint_path=config.clip_checkpoint,
        tokenizer_path=config.clip_tokenizer,
    )

    target_images = _collect_files(
        args.first_frame_image, args.target_image_dir,
        {".png", ".jpg", ".jpeg", ".bmp", ".webp"})
    ref_videos = _collect_files(
        args.reference_video, args.ref_video_dir,
        {".mp4", ".avi", ".mov", ".mkv", ".webm"})

    if not target_images:
        raise ValueError("No target images found. Provide --first_frame_image and/or --target_image_dir")
    if not ref_videos:
        raise ValueError("No reference video found. Provide --reference_video and/or --ref_video_dir")

    os.makedirs(args.output_folder, exist_ok=True)

    lam = FrozenLAMEncoder(ckpt_path=args.lam_ckpt, variant=args.lam_variant, device=str(device))
    compression_ratio = 4

    def run_for_one_motion_source(
        vids_btHWC_01: torch.Tensor,
        vis_tHWC_u8: Optional[torch.Tensor],
        ref_tag: str,
    ):
        lam_actions = lam(vids_btHWC_01)
        T_ref = vids_btHWC_01.shape[1]
        B_lam, Tm1, D = lam_actions.shape
        assert Tm1 == T_ref - 1, (Tm1, T_ref)

        pad_last = torch.zeros(B_lam, 1, D, device=lam_actions.device, dtype=lam_actions.dtype)
        lam_actions_full = torch.cat([lam_actions, pad_last], dim=1)

        num_latent_frames = int(args.num_output_frames)
        ideal_num_latent_frames = 1 + int(np.ceil((T_ref - 1) / compression_ratio))
        if ideal_num_latent_frames != num_latent_frames:
            print(
                f"[Adjust] num_output_frames: {num_latent_frames} -> {ideal_num_latent_frames} "
                f"to match T_ref={T_ref} with compression_ratio={compression_ratio}"
            )
            num_latent_frames = ideal_num_latent_frames

        F_out = (num_latent_frames - 1) * compression_ratio + 1

        action_grps = process_actions_for_latent_space(
            lam_actions_full.to(torch.bfloat16),
            num_latent_frames=num_latent_frames,
            compression_ratio=compression_ratio,
        )

        for img_idx, img_path in enumerate(target_images):
            img_stem = Path(img_path).stem
            print(
                f"[Run] motion '{ref_tag}' ({T_ref}f) -> target '{img_stem}' "
                f"[pair {img_idx + 1}/{len(target_images)}]"
            )

            image_bchw = _load_image_bchw(img_path, (config.height, config.width)).to(
                device=device, dtype=torch.bfloat16
            )
            B_img, _, H, W = image_bchw.shape
            img_1f = image_bchw.unsqueeze(2)
            pad = torch.zeros(B_img, 3, F_out - 1, H, W, device=device, dtype=torch.bfloat16)
            vid = torch.cat([img_1f, pad], dim=2)

            img_lat_bfchw = pipeline.vae.encode_to_latent(vid).to(
                device=device, dtype=torch.bfloat16
            )

            save_side_by_side = args.save_side_by_side and (vis_tHWC_u8 is not None)
            ref_stem = ref_tag
            model_tag = "ema" if args.use_ema else "regular"

            pending_seeds = []
            for seed_idx in range(args.num_samples):
                if save_side_by_side:
                    sbs_root = os.path.join(args.output_folder, "side_by_side")
                    sbs_dir = os.path.join(sbs_root, ref_stem)
                    sbs_fname = f"{ref_stem}__{img_stem}__seed{seed_idx}_{model_tag}_concat.mp4"
                    final_path = os.path.join(sbs_dir, sbs_fname)
                    pending_seeds.append({
                        "seed_idx": seed_idx, "path": final_path,
                        "mode": "sbs", "dir": sbs_dir,
                    })
                else:
                    out_fname = f"{ref_stem}__{img_stem}__seed{seed_idx}_{model_tag}_bidirectional.mp4"
                    final_path = os.path.join(args.output_folder, out_fname)
                    pending_seeds.append({
                        "seed_idx": seed_idx, "path": final_path, "mode": "gen",
                    })

            existing = [s for s in pending_seeds if os.path.exists(s["path"])]
            for s in existing:
                print(f"  [Skip] exists, skipping seed {s['seed_idx']}: {s['path']}")
            pending_seeds = [s for s in pending_seeds if not os.path.exists(s["path"])]

            if not pending_seeds:
                print(f"  [Skip] motion '{ref_tag}' -> target '{img_stem}' already generated for all seeds.")
                continue

            num_to_generate = len(pending_seeds)
            img_cond = img_lat_bfchw.repeat(num_to_generate, 1, 1, 1, 1).permute(0, 2, 1, 3, 4)
            initial_latent = img_cond[:, :, 0:1, :, :].permute(0, 2, 1, 3, 4)

            clip_feats = clip_wrapper.visual(img_1f.to(device=device, dtype=torch.bfloat16))
            clip_feats = clip_feats.repeat(num_to_generate, 1, 1)

            prompts = [""] * num_to_generate

            sampled_noise = torch.randn(
                [num_to_generate, num_latent_frames, 16, 68, 120],
                device=device, dtype=torch.bfloat16,
            )
            video, _ = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                first_latent=initial_latent,
                init_latent=img_cond,
                clip_feats=clip_feats,
                actions=action_grps,
                return_latents=True,
            )

            video_bthwc = rearrange(video, "b t c h w -> b t h w c").cpu()
            out = 255.0 * video_bthwc

            for gen_idx, spec in enumerate(pending_seeds):
                gen_hwcv = out[gen_idx].to(torch.uint8)
                T_gen = gen_hwcv.shape[0]

                if spec["mode"] == "sbs":
                    vis_hwcv = vis_tHWC_u8
                    if vis_hwcv.shape[0] != T_gen:
                        idxs = torch.linspace(0, vis_hwcv.shape[0] - 1, steps=T_gen).round().long()
                        ref_for_concat = vis_hwcv[idxs]
                    else:
                        ref_for_concat = vis_hwcv

                    if (ref_for_concat.shape[1] != gen_hwcv.shape[1]) or \
                       (ref_for_concat.shape[2] != gen_hwcv.shape[2]):
                        ref_nchw = ref_for_concat.permute(0, 3, 1, 2).float()
                        gen_h, gen_w = gen_hwcv.shape[1], gen_hwcv.shape[2]
                        ref_res = F.interpolate(ref_nchw, size=(gen_h, gen_w),
                                                mode="bilinear", align_corners=False)
                        ref_for_concat = ref_res.clamp(0, 255).byte().permute(0, 2, 3, 1)

                    side_by_side = torch.cat([ref_for_concat, gen_hwcv], dim=2)

                    os.makedirs(spec["dir"], exist_ok=True)
                    write_video(spec["path"], side_by_side.cpu(), fps=args.ref_fps)
                    print(f"  [SBS] {spec['path']}")
                else:
                    write_video(spec["path"], gen_hwcv.cpu(), fps=args.ref_fps)
                    print(f"  [Save] {spec['path']}")

    for v_idx, ref_path in enumerate(ref_videos):
        ref_stem = Path(ref_path).stem
        print(f"[Motion] {v_idx + 1}/{len(ref_videos)}: {ref_path}")

        vids_btHWC_01, vis_tHWC_u8 = _prepare_lam_video(
            ref_path,
            num_frames=args.ref_num_frames,
            rand_start=args.ref_rand_start,
        )

        if not args.save_side_by_side:
            ref_vis_out_path = os.path.join(
                args.output_folder,
                f"ref_{ref_stem}_processed_{vis_tHWC_u8.shape[0]}f.mp4",
            )
            write_video(ref_vis_out_path, vis_tHWC_u8.cpu(), fps=args.ref_fps)
            print(f"[Ref] saved processed reference clip: {ref_vis_out_path}")

        run_for_one_motion_source(vids_btHWC_01, vis_tHWC_u8, ref_stem)


if __name__ == "__main__":
    main()
