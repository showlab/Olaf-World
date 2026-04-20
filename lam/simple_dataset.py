import os
import math
import random
from typing import List, Tuple, Optional, Dict, Union

import cv2
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
import numpy as np

# pytorchvideo for robust decoding
from pytorchvideo.data.encoded_video import EncodedVideo

def _list_videos(root: str, exts=(".mp4", ".webm", ".mov", ".mkv", ".avi")) -> List[str]:
    """List video files in root; if none, include videos from immediate subfolders.

    This function preserves backward compatibility: if there are videos directly
    under root, it returns those. If not, it searches one level deep (immediate
    subdirectories) and returns any videos found there.
    """
    try:
        entries = os.listdir(root)
    except FileNotFoundError:
        return []

    # First, look for videos directly in root
    direct_videos = [
        os.path.join(root, f)
        for f in entries
        if isinstance(f, str)
        and f.lower().endswith(exts)
        and os.path.isfile(os.path.join(root, f))
    ]
    if direct_videos:
        return sorted(direct_videos)

    # If none, look into immediate subfolders (one level deep)
    videos: List[str] = []
    for name in entries:
        subpath = os.path.join(root, name)
        if not os.path.isdir(subpath):
            continue
        try:
            subentries = os.listdir(subpath)
        except Exception:
            continue
        for f in subentries:
            if not isinstance(f, str):
                continue
            if f.lower().endswith(exts):
                full = os.path.join(subpath, f)
                if os.path.isfile(full):
                    videos.append(full)

    return sorted(videos)


def _ensure_resolution(resolution: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(resolution, int):
        return resolution, resolution
    assert isinstance(resolution, (list, tuple)) and len(resolution) == 2
    return int(resolution[0]), int(resolution[1])


class SingleVideoDataset(Dataset):
    """
    Single-folder video dataset.

    Features:
    - Uniformly re-samples frames to target_fps (≈30) by temporal skipping / duplication.
    - Supports arbitrary (H, W) (rectangular) resizing (no forced square crop).
    - Optional center crop to preserve aspect ratio (if center_crop=True) before resize.
    - Optional brightness augmentation.
    - Optional spatial divisibility by a patch size (crop bottom/right).
    - skip_initial_seconds: avoid sampling first N seconds (e.g. black intro).
    - Returns tensor in (T,H,W,C) or (T,C,H,W).

    Notes:
    - If original FPS unknown (0), falls back to sequential frames (no skip).
    - If video shorter than required indices, last frame is repeated.
    - For FPS downsampling we select frames at indices round(start + i * (orig_fps/target_fps)).
    - For FPS upsampling (orig_fps < target_fps) frames are duplicated via rounding.
    """

    def __init__(
        self,
        root: str,
        num_frames: int,
        resolution: Union[int, Tuple[int, int]] = (540, 966),
        target_fps: int = 30,
        random_start: bool = True,
        output_format: str = "t h w c",
        center_crop: bool = False,
        color_aug: bool = True,
        brightness_jitter: float = 0.1,
        ensure_divisible_by: Optional[int] = None,
        seed: Optional[int] = None,
        skip_initial_seconds: float = 1.0    # NEW: avoid first ~N seconds (black frames)
    ):
        super().__init__()
        self.files = _list_videos(root)
        if not self.files:
            raise ValueError(f"No video files found in {root}")
        self.num_frames = num_frames
        self.resolution = _ensure_resolution(resolution)  # (H, W)
        self.target_fps = target_fps
        self.random_start = random_start
        self.output_format = output_format
        self.center_crop = center_crop
        self.color_aug = color_aug
        self.brightness_jitter = brightness_jitter
        self.ensure_divisible_by = ensure_divisible_by
        self.skip_initial_seconds = max(0.0, skip_initial_seconds)
        if seed is not None:
            random.seed(seed)
        print(f"SingleVideoDataset: found {len(self.files)} videos in {root}")

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _compute_indices(total_frames: int, orig_fps: float, target_fps: int, num_frames: int, start: int) -> List[int]:
        if orig_fps <= 0:
            # Fallback to contiguous frames
            end = min(total_frames, start + num_frames)
            indices = list(range(start, end))
        else:
            ratio = orig_fps / target_fps
            indices = []
            for i in range(num_frames):
                idx = int(round(start + i * ratio))
                if idx >= total_frames:
                    break
                indices.append(idx)
        # Pad by repeating last
        while len(indices) < num_frames and len(indices) > 0:
            indices.append(indices[-1])
        if not indices:
            indices = [0] * num_frames  # degenerate case
        return indices[:num_frames]

    # Removed black-frame detection (no longer needed with EncodedVideo)

    @staticmethod
    def _center_crop(frame: np.ndarray, target_aspect: float) -> np.ndarray:
        h, w = frame.shape[:2]
        current_aspect = h / w
        if math.isclose(current_aspect, target_aspect, rel_tol=1e-3):
            return frame
        if current_aspect > target_aspect:
            # Too tall: crop height
            new_h = int(round(w * target_aspect))
            top = (h - new_h) // 2
            return frame[top:top+new_h, :]
        else:
            # Too wide: crop width
            new_w = int(round(h / target_aspect))
            left = (w - new_w) // 2
            return frame[:, left:left+new_w]

    def _read_with_encodedvideo(self, path: str) -> Optional[np.ndarray]:
        """Decode frames using pytorchvideo EncodedVideo. Returns (T,H,W,3) uint8 or None on failure."""
        try:
            ev = EncodedVideo.from_path(path, decode_audio=False)
        except Exception:
            return None

        # Determine temporal window
        try:
            duration = float(getattr(ev, "duration", 0.0) or 0.0)
        except Exception:
            duration = 0.0

        span_sec = max(1e-3, self.num_frames / float(self.target_fps))

        # If duration is unknown, try a reasonable window after skip
        if duration > 0:
            min_start = max(0.0, float(self.skip_initial_seconds))
            max_start = max(min_start, duration - span_sec)
            if self.random_start and max_start > min_start:
                start = random.uniform(min_start, max_start)
            else:
                start = min_start
            end = min(duration, start + span_sec)
            # Ensure end > start
            if end <= start:
                end = min(duration, start + max(1e-2, span_sec))
        else:
            # Fallback window when duration not available
            start = float(self.skip_initial_seconds)
            end = start + span_sec

        try:
            clip = ev.get_clip(start, end)
        except Exception:
            return None

        if not isinstance(clip, dict) or "video" not in clip:
            return None

        vid = clip["video"]  # (C, T, H, W), usually uint8
        if vid is None or vid.numel() == 0:
            return None

        # Move to CPU numpy uint8 (T,H,W,C)
        vid = vid.detach().cpu()
        if vid.dtype != torch.uint8:
            vid = vid.clamp(0, 255).to(torch.uint8)
        vid = vid.permute(1, 2, 3, 0)  # (T,H,W,C)

        t = vid.shape[0]
        if t <= 0:
            return None

        # Uniformly sample to exactly self.num_frames
        if t != self.num_frames:
            idxs = np.linspace(0, t - 1, num=self.num_frames)
            idxs = np.clip(np.round(idxs).astype(int), 0, t - 1)
            vid = vid[idxs]

        return vid.numpy()

    def _read_video_frames(self, path: str) -> np.ndarray:
        """Read video frames using EncodedVideo. Returns (T,H,W,3) uint8 numpy array."""
        # Try the requested video first
        arr = self._read_with_encodedvideo(path)
        if isinstance(arr, np.ndarray) and arr.size > 0:
            return arr

        # Warn and resample from other videos if decoding fails
        print(f"[WARN] EncodedVideo failed to decode: {path} — resampling another video")

        # If there's only one file, we cannot resample; raise
        if len(self.files) <= 1:
            raise RuntimeError(f"Failed to decode video via EncodedVideo (no alternatives): {path}")

        tried = {path}
        # Try a few alternative videos to keep the dataloader robust
        max_attempts = min(5, max(1, len(self.files) - 1))
        attempts = 0
        while attempts < max_attempts:
            alt = random.choice(self.files)
            if alt in tried:
                continue
            tried.add(alt)
            attempts += 1
            alt_arr = self._read_with_encodedvideo(alt)
            if isinstance(alt_arr, np.ndarray) and alt_arr.size > 0:
                print(f"[INFO] Replaced failed sample {os.path.basename(path)} with {os.path.basename(alt)}")
                return alt_arr

        # If all attempts fail, raise an error with context
        raise RuntimeError(
            f"Failed to decode via EncodedVideo after {attempts} resample attempts. Last failed: {os.path.basename(alt)}; original: {path}"
        )

    def _spatial_process(self, video: np.ndarray) -> np.ndarray:
        # Optional center crop to target aspect ratio
        target_h, target_w = self.resolution
        target_aspect = target_h / target_w
        if self.center_crop:
            processed = []
            for f in video:
                f2 = self._center_crop(f, target_aspect)
                processed.append(f2)
            video = np.stack(processed, axis=0)

        # Resize (H,W) -> (target_h,target_w)
        if (video.shape[1], video.shape[2]) != (target_h, target_w):
            resized = []
            for f in video:
                r = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                resized.append(r)
            video = np.stack(resized, axis=0)

        # Optional enforce divisibility (crop bottom/right)
        if self.ensure_divisible_by:
            ps = self.ensure_divisible_by
            h, w = video.shape[1:3]
            new_h = (h // ps) * ps
            new_w = (w // ps) * ps
            if new_h != h or new_w != w:
                video = video[:, :new_h, :new_w]

        return video

    def _augment(self, video: Tensor) -> Tensor:
        if not self.color_aug:
            return video
        if self.brightness_jitter > 0:
            delta = (torch.rand(1, dtype=video.dtype) * 2 - 1.0) * self.brightness_jitter
            video = (video + delta).clamp(0, 1)
        return video

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        path = self.files[idx]
        raw_video = self._read_video_frames(path)  # (T,H,W,3) uint8
        raw_video = self._spatial_process(raw_video).astype(np.float32) / 255.0  # [0,1]
        video = torch.from_numpy(raw_video)  # (T,H,W,3)
        video = self._augment(video)

        if self.output_format == "t h w c":
            pass
        elif self.output_format == "t c h w":
            video = video.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(f"Unsupported output_format: {self.output_format}")

        return {"videos": video}


class SingleVideoDataModule(LightningDataModule):
    """
    DataModule for SingleVideoDataset.
    Splits a single folder into train/val by index (random split).
    """

    def __init__(
        self,
        data_root: str,
        num_frames: int,
        resolution: Union[int, Tuple[int, int]] = (540, 966),
        target_fps: int = 30,
        batch_size: int = 4,
        val_split: float = 0.05,
        num_workers: int = 8,
        random_start: bool = True,
        output_format: str = "t h w c",
        center_crop: bool = False,
        color_aug: bool = True,
        brightness_jitter: float = 0.1,
        ensure_divisible_by: Optional[int] = None,
        seed: Optional[int] = 42,
        skip_initial_seconds: float = 1.0   # NEW
    ):
        super().__init__()
        self.save_hyperparameters()
        self._train_dataset = None
        self._val_dataset = None

    def setup(self, stage: Optional[str] = None):
        full = SingleVideoDataset(
            root=self.hparams.data_root,
            num_frames=self.hparams.num_frames,
            resolution=self.hparams.resolution,
            target_fps=self.hparams.target_fps,
            random_start=self.hparams.random_start,
            output_format=self.hparams.output_format,
            center_crop=self.hparams.center_crop,
            color_aug=self.hparams.color_aug if stage != "test" else False,
            brightness_jitter=self.hparams.brightness_jitter,
            ensure_divisible_by=self.hparams.ensure_divisible_by,
            seed=self.hparams.seed,
            skip_initial_seconds=self.hparams.skip_initial_seconds  # pass through
        )
        n = len(full)
        val_n = max(1, int(round(n * self.hparams.val_split)))
        indices = list(range(n))
        random.Random(self.hparams.seed).shuffle(indices)
        val_indices = set(indices[:val_n])
        train_indices = [i for i in indices if i not in val_indices]

        self._train_dataset = torch.utils.data.Subset(full, train_indices)
        self._val_dataset = torch.utils.data.Subset(full, list(val_indices))

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=False
        )

    def test_dataloader(self):
        # Reuse val split for simplicity
        return self.val_dataloader()