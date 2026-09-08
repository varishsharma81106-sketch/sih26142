"""
PyTorch Dataset for LR/HR Sentinel-2 patch pairs.

Two sources this should support (see research brief section 3):
  1. WorldStrat pairs (real, pre-aligned Sentinel-2 <-> SPOT) — preferred
  2. Synthetically degraded pairs (your own AOI, downsampled from a higher-res
     source) — use the FIXED degradation model from research brief section 2,
     not Real-ESRGAN's default natural-image degradation pipeline.

Convention (matches augmentation.py): `patch_size` is the HR training patch
size; the LR counterpart is `patch_size // scale`. Save your raw tiled
patches LARGER than this (e.g. 512x512 HR / 128x128 LR from
data_pipeline.tile_into_patches), so PairedAugmentation has room to take a
different random crop each time __getitem__ is called — this is what turns
a fixed set of saved files into an effectively unbounded stream of distinct
training views (see augmentation.py docstring for why this matters given
limited real data).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from augmentation import PairedAugmentation


class SentinelSRDataset(Dataset):
    def __init__(self, lr_dir: str, hr_dir: str, patch_size: int = 128, scale: int = 4,
                 transform=None, augment: bool = True, seed: int | None = None):
        """
        Args:
            lr_dir: directory of low-res (Sentinel-2-like) patches (.npy),
                    each LARGER than patch_size // scale so random-crop
                    augmentation has room to move
            hr_dir: directory of matching high-res patches, same filename stems
            patch_size: FINAL HR training patch size in pixels (LR counterpart
                    is patch_size // scale) — same convention as augmentation.py
            scale: upsampling factor
            transform: override the default augmentation entirely if you pass
                    your own callable(lr, hr) -> (lr, hr)
            augment: if True (default) and transform is None, wraps a
                    PairedAugmentation automatically — this is what actually
                    multiplies your limited real data. Set False for a
                    validation/held-out split, where you want the same crop
                    every time, not a random one.
        """
        self.lr_paths = sorted(Path(lr_dir).glob("*.npy"))
        self.hr_paths = sorted(Path(hr_dir).glob("*.npy"))
        assert len(self.lr_paths) == len(self.hr_paths), \
            "LR/HR count mismatch — check your pairing/preprocessing step"
        assert len(self.lr_paths) > 0, \
            f"No .npy patches found in {lr_dir} / {hr_dir} — run data_pipeline.py first"
        self.patch_size = patch_size
        self.scale = scale

        if transform is not None:
            self.transform = transform
        elif augment:
            self.transform = PairedAugmentation(patch_size=patch_size, scale=scale, seed=seed)
        else:
            self.transform = None  # validation mode: use saved patches as-is, no randomness

    def __len__(self) -> int:
        return len(self.lr_paths)

    def __getitem__(self, idx: int):
        lr = np.load(self.lr_paths[idx]).astype(np.float32)   # (H, W, C)
        hr = np.load(self.hr_paths[idx]).astype(np.float32)   # (H*scale, W*scale, C)

        # NOTE: if your data_pipeline.py wrote raw Sentinel-2 L2A reflectance
        # values (e.g. 0-10000 range) instead of pre-normalized [0,1] floats,
        # normalize here before augmentation/training — PairedAugmentation's
        # brightness jitter and clip(0,1) assume [0,1] input.

        if self.transform:
            lr, hr = self.transform(lr, hr)
        else:
            # validation mode still needs a fixed-size crop, just not a random one
            lr_size = self.patch_size // self.scale
            lr, hr = lr[:lr_size, :lr_size], hr[:self.patch_size, :self.patch_size]

        lr_tensor = torch.from_numpy(lr.copy()).permute(2, 0, 1)  # CHW
        hr_tensor = torch.from_numpy(hr.copy()).permute(2, 0, 1)

        return lr_tensor, hr_tensor

# Note: paired random-crop logic now lives in augmentation.py
# (PairedAugmentation.random_crop_pair) so there's one source of truth for
# the LR/HR alignment math instead of two slightly-different copies.
