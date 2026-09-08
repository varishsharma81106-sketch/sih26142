"""
Data augmentation for LR/HR super-resolution pairs.

Why this matters for THIS project specifically: WorldStrat gives you real,
correctly co-registered pairs, but "real and correctly co-registered" data
for any single AOI is still finite — a few thousand patches at most once
you've tiled it. Augmentation is the standard, accepted way to multiply
that into an effectively much larger training set without needing any
more real imagery.

Rules that matter for SATELLITE imagery specifically (this is not the same
as augmenting natural photos):
  1. Any geometric transform applied to the LR patch MUST be applied
     identically to its paired HR patch, or you break the pixel-alignment
     the whole training signal depends on.
  2. Only use transforms that are physically valid for overhead imagery:
     - 90/180/270 degree rotations: valid (a field looks like a field
       rotated 90 degrees — nadir satellite view has no "up")
     - Horizontal/vertical flips: valid for the same reason
     - Arbitrary-angle rotation: AVOID by default — introduces resampling
       blur into ground truth and complicates the LR/HR pixel mapping.
       If you need it, resample HR first then re-derive LR by repeating
       the degradation step, never rotate LR and HR independently.
  3. Radiometric (brightness/contrast) jitter should be SMALL and applied
     to BOTH LR and HR consistently — you're teaching the model to be
     robust to seasonal/atmospheric variation, not inventing spectral
     signatures that don't correspond to anything physical. Keep jitter
     within +/-10% — this is not natural-photo color augmentation where
     wide jitter is fine; here the spectral values carry meaning the PS
     explicitly asks you to preserve ("spectral consistency").
  4. Random crop is free multiplication if your source scenes are larger
     than your patch size — always do this before other augmentations.

Usage:
    aug = PairedAugmentation(patch_size=128, scale=4)
    lr_aug, hr_aug = aug(lr_patch, hr_patch)   # one random augmented view
    # call it again for a different random view of the SAME source pair —
    # this is how a few hundred real pairs become an effectively much
    # larger training set.
"""

import numpy as np


class PairedAugmentation:
    def __init__(self, patch_size: int = 128, scale: int = 4,
                 brightness_jitter: float = 0.1, seed: int | None = None):
        """
        Args:
            patch_size: HR output patch size in pixels (LR patch = patch_size // scale)
            scale: LR->HR scale factor (e.g. 4 for 10m -> ~2.5m style ratios; use
                   the ratio matching your actual data, e.g. round(10/4)=2 or as configured
                   in your data_pipeline synthetic_degrade call)
            brightness_jitter: max fractional brightness change, applied identically
                   to LR and HR (default +/-10%, kept small deliberately — see module docstring)
        """
        self.patch_size = patch_size
        self.scale = scale
        self.brightness_jitter = brightness_jitter
        self.rng = np.random.default_rng(seed)

    def random_crop_pair(self, lr: np.ndarray, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Crop a random aligned LR/HR patch pair from larger source images.
        This is the main 'free data multiplication' step — every scene can
        yield many distinct patches, not just one."""
        lr_size = self.patch_size // self.scale
        lh, lw = lr.shape[:2]
        if lh < lr_size or lw < lr_size:
            raise ValueError(f"Source LR image ({lh}x{lw}) smaller than required patch ({lr_size}x{lr_size})")

        ly = self.rng.integers(0, lh - lr_size + 1)
        lx = self.rng.integers(0, lw - lr_size + 1)
        hy, hx = ly * self.scale, lx * self.scale

        lr_patch = lr[ly:ly + lr_size, lx:lx + lr_size]
        hr_patch = hr[hy:hy + self.patch_size, hx:hx + self.patch_size]
        return lr_patch, hr_patch

    def _random_dihedral(self, lr: np.ndarray, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One of the 8 dihedral symmetries (identity, 3 rotations, and their
        mirror versions) — all physically valid for nadir overhead imagery,
        applied identically to both LR and HR so alignment is preserved."""
        k = self.rng.integers(0, 4)          # 0/90/180/270 degree rotation
        flip = self.rng.random() < 0.5        # optional mirror

        lr_out = np.rot90(lr, k=k, axes=(0, 1))
        hr_out = np.rot90(hr, k=k, axes=(0, 1))
        if flip:
            lr_out = np.flip(lr_out, axis=1)
            hr_out = np.flip(hr_out, axis=1)
        return np.ascontiguousarray(lr_out), np.ascontiguousarray(hr_out)

    def _radiometric_jitter(self, lr: np.ndarray, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Small, IDENTICAL brightness scaling on both — simulates natural
        atmospheric/seasonal variation without inventing fake spectral
        signatures. Assumes inputs are float in [0, 1]."""
        factor = 1.0 + self.rng.uniform(-self.brightness_jitter, self.brightness_jitter)
        return np.clip(lr * factor, 0, 1), np.clip(hr * factor, 0, 1)

    def __call__(self, lr: np.ndarray, hr: np.ndarray,
                 do_crop: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Apply one random augmented view. Call repeatedly on the same
        source pair to generate many distinct training samples from it."""
        if do_crop:
            lr, hr = self.random_crop_pair(lr, hr)
        lr, hr = self._random_dihedral(lr, hr)
        lr, hr = self._radiometric_jitter(lr, hr)
        return lr, hr

    def effective_multiplier(self, n_source_scenes: int, crops_per_scene: int = 20) -> dict:
        """Rough estimate of effective dataset size after augmentation —
        useful for your idea-PPT slide answering 'is this enough data'."""
        dihedral_variants = 8
        base_patches = n_source_scenes * crops_per_scene
        return {
            "source_scenes": n_source_scenes,
            "crops_per_scene": crops_per_scene,
            "base_patches": base_patches,
            "with_dihedral_augmentation": base_patches * dihedral_variants,
            "note": ("Radiometric jitter is continuous (not counted as discrete "
                     "multiplier) — every epoch sees slightly different brightness "
                     "even for the 'same' crop+rotation combination."),
        }


if __name__ == "__main__":
    # smoke test
    rng = np.random.default_rng(0)
    fake_hr = rng.random((256, 256, 3)).astype(np.float32)
    fake_lr = rng.random((64, 64, 3)).astype(np.float32)

    aug = PairedAugmentation(patch_size=128, scale=4, seed=1)
    lr_out, hr_out = aug(fake_lr, fake_hr)
    print("LR patch:", lr_out.shape, "HR patch:", hr_out.shape)
    print(aug.effective_multiplier(n_source_scenes=40, crops_per_scene=20))
