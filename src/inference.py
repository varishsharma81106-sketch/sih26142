"""
Full-scene inference: tile a large Sentinel-2 scene, run SR on each tile,
stitch back together with overlap blending to avoid visible seams at
patch boundaries.
"""

import numpy as np
import torch


def tile_with_overlap(image: np.ndarray, tile_size: int, overlap: int):
    """Yield (y, x, patch) for a scene, with overlapping edges so stitching
    can blend seams instead of showing hard patch boundaries."""
    h, w = image.shape[:2]
    stride = tile_size - overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            yield y_start, x_start, image[y_start:y_end, x_start:x_end]


def _feather_window(size: int, overlap: int) -> np.ndarray:
    """1D ramp: rises 0->1 over the first `overlap` pixels, stays at 1
    through the middle, falls 1->0 over the last `overlap` pixels.

    CORRECTED: the previous version of `run_full_scene` used
    `feather = np.ones(...)` — literally a uniform weight, not a feather at
    all, despite the comment claiming otherwise. A flat weight still
    blends overlap zones (via the weighted average below), but it does so
    abruptly: pixels covered by 2 tiles get straight-averaged while
    adjacent pixels covered by only 1 tile don't, and that step in
    effective smoothing (averaging suppresses per-tile noise) is exactly
    what shows up as a faint seam at tile boundaries in a real full-scene
    mosaic — the artifact this function's docstring says it's avoiding.
    A real ramp makes each tile's influence fade out smoothly through the
    overlap region instead of cutting off at a hard edge.
    """
    if overlap <= 0:
        return np.ones(size, dtype=np.float32)
    # Pixel-center ramp (never touches exactly 0.0 or 1.0): a plain
    # linspace(0, 1, overlap, endpoint=False) hits exactly 0.0 at the very
    # first sample, which is fine at an INTERNAL tile seam (another tile's
    # nonzero weight covers that pixel too, and the weighted average is
    # still well-defined) but is wrong at the true OUTER border of the
    # whole scene, where only one tile covers that pixel at all — a
    # literal 0.0 there means output=0 and weight=0 simultaneously, and the
    # divide-by-zero guard below then reports that border pixel as 0
    # instead of the tile's real prediction. Caught by testing this against
    # a known-flat scene before trusting it, not assumed correct.
    ramp = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
    window = np.ones(size, dtype=np.float32)
    window[:overlap] = ramp
    window[-overlap:] = ramp[::-1]
    return window


@torch.no_grad()
def run_full_scene(model: torch.nn.Module, scene: np.ndarray, scale: int,
                    tile_size: int = 256, overlap: int = 32,
                    device: str = "cuda" if torch.cuda.is_available() else "cpu") -> np.ndarray:
    """Run the SR model over a full scene too large to fit in memory at once."""
    model.eval()
    h, w, c = scene.shape
    out_h, out_w = h * scale, w * scale
    output = np.zeros((out_h, out_w, c), dtype=np.float32)
    weight = np.zeros((out_h, out_w, 1), dtype=np.float32)

    # separable 2D feather = outer product of the same 1D ramp on each axis,
    # in output (post-scale) pixel units
    out_tile, out_overlap = tile_size * scale, overlap * scale
    win_1d = _feather_window(out_tile, out_overlap)
    feather = (win_1d[:, None] * win_1d[None, :])[..., None]  # (out_tile, out_tile, 1)

    for y, x, patch in tile_with_overlap(scene, tile_size, overlap):
        patch_t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float().to(device)
        pred = model(patch_t).squeeze(0).permute(1, 2, 0).cpu().numpy()

        y_out, x_out = y * scale, x * scale
        ph, pw = pred.shape[:2]

        output[y_out:y_out + ph, x_out:x_out + pw] += pred * feather[:ph, :pw]
        weight[y_out:y_out + ph, x_out:x_out + pw] += feather[:ph, :pw]

    weight[weight == 0] = 1.0  # avoid divide-by-zero on any uncovered edge
    return output / weight
