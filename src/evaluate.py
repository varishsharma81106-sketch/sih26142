"""
Evaluation metrics for the SIH26142 super-resolution pipeline.

Reports PSNR, SSIM (pixel/structural fidelity) AND SAM, ERGAS (spectral
fidelity) — the PS explicitly asks for "geospatial and spectral
consistency", and PSNR/SSIM alone don't demonstrate that.

Usage:
    from evaluate import evaluate_pair, bicubic_baseline_comparison
    metrics = evaluate_pair(pred, gt)          # dict of scores for one pair
    report  = bicubic_baseline_comparison(lr, pred, gt)  # model vs baseline
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim
from PIL import Image


def psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio. Higher is better."""
    return float(_psnr(gt, pred, data_range=data_range))


def ssim(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """Structural Similarity Index. 0-1, closer to 1 is better.
    Expects HWC arrays; uses channel_axis for multi-band images."""
    channel_axis = -1 if pred.ndim == 3 else None
    return float(_ssim(gt, pred, data_range=data_range, channel_axis=channel_axis))


def sam(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    """Spectral Angle Mapper, in degrees, averaged over all pixels.
    pred, gt: HxWxB arrays (B = number of spectral bands).
    Near 0 = spectrally faithful; large angle = invented/incorrect spectral signature."""
    assert pred.shape == gt.shape, "pred and gt must have the same shape"
    p = pred.reshape(-1, pred.shape[-1]).astype(np.float64)
    g = gt.reshape(-1, gt.shape[-1]).astype(np.float64)

    dot = np.sum(p * g, axis=1)
    p_norm = np.linalg.norm(p, axis=1)
    g_norm = np.linalg.norm(g, axis=1)

    cos_angle = dot / (p_norm * g_norm + eps)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angles_rad = np.arccos(cos_angle)
    return float(np.degrees(np.mean(angles_rad)))


def ergas(pred: np.ndarray, gt: np.ndarray, resolution_ratio: float, eps: float = 1e-8) -> float:
    """Erreur Relative Globale Adimensionnelle de Synthese.
    resolution_ratio = low_res / high_res, e.g. 10/4 = 2.5 for a 10m->4m task.
    Lower is better (0 = perfect)."""
    assert pred.shape == gt.shape, "pred and gt must have the same shape"
    n_bands = pred.shape[-1] if pred.ndim == 3 else 1
    pred_b = pred.reshape(-1, n_bands).astype(np.float64)
    gt_b = gt.reshape(-1, n_bands).astype(np.float64)

    band_terms = []
    for k in range(n_bands):
        rmse_k = np.sqrt(np.mean((pred_b[:, k] - gt_b[:, k]) ** 2))
        mean_k = np.mean(gt_b[:, k])
        band_terms.append((rmse_k / (mean_k + eps)) ** 2)

    return float(100.0 * resolution_ratio * np.sqrt(np.mean(band_terms)))


def evaluate_pair(pred: np.ndarray, gt: np.ndarray, resolution_ratio: float = 2.5) -> dict:
    """Run all four metrics on one predicted/ground-truth pair.
    Arrays should be float, normalized to [0, 1], shape HxWxB."""
    return {
        "PSNR": psnr(pred, gt),
        "SSIM": ssim(pred, gt),
        "SAM_deg": sam(pred, gt),
        "ERGAS": ergas(pred, gt, resolution_ratio),
    }


def bicubic_upsample(lr: np.ndarray, scale: int) -> np.ndarray:
    """Simple bicubic baseline to prove the model adds value beyond upscaling."""
    img = Image.fromarray((lr * 255).astype(np.uint8))
    h, w = lr.shape[0] * scale, lr.shape[1] * scale
    upsampled = img.resize((w, h), Image.BICUBIC)
    return np.asarray(upsampled).astype(np.float64) / 255.0


def bicubic_baseline_comparison(lr: np.ndarray, pred: np.ndarray, gt: np.ndarray,
                                 scale: int = 4, resolution_ratio: float = 2.5) -> dict:
    """Put on your results slide: model metrics next to the bicubic baseline,
    so it's clear the deep learning approach is adding real value."""
    baseline = bicubic_upsample(lr, scale)
    return {
        "model": evaluate_pair(pred, gt, resolution_ratio),
        "bicubic_baseline": evaluate_pair(baseline, gt, resolution_ratio),
    }


if __name__ == "__main__":
    # quick smoke test with random data
    rng = np.random.default_rng(0)
    gt_img = rng.random((64, 64, 3))
    pred_img = gt_img + rng.normal(0, 0.02, gt_img.shape)
    pred_img = np.clip(pred_img, 0, 1)
    print(evaluate_pair(pred_img, gt_img))
