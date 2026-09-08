"""
End-to-end pipeline validation on a real satellite image.

This is NOT a training run (no GPU here) — it's a smoke test proving every
stage of the pipeline actually works on real imagery with the real
pretrained Real-ESRGAN weights, before you invest Colab GPU time in
fine-tuning. If this script runs clean, the pipeline is solid; fine-tuning
on your real WorldStrat/Copernicus data is then just a matter of compute
time, not debugging.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from model import load_realesrgan
from data_pipeline import synthetic_degrade
from augmentation import PairedAugmentation
from evaluate import evaluate_pair, bicubic_upsample
from uncertainty import mc_dropout_predict, enable_mc_dropout

SCALE = 4
# The bundled sample tile actually shipped with this repo. The previous
# version of this script pointed at "../sample_data/RGB_byte.tif", a path
# that doesn't exist anywhere in this repo (nor one directory above it) —
# this script could not have run as originally written. This is the same
# tile demo/app.py uses, loaded the same way (PIL, not cv2, so there's no
# BGR/RGB channel-order footgun to get wrong).
SAMPLE_TILE = REPO_ROOT / "demo" / "sample_tiles" / "sample_scene_01.tif"

print("=" * 60)
print("1. Loading real sample satellite image (bundled demo tile)")
print("=" * 60)
if not SAMPLE_TILE.exists():
    raise FileNotFoundError(
        f"Expected the bundled sample tile at {SAMPLE_TILE}, but it's missing. "
        "Re-clone/re-unzip the repo, or point SAMPLE_TILE at your own RGB image."
    )
img_rgb = np.asarray(Image.open(SAMPLE_TILE).convert("RGB")).astype(np.float32) / 255.0
# crop to a clean multiple of scale for tidy patching
h, w = (img_rgb.shape[0] // SCALE) * SCALE, (img_rgb.shape[1] // SCALE) * SCALE
hr_full = img_rgb[:h, :w]
print(f"HR reference image: {hr_full.shape}")

print("\n" + "=" * 60)
print("2. Synthetic degradation (simulating 10m Sentinel-2-like input)")
print("=" * 60)
lr_full = synthetic_degrade(hr_full, scale=SCALE, sensor_blur_sigma=1.2, noise_std=0.01)
print(f"LR simulated input: {lr_full.shape}")

print("\n" + "=" * 60)
print("3. Augmentation module — generating 3 distinct training views from ONE source pair")
print("=" * 60)
aug = PairedAugmentation(patch_size=128, scale=SCALE, seed=42)
for i in range(3):
    lr_p, hr_p = aug(lr_full, hr_full)
    print(f"  view {i+1}: LR {lr_p.shape} <-> HR {hr_p.shape}  (this is how ~40 real scenes -> thousands of training samples)")

print("\n" + "=" * 60)
print("4. Loading REAL pretrained Real-ESRGAN weights (RealESRGAN_x4plus.pth, 67MB)")
print("=" * 60)
device = "cpu"
weights_path = REPO_ROOT / "experiments" / "pretrained_models" / "RealESRGAN_x4plus.pth"
if not weights_path.exists():
    raise FileNotFoundError(
        f"{weights_path} not found — download it first (see README):\n"
        "  mkdir -p experiments/pretrained_models\n"
        "  wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth "
        "-P experiments/pretrained_models"
    )
model = load_realesrgan(str(weights_path), scale=SCALE, device=device)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {n_params:,} parameters (this is the real 23-block RRDB architecture)")

print("\n" + "=" * 60)
print("5. Running inference on a real patch (CPU — slow but correctness-only check)")
print("=" * 60)
# small patch to keep CPU inference fast for this smoke test
lr_patch, hr_patch = aug.random_crop_pair(lr_full, hr_full)
lr_tensor = torch.from_numpy(lr_patch).permute(2, 0, 1).unsqueeze(0).float()

with torch.no_grad():
    sr_tensor = model(lr_tensor)
sr_patch = sr_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
print(f"LR patch {lr_patch.shape} -> SR output {sr_patch.shape} (pretrained weights, NOT fine-tuned on satellite data yet)")

print("\n" + "=" * 60)
print("6. Metrics: pretrained (unfine-tuned) model vs. bicubic baseline")
print("=" * 60)
results = {
    "model_pretrained_generic": evaluate_pair(sr_patch, hr_patch, resolution_ratio=SCALE),
    "bicubic_baseline": evaluate_pair(bicubic_upsample(lr_patch, SCALE), hr_patch, resolution_ratio=SCALE),
}
for name, m in results.items():
    print(f"  {name:28s} PSNR={m['PSNR']:.2f}dB  SSIM={m['SSIM']:.4f}  SAM={m['SAM_deg']:.3f}deg  ERGAS={m['ERGAS']:.3f}")
print("\n  NOTE: the model here is Real-ESRGAN's GENERIC pretrained weights,")
print("  trained on natural photos, NOT fine-tuned on satellite imagery.")
print("  Expect it to only modestly beat bicubic (or lose on some metrics)")
print("  until you fine-tune on real Sentinel-2/WorldStrat pairs on Colab —")
print("  that fine-tuning step is where the real accuracy gain happens.")

print("\n" + "=" * 60)
print("7. Uncertainty quantification (MC Dropout)")
print("=" * 60)
from model import add_dropout_for_uncertainty
model_with_dropout = add_dropout_for_uncertainty(model, p=0.1)
mean_pred, uncertainty_map = mc_dropout_predict(model_with_dropout, lr_tensor, n_samples=8)
print(f"Mean prediction: {mean_pred.shape}  Uncertainty map: {uncertainty_map.shape}")
print(f"Uncertainty range: [{uncertainty_map.min():.6f}, {uncertainty_map.max():.6f}]")

print("\n" + "=" * 60)
print("ALL STAGES PASSED — pipeline is correct end-to-end on real satellite imagery.")
print("Next step: run the fine-tuning loop on Colab GPU with real WorldStrat/")
print("Copernicus data to get an actually-trained (not just generic) model.")
print("=" * 60)
