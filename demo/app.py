"""
SIH26142 — Satellite Super-Resolution — Jury Demo

Real inference, real metrics, real uncertainty map. No placeholders.

Run with:
    streamlit run demo/app.py

Model checkpoint priority (edit CHECKPOINT_PATH below once you've fine-tuned
on Colab): falls back to the generic pretrained weights if no fine-tuned
checkpoint exists yet, so the demo always runs even before training is done.

REDESIGN NOTE: the backend logic below (model loading, run_sr, run_sr_any_size,
the checkpoint-priority fallback, every honesty disclosure about generic vs
fine-tuned weights and MC-Dropout being an approximation) is functionally
the same as the previous version of this file — nothing about what the
system actually does has changed. What changed is presentation: this is
restyled to the same brand as SIH26142_Jury_Report.html (so the two
artifacts read as one submission, not two mismatched demos), restructured
as a single narrative instead of developer-style tabs, tied explicitly to
the problem statement's own asks via a live checklist, and given one real
interactive moment (a drag-to-compare slider) plus one orchestrated reveal
(the metrics counting up) instead of static column dumps. See demo/theme.py
for the shared design system.
"""

import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import load_realesrgan, add_dropout_for_uncertainty          # noqa: E402
from data_pipeline import synthetic_degrade                              # noqa: E402
from evaluate import evaluate_pair, bicubic_upsample                     # noqa: E402
from uncertainty import mc_dropout_predict, normalize_uncertainty_for_display  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import theme  # noqa: E402

SCALE = 4
REPO_ROOT = Path(__file__).parent.parent
PRETRAINED_PATH = REPO_ROOT / "experiments" / "pretrained_models" / "RealESRGAN_x4plus.pth"
FINETUNED_PATH = REPO_ROOT / "checkpoints" / "best.pth"
SAMPLE_DIR = Path(__file__).parent / "sample_tiles"
MAX_SINGLE_PASS_LR_SIDE = 512  # keep single-shot MC-Dropout inference responsive
                                # on CPU; bigger uploads get tiled instead (below)

st.set_page_config(page_title="SIH26142 — Satellite Super-Resolution", layout="centered", page_icon="🛰️")
st.markdown(theme.PAGE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading (cached — only happens once per session) — unchanged logic
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if FINETUNED_PATH.exists():
        model = load_realesrgan(str(FINETUNED_PATH), scale=SCALE, device=device)
        status = "finetuned"
    elif PRETRAINED_PATH.exists():
        model = load_realesrgan(str(PRETRAINED_PATH), scale=SCALE, device=device)
        status = "generic"
    else:
        return None, None, device
    model.eval()
    return model, status, device


@st.cache_resource(show_spinner=False)
def get_sample_tiles():
    return sorted(SAMPLE_DIR.glob("*.tif")) + sorted(SAMPLE_DIR.glob("*.png"))


def to_float01(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    return arr


def run_sr(model, lr_patch: np.ndarray, device: str, n_mc_samples: int = 10):
    """Real inference: mean SR prediction + real MC-dropout uncertainty map."""
    lr_tensor = torch.from_numpy(lr_patch).permute(2, 0, 1).unsqueeze(0).float().to(device)
    model_mc = add_dropout_for_uncertainty(model, p=0.1)
    mean_pred, uncertainty_map = mc_dropout_predict(model_mc, lr_tensor, n_samples=n_mc_samples)
    sr_image = np.clip(mean_pred.transpose(1, 2, 0), 0, 1)
    return sr_image, uncertainty_map


def run_sr_any_size(model, lr_full: np.ndarray, device: str, n_mc_samples: int):
    """Sharpen an LR image of ANY size, not just a small fixed crop.

    Small/typical inputs (<= MAX_SINGLE_PASS_LR_SIDE per side): single-shot
    MC-Dropout inference (mean prediction + real per-pixel uncertainty map).

    Larger uploads: sharpened output comes from tiled full-scene inference
    (a single deterministic pass per tile with overlap-blended stitching) so
    the WHOLE image gets sharpened. Running N MC-Dropout passes over every
    tile of a large image would be too slow for a live CPU demo, so the
    uncertainty map in that case comes from one representative center tile
    only — the returned `tile_box` tells the caller exactly which region of
    the OUTPUT that representative map corresponds to, so it can be overlaid
    honestly on just that region rather than implying full-scene coverage.

    Returns (sr_image, uncertainty_map, tiled, tile_box) where tile_box is
    None when not tiled, else (y0, x0, y1, x1) in OUTPUT pixel coordinates.
    """
    h, w = lr_full.shape[:2]
    if max(h, w) <= MAX_SINGLE_PASS_LR_SIDE:
        sr_image, uncertainty_map = run_sr(model, lr_full, device, n_mc_samples)
        return sr_image, uncertainty_map, False, None

    from inference import run_full_scene
    sr_image = np.clip(
        run_full_scene(model, lr_full, scale=SCALE, tile_size=128, overlap=16, device=device),
        0, 1,
    )
    ch, cw = min(128, h), min(128, w)
    cy, cx = (h - ch) // 2, (w - cw) // 2
    _, uncertainty_map = run_sr(model, lr_full[cy:cy + ch, cx:cx + cw], device, n_mc_samples)
    tile_box = (cy * SCALE, cx * SCALE, (cy + ch) * SCALE, (cx + cw) * SCALE)
    return sr_image, uncertainty_map, True, tile_box


# ---------------------------------------------------------------------------
# Model status (no sidebar — this becomes part of the hero, not an admin panel)
# ---------------------------------------------------------------------------
model, status, device = get_model()

st.markdown(f"""
<div class="sih-kicker">SIH26142 · National Technical Research Organisation (NTRO) · Space Technology</div>
<h1 style="font-size:38px; margin:0 0 12px;">Deep Learning Super-Resolution Mapping</h1>
<p style="color:var(--text-muted); font-size:16px; max-width:60ch; margin:0 0 6px;">
Sharpening free 10&nbsp;m Sentinel-2 imagery toward &lt;4&nbsp;m — while keeping every
pixel's geospatial position and spectral signature honest, not just sharper-looking.
</p>
<div class="sih-hero-meta">
  <span>Base model <b>Real-ESRGAN / RRDBNet</b></span>
  <span>Scale <b>×4</b></span>
  <span>Checkpoint <b>{"fine-tuned" if status == "finetuned" else ("generic pretrained" if status else "not found")}</b></span>
  <span>Device <b>{device}</b></span>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.markdown('<div class="sih-section"></div>', unsafe_allow_html=True)
    st.error("No model weights found. Download RealESRGAN_x4plus.pth into "
             "`experiments/pretrained_models/` (see README) before running this demo.")
    st.stop()

if status == "generic":
    st.markdown(
        '<span class="sih-badge sih-badge-warm">⚠ generic weights — not yet fine-tuned on satellite imagery</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span class="sih-badge sih-badge-good">✓ fine-tuned checkpoint loaded</span>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Section: what this needs to prove (tied directly to the problem statement)
# ---------------------------------------------------------------------------
if "sih_result" not in st.session_state:
    st.session_state.sih_result = None

result = st.session_state.sih_result
has_metrics = result is not None and result.get("metrics") is not None
has_uncertainty = result is not None

def check_item(done: bool, title: str, sub: str) -> str:
    mark = '<div class="sih-check-mark done">✓</div>' if done else '<div class="sih-check-mark pending">•</div>'
    return f"""<div class="sih-check-item">{mark}
      <div class="sih-check-text"><b>{title}</b><span class="sih-check-sub">{sub}</span></div>
    </div>"""

st.markdown('<div class="sih-section"><div class="sih-kicker">WHAT THIS NEEDS TO PROVE</div></div>', unsafe_allow_html=True)
st.markdown('<p class="sih-dek">Four specific, checkable claims — the same ones a jury will ask about.</p>', unsafe_allow_html=True)

checklist_html = '<div class="sih-checklist">'
checklist_html += check_item(
    True, "Real resolution enhancement, ×4",
    "Built into the model architecture (verified against the actual RealESRGAN_x4plus.pth checkpoint structure) — not a claim, a config.",
)
checklist_html += check_item(
    True, "Geospatial consistency preserved",
    "CRS-aware reprojection and a real (ramped, not flat) tile-stitching feather, tested against synthetic CRS-mismatched and per-tile-biased cases before shipping.",
)
if has_metrics:
    m = result["metrics"]
    checklist_html += check_item(
        True, "Spectral consistency, not just pixel fidelity",
        f"SAM {m['SAM_deg']:.2f}° · ERGAS {m['ERGAS']:.1f} on the last run below — most SR demos stop at PSNR/SSIM, which doesn't show this.",
    )
else:
    checklist_html += check_item(
        False, "Spectral consistency, not just pixel fidelity",
        "SAM + ERGAS, computed alongside PSNR/SSIM — run the demo below to compute this live.",
    )
if has_uncertainty:
    checklist_html += check_item(
        True, "Quantified uncertainty, honestly caveated",
        "MC-Dropout confidence map computed on the last run below — a fast approximation, not a calibrated probability (see note below).",
    )
else:
    checklist_html += check_item(
        False, "Quantified uncertainty, honestly caveated",
        "MC-Dropout per-pixel confidence map — run the demo below to compute this live.",
    )
checklist_html += "</div>"
st.markdown(checklist_html, unsafe_allow_html=True)

if status == "generic":
    st.markdown('<div class="sih-section"></div>', unsafe_allow_html=True)
    st.info(
        "An honest finding, not a bug: generic Real-ESRGAN (trained on natural photos) "
        "actually **loses to plain bicubic upsampling** on real satellite content in our "
        "validation. That's expected — it's the core argument for this project. "
        "Domain-specific fine-tuning on real satellite pairs isn't a polish step, "
        "it's the entire point. The demo below shows this comparison live; the numbers "
        "should flip once a fine-tuned checkpoint (`Finetune_On_Colab.ipynb`) is dropped in."
    )

# ---------------------------------------------------------------------------
# Section: try it
# ---------------------------------------------------------------------------
st.markdown('<div class="sih-section"><div class="sih-kicker">TRY IT</div></div>', unsafe_allow_html=True)
st.markdown('<p class="sih-dek">Real inference on real pixels — nothing here is pre-rendered.</p>', unsafe_allow_html=True)

with st.expander("Advanced (for the team)"):
    n_mc_samples = st.slider("MC-Dropout samples", 5, 20, 10,
                              help="More samples = smoother uncertainty map, slower inference.")

mode = st.radio("mode", ["Validated sample (real ground truth)", "Upload your own image"],
                 horizontal=True, label_visibility="collapsed")

lr_input = hr_reference = None
run_clicked = False
run_label = "Run super-resolution"

if mode == "Validated sample (real ground truth)":
    samples = get_sample_tiles()
    if not samples:
        st.warning("No bundled sample tiles found in demo/sample_tiles/")
    else:
        chosen = st.selectbox("Sample scene", [s.name for s in samples], label_visibility="collapsed")
        hr_path = next(s for s in samples if s.name == chosen)
        hr_full = to_float01(Image.open(hr_path))
        h, w = (hr_full.shape[0] // SCALE) * SCALE, (hr_full.shape[1] // SCALE) * SCALE
        hr_full = hr_full[:h, :w]

        patch_size = min(256, h, w)
        cy, cx = (h - patch_size) // 2, (w - patch_size) // 2
        hr_reference = hr_full[cy:cy + patch_size, cx:cx + patch_size]
        lr_input = synthetic_degrade(hr_reference, scale=SCALE, sensor_blur_sigma=1.2, noise_std=0.01)

        st.image(lr_input, caption="Simulated 10 m input (what the model actually starts from)", width=180)
        run_clicked = st.button(run_label, type="primary", key="run_sample")

else:
    sub_mode = st.radio(
        "sub_mode",
        ["It's a high-res image — simulate degradation + show metrics",
         "It's already low-res satellite input — just sharpen it"],
        label_visibility="collapsed",
    )
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif", "tiff"], label_visibility="collapsed")

    if uploaded is not None:
        img = to_float01(Image.open(uploaded))
        if sub_mode.startswith("It's a high-res"):
            h, w = (img.shape[0] // SCALE) * SCALE, (img.shape[1] // SCALE) * SCALE
            img_c = img[:h, :w]
            patch = min(256, h, w)
            cy, cx = (h - patch) // 2, (w - patch) // 2
            hr_reference = img_c[cy:cy + patch, cx:cx + patch]
            lr_input = synthetic_degrade(hr_reference, scale=SCALE, sensor_blur_sigma=1.2, noise_std=0.01)
        else:
            lr_input = img
            hr_reference = None

        st.image(lr_input, caption="Input", width=220)
        run_label = "Sharpen this image"
        run_clicked = st.button(run_label, type="primary", key="run_upload")
    else:
        st.caption("Upload a tile above, or switch to the validated sample for a guaranteed-working example.")

# ---------------------------------------------------------------------------
# Inference (only on click — sliders/reruns below reuse session_state)
# ---------------------------------------------------------------------------
if run_clicked and lr_input is not None:
    t0 = time.time()
    with st.spinner("Running inference on real pixels..."):
        h, w = lr_input.shape[:2]
        if max(h, w) <= MAX_SINGLE_PASS_LR_SIDE:
            sr_image, uncertainty_map = run_sr(model, lr_input, device, n_mc_samples)
            tiled, tile_box = False, None
        else:
            sr_image, uncertainty_map, tiled, tile_box = run_sr_any_size(model, lr_input, device, n_mc_samples)
        bicubic_image = bicubic_upsample(lr_input, SCALE)
    elapsed = time.time() - t0

    metrics = deltas = None
    if hr_reference is not None:
        metrics = evaluate_pair(sr_image, hr_reference, resolution_ratio=SCALE)
        baseline_m = evaluate_pair(bicubic_image, hr_reference, resolution_ratio=SCALE)
        deltas = {k: metrics[k] - baseline_m[k] for k in metrics}

    st.session_state.sih_result = {
        "sr_image": sr_image, "bicubic_image": bicubic_image,
        "hr_reference": hr_reference, "uncertainty_map": normalize_uncertainty_for_display(uncertainty_map),
        "tiled": tiled, "tile_box": tile_box, "elapsed": elapsed,
        "n_mc_samples": n_mc_samples, "metrics": metrics, "deltas": deltas,
        "status": status,
    }
    result = st.session_state.sih_result

# ---------------------------------------------------------------------------
# Results (reads from session_state — safe to rerun via the opacity slider
# below without re-running MC-Dropout inference)
# ---------------------------------------------------------------------------
if result is not None:
    st.markdown('<div class="sih-section"><div class="sih-kicker">RESULT</div></div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="sih-dek">Inference took {result["elapsed"]:.1f}s on {device} '
        f'({result["n_mc_samples"]}× MC-Dropout for the confidence map). '
        f'Drag to compare — this is the measured gap a naive upsample leaves.</p>',
        unsafe_allow_html=True,
    )

    components.html(
        theme.render_compare_slider(result["bicubic_image"], result["sr_image"],
                                     "bicubic", "our model", key="result"),
        height=530,
    )

    if result["metrics"] is not None:
        st.markdown('<div class="sih-section"><div class="sih-kicker">METRICS vs. GROUND TRUTH</div></div>',
                     unsafe_allow_html=True)
        components.html(
            theme.render_metrics_reveal(result["metrics"], result["deltas"], key="metrics"),
            height=175,
        )
    else:
        st.caption("No ground truth for this upload, so no PSNR/SSIM/SAM/ERGAS here — qualitative only.")

    st.markdown('<div class="sih-section"><div class="sih-kicker">UNCERTAINTY</div></div>', unsafe_allow_html=True)
    alpha = st.slider("Overlay opacity", 0, 100, 40, key="overlay_alpha",
                       help="Blends the confidence heatmap onto the output — brighter amber = less confident.") / 100.0

    heat = theme.colorize_uncertainty(result["uncertainty_map"])
    if not result["tiled"]:
        blended = theme.blend_overlay(result["sr_image"], heat, alpha)
        st.image(blended, caption="Model output with confidence overlay", use_container_width=True)
    else:
        y0, x0, y1, x1 = result["tile_box"]
        crop = result["sr_image"][y0:y1, x0:x1]
        blended = theme.blend_overlay(crop, heat, alpha)
        st.image(blended, caption="Confidence overlay — one representative center tile only, "
                                    "not the full scene (full-scene MC-Dropout would be too slow live on CPU)",
                  width=320)

    st.caption(
        "Calibration caveat, stated plainly: MC-Dropout variance is a fast approximation, "
        "not a calibrated probability. The frontier approach is self-supervised Bayesian "
        "risk minimization (Zheng, Dewil & Arias, UAI 2026) — too heavy for a 3-week build, "
        "but worth naming if a judge asks how rigorous this is."
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="sih-footer">
    Training data: real, co-registered Sentinel-2 ↔ SPOT pairs from <b>WorldStrat</b>
    (~10,000 km², CC-BY), multiplied via geometry-safe augmentation (90°-rotation/flip +
    small radiometric jitter — never arbitrary-angle rotation, which would corrupt
    LR/HR pixel alignment for nadir overhead imagery).<br><br>
    Cites RS-ESRGAN (Salgueiro Romero et al., <i>Remote Sensing</i> 2020) and Sen2-RDSR
    (same authors, 2021) as prior art for Sentinel-2-specific super-resolution.<br><br>
    SIH26142 · National Technical Research Organisation (NTRO) · Team submission —
    real inference, no mocked outputs.
    </div>
    """,
    unsafe_allow_html=True,
)
