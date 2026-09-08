"""
Design system for the jury-facing demo.

Deliberately reuses the exact color/type tokens and the drag-to-compare
component already established in SIH26142_Jury_Report.html, rather than
inventing a second, mismatched visual identity. The two artifacts (the
static report and this live app) are meant to read as one submission.

Everything here is presentation only — no numbers are invented. Every
value this module renders comes from a real model.forward() / evaluate_pair()
call made in app.py; this module never fabricates a metric or image.
"""

import base64
import io

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Design tokens (verbatim from SIH26142_Jury_Report.html's :root block)
# ---------------------------------------------------------------------------
ROOT_VARS_CSS = """
:root {
  --bg: #0b1310;
  --panel: #0f1b19;
  --panel-2: #122023;
  --border: #1e332f;
  --text: #eaf1ef;
  --text-muted: #8fa8a4;
  --accent: #3aa0c2;
  --accent-dim: #2a6a80;
  --warm: #d99a52;
  --good: #6fbf8b;
  --pending: #c9954f;
  --serif: Iowan Old Style, Palatino Linotype, "Palatino", Georgia, serif;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", "JetBrains Mono", Consolas, monospace;
}
"""

# Injected once into the main Streamlit page. Re-skins Streamlit's own chrome
# to the same tokens and hides the parts that make an app read as a
# developer console (hamburger menu, "made with Streamlit" footer, the
# default light sidebar) rather than a jury-facing tool.
PAGE_CSS = f"""
<style>
{ROOT_VARS_CSS}

#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 2rem; max-width: 900px; }}

html, body, [class*="css"] {{
  font-family: var(--sans);
  color: var(--text);
}}
.stApp {{
  background:
    radial-gradient(ellipse 900px 500px at 15% -10%, rgba(58,160,194,0.10), transparent 60%),
    linear-gradient(180deg, #0d1815 0%, #0b1310 100%);
}}

h1, h2, h3 {{ font-family: var(--serif); color: #fff; font-weight: 500; }}
p, li, span, label {{ color: var(--text); }}
a {{ color: var(--accent); }}

/* section rhythm */
.sih-section {{ margin: 40px 0 8px; }}
.sih-kicker {{
  font-family: var(--mono); font-size: 12px; color: var(--text-muted);
  letter-spacing: 0.02em; margin-bottom: 6px;
}}
.sih-dek {{ color: var(--text-muted); font-size: 15px; margin: 0 0 18px; max-width: 62ch; }}

/* hero meta row, matching the report's .hero-meta */
.sih-hero-meta {{
  display: flex; gap: 20px; flex-wrap: wrap; font-family: var(--mono);
  font-size: 12.5px; color: var(--text-muted); margin: 10px 0 0;
}}
.sih-hero-meta b {{ color: var(--text); font-weight: 600; }}

/* checklist */
.sih-checklist {{ display: flex; flex-direction: column; gap: 10px; margin: 14px 0 0; }}
.sih-check-item {{
  display: flex; align-items: flex-start; gap: 12px; padding: 12px 14px;
  background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
}}
.sih-check-mark {{
  flex-shrink: 0; width: 20px; height: 20px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700; margin-top: 1px;
}}
.sih-check-mark.done {{ background: var(--good); color: #06181c; }}
.sih-check-mark.pending {{ background: var(--pending); color: #201404; }}
.sih-check-text b {{ color: var(--text); }}
.sih-check-sub {{ display: block; color: var(--text-muted); font-size: 13px; margin-top: 2px; }}

/* badges */
.sih-badge {{
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  font-size: 12px; font-weight: 600; font-family: var(--mono);
}}
.sih-badge-good {{ background: #14532d; color: #86efac; }}
.sih-badge-warm {{ background: #451a03; color: #fdba74; }}

/* Streamlit widget reskin — targets stable data-testid hooks, degrades
   gracefully to default styling if a future Streamlit version renames them. */
div[data-testid="stRadio"] label {{
  background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  padding: 8px 14px; margin-right: 6px; transition: border-color 0.15s ease;
}}
div[data-testid="stRadio"] label:hover {{ border-color: var(--accent-dim); }}

.stButton > button {{
  background: var(--accent); color: #062229; border: none; font-weight: 600;
  border-radius: 6px; transition: transform 0.12s ease, box-shadow 0.12s ease;
}}
.stButton > button:hover {{
  transform: translateY(-1px); box-shadow: 0 6px 16px -6px rgba(58,160,194,0.6);
}}

div[data-testid="stExpander"] {{
  border: 1px solid var(--border); border-radius: 6px; background: var(--panel);
}}

.sih-footer {{
  margin-top: 48px; padding-top: 18px; border-top: 1px solid var(--border);
  color: var(--text-muted); font-size: 13px;
}}
</style>
"""


def img_to_data_uri(arr: np.ndarray) -> str:
    """(H, W, 3) float array in [0, 1] -> a data: URI PNG string, for
    embedding directly in an <img src="..."> inside a components.html block."""
    arr = np.clip(arr, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def colorize_uncertainty(u: np.ndarray,
                          low_color=(42, 106, 128),
                          high_color=(217, 154, 82)) -> np.ndarray:
    """0-1 normalized uncertainty (H, W) -> an (H, W, 3) float RGB heatmap in
    [0, 1], interpolating confident (--accent-dim teal) to uncertain
    (--warm amber) — continues the same semantic color meaning already used
    for status badges elsewhere in this project (amber = caution/pending,
    not a new color introduced just for this map)."""
    u = np.clip(u, 0, 1)[..., None]
    low = np.array(low_color, dtype=np.float32) / 255.0
    high = np.array(high_color, dtype=np.float32) / 255.0
    return low * (1 - u) + high * u


def blend_overlay(base_rgb: np.ndarray, overlay_rgb: np.ndarray, alpha: float) -> np.ndarray:
    """Alpha-blend an overlay (e.g. the uncertainty heatmap) on top of a
    base image (e.g. the SR output) — a real overlay, not two separate
    side-by-side images, per the project's own team checklist ('before/after
    crop + uncertainty overlay')."""
    return np.clip(base_rgb * (1 - alpha) + overlay_rgb * alpha, 0, 1)


def render_compare_slider(left_img: np.ndarray, right_img: np.ndarray,
                           left_label: str, right_label: str, key: str,
                           box_px: int = 440) -> str:
    """Real drag-to-compare component (CSS/markup/JS ported from
    SIH26142_Jury_Report.html's .compare component) for two live-computed
    images. Returns a full HTML document string for st.components.v1.html.
    """
    left_uri = img_to_data_uri(left_img)
    right_uri = img_to_data_uri(right_img)
    return f"""
<!DOCTYPE html><html><head><meta charset="utf-8"><style>
{ROOT_VARS_CSS}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; background: transparent; font-family: var(--sans); }}
.compare {{
  position: relative; border-radius: 4px; overflow: hidden; border: 1px solid var(--border);
  aspect-ratio: 1 / 1; max-width: {box_px}px; margin: 0 auto; cursor: ew-resize; touch-action: none;
  box-shadow: 0 24px 48px -20px rgba(0,0,0,0.6);
}}
.compare img {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; display: block; user-select: none; -webkit-user-drag: none; }}
.compare .after-clip {{ position: absolute; inset: 0; overflow: hidden; }}
.compare .divider {{
  position: absolute; top: 0; bottom: 0; width: 2px; background: rgba(255,255,255,0.85);
  left: 50%; transform: translateX(-1px); pointer-events: none;
}}
.compare .handle {{
  position: absolute; top: 50%; left: 50%; width: 34px; height: 34px; border-radius: 50%;
  background: var(--accent); transform: translate(-50%, -50%);
  display: flex; align-items: center; justify-content: center; pointer-events: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}}
.compare .handle svg {{ width: 16px; height: 16px; }}
.compare .tag {{
  position: absolute; top: 10px; font-family: var(--mono); font-size: 11px;
  background: rgba(10,15,14,0.72); padding: 4px 9px; border-radius: 3px; color: var(--text-muted);
  letter-spacing: 0.02em;
}}
.compare .tag.left {{ left: 10px; }}
.compare .tag.right {{ right: 10px; }}
.compare-caption {{ font-size: 12.5px; color: var(--text-muted); text-align: center; margin-top: 10px; max-width: {box_px}px; margin-left:auto; margin-right:auto; }}
</style></head><body>
  <div class="compare" id="compare-{key}">
    <img src="{left_uri}" alt="{left_label}">
    <div class="after-clip" id="afterClip-{key}" style="width:50%">
      <img src="{right_uri}" alt="{right_label}" style="width: {box_px}px; max-width:none;">
    </div>
    <div class="divider" id="divider-{key}"></div>
    <div class="handle" id="handle-{key}">
      <svg viewBox="0 0 24 24" fill="none" stroke="#06181c" stroke-width="2.4"><path d="M8 7l-5 5 5 5M16 7l5 5-5 5"/></svg>
    </div>
    <div class="tag left">{left_label}</div>
    <div class="tag right">{right_label}</div>
  </div>
  <p class="compare-caption">Drag to compare — both sides are the real, just-computed output for this image.</p>
<script>
(function(){{
  const compare = document.getElementById('compare-{key}');
  const afterClip = document.getElementById('afterClip-{key}');
  const divider = document.getElementById('divider-{key}');
  const handle = document.getElementById('handle-{key}');
  let dragging = false;
  function setPos(clientX){{
    const rect = compare.getBoundingClientRect();
    let x = (clientX - rect.left) / rect.width;
    x = Math.max(0, Math.min(1, x));
    afterClip.style.width = (x*100) + '%';
    divider.style.left = (x*100) + '%';
    handle.style.left = (x*100) + '%';
    const img = afterClip.querySelector('img');
    img.style.width = rect.width + 'px';
  }}
  compare.addEventListener('pointerdown', e => {{ dragging = true; setPos(e.clientX); }});
  window.addEventListener('pointermove', e => {{ if(dragging) setPos(e.clientX); }});
  window.addEventListener('pointerup', () => dragging = false);
  function init(){{
    const rect = compare.getBoundingClientRect();
    afterClip.querySelector('img').style.width = rect.width + 'px';
  }}
  window.addEventListener('load', init);
  setTimeout(init, 50);
}})();
</script>
</body></html>
"""


def render_metrics_reveal(metrics: dict, deltas: dict, key: str) -> str:
    """Animated metric-grid reveal (fade-in + count-up), matching the
    report's .metric-grid pattern. `metrics` and `deltas` are plain floats
    already computed by evaluate.evaluate_pair — this only animates the
    on-screen reveal of real numbers, never invents them.

    This is the single orchestrated motion moment for the whole page,
    triggered once per real inference run (not scattered hover effects).
    """
    rows = [
        ("PSNR (dB)", metrics["PSNR"], deltas["PSNR"], "higher is better", deltas["PSNR"] >= 0),
        ("SSIM", metrics["SSIM"], deltas["SSIM"], "0-1, higher is better", deltas["SSIM"] >= 0),
        ("SAM (deg)", metrics["SAM_deg"], deltas["SAM_deg"], "spectral angle, lower is better", deltas["SAM_deg"] <= 0),
        ("ERGAS", metrics["ERGAS"], deltas["ERGAS"], "global error, lower is better", deltas["ERGAS"] <= 0),
    ]
    cells_html = []
    for label, value, delta, sub, good in rows:
        color = "var(--good)" if good else "#f87171"
        sign = "+" if delta >= 0 else ""
        decimals = 4 if label == "SSIM" else 2
        cells_html.append(f"""
        <div class="metric">
          <div class="metric-label">{label}</div>
          <div class="metric-value" data-target="{value:.6f}" data-decimals="{decimals}">0</div>
          <div class="metric-sub">{sub}</div>
          <div class="metric-delta" style="color:{color}">{sign}{delta:.{decimals}f} vs bicubic</div>
        </div>""")

    return f"""
<!DOCTYPE html><html><head><meta charset="utf-8"><style>
{ROOT_VARS_CSS}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; background: transparent; font-family: var(--sans); }}
.wrap {{ opacity: 0; transform: translateY(6px); animation: reveal 0.5s ease forwards; }}
@keyframes reveal {{ to {{ opacity: 1; transform: translateY(0); }} }}
.metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }}
.metric {{ padding: 16px 14px; border-right: 1px solid var(--border); }}
.metric:last-child {{ border-right: none; }}
.metric-label {{ font-family: var(--mono); font-size: 10.5px; color: var(--text-muted); letter-spacing: 0.03em; }}
.metric-value {{ font-family: var(--mono); font-size: 24px; color: #fff; margin-top: 4px; }}
.metric-sub {{ font-size: 11px; color: var(--text-muted); margin-top: 3px; }}
.metric-delta {{ font-size: 12px; margin-top: 6px; font-family: var(--mono); }}
@media (max-width: 640px) {{ .metric-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
</style></head><body>
<div class="wrap">
  <div class="metric-grid">{''.join(cells_html)}</div>
</div>
<script>
document.querySelectorAll('.metric-value').forEach(el => {{
  const target = parseFloat(el.getAttribute('data-target'));
  const decimals = parseInt(el.getAttribute('data-decimals'));
  const duration = 650;
  const start = performance.now();
  function tick(now){{
    const p = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = (target * eased).toFixed(decimals);
    if (p < 1) requestAnimationFrame(tick);
  }}
  requestAnimationFrame(tick);
}});
</script>
</body></html>
"""
