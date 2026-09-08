# MERGE_NOTES — how this version was assembled

You uploaded four things: two independent "fix" passes on the original
repo (each as a zip + a changelog + a copy of the idea PPT), plus a
standalone copy of one of the fix zips. This version merges the two fix
passes into one, picking the better implementation wherever they
overlapped, and verifies the result actually runs rather than trusting
either changelog's claims at face value.

## Where each file came from

| File | Source | Why |
|---|---|---|
| `src/data_pipeline.py` | **Fixed_Delivery** | Real WorldStrat archive-layout fix, `opensr-test` multi-dataset combiner (119 real scenes), band-order fix, all independently verified against real package docs |
| `src/model.py`, `evaluate.py`, `uncertainty.py`, `inference.py`, `dataset.py`, `augmentation.py`, `validate_pipeline.py` | **Identical in both** forks | No conflict — both forks left these untouched from the shared base, which already had the MC-Dropout placement fix, the tile-feather fix, and the correct sample path |
| `requirements.txt` | **Fixed_Delivery** | Comments out `GDAL`/`rasterio`/`sentinelhub` by default — this is the actual fix for the Visual C++ build error you hit on Windows. `opensr-test` itself still pulls in a working `rasterio` wheel automatically when needed (verified — installs clean, no Windows build tools required for that path), just not the standalone `GDAL` package |
| `src/train.py` | **Merged, hand-combined** | See below — neither fork alone had the best version |
| `demo/app.py`, `demo/theme.py` | **v3** | Animated compare-slider, metric-reveal, uncertainty blend. Verified this did NOT reintroduce the "hardcoded small crop on upload" bug Fixed_Delivery's changelog described fixing — v3 independently handles it correctly (and more thoroughly: proper tiled inference for large uploads via `inference.run_full_scene`, with an honestly-labeled partial-uncertainty map when tiling kicks in, since running full MC-Dropout over every tile of a large image would be too slow for a live demo) |
| `Finetune_On_Colab.ipynb` | **Fixed_Delivery's FAST notebook**, renamed | Fixed_Delivery kept both an old (slow-by-default) and new (FAST) notebook side by side "for reference." That's confusing baggage for a team under deadline pressure — one notebook, the working one, named the obvious thing. The old notebook's own opensr-test-first fix is preserved as history in git if you ever want it; it's not shipped here |
| PPT, `SIH26142_Deep_Research_Brief.md` | Carried over as-is | Not modified this pass |

## `train.py` — what was actually merged and why

Both forks independently added mixed precision + faster checkpointing on
top of the shared base, but each added a DIFFERENT additional idea neither
had:

- **Fixed_Delivery had:** `freeze_backbone` (transfer-learning: train only
  the last 4 RRDB blocks + upsampling head — verified by running it:
  3,028,931 / 16,697,987 params trainable), auto `num_workers` +
  `pin_memory` + `persistent_workers`
- **v3 had:** early stopping (`patience` — stop once the tracked loss
  hasn't improved for N epochs instead of always running the full `epochs`
  ceiling), and the non-deprecated `torch.amp.GradScaler("cuda", ...)` API
  with an explicit CPU/CUDA guard

**Bug found and fixed during the merge:** Fixed_Delivery's AMP code called
`torch.cuda.amp.GradScaler(...)` — the deprecated form. Running it with
`python -W error::FutureWarning` confirmed this would raise a warning today
and could break outright in a future torch release. The merged version
uses v3's non-deprecated pattern instead, and adds an explicit
`amp_enabled = (device == "cuda" and torch.cuda.is_available())` guard so
it's also safe to run on CPU (e.g. for a local smoke test) without a
mismatched `device_type`.

The merged `train.py` has BOTH `freeze_backbone` and `patience` now,
neither fork had both.

## What was actually verified this pass (not just read)

- Ran the merged `train.py` end-to-end on CPU with synthetic data, with
  `python -W error::FutureWarning` — confirms no deprecation warning fires,
  confirms `freeze_backbone` reports the correct trainable-param split,
  confirms `patience=2` correctly stops training early (fired at epoch 4 of
  a 10-epoch ceiling in the test run) instead of always running to the cap
- Confirmed `opensr-test` actually installs cleanly from PyPI (its own
  dependency chain pulls in `rasterio`, `kornia`, `lpips`, `open-clip-torch`
  and others — that's a real, fairly heavy chain; worth knowing before you
  commit to it on a slow connection, though none of it needs Windows build
  tools)
- Booted the merged `demo/app.py` (v3's UI + Fixed_Delivery's backend)
  with the real 67MB Real-ESRGAN checkpoint present — clean boot, no
  tracebacks
- Directly called every function in `theme.py` (`colorize_uncertainty`,
  `blend_overlay`, `render_compare_slider`, `render_metrics_reveal`) with
  real pipeline output (real model, real uncertainty map, real metrics) —
  all four work correctly end to end
- Read `app.py`'s upload-handling path line by line to confirm it does NOT
  have the hardcoded-crop bug Fixed_Delivery's changelog flagged, and that
  its large-image path genuinely uses tiled inference rather than a naive
  single forward pass

## What's still NOT independently verified

- `opensr-test`'s actual `.load(name)` data download — needs Hugging Face
  access, which this sandbox can't reach (same limitation as the original
  WorldStrat/Copernicus data, just a different host this time). The code
  path was checked against the package's own documented API and dataset
  facts, not run against a live download
- The real GPU training run on Colab — still needs to happen on your end;
  everything here de-risks that run, it doesn't replace doing it
