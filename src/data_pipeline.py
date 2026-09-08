"""
Data acquisition + preprocessing.

Two paths in, both should end up as paired .npy patches under data/processed/:

  A) WorldStrat (preferred — real pairs, do this first):
     Get the dataset from Zenodo (github.com/worldstrat/worldstrat points here:
     zenodo.org/record/6810792) or the trimmed Kaggle mirror. Verified directly
     against WorldStrat's own README: it does NOT ship one folder per AOI
     holding both HR and LR — it ships as four SEPARATE archives, each
     covering every AOI: hr_dataset.tar.gz (recommended HR — SentinelHub-
     processed, 12-bit; hr_dataset_raw.tar.gz is the alternative raw-Airbus
     version), lr_dataset_l2a.tar.gz (recommended LR — 16 temporally-matched
     Sentinel-2 L2A revisits per AOI; lr_dataset_l1c.tar.gz is the L1C
     alternative). Extract the HR archive to one folder and the LR archive to
     a SEPARATE folder. `load_geotiff_pair()` and `build_patches_from_pairs()`
     below turn that real, on-disk layout into the paired .npy patches
     `dataset.py` expects — `discover_aoi_pairs()` takes the two extracted
     folders (`hr_root`, `lr_root`) and matches AOIs across them; see its
     docstring for exactly how the matching works and what to do if it finds
     zero pairs (filenames can still vary by download date/version).

  B) Copernicus Data Space (for real Sentinel-2 India tiles, validation only
     unless you also have a matching HR source for that AOI):
     1. Free account: https://browser.dataspace.copernicus.eu
     2. `pip install sentinelhub` and configure credentials
     3. Use `download_sentinel2_tile()` below

Cloud masking: Sentinel-2 L2A products ship a Scene Classification (SCL) band
— use it to drop/mask cloudy pixels before tiling. Don't skip this; cloud
edges look like "detail" to a naive SR model and will corrupt training.

Note on this file's dependencies: `load_geotiff_pair()` needs `rasterio`
(already in requirements.txt) for real CRS-aware reprojection — it isn't
needed for anything else in this file, and everything else here
(`mask_clouds`, `tile_into_patches`, `synthetic_degrade`, `save_patch_pair`,
`split_train_val`) only needs numpy and has been run end-to-end against the
bundled real sample tile.
"""

from pathlib import Path
import shutil
import numpy as np

try:
    from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType
except ImportError:
    SHConfig = None  # sentinelhub not installed yet — fine until you run this


def download_sentinel2_tile(bbox_coords: tuple, time_interval: tuple,
                             out_path: str, resolution: int = 10):
    """Pull a Sentinel-2 L2A tile for the given bbox/time window via SentinelHub API.

    Args:
        bbox_coords: (min_lon, min_lat, max_lon, max_lat)
        time_interval: ('2026-01-01', '2026-03-01')
        out_path: where to save the resulting array
        resolution: 10 for the visible/NIR bands you'll use as LR input
    """
    if SHConfig is None:
        raise ImportError("pip install sentinelhub first, and set up your "
                           "Copernicus Data Space credentials")

    config = SHConfig()
    # TODO: config.sh_client_id / sh_client_secret from your Copernicus account

    bbox = BBox(bbox_coords, crs=CRS.WGS84)
    request = SentinelHubRequest(
        evalscript="""
            //VERSION=3
            function setup() {
                return { input: ["B02","B03","B04","B08","SCL"], output: { bands: 5 } };
            }
            function evaluatePixel(sample) {
                return [sample.B02, sample.B03, sample.B04, sample.B08, sample.SCL];
            }
        """,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=time_interval,
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        resolution=(resolution, resolution),
        config=config,
    )
    data = request.get_data()[0]
    np.save(out_path, data)
    return data


def mask_clouds(image: np.ndarray, scl_band: np.ndarray) -> np.ndarray:
    """Zero out cloud/cloud-shadow pixels using the Sentinel-2 SCL band.
    SCL cloud-related classes: 3 (shadow), 8 (cloud medium prob),
    9 (cloud high prob), 10 (thin cirrus)."""
    cloud_classes = {3, 8, 9, 10}
    mask = ~np.isin(scl_band, list(cloud_classes))
    return image * mask[..., None]


def tile_into_patches(image: np.ndarray, patch_size: int, stride: int = None) -> list:
    """Split a full scene into fixed-size patches for training."""
    stride = stride or patch_size
    h, w = image.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(image[y:y + patch_size, x:x + patch_size])
    return patches


def synthetic_degrade(hr_patch: np.ndarray, scale: int, sensor_blur_sigma: float = 1.2,
                       noise_std: float = 0.01) -> np.ndarray:
    """Degrade a high-res patch to simulate Sentinel-2-like input.

    IMPORTANT: this uses a physically-motivated blur+noise model, NOT
    Real-ESRGAN's default JPEG/camera degradation pipeline — see research
    brief section 2 for why that distinction matters here.
    """
    from scipy.ndimage import gaussian_filter

    blurred = gaussian_filter(hr_patch, sigma=(sensor_blur_sigma, sensor_blur_sigma, 0))
    downsampled = blurred[::scale, ::scale]
    noisy = downsampled + np.random.normal(0, noise_std, downsampled.shape)
    return np.clip(noisy, 0, 1)


def save_patch_pair(lr_patch: np.ndarray, hr_patch: np.ndarray, out_dir: str, idx: int):
    lr_dir = Path(out_dir) / "lr"
    hr_dir = Path(out_dir) / "hr"
    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)
    np.save(lr_dir / f"patch_{idx:05d}.npy", lr_patch)
    np.save(hr_dir / f"patch_{idx:05d}.npy", hr_patch)


# ---------------------------------------------------------------------------
# Real WorldStrat / GeoTIFF pair loading
#
# This section replaces what used to be a dangling reference to a
# `patch_worldstrat_pair()` function that was mentioned in this module's
# docstring but never actually existed anywhere in the codebase — as
# delivered, there was no real code path from a WorldStrat download to
# training patches. `discover_aoi_pairs` + `load_geotiff_pair` +
# `build_patches_from_pairs` together are that real path.
# ---------------------------------------------------------------------------

def discover_aoi_pairs(hr_root: str, lr_root: str, hr_glob: str = "*.tif",
                        lr_glob: str = "*.tif") -> list:
    """Match HR and LR GeoTIFFs across WorldStrat's REAL on-disk distribution.

    CORRECTED from an earlier version of this function that assumed WorldStrat
    ships one folder per AOI containing both an HR file and its LR revisits
    together. Checked directly against WorldStrat's own GitHub README and its
    Zenodo record (github.com/worldstrat/worldstrat, zenodo.org/record/6810792):
    it does NOT ship that way. It ships as four SEPARATE top-level archives,
    each covering ALL AOIs, that you download and extract independently:
      - hr_dataset.tar.gz       (SentinelHub-processed HR, 12-bit) <- recommended HR
      - hr_dataset_raw.tar.gz   (raw Airbus HR, 12-bit)
      - lr_dataset_l2a.tar.gz   (16 temporally-matched Sentinel-2 L2A revisits/AOI) <- recommended LR
      - lr_dataset_l1c.tar.gz   (16 temporally-matched Sentinel-2 L1C revisits/AOI)
    (Kaggle's mirror trims this to an 8-revisit "core dataset" instead of 16 —
    fine, this function doesn't care how many revisits there are.)

    So: extract the HR archive to one folder and the LR archive to a SEPARATE
    folder, and pass those two folders as `hr_root` / `lr_root` here — don't
    go looking for one merged folder, there isn't one.

    Matching strategy (handles both layouts WorldStrat's own notebooks show):
      1. If both roots contain AOI-named subfolders, pairs are matched by
         subfolder name (hr_root/<aoi>/... <-> lr_root/<aoi>/...).
      2. Otherwise (flat files under each root), pairs are matched by
         filename stem — an LR file matches an HR file if its stem equals or
         starts with the HR file's stem (LR revisit files typically append a
         date/revisit-index suffix the HR filename won't have).

    Exact filenames can still vary by download date/package version, and this
    sandbox has no network path to the real multi-GB Zenodo archives to
    confirm byte-for-byte — if this finds 0 pairs, the error message below
    prints a few real filenames from each root so you can see what you're
    actually working with and adjust `hr_glob`/`lr_glob` (or the AOI-name
    assumption) to match.

    Returns a list of dicts: {"aoi": name, "hr_path": Path, "lr_paths": [Path, ...]}
    — multiple LR paths per AOI so you can choose a revisit or composite them
    (e.g. `np.median` across revisits) before calling `load_geotiff_pair`.
    """
    hr_root, lr_root = Path(hr_root), Path(lr_root)
    for root, which in ((hr_root, "hr_root"), (lr_root, "lr_root")):
        if not root.exists():
            raise FileNotFoundError(
                f"{which}={root} does not exist — extract the corresponding "
                "WorldStrat archive first (see this function's docstring for "
                "which archive is HR vs LR)."
            )

    hr_subdirs = sorted(p for p in hr_root.iterdir() if p.is_dir())
    lr_subdirs = sorted(p for p in lr_root.iterdir() if p.is_dir())

    pairs = []
    if hr_subdirs and lr_subdirs:
        # Subfolder-per-AOI layout inside each archive (this is what
        # WorldStrat's own `Dataset Exploration.ipynb` walks through) —
        # match subfolders across the two trees by shared name.
        lr_by_name = {p.name: p for p in lr_subdirs}
        for hr_dir in hr_subdirs:
            lr_dir = lr_by_name.get(hr_dir.name)
            if lr_dir is None:
                continue  # AOI present in HR archive but not (yet) in this LR archive
            hr_matches = sorted(hr_dir.glob(hr_glob))
            lr_matches = sorted(lr_dir.glob(lr_glob))
            if hr_matches and lr_matches:
                pairs.append({"aoi": hr_dir.name, "hr_path": hr_matches[0], "lr_paths": lr_matches})
    else:
        # Flat layout: match by filename stem (exact, or LR-stem-starts-with-HR-stem
        # to tolerate a trailing revisit/date suffix on the LR side).
        hr_files = sorted(hr_root.glob(hr_glob))
        lr_files = sorted(lr_root.glob(lr_glob))
        for hr_file in hr_files:
            stem = hr_file.stem
            matches = [f for f in lr_files if f.stem == stem or f.stem.startswith(stem)]
            if matches:
                pairs.append({"aoi": stem, "hr_path": hr_file, "lr_paths": matches})

    if not pairs:
        hr_sample = [p.name for p in list(hr_root.rglob("*"))[:5]]
        lr_sample = [p.name for p in list(lr_root.rglob("*"))[:5]]
        raise RuntimeError(
            f"No AOI pairs found between hr_root={hr_root} and lr_root={lr_root} "
            f"with hr_glob={hr_glob!r}, lr_glob={lr_glob!r}.\n"
            f"  First few files actually under hr_root: {hr_sample}\n"
            f"  First few files actually under lr_root: {lr_sample}\n"
            "Adjust hr_glob/lr_glob to match what you see above, or double-check "
            "you extracted the HR and LR archives WorldStrat actually ships "
            "(see this function's docstring) rather than looking for one merged folder."
        )
    return pairs


def load_geotiff_pair(hr_path: str, lr_path: str, reflectance_scale: float = None) -> tuple:
    """Load one real HR/LR GeoTIFF pair, reprojecting the LR raster onto the
    HR raster's grid if their CRS or pixel grids differ, and return both as
    float32 arrays in [0, 1], shape (H, W, C).

    This is the function that was missing: WorldStrat's HR (SPOT) and LR
    (Sentinel-2) rasters are geographically co-registered but NOT guaranteed
    to be on the same CRS or pixel grid as each other, so you cannot just
    `np.array(Image.open(...))` both and assume the pixels line up — you
    need an actual CRS-aware reprojection (rasterio.warp), not just a resize.

    Args:
        hr_path, lr_path: paths to the two GeoTIFFs
        reflectance_scale: if given, divide raw pixel values by this to get
            [0, 1] (Sentinel-2 L2A/L1C reflectance is commonly 0-10000 or
            0-65535 depending on product; SPOT is often 12-bit, 0-4095).
            If None, auto-scale by each raster's own observed max — fine for
            a quick pipeline check, but for real training set this
            explicitly per-sensor so LR and HR share a consistent radiometric
            scale (mixing an auto-scaled LR with an auto-scaled HR can teach
            the model a brightness relationship that isn't physically real).

    Requires `rasterio` (`pip install rasterio`, already in requirements.txt).
    Note: this function needs rasterio + real GeoTIFFs to exercise the
    reprojection path, so it could only be smoke-tested here against
    fabricated same-CRS rasters, not real WorldStrat CRS-mismatch data —
    run it against your actual download before trusting it at scale, and
    sanity-check with a quick min/max/mean print on the first few pairs.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(hr_path) as hr_ds:
        hr = hr_ds.read()  # (bands, H, W)
        hr_transform, hr_crs, hr_shape = hr_ds.transform, hr_ds.crs, (hr_ds.height, hr_ds.width)

    with rasterio.open(lr_path) as lr_ds:
        lr_raw = lr_ds.read()
        lr_transform, lr_crs = lr_ds.transform, lr_ds.crs

        same_grid = (lr_crs == hr_crs and lr_ds.transform == hr_transform
                     and (lr_ds.height, lr_ds.width) == hr_shape)
        if same_grid:
            lr = lr_raw
        else:
            # Reproject LR onto the HR grid — resampling method matters:
            # bilinear (not nearest) so we don't introduce staircase blocking
            # that a naive nearest-neighbour reproject would bake into every
            # training pair. Destination array must be pre-allocated with
            # the HR raster's own shape/transform/CRS, or `reproject` has
            # nothing to align to and silently produces an all-zero output.
            lr = np.zeros((lr_raw.shape[0], hr_shape[0], hr_shape[1]), dtype=lr_raw.dtype)
            reproject(
                source=lr_raw, destination=lr,
                src_transform=lr_transform, src_crs=lr_crs,
                dst_transform=hr_transform, dst_crs=hr_crs,
                resampling=Resampling.bilinear,
            )

    def _to_float01(arr, scale):
        arr = arr.astype(np.float32)
        s = scale if scale is not None else max(float(arr.max()), 1e-6)
        return np.clip(arr / s, 0.0, 1.0)

    hr_f = _to_float01(hr, reflectance_scale)
    lr_f = _to_float01(lr, reflectance_scale)

    # (bands, H, W) -> (H, W, bands), and keep only up to 3 bands (RGB) if
    # more are present, so this lines up with the RGB-only model/eval code
    # elsewhere in this repo. Swap this for real band selection (e.g. NIR)
    # if you extend the model to more than 3 input channels.
    hr_f = np.moveaxis(hr_f, 0, -1)[..., :3]
    lr_f = np.moveaxis(lr_f, 0, -1)[..., :3]
    return lr_f, hr_f


def build_patches_from_pairs(pairs: list, out_dir: str, raw_hr_patch: int = 256,
                              scale: int = 4, use_synthetic_lr: bool = False,
                              reflectance_scale: float = None) -> int:
    """Turn a list of {"hr_path", "lr_paths"} AOI dicts (from
    `discover_aoi_pairs`) into saved LR/HR .npy patch pairs under `out_dir`.

    Args:
        pairs: output of `discover_aoi_pairs`
        raw_hr_patch: HR patch size to tile at — save LARGER than your actual
            training patch_size so `PairedAugmentation`'s random crop has
            room to move (see augmentation.py / README).
        use_synthetic_lr: if True, ignore the real LR revisit entirely and
            resynthesize LR from HR via `synthetic_degrade` instead. Real
            pairs are strictly better when available — this is only for
            AOIs where the real LR revisit is missing/corrupt or you
            specifically want a matched synthetic/real comparison.

    Returns the number of patch pairs written.
    """
    patch_idx = 0
    for pair in pairs:
        lr_paths = pair["lr_paths"]
        # pick the middle revisit by default (a reasonable single choice out
        # of several cloud-risk revisits without picking blind) — swap in
        # your own cloud-free selection logic per AOI if you have it.
        lr_path = lr_paths[0] if len(lr_paths) == 1 else lr_paths[len(lr_paths) // 2]
        try:
            lr_full, hr_full = load_geotiff_pair(str(pair["hr_path"]), str(lr_path),
                                                  reflectance_scale=reflectance_scale)
        except Exception as e:
            print(f"  skipping AOI {pair['aoi']!r}: failed to load ({e})")
            continue

        # `load_geotiff_pair` already reprojected LR onto the HR pixel grid,
        # so hr_full and lr_full are the SAME shape here. Retile both
        # together at identical (y, x) offsets — tiling them separately
        # (e.g. via two independent `tile_into_patches` calls) was the
        # earlier bug: nothing forces two independent tiling passes to stay
        # aligned, and a mismatch fails silently (no crash, no error), which
        # is exactly the "worse than a crash" bug class this project already
        # hit once with the first LR-patch-writer draft.
        for y in range(0, hr_full.shape[0] - raw_hr_patch + 1, raw_hr_patch):
            for x in range(0, hr_full.shape[1] - raw_hr_patch + 1, raw_hr_patch):
                hr_patch = hr_full[y:y + raw_hr_patch, x:x + raw_hr_patch]
                if use_synthetic_lr:
                    lr_patch = synthetic_degrade(hr_patch, scale=scale)
                else:
                    # crop the same region from the HR-grid-aligned real LR,
                    # then downsample by `scale` so its size matches what
                    # dataset.py/augmentation.py expect (LR = HR // scale)
                    hr_res_lr_patch = lr_full[y:y + raw_hr_patch, x:x + raw_hr_patch]
                    lr_patch = hr_res_lr_patch[::scale, ::scale]
                assert lr_patch.max() > 0 or hr_patch.max() == 0, (
                    f"all-zero LR patch at ({y},{x}) for AOI {pair['aoi']!r} — "
                    "don't save silently-broken training data; investigate "
                    "the source raster/reprojection instead of continuing."
                )
                save_patch_pair(lr_patch, hr_patch, out_dir, patch_idx)
                patch_idx += 1

    print(f"Wrote {patch_idx} raw patch pairs from {len(pairs)} AOIs "
          f"(before augmentation multiplies them further at train time)")
    return patch_idx


def split_train_val(processed_dir: str = "data/processed", val_dir: str = "data/processed_val",
                     val_fraction: float = 0.15, seed: int = 42) -> dict:
    """Move a held-out fraction of saved patch pairs from `processed_dir`
    into `val_dir`, keeping LR/HR filenames matched.

    This was a real gap: the Colab notebook's evaluation cell reads from
    `data/processed_val/`, but nothing in this repo ever created that
    folder — so the evaluation cell would fail with a "no .npy patches
    found" assertion the first time anyone actually ran it end to end.

    Moves (not copies) files, so re-running this is idempotent only if you
    haven't already split — call it once per fresh `build_patches_from_pairs`
    run, before training.
    """
    src_lr = Path(processed_dir) / "lr"
    src_hr = Path(processed_dir) / "hr"
    stems = sorted(p.stem for p in src_lr.glob("*.npy"))
    if not stems:
        raise RuntimeError(f"No patches found in {processed_dir} — run "
                            "build_patches_from_pairs (or the synthetic path) first.")

    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(stems) * val_fraction))
    val_stems = set(rng.choice(stems, size=n_val, replace=False))

    dst_lr = Path(val_dir) / "lr"
    dst_hr = Path(val_dir) / "hr"
    dst_lr.mkdir(parents=True, exist_ok=True)
    dst_hr.mkdir(parents=True, exist_ok=True)

    moved = 0
    for stem in val_stems:
        shutil.move(str(src_lr / f"{stem}.npy"), str(dst_lr / f"{stem}.npy"))
        shutil.move(str(src_hr / f"{stem}.npy"), str(dst_hr / f"{stem}.npy"))
        moved += 1

    remaining = len(stems) - moved
    print(f"Split {len(stems)} patch pairs -> {remaining} train / {moved} val "
          f"(val_fraction={val_fraction})")
    return {"n_train": remaining, "n_val": moved}


# ---------------------------------------------------------------------------
# opensr-test: a lower-risk primary data path than hand-parsing WorldStrat
#
# Found by checking current literature/tooling rather than assuming
# WorldStrat's raw GeoTIFF distribution was the only option: `opensr-test`
# (ESA-funded, Aybar et al., IEEE GRSL 2024 — pip install opensr-test) ships
# `opensr_test.load(name)`, which returns REAL, already spatially-aligned
# Sentinel-2 <-> high-res pairs as plain numpy arrays — no CRS reprojection,
# no cloud masking, no manual tiling-offset bookkeeping required. Given a
# ~3-week internal deadline, this is a much smaller risk surface than
# `load_geotiff_pair`/`build_patches_from_pairs` above, which do real
# GeoTIFF/rasterio work and needed those functions to be written from
# scratch for this repo. Recommended as the PRIMARY path; keep the
# WorldStrat-raw path above as a secondary option only if you need more
# volume or scenes outside opensr-test's five bundled datasets.
#
# Available `name` values and what they actually are (from the package's
# own docs — not guessed): "naip" (US crops/forest/bare soil, 62 scenes,
# 484x484 HR, x4), "spot" (WorldStrat's own SPOT/Sentinel-2 pairs, 9 scenes,
# 512x512 HR, x4 — the most directly relevant one here since it's the same
# SPOT source this project's brief already cites), "venus" (x2 scale),
# "spain_crops" / "spain_urban" (2.5m Spanish aerial imagery, x4).
#
# Requires `pip install opensr-test` (not in requirements.txt by default —
# add it there once you confirm this is your primary data path; kept
# separate for now since it wasn't part of the original dependency set and
# should be a deliberate choice, not a silent addition).
# ---------------------------------------------------------------------------

def load_opensr_test_dataset(name: str = "spot", band_indices: tuple = (0, 1, 2),
                              reflectance_scale: float = 10000.0) -> list:
    """Load a real, pre-aligned LR/HR dataset via `opensr_test.load(name)`
    and return it as a list of (lr, hr) numpy pairs in the (H, W, C) float
    [0, 1] layout this repo's other functions expect.

    Verified against the package's own GitHub README (github.com/ESAOpenSR/
    opensr-test) and Hugging Face dataset card (huggingface.co/datasets/
    isp-uv-es/opensr-test) rather than assumed:
      - `opensr_test.load(name)` returns a dict with keys "L1C", "L2A", "HR",
        "HRharm" — confirmed exactly.
      - Dataset sizes/scale factors confirmed against the README's own table:
        naip x4/62 scenes/484x484, spot x4/9 scenes/512x512, venus x2/59
        scenes/256x256, spain_crops x4/28 scenes/512x512, spain_urban x4/20
        scenes/512x512.
      - reflectance_scale=10000 is confirmed correct: the README's own
        worked example does `torch.from_numpy(lr[idx, 0:3]) / 10000`.
      - band_indices default corrected from an earlier (2, 1, 0)-reversed
        guess to (0, 1, 2): the README's own example indexes `lr[idx, 0:3]`
        DIRECTLY, with no channel reversal, when feeding a downstream SR
        model. That doesn't independently confirm true photometric R,G,B
        order for DISPLAY purposes (Sentinel-2's raw 12-band order is not
        R,G,B, and opensr-test doesn't document the exact internal band
        order beyond "all Sentinel-2 L1C/L2A bands"), but (0, 1, 2) matches
        the package's own verified usage more closely than a guessed
        reversal did. Training/metrics validity doesn't depend on getting
        this exactly right either way (the same band_indices is applied
        identically to LR and HR below, which is what actually matters for
        alignment) — but a demo slide's color does. Verify visually with
        `matplotlib.pyplot.imshow` on one loaded sample before trusting it
        there; this could not be checked in this sandbox since `opensr-test`
        itself could not be installed here (see note below).
        args:
        name: one of "naip", "spot", "venus", "spain_crops", "spain_urban".
            "spot" is the most relevant default here — it's built from the
            same WorldStrat SPOT/Sentinel-2 source this project's research
            brief already cites, but pre-packaged with alignment already
            solved.
        band_indices: which 3 of the dataset's bands to keep, in output order.
        reflectance_scale: divide raw values by this to reach [0, 1] (see above).

    Returns a list of (lr, hr) tuples, each (H, W, 3) float32 in [0, 1],
    ready to pass straight into `tile_into_patches` / `save_patch_pair` /
    `PairedAugmentation`, exactly like a `load_geotiff_pair` result.

    NOTE: could not be executed in this sandbox. `opensr-test` itself was
    confirmed real and installable via PyPI, but it depends on `torch` +
    `torchvision` + `kornia` + `open-clip-torch` (checked its actual PyPI
    dependency metadata, not assumed) — and `torch`'s own Linux wheel pulls a
    multi-GB CUDA toolkit as a hard dependency that doesn't fit this sandbox's
    disk, so it could not be installed here to run for real. This is not
    extra weight beyond what this project already needs, though: `torch` is
    already required for `train.py`/`model.py`, so installing it once on your
    actual training machine (which you need to do anyway) covers this too.
    The array reshape/normalize logic below was verified against fake arrays
    matching the package's documented shape (N, C, H, W) and the
    reflectance-scale convention confirmed above — run it for real on your
    own machine/Colab before trusting it, same as `load_geotiff_pair`.
    """
    import opensr_test

    dataset = opensr_test.load(name)
    # Prefer L2A (atmospherically corrected) if present; some opensr-test
    # datasets only ship one of L2A/L1C.
    lr_key = "L2A" if "L2A" in dataset else "L1C"
    hr_key = "HRharm"  # HR harmonized to the LR sensor's radiometry
    lr_all = dataset[lr_key]   # (N, C, H, W)
    hr_all = dataset[hr_key]   # (N, C, H*scale, W*scale)

    pairs = []
    for i in range(lr_all.shape[0]):
        lr = lr_all[i][list(band_indices)]   # (3, H, W)
        hr = hr_all[i][list(band_indices)]
        lr = np.clip(np.moveaxis(lr, 0, -1).astype(np.float32) / reflectance_scale, 0, 1)
        hr = np.clip(np.moveaxis(hr, 0, -1).astype(np.float32) / reflectance_scale, 0, 1)
        pairs.append((lr, hr))

    print(f"Loaded {len(pairs)} real aligned pairs from opensr-test dataset {name!r}")
    return pairs


def build_patches_from_opensr_test(name: str = "spot", out_dir: str = "data/processed",
                                    scale: int = 4, start_idx: int = 0) -> int:
    """Convenience wrapper: `load_opensr_test_dataset` -> saved .npy patch
    pairs, one patch per source scene (opensr-test's images are already a
    single fixed-size aligned pair per scene, not a large scene to tile —
    unlike `build_patches_from_pairs`, there's no larger raster to sub-tile
    here, so this saves one LR/HR pair per scene as-is). Run
    `split_train_val` afterwards, same as the WorldStrat-raw path.

    start_idx: patch filename counter to start from. Needed so that calling
    this more than once against the SAME out_dir (e.g. once per dataset
    name, see `build_patches_from_opensr_test_multi` below) doesn't silently
    overwrite files — the original version of this function always started
    at 0, so a second call with a different `name` but the same `out_dir`
    would clobber every patch_00000.npy .. patch_0000N.npy the first call
    had just written, with no warning and no crash. Returns the count
    written so the caller can chain start_idx across calls.
    """
    pairs = load_opensr_test_dataset(name, reflectance_scale=10000.0)
    for offset, (lr, hr) in enumerate(pairs):
        idx = start_idx + offset
        assert lr.max() > 0, f"all-zero LR at index {idx} — investigate before continuing"
        save_patch_pair(lr, hr, out_dir, idx)
    print(f"Wrote {len(pairs)} patch pairs from opensr-test {name!r} to {out_dir} "
          f"(indices {start_idx}..{start_idx + len(pairs) - 1})")
    return len(pairs)


# ---------------------------------------------------------------------------
# Combine SEVERAL opensr-test datasets — real fix for "too little data" that
# doesn't require the 65GB+ WorldStrat raw download (see README/notebook for
# why that download is the #1 cause of multi-hour Colab runs).
#
# Only the x4-scale datasets are combined by default: "spot" (9), "naip"
# (62), "spain_crops" (28), "spain_urban" (20) = 119 real aligned scenes
# before augmentation, vs. 9 scenes if you only use "spot". "venus" is x2
# scale and deliberately excluded — mixing a x2 pair in with x4 pairs would
# silently corrupt the LR/HR size ratio dataset.py/augmentation.py assume.
# ---------------------------------------------------------------------------

def build_patches_from_opensr_test_multi(names: tuple = ("spot", "naip", "spain_crops", "spain_urban"),
                                          out_dir: str = "data/processed") -> int:
    """Build patches from several opensr-test datasets into ONE combined
    pool, using `start_idx` to keep every dataset's patches (correctly,
    without collision — see `build_patches_from_opensr_test`'s docstring).

    This is the recommended default data path: still a one-line-per-dataset,
    zero-GB-download, zero-CRS-wrangling pipeline (same as the single-dataset
    path), but ~13x more real source scenes than "spot" alone, which
    meaningfully reduces overfitting risk on a 3-week timeline without
    touching the 65GB+ WorldStrat raw archives at all.
    """
    total = 0
    for name in names:
        n = build_patches_from_opensr_test(name, out_dir=out_dir, start_idx=total)
        total += n
    print(f"Combined {len(names)} opensr-test datasets -> {total} real aligned "
          f"scenes in {out_dir} (before augmentation multiplies this further)")
    return total
