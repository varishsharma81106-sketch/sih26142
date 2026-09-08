"""
Model wrapper — load pretrained Real-ESRGAN (default) or SwinIR (comparison),
ready for fine-tuning.

Setup:
    git clone https://github.com/xinntao/Real-ESRGAN
    cd Real-ESRGAN && pip install basicsr facexlib gfpgan -r requirements.txt
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
        -P experiments/pretrained_models

For fine-tuning specifically, edit Real-ESRGAN's
`options/finetune_realesrgan_x4plus.yml`:
  - point `dataroot_gt` / `dataroot_lq` at your data/processed patches
  - REPLACE the default degradation params with the physically-motivated
    ones from research brief section 2 (sensor blur + Poisson-like noise,
    not JPEG/camera degradation)
"""

import sys
import types
import torch


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _patch_basicsr_torchvision_compat():
    """basicsr (Real-ESRGAN's dependency) imports
    `torchvision.transforms.functional_tensor`, which was removed in
    current torchvision releases (the function moved to
    `torchvision.transforms.functional`). Without this shim, `import
    basicsr` crashes on any recent torchvision — including current
    Colab images. Safe no-op if the module already exists.

    Verified real, not assumed: this is a well-documented, widely-hit
    issue (XPixelGroup/BasicSR#650, #659, #677; torchvision removed the
    module in the v0.17.0 release, Jan 2024) — basicsr 1.4.2's
    data/degradations.py still does `from torchvision.transforms.
    functional_tensor import rgb_to_grayscale`, which is exactly what this
    shim satisfies by pre-registering a fake module with that one
    function copied over from the current location."""
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    try:
        import torchvision.transforms.functional as F
        shim = types.ModuleType("torchvision.transforms.functional_tensor")
        shim.rgb_to_grayscale = F.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = shim
    except ImportError:
        pass  # torchvision not installed yet — will fail later with a clearer error


_patch_basicsr_torchvision_compat()


def load_realesrgan(weights_path: str, scale: int = 4, device: str = None):
    """Load a Real-ESRGAN generator from the official repo's architecture.
    Only needs `basicsr` installed (`pip install basicsr --no-deps` is
    enough — you don't need its full training-only dependency chain just
    to load the architecture and run inference/fine-tuning here).

    device: if not given, auto-detects (`cuda` if available, else `cpu`) —
    previously defaulted to a hardcoded `"cuda"`, which crashed with
    "Torch not compiled with CUDA enabled" / no-GPU errors on any team
    member's CPU-only laptop who called this directly without passing
    device= explicitly (every CALLER in this repo already passes device
    explicitly, so this was a footgun for ad-hoc/notebook use, not a bug
    that was firing silently in the shipped code paths).

    Architecture parameters (num_in_ch=3, num_out_ch=3, num_feat=64,
    num_block=23, num_grow_ch=32) and the `"params_ema"` key check below
    were verified against the REAL RealESRGAN_x4plus.pth checkpoint
    (downloaded and inspected its pickle structure directly, without
    needing torch to do it): conv_first.weight is shape (64, 3, 3, 3),
    body.0 through body.22 exist (23 blocks, each with rdb1/rdb2/rdb3 ->
    conv1..conv5, matching RRDBNet's real ResidualDenseBlock/RRDB
    structure), and the only top-level key present is "params_ema" — this
    function's assumptions line up exactly with the actual file, not just
    the docs.
    """
    from basicsr.archs.rrdbnet_arch import RRDBNet

    device = device or _default_device()
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                     num_block=23, num_grow_ch=32, scale=scale)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict["params_ema"] if "params_ema" in state_dict else state_dict)
    model.to(device)
    return model


def load_swinir(weights_path: str, scale: int = 4, device: str = None):
    """Load a SwinIR model for real-world SR.
    Requires the SwinIR repo (JingyunLiang/SwinIR) cloned alongside this
    project so `models.network_swinir` is importable.

    device: auto-detects if not given — see load_realesrgan's docstring.
    """
    from models.network_swinir import SwinIR

    device = device or _default_device()
    model = SwinIR(upscale=scale, img_size=64, window_size=8, img_range=1.0,
                    depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
                    upsampler="nearest+conv", resi_connection="1conv")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict["params"] if "params" in state_dict else state_dict)
    model.to(device)
    return model


def add_dropout_for_uncertainty(model: torch.nn.Module, p: float = 0.1,
                                 n_dropout_blocks: int = 4) -> torch.nn.Module:
    """Insert Dropout2d INSIDE the network's trunk (RRDBNet's `body`, the
    stack of 23 RRDB blocks — verified against the real checkpoint, see
    load_realesrgan's docstring) rather than only after the final 3-channel
    RGB output.

    CORRECTED from an earlier version that wrapped the whole model as
    `nn.Sequential(model, nn.Dropout2d(p))`, applying dropout only to the
    final 3-channel image. Dropout2d zeros entire CHANNELS at a time — with
    only 3 channels to drop, that gives at most 3 coarse, all-or-nothing
    outcomes (an entire color channel blacked out, or nothing), essentially
    content-independent, not a real per-pixel confidence signal. Inserting
    it instead after a handful of the 64-channel intermediate `body[i]`
    blocks gives dropout many more "channels to flip" on feature maps that
    actually vary with image content (edges, texture, land-cover type) —
    the difference between a meaningful spatial uncertainty heatmap and one
    that's nearly uniform except for occasional whole-channel flicker.

    Idempotent and mutates `model` in place (rather than wrapping it in a
    new module): safe to call on the same cached model instance every time
    a user clicks "run" in the Streamlit demo without stacking additional
    dropout layers on each click, and `uncertainty.py`'s `enable_mc_dropout`
    (which walks `model.modules()` for any Dropout/Dropout2d/Dropout3d
    instance) finds these newly-inserted layers with no changes needed on
    that end — the interface every existing caller (validate_pipeline.py,
    demo/app.py) uses is unchanged.
    """
    if getattr(model, "_mc_dropout_injected", False):
        return model

    body = model.body  # nn.Sequential of RRDB blocks (verified: 23 of them, each ending in a 64-channel feature map after its own internal residual add)
    n = len(body)
    step = max(1, n // n_dropout_blocks)
    idxs = list(range(step - 1, n, step))[:n_dropout_blocks]
    for i in idxs:
        body[i] = torch.nn.Sequential(body[i], torch.nn.Dropout2d(p=p))

    model._mc_dropout_injected = True
    return model
