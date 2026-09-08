"""
Monte Carlo Dropout uncertainty estimation.

Cheapest legitimate way to produce a per-pixel confidence/uncertainty map:
keep dropout ACTIVE at inference (model.train() on dropout layers only, or a
custom forward pass), run the same input through the model N times, and
compute the per-pixel variance across runs.

Caveat to state honestly in the pitch: this is a fast approximation, not a
calibrated probability. See SIH26142_Deep_Research_Brief.md section 6 for
the more rigorous (but heavier) self-supervised Bayesian alternative.
"""

import torch
import numpy as np


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """Set the whole model to eval() (batchnorm etc. behave correctly)
    but force any Dropout layers back into train mode so they stay active."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model: torch.nn.Module, input_tensor: torch.Tensor,
                        n_samples: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Run the model n_samples times with dropout active.

    Args:
        model: your SR model, must contain nn.Dropout layers to be meaningful
        input_tensor: shape (1, C, H, W), already on the right device
        n_samples: 10-20 is typically enough (brief recommends this range)

    Returns:
        mean_pred: (C, H, W) numpy array — the sharpened output to show
        uncertainty_map: (H, W) numpy array — per-pixel std dev across samples,
                          normalize this for the heatmap overlay in the demo
    """
    enable_mc_dropout(model)

    samples = []
    for _ in range(n_samples):
        output = model(input_tensor)
        samples.append(output.squeeze(0).cpu().numpy())

    stacked = np.stack(samples, axis=0)  # (n_samples, C, H, W)
    mean_pred = stacked.mean(axis=0)
    # collapse channel dim for a single-band confidence heatmap
    uncertainty_map = stacked.std(axis=0).mean(axis=0)

    return mean_pred, uncertainty_map


def normalize_uncertainty_for_display(uncertainty_map: np.ndarray) -> np.ndarray:
    """0-1 normalize for overlaying as a heatmap in the Streamlit demo."""
    u_min, u_max = uncertainty_map.min(), uncertainty_map.max()
    if u_max - u_min < 1e-8:
        return np.zeros_like(uncertainty_map)
    return (uncertainty_map - u_min) / (u_max - u_min)
