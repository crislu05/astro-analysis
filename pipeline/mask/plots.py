"""Visual QA for the mask pipeline (no source-detection validation)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ZScaleInterval
from matplotlib.patches import Patch

from pipeline.mask.config import LAYER_NAMES
from pipeline.mask.sky import fit_sky_histogram, gauss
from pipeline.mask.state import MaskState

LAYER_CMAP = ["#999999", "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]

DEFAULT_ZOOMS = {
    "central star": (1435, 3200, 450),
    "bleed bottom": (1435, 700, 450),
    "star (970,2780)": (970, 2780, 120),
    "star (2130,2310)": (2130, 2310, 120),
}


def _layer_index_map(state: MaskState) -> np.ndarray:
    ny, nx = state.data.shape
    out = np.zeros((ny, nx))
    for i, name in enumerate(LAYER_NAMES, start=1):
        layer = getattr(state.layers, name)
        out[layer & (out == 0)] = i
    return out


def save_inspection_plots(
    state: MaskState,
    outdir: str | Path,
    zooms: dict[str, tuple[int, int, int]] | None = None,
) -> None:
    """Write overlay, layer, zoom, and sky-histogram figures."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = state.data
    mask = state.mask
    zooms = DEFAULT_ZOOMS if zooms is None else zooms
    vmin, vmax = ZScaleInterval().get_limits(data[~mask])

    fig, axes = plt.subplots(1, 2, figsize=(14, 12))
    for ax, show_mask in zip(axes, (False, True)):
        ax.imshow(data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        if show_mask:
            ax.imshow(
                np.ma.masked_where(~mask, mask),
                cmap="autumn",
                origin="lower",
                alpha=0.45,
            )
        ax.set_title("zscale, mask overlaid" if show_mask else "zscale, no mask")
    plt.tight_layout()
    plt.savefig(outdir / "mask_overlay.png", dpi=90)
    plt.close()

    layer_map = _layer_index_map(state)
    fig, ax = plt.subplots(figsize=(7, 12))
    ax.imshow(data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    cm = matplotlib.colors.ListedColormap(LAYER_CMAP)
    ax.imshow(
        np.ma.masked_where(layer_map == 0, layer_map),
        cmap=cm,
        vmin=0.5,
        vmax=5.5,
        origin="lower",
        alpha=0.55,
    )
    ax.legend(
        handles=[
            Patch(color=c, label=n) for c, n in zip(cm.colors, LAYER_NAMES)
        ],
        loc="upper right",
    )
    ax.set_title("Mask layers")
    plt.tight_layout()
    plt.savefig(outdir / "mask_layers.png", dpi=90)
    plt.close()

    ny, nx = data.shape
    fig, axes = plt.subplots(2, len(zooms), figsize=(4.5 * len(zooms), 9))
    if len(zooms) == 1:
        axes = axes.reshape(2, 1)
    for j, (name, (xc, yc, h)) in enumerate(zooms.items()):
        y0, y1 = max(0, yc - h), min(ny, yc + h)
        x0, x1 = max(0, xc - h), min(nx, xc + h)
        for i in range(2):
            ax = axes[i, j]
            ax.imshow(
                data[y0:y1, x0:x1],
                cmap="gray",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                extent=[x0, x1, y0, y1],
            )
            if i == 1:
                m = mask[y0:y1, x0:x1]
                ax.imshow(
                    np.ma.masked_where(~m, m),
                    cmap="autumn",
                    origin="lower",
                    alpha=0.45,
                    extent=[x0, x1, y0, y1],
                )
            ax.set_title(name + (" + mask" if i else ""))
    plt.tight_layout()
    plt.savefig(outdir / "mask_zooms.png", dpi=80)
    plt.close()

    sky, sigma, centres, counts, p = fit_sky_histogram(data[~mask])
    model = gauss(centres, *p)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(centres, counts, where="mid", label="unmasked pixels")
    ax.plot(
        centres,
        model,
        "r",
        label=rf"Gaussian fit: $\mu$={sky:.1f}, $\sigma$={sigma:.1f}",
    )
    ax.set_yscale("log")
    ax.set_ylim(1, counts.max() * 2)
    ax.set_xlabel("pixel value (counts)")
    ax.set_ylabel("number of pixels")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "good_pixel_histogram.png", dpi=100)
    plt.close()