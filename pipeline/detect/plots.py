"""Validation plots for the detect pipeline."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ZScaleInterval

from pipeline.detect.util import instrumental_mag


def save_validation_plots(
    out: Path,
    data: np.ndarray,
    mask: np.ndarray,
    res,
    cat,
    neg_cat,
    rec_rows: list[dict],
    results: dict,
    zp: float,
) -> None:
    segm = res.segmentation.segm
    vmin, vmax = ZScaleInterval().get_limits(data[~mask])

    regions = {
        "open field": (700, 1900, 150),
        "crowded / bright galaxies": (2150, 1500, 150),
        "edge of central-star mask": (1436, 2650, 150),
        "next to bleed column": (1500, 1000, 150),
        "lower-left noisier edge": (330, 600, 150),
        "near bright star (904, 2286)": (904, 2286, 150),
    }
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    for ax, (name, (xc, yc, h)) in zip(axes.ravel(), regions.items()):
        y0, y1 = max(0, yc - h), min(data.shape[0], yc + h)
        x0, x1 = max(0, xc - h), min(data.shape[1], xc + h)
        ext = [x0, x1, y0, y1]
        ax.imshow(
            data[y0:y1, x0:x1],
            cmap="gray",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            extent=ext,
        )
        ax.imshow(
            np.ma.masked_where(~mask[y0:y1, x0:x1], mask[y0:y1, x0:x1]),
            cmap="autumn",
            alpha=0.35,
            origin="lower",
            extent=ext,
        )
        if segm is not None:
            seg = segm.data[y0:y1, x0:x1]
            edges = (seg > 0) & (
                (np.roll(seg, 1, 0) != seg) | (np.roll(seg, 1, 1) != seg)
            )
            ax.imshow(
                np.ma.masked_where(~edges, edges),
                cmap="cool",
                origin="lower",
                extent=ext,
                interpolation="nearest",
            )
        sel = (cat["x"] > x0) & (cat["x"] < x1) & (cat["y"] > y0) & (cat["y"] < y1)
        ax.plot(cat["x"][sel], cat["y"][sel], "+", color="lime", ms=5, mew=0.8)
        nsel = (neg_cat["x"] > x0) & (neg_cat["x"] < x1) & (neg_cat["y"] > y0) & (neg_cat["y"] < y1)
        ax.plot(neg_cat["x"][nsel], neg_cat["y"][nsel], "x", color="orange", ms=8, mew=1.5)
        ax.set_title(f"{name}: {int(sel.sum())} sources")
    fig.suptitle(
        "cyan: segment outlines, green +: centroids, red: mask, orange x: negative-image detections"
    )
    plt.tight_layout()
    plt.savefig(out / "detection_cutouts.png", dpi=75)
    plt.close()

    mags = np.array([r["mag"] for r in rec_rows])
    comp = np.array([r["completeness"] for r in rec_rows])
    cerr = np.array([r["completeness_err"] for r in rec_rows])
    fr = np.array([r["median_flux_ratio"] for r in rec_rows])
    f16 = np.array([r["flux_ratio_p16"] for r in rec_rows])
    f84 = np.array([r["flux_ratio_p84"] for r in rec_rows])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    a1.errorbar(mags, comp, yerr=cerr, fmt="o-")
    for lvl in (0.9, 0.5):
        a1.axhline(lvl, color="grey", ls=":")
    a1.set_xlabel(f"injected magnitude (ZP = {zp}, lab eq. 5)")
    a1.set_ylabel("fraction recovered")
    a1.set_title("Completeness from injected point sources")
    a2.plot(mags, fr, "o-", label="median")
    a2.fill_between(mags, f16, f84, alpha=0.3, label="16 to 84%")
    a2.axhline(1, color="k", ls="--")
    a2.set_xlabel("injected magnitude")
    a2.set_ylabel("isophotal flux / true flux")
    a2.set_title("Flux recovered inside the detection segment")
    a2.legend()
    plt.tight_layout()
    plt.savefig(out / "completeness.png", dpi=100)
    plt.close()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = np.arange(8, 22, 0.25)
    ax.hist(np.asarray(cat["mag"]), bins=bins, histtype="step", lw=1.5, label=f"real ({len(cat)})")
    neg_m = instrumental_mag(neg_cat["flux"], zp)
    ax.hist(
        np.asarray(neg_m),
        bins=bins,
        histtype="stepfilled",
        alpha=0.6,
        color="orange",
        label=f"negative image ({len(neg_cat)})",
    )
    lim = results["completeness_limits"]
    for key, ls in (("mag_90", "--"), ("mag_50", ":")):
        if np.isfinite(lim[key]):
            ax.axvline(lim[key], color="k", ls=ls, label=f"{key[-2:]}% complete")
    ax.set_yscale("log")
    ax.set_xlabel("isophotal magnitude (lab eq. 5)")
    ax.set_ylabel("sources per 0.25 mag")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "magnitude_histogram.png", dpi=100)
    plt.close()