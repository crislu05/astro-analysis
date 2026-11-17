"""Diagnostic plots for photometry."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.photometry import config


def save_photometry_plots(out, phot, cog, ap_frac, empty_sums, expected, measured, summary):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    ax = axes[0, 0]
    for c in cog:
        ax.plot(config.COG_RADII, c, color="grey", alpha=0.3, lw=0.8)
    if len(cog):
        ax.plot(config.COG_RADII, np.median(cog, axis=0), "k-", lw=2, label="median of stars")
    ax.axvline(
        config.R_AP,
        color="r",
        ls="--",
        label=f"aperture r = {config.R_AP:g} px: {100 * ap_frac:.1f}%",
    )
    ax.set_xlabel("radius (px)")
    ax.set_ylabel(f"enclosed flux / flux within {config.COG_R_TOTAL:g} px")
    ax.set_title("Curve of growth (aperture correction)")
    ax.legend()

    ax = axes[0, 1]
    ax.hist(empty_sums, bins=60, histtype="step", density=True, label="empty apertures")
    xg = np.linspace(empty_sums.min(), empty_sums.max(), 300)
    for s, lab in ((expected, "independent-pixel prediction"), (measured, "measured")):
        ax.plot(
            xg,
            np.exp(-xg**2 / (2 * s**2)) / (s * np.sqrt(2 * np.pi)),
            label=f"{lab}: sigma = {s:.0f}",
        )
    ax.set_xlabel("sky-subtracted sum in an r = 6 px aperture (counts)")
    ax.set_title(f"Empty apertures: correlated-noise factor {summary['correlated_noise_factor']}")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    good = phot["flag"] == 0
    ax.scatter(phot["mag_ap"][good], phot["mag_ap_err"][good], s=2, alpha=0.4, label="flag = 0")
    ax.scatter(
        phot["mag_ap"][~good], phot["mag_ap_err"][~good], s=2, alpha=0.4, color="orange", label="flagged"
    )
    ax.axhline(2.5 / np.log(10) / 5, color="k", ls=":", label="S/N = 5")
    ax.set_yscale("log")
    ax.set_xlabel("mag_ap (lab eq. 5)")
    ax.set_ylabel("magnitude error")
    ax.set_title("Photometric errors")
    ax.legend()

    ax = axes[1, 1]
    if "injection" in summary:
        inj = summary["injection"]
        m = [r["mag_in"] for r in inj]
        ax.errorbar(
            m,
            [r["median_dmag"] for r in inj],
            yerr=[r["dmag_scatter"] for r in inj],
            fmt="o-",
            label="mag_psf_total - input (scatter as bars)",
        )
        ax.plot(m, [r["median_mag_err"] for r in inj], "s--", label="median quoted error")
        ax.plot(m, [r["pull_std"] for r in inj], "^-", label="pull std (1 = errors right)")
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(1, color="grey", ls=":", lw=0.8)
        ax.set_xlabel("injected magnitude")
        ax.set_title("Injected stars: bias, scatter and error calibration")
        ax.legend(fontsize=8)
    else:
        ax.hist(phot["mag_ap"][np.isfinite(phot["mag_ap"])], bins=60, histtype="step")
        ax.set_yscale("log")
        ax.set_xlabel("mag_ap")
        ax.set_title("Aperture magnitudes")
    plt.tight_layout()
    plt.savefig(Path(out) / "photometry_diagnostics.png", dpi=90)
    plt.close()