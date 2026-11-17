"""Star/galaxy diagnostic figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.classifier import config


def save_star_galaxy_plot(
    out,
    mag,
    r50,
    conc,
    cls,
    star_idx,
    locus,
    loc_tab,
    mag_limit,
    model,
) -> None:
    out = Path(out)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})
    colours = {"galaxy": "C0", "star": "C3", "artefact": "C2", "unclassified": "0.7"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    m = star_idx[star_idx >= 0]
    c_grid = np.asarray(loc_tab["mag_lo"]) + config.BIN / 2
    sig = np.asarray(loc_tab["sigma_used"])
    for panel, (y, ylab, ylim) in enumerate(
        [
            (conc, r"concentration $c$ (mag)", (0.3, 1.8)),
            (r50, r"half-light radius $r_{50}$ (px)", (1, 12)),
        ]
    ):
        a = ax[panel]
        for name in ("unclassified", "galaxy", "star", "artefact"):
            s = cls == name
            a.scatter(
                mag[s],
                y[s],
                s=2,
                color=colours[name],
                alpha=0.5,
                label=f"{name} ({s.sum()})",
                rasterized=True,
            )
        a.scatter(mag[m], y[m], s=22, facecolors="none", edgecolors="k", lw=0.8, label="SDSS stars")
        a.axvline(mag_limit, color="k", ls=":", lw=1)
        a.set_xlim(config.MAG_RANGE)
        a.set_ylim(ylim)
        a.set_xlabel(r"$r_{\rm point}$ (SDSS AB)")
        a.set_ylabel(ylab)
    ax[0].plot(c_grid, locus(c_grid), "k", lw=0.8)
    ax[0].plot(c_grid, locus(c_grid) + config.N_SIGMA * sig, "k--", lw=1)
    ax[0].plot(c_grid, locus(c_grid) - config.N_SIGMA * sig, "k--", lw=1)
    ax[0].legend(fontsize=7, markerscale=3, loc="upper left")
    ax[0].set_title("Concentration against magnitude (classifier)", fontsize=10)
    ax[1].set_title("Half-light radius (not used: too noisy)", fontsize=10)
    cc = (np.asarray(model["mag_lo"]) + np.asarray(model["mag_hi"])) / 2
    n = np.asarray(model["n_star_classified"], float)
    ok = n > 0
    ax[2].errorbar(cc[ok], n[ok], np.sqrt(n[ok]), fmt="o", color="C3", mfc="none", label="classified as star")
    nr = np.asarray(model["n_star_sdss"], float)
    okr = nr > 0
    ax[2].errorbar(cc[okr] + 0.05, nr[okr], np.sqrt(nr[okr]), fmt="s", color="k", label="SDSS stars (unmasked area)")
    n_all = np.histogram(
        mag,
        np.r_[np.asarray(model["mag_lo"]), np.asarray(model["mag_hi"])[-1]],
    )[0]
    ax[2].step(cc, n_all, where="mid", color="0.5", label="all sources")
    ax[2].plot(cc, model["n_star_model"], "k--", label="fit to SDSS, extrapolated (upper limit)")
    ax[2].axvline(mag_limit, color="k", ls=":", lw=1)
    ax[2].set_yscale("log")
    ax[2].set_xlabel(r"$r_{\rm point}$")
    ax[2].set_ylabel("stars per 0.5 mag")
    ax[2].set_title("Star counts", fontsize=10)
    ax[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "star_galaxy.png", dpi=120)
    plt.close()