"""Number-count figure with literature comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.counts import config


def save_number_counts_plot(out, t, centres, n, sn, base, fits_, comps) -> None:
    out = Path(out)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    good = (n > 0) & (np.asarray(t["C_area_weighted"]) >= config.C_PLOT_MIN)
    ax.errorbar(
        centres[good & base],
        n[good & base],
        sn[good & base],
        fmt="o",
        color="k",
        ms=5,
        label="this work (used in fit)",
    )
    other = good & ~base
    ax.errorbar(
        centres[other],
        n[other],
        sn[other],
        fmt="o",
        mfc="none",
        color="k",
        ms=5,
        label="this work (not fitted: completeness < 70% or < 5 galaxies)",
    )
    ax.plot(config.YASUDA[:, 0], config.YASUDA[:, 1], "s", color="C0", ms=5, label="Yasuda et al. 2001, SDSS $r^*$")
    ax.plot(
        config.METCALFE[:, 0] + config.R_TO_R_SHIFT,
        config.METCALFE[:, 1],
        "^",
        color="C1",
        ms=5,
        label=f"Metcalfe et al. 2001, $R$ + {config.R_TO_R_SHIFT}",
    )
    xx = np.linspace(17.5, 26.5, 50)
    for key, col in (("bright", "C2"), ("faint", "C3")):
        f = fits_.get(key)
        if f:
            x = np.linspace(f["range"][0], f["range"][1], 10)
            ax.plot(
                x,
                10 ** (f["intercept"] + f["slope"] * x),
                color=col,
                lw=1.5,
                label=f"fit {f['range'][0]:.1f} to {f['range'][1]:.1f}: "
                f"slope {f['slope']:.2f} $\\pm$ {f['slope_err']:.2f}",
            )
    if fits_.get("bright"):
        f = fits_["bright"]
        m0 = f["range"][0] + config.BIN / 2
        n0 = 10 ** (f["intercept"] + f["slope"] * m0)
        ax.plot(xx, n0 * 10 ** (config.EUCLID * (xx - m0)), "k:", lw=1, label="Euclidean slope 0.6")
    ax.set_yscale("log")
    ax.set_ylim(20, 2e5)
    ax.set_ylabel(r"$n(m)$ (galaxies per 0.5 mag per deg$^2$)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("Differential galaxy counts, Lockman Hole, KPNO $r$", fontsize=10)
    for key, col, mk in (("yasuda2001", "C0", "s"), ("metcalfe2001", "C1", "^")):
        c = comps.get(key)
        if c:
            ax2.errorbar(c["centres"], c["ratio"], c["ratio_err"], fmt=mk, color=col, ms=4)
    ax2.axhline(1, color="k", lw=0.8)
    ax2.set_ylim(0, 2.5)
    ax2.set_xlim(17.5, 26.5)
    ax2.set_xlabel(r"$r$ (SDSS AB)")
    ax2.set_ylabel("this work / published")
    plt.tight_layout()
    plt.savefig(out / "number_counts.png", dpi=130)
    plt.close()