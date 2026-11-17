"""Calibration diagnostic figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_calibration_plots(out, zp, zp_plot, comp, transfer, lim90, header) -> None:
    out = Path(out)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(
        zp_plot["r"],
        zp_plot["zp"],
        "o",
        ms=4,
        label=f"aperture: ZP$_{{ap}}$ = {zp['ZP_ap']:.3f} $\\pm$ {zp['ZP_ap_err']:.3f}",
    )
    ax.plot(
        zp_plot["kron_r"],
        zp_plot["kron_zp"],
        "s",
        ms=4,
        label=f"Kron: ZP$_{{kron}}$ = {zp['ZP_kron']:.3f} $\\pm$ {zp['ZP_kron_err']:.3f}",
    )
    ax.axhline(zp["ZP_ap"], color="C0", ls="--")
    ax.axhline(zp["ZP_kron"], color="C1", ls="--")
    ax.axhline(
        float(header["MAGZPT"]) + 2.5 * np.log10(float(header["EXPTIME"])),
        color="C3",
        ls=":",
        label="MAGZPT + 2.5 log$_{10}$(EXPTIME)",
    )
    ax.set_xlabel(r"SDSS $r_{\rm PSF}$")
    ax.set_ylabel(r"$r_{\rm SDSS} + 2.5\log_{10}F$")
    ax.legend(fontsize=7)
    ax.set_title("Zero points from SDSS stars", fontsize=10)
    plt.tight_layout()
    plt.savefig(out / "calibration_zeropoint.png", dpi=130)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for zone, mk in (("inner", "o-"), ("edge", "s-"), ("all", "k^--")):
        c = comp[comp["zone"] == zone]
        ax.errorbar(c["mag"], c["C"], c["C_err"], fmt=mk, ms=4, label=zone)
    for lvl in (0.9, 0.5):
        ax.axhline(lvl, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("injected galaxy $r$ (SDSS AB)")
    ax.set_ylabel("fraction recovered")
    ax.set_title("Galaxy completeness by zone", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    for est, lab, mk in (("m_ap", "point-source aperture total", "o-"), ("m_kron", "Kron", "s-")):
        ax.errorbar(
            transfer["mag_in"],
            transfer[f"{est}_bias"],
            transfer[f"{est}_scatter"],
            fmt=mk,
            ms=4,
            capsize=2,
            label=lab,
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(lim90, color="grey", ls=":", lw=0.8)
    ax.text(lim90 + 0.05, -0.9, "90% complete", fontsize=7, color="grey")
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("injected galaxy $r$")
    ax.set_ylabel(r"$m_{\rm measured} - m_{\rm true}$ (median; bars: scatter)")
    ax.set_title("Galaxy magnitude bias", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "calibration_galaxies.png", dpi=130)
    plt.close()