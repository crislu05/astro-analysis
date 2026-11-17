"""Synthetic self-tests for photometry (lab B2)."""

import sys

import numpy as np
from astropy.table import Table

from pipeline.photometry import config
from pipeline.photometry.aperture import run_photometry
from pipeline.photometry.calibration import counts_to_mag
from pipeline.photometry.cog import curve_of_growth
from pipeline.photometry.inputs import PhotometryInputs
from pipeline.photometry.synthetic import add_gaussian, gaussian_fraction_within


def selftest() -> None:
    rng = np.random.default_rng(1)
    shape = (300, 300)
    sky_level, noise, fwhm, gain = 3420.0, 10.0, 5.3, 3.1
    fails = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
        if not cond:
            fails.append(name)

    img = rng.normal(sky_level, noise, shape)
    inp = PhotometryInputs(img, np.zeros(shape, bool), np.zeros(shape, int), gain)
    t = run_photometry(inp, [150], [150], [1])
    check(
        "blank sky: |flux| < 4 sigma",
        abs(t["ap_flux"][0]) < 4 * t["ap_flux_err"][0],
        f"(flux {t['ap_flux'][0]:.1f} +- {t['ap_flux_err'][0]:.1f})",
    )
    check(
        "blank sky: sky level within 1 count",
        abs(t["sky"][0] - sky_level) < 1.0,
        f"(sky {t['sky'][0]:.2f})",
    )

    img = rng.normal(sky_level, noise, shape)
    F = 50000.0
    add_gaussian(img, 150.3, 149.6, F, fwhm)
    seg = np.zeros(shape, int)
    yy, xx = np.mgrid[0:300, 0:300]
    seg[np.hypot(xx - 150.3, yy - 149.6) < 8] = 1
    inp = PhotometryInputs(img, np.zeros(shape, bool), seg, gain)
    t = run_photometry(inp, [150.3], [149.6], [1])
    expect = F * gaussian_fraction_within(config.R_AP, fwhm)
    check(
        "star: aperture flux matches analytic enclosed flux (1%)",
        abs(t["ap_flux"][0] / expect - 1) < 0.01,
        f"(ratio {t['ap_flux'][0] / expect:.4f})",
    )

    img2 = img.copy()
    add_gaussian(img2, 170, 150, 200000, fwhm)
    seg2 = seg.copy()
    seg2[np.hypot(xx - 170, yy - 150) < 12] = 2
    inp2 = PhotometryInputs(img2, np.zeros(shape, bool), seg2, gain)
    t2 = run_photometry(inp2, [150.3], [149.6], [1])
    check(
        "neighbour in annulus: flux unchanged (1%)",
        abs(t2["ap_flux"][0] / t["ap_flux"][0] - 1) < 0.01,
        f"(ratio {t2['ap_flux'][0] / t['ap_flux'][0]:.4f})",
    )

    m = np.zeros(shape, bool)
    m[148:151, 152:156] = True
    inp3 = PhotometryInputs(img, m, seg, gain)
    t3 = run_photometry(inp3, [150.3], [149.6], [1])
    check("masked pixels in aperture: flag 1 set", bool(t3["flag"][0] & config.FLAG_MASKED))

    pulls = []
    for _ in range(200):
        img = rng.normal(sky_level, noise, shape)
        F = 800.0
        add_gaussian(img, 150, 150, F, fwhm)
        seg = np.zeros(shape, int)
        seg[np.hypot(xx - 150, yy - 150) < 7] = 1
        inp = PhotometryInputs(img, np.zeros(shape, bool), seg, gain)
        t = run_photometry(inp, [150], [150], [1])
        pulls.append(
            (t["ap_flux"][0] - F * gaussian_fraction_within(config.R_AP, fwhm))
            / t["ap_flux_err"][0]
        )
    s = np.std(pulls)
    check(
        "faint stars: pull std between 0.85 and 1.15 (errors are right)",
        0.85 < s < 1.15,
        f"(std {s:.2f}, mean {np.mean(pulls):+.2f})",
    )

    img = rng.normal(sky_level, noise, (300, 300))
    add_gaussian(img, 150, 150, 2e5, fwhm)
    seg = np.zeros(shape, int)
    seg[np.hypot(xx - 150, yy - 150) < 12] = 1
    inp = PhotometryInputs(img, np.zeros(shape, bool), seg, gain)
    stars = Table({"x": [150.0], "y": [150.0], "label": [1]})
    cog, _ = curve_of_growth(inp, stars)
    i_ap = int(np.argmin(abs(config.COG_RADII - config.R_AP)))
    expect = gaussian_fraction_within(config.R_AP, fwhm) / gaussian_fraction_within(
        config.COG_R_TOTAL, fwhm
    )
    check(
        "curve of growth at R_AP matches Gaussian (1%)",
        abs(cog[0, i_ap] / expect - 1) < 0.01,
        f"(measured {cog[0, i_ap]:.4f}, expected {expect:.4f})",
    )

    check(
        "mag: 10^((ZP-20)/2.5) counts -> m = 20",
        abs(counts_to_mag(10 ** ((25.3 - 20) / 2.5), 25.3) - 20) < 1e-9,
    )

    print(f"\n{len(fails)} failed" if fails else "\nall self-tests passed")
    if fails:
        sys.exit(1)