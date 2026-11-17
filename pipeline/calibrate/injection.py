"""Injected galaxies for completeness and magnitude transfer."""

from __future__ import annotations

import numpy as np
from astropy.table import Table
from scipy import ndimage
from scipy.signal import fftconvolve

from pipeline.calibrate import config
from pipeline.calibrate.log import log
from pipeline.calibrate.stats import robust
from pipeline.calibrate.zones import zone_of
from pipeline.photometry.aperture import run_photometry
from pipeline.photometry.inputs import PhotometryInputs


def sersic_stamp(r_e, q, pa, n, psf):
    from astropy.modeling.models import Sersic2D

    half = int(np.ceil(max(6 * r_e, 4) + psf.shape[0] // 2))
    size = 2 * half + 1
    fine = np.linspace(
        -half - 0.5 + 0.5 / config.OVERSAMPLE,
        half + 0.5 - 0.5 / config.OVERSAMPLE,
        size * config.OVERSAMPLE,
    )
    xx, yy = np.meshgrid(fine, fine)
    model = Sersic2D(amplitude=1.0, r_eff=r_e, n=n, x_0=0, y_0=0, ellip=1 - q, theta=pa)(xx, yy)
    img = model.reshape(size, config.OVERSAMPLE, size, config.OVERSAMPLE).sum(axis=(1, 3))
    img /= img.sum()
    img = fftconvolve(img, psf / psf.sum(), mode="same")
    img = np.clip(img, 0, None)
    return img / img.sum()


def add_stamp(image, x, y, stamp, flux, gain, rng):
    half = stamp.shape[0] // 2
    xi, yi = int(round(x)), int(round(y))
    shifted = ndimage.shift(stamp, (y - yi, x - xi), order=1, mode="constant") * flux
    noisy = rng.poisson(np.clip(shifted, 0, None) * gain) / gain
    y0, x0 = yi - half, xi - half
    sl = (
        slice(max(0, y0), min(image.shape[0], y0 + stamp.shape[0])),
        slice(max(0, x0), min(image.shape[1], x0 + stamp.shape[1])),
    )
    image[sl] += noisy[sl[0].start - y0 : sl[0].stop - y0, sl[1].start - x0 : sl[1].stop - x0]


def measure_size_distribution(data_sub, segm_img, mask, r_psf_like, psf_r50):
    from photutils.segmentation import SourceCatalog

    cat = SourceCatalog(data_sub, segm_img, mask=mask)
    r50 = np.asarray(
        cat.fluxfrac_radius(0.5).value
        if hasattr(cat.fluxfrac_radius(0.5), "value")
        else cat.fluxfrac_radius(0.5),
        float,
    )
    q = 1 - np.asarray(
        cat.ellipticity.value if hasattr(cat.ellipticity, "value") else cat.ellipticity,
        float,
    )
    sel = (
        (r_psf_like > config.SIZE_SAMPLE_RANGE[0])
        & (r_psf_like < config.SIZE_SAMPLE_RANGE[1])
        & np.isfinite(r50)
    )
    sel &= r50 > 1.15 * psf_r50
    r_int = np.sqrt(np.clip(r50[sel] ** 2 - psf_r50 ** 2, 0.25, None))
    return r_int, np.clip(q[sel], 0.2, 1.0)


def inject_galaxies(
    detect_fn,
    BackgroundResult,
    data,
    mask,
    bkg,
    gain,
    psf,
    zp_ap,
    zp_kron,
    ap_frac,
    sizes,
    qs,
    mags,
    n_per_mag,
    seed=11,
):
    from photutils.segmentation import SourceCatalog

    rng = np.random.default_rng(seed)
    allowed = ~ndimage.binary_dilation(mask, iterations=10)
    ny, nx = data.shape
    rows = []
    for m_in in mags:
        flux = 10 ** ((zp_ap - m_in) / 2.5) / ap_frac
        pts = []
        while len(pts) < n_per_mag:
            x, y = rng.uniform(60, nx - 60), rng.uniform(60, ny - 60)
            if allowed[int(y), int(x)] and all(
                np.hypot(x - a, y - b) > config.INJ_MIN_SEP for a, b, *_ in pts
            ):
                j = rng.integers(len(sizes))
                pts.append((x, y, sizes[j], qs[j], rng.uniform(0, np.pi)))
        img = data.copy()
        for x, y, re, q, pa in pts:
            add_stamp(
                img,
                x,
                y,
                sersic_stamp(re, q, pa, config.SERSIC_N, psf),
                flux,
                gain,
                rng,
            )
        sub = img - bkg.background
        res = detect_fn(
            img,
            mask,
            gain=gain,
            background=BackgroundResult(bkg.background, bkg.rms, sub),
        )
        cat = res.catalogue
        segm = res.segmentation.segm
        kcat = SourceCatalog(sub, segm, mask=mask)
        kron = np.asarray(kcat.kron_flux, float)
        cx, cy = np.asarray(cat["x"]), np.asarray(cat["y"])
        pts_arr = np.array([(p[0], p[1]) for p in pts])
        found_idx = np.full(len(pts), -1)
        for i, (x, y) in enumerate(pts_arr):
            d = np.hypot(cx - x, cy - y)
            j = int(np.argmin(d))
            if d[j] <= config.MATCH_RADIUS:
                found_idx[i] = j
        found = found_idx >= 0
        lab = np.asarray(cat["label"])[found_idx[found]]
        inp = PhotometryInputs(img, mask, segm.data, gain)
        t = run_photometry(
            inp,
            cx[found_idx[found]],
            cy[found_idx[found]],
            lab,
            corr_factor=config.AP_CORR_FACTOR,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            m_ap = zp_ap - 2.5 * np.log10(np.asarray(t["ap_flux"]))
            m_kron = zp_kron - 2.5 * np.log10(kron[found_idx[found]])
        zones = zone_of(pts_arr[:, 0], pts_arr[:, 1], data.shape)
        mi = 0
        for i in range(len(pts)):
            row = {
                "mag_in": float(m_in),
                "x": pts_arr[i, 0],
                "y": pts_arr[i, 1],
                "zone": str(zones[i]),
                "r_e": pts[i][2],
                "found": bool(found[i]),
                "m_ap": np.nan,
                "m_kron": np.nan,
            }
            if found[i]:
                row["m_ap"], row["m_kron"] = float(m_ap[mi]), float(m_kron[mi])
                mi += 1
            rows.append(row)
        log(f"  injected r = {m_in:5.2f}: recovered {found.mean():.2f}")
    return Table(rows=rows)


def summarise_injection(inj) -> tuple[Table, Table]:
    comp_rows, transfer_rows = [], []
    for m in np.unique(inj["mag_in"]):
        at = inj[inj["mag_in"] == m]
        for zone in ("inner", "edge", "all"):
            z = at if zone == "all" else at[at["zone"] == zone]
            n = len(z)
            c = float(np.mean(z["found"])) if n else np.nan
            comp_rows.append(
                {
                    "mag": float(m),
                    "zone": zone,
                    "n": n,
                    "C": c,
                    "C_err": float(np.sqrt(c * (1 - c) / n)) if n else np.nan,
                }
            )
        f = at[at["found"]]
        row = {"mag_in": float(m), "n_found": len(f)}
        for est in ("m_ap", "m_kron"):
            d = np.asarray(f[est]) - m
            d = d[np.isfinite(d)]
            if len(d) >= 5:
                med, sig, err = robust(d)
                row.update(
                    {
                        f"{est}_bias": med,
                        f"{est}_bias_err": err,
                        f"{est}_scatter": sig,
                    }
                )
            else:
                row.update(
                    {
                        f"{est}_bias": np.nan,
                        f"{est}_bias_err": np.nan,
                        f"{est}_scatter": np.nan,
                    }
                )
        transfer_rows.append(row)
    return Table(rows=comp_rows), Table(rows=transfer_rows)


def limit_at(comp, zone, level) -> float:
    c = comp[comp["zone"] == zone]
    m, c_val = np.asarray(c["mag"]), np.asarray(c["C"])
    below = np.where(c_val < level)[0]
    if len(below) == 0:
        return float(m[-1])
    i = below[0]
    if i == 0:
        return float(m[0])
    return float(np.interp(level, [c_val[i], c_val[i - 1]], [m[i], m[i - 1]]))