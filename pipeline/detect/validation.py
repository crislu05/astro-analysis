"""Detection validation checks (negative image, injection, uniformity)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from pipeline.detect import config as dconfig
from pipeline.detect import detect
from pipeline.detect.background import BackgroundResult
from pipeline.detect.pipeline import DetectionResult
from pipeline.detect.synthetic import add_gaussian, match, random_positions
from pipeline.detect.util import expected_white_noise_ratio, instrumental_mag


def run_negative_image_test(
    data: np.ndarray,
    mask: np.ndarray,
    bkg: BackgroundResult,
    gain: float,
    zp: float,
    n_real: int,
    good_px: int,
) -> tuple[dict, object]:
    neg = 2 * bkg.background - data
    neg_bkg = BackgroundResult(bkg.background, bkg.rms, neg - bkg.background)
    neg_cat = detect(neg, mask, gain=gain, background=neg_bkg).catalogue
    n_neg = len(neg_cat)
    neg_mag = instrumental_mag(neg_cat["flux"], zp)
    block = {
        "n_false": n_neg,
        "false_fraction": round(n_neg / n_real, 4) if n_real else 0.0,
        "false_per_mpix": round(n_neg / (good_px / 1e6), 2),
        "false_near_mask": int(np.sum(neg_cat["near_mask"])),
        "false_mags": [round(float(m), 2) for m in np.sort(neg_mag)],
    }
    return block, neg_cat


def injection_recovery(
    data: np.ndarray,
    mask: np.ndarray,
    cat,
    bkg: BackgroundResult,
    gain: float,
    zp: float,
    stellar_fwhm: float,
    log,
) -> tuple[list[dict], dict]:
    fwhm_inj = stellar_fwhm if np.isfinite(stellar_fwhm) else dconfig.KERNEL_FWHM_PX
    mags_inj = np.arange(14.0, 19.76, 0.25)
    n_per_run = 300
    allowed = ~ndimage.binary_dilation(mask, iterations=10)
    real_x, real_y = np.asarray(cat["x"]), np.asarray(cat["y"])
    rec_rows = []
    for k, m_in in enumerate(mags_inj):
        flux_in = 10 ** ((zp - m_in) / 2.5)
        pos = random_positions(n_per_run, allowed, min_sep=40, seed=100 + k)
        img = data.copy()
        for x, y in pos:
            add_gaussian(img, x, y, flux_in, fwhm_inj)
        inj_bkg = BackgroundResult(bkg.background, bkg.rms, img - bkg.background)
        c = detect(img, mask, gain=gain, background=inj_bkg).catalogue
        cx, cy = np.asarray(c["x"]), np.asarray(c["y"])
        idx = match(pos[:, 0], pos[:, 1], cx, cy, radius=2.0)
        found = idx >= 0
        pre = match(pos[:, 0], pos[:, 1], real_x, real_y, radius=2.0) >= 0
        clean = found & ~pre
        dpos = np.hypot(cx[idx[clean]] - pos[clean, 0], cy[idx[clean]] - pos[clean, 1])
        fratio = np.asarray(c["flux"])[idx[clean]] / flux_in
        rec_rows.append(
            {
                "mag": float(m_in),
                "flux": float(flux_in),
                "n": int(len(pos)),
                "completeness": float(found.mean()),
                "completeness_err": float(
                    np.sqrt(found.mean() * (1 - found.mean()) / len(pos))
                ),
                "median_pos_err_px": float(np.median(dpos)) if len(dpos) else float("nan"),
                "median_flux_ratio": float(np.median(fratio)) if len(fratio) else float("nan"),
                "flux_ratio_p16": float(np.percentile(fratio, 16)) if len(fratio) else float("nan"),
                "flux_ratio_p84": float(np.percentile(fratio, 84)) if len(fratio) else float("nan"),
            }
        )
        log(f"  inject m={m_in:.2f}: completeness {found.mean():.2f}")
    comp = np.array([r["completeness"] for r in rec_rows])

    def mag_at(level):
        below = np.where(comp < level)[0]
        if len(below) == 0 or below[0] == 0:
            return float("nan")
        i = below[0]
        return float(np.interp(level, [comp[i], comp[i - 1]], [mags_inj[i], mags_inj[i - 1]]))

    limits = {"mag_90": round(mag_at(0.9), 2), "mag_50": round(mag_at(0.5), 2)}
    return rec_rows, limits


def spatial_uniformity(
    data: np.ndarray,
    mask: np.ndarray,
    cat,
    bkg: BackgroundResult,
    mag_90: float,
) -> dict:
    ny, nx = data.shape
    xs, ys = np.asarray(cat["x"]), np.asarray(cat["y"])
    yy, xx = np.mgrid[0:ny, 0:nx]
    edge_dist = np.minimum.reduce([xx, nx - 1 - xx, yy, ny - 1 - yy])
    zones = {"edge (150 to 450 px from border)": (150, 450), "inner (> 450 px)": (450, 10**6)}
    m_bright = np.asarray(cat["mag"]) < mag_90
    uni = {}
    src_edge = edge_dist[
        np.clip(ys.round().astype(int), 0, ny - 1),
        np.clip(xs.round().astype(int), 0, nx - 1),
    ]
    for name, (lo, hi) in zones.items():
        zone = (edge_dist >= lo) & (edge_dist < hi) & ~mask
        in_zone = (src_edge >= lo) & (src_edge < hi)
        uni[name] = {
            "good_px": int(zone.sum()),
            "median_rms": round(float(np.median(bkg.rms[zone])), 2),
            "density_all_per_mpix": round(in_zone.sum() / (zone.sum() / 1e6), 1),
            "density_bright_per_mpix": round((in_zone & m_bright).sum() / (zone.sum() / 1e6), 1),
        }
    return uni


def detection_summary(res: DetectionResult, area_deg2: float, good_px: int) -> dict:
    cat = res.catalogue
    n_real = len(cat)
    return {
        "n_sources": n_real,
        "density_per_deg2": round(n_real / area_deg2) if area_deg2 else 0,
        "filter_noise_ratio": round(res.segmentation.filter_noise_ratio, 3),
        "white_noise_ratio_expected": round(expected_white_noise_ratio(), 3),
        "n_near_mask": int(np.sum(cat["near_mask"])),
        "median_snr": round(float(np.median(cat["snr"])), 1) if n_real else 0.0,
        "nsigma": dconfig.DETECT_NSIGMA,
        "npixels": dconfig.DETECT_NPIXELS,
        "kernel_fwhm_px": dconfig.KERNEL_FWHM_PX,
        "good_pixels": good_px,
    }