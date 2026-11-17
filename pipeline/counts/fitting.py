"""Weighted log-log fits and literature comparison."""

from __future__ import annotations

import numpy as np
from scipy import stats

from pipeline.counts import config


def weighted_line(x, y, sy):
    w = 1 / sy**2
    design = np.c_[np.ones_like(x), x]
    cov = np.linalg.inv(design.T @ (design * w[:, None]))
    p = cov @ (design.T @ (w * y))
    chi2 = float(np.sum(w * (y - design @ p) ** 2))
    dof = len(x) - 2
    return p, cov, chi2, dof


def fit_range(centres, logn, slog, sel, label):
    if sel.sum() < 3:
        return None
    p, cov, chi2, dof = weighted_line(centres[sel], logn[sel], slog[sel])
    scale = max(1.0, np.sqrt(chi2 / dof)) if dof > 0 else 1.0
    slope, err = float(p[1]), float(np.sqrt(cov[1, 1]) * scale)
    z = (slope - config.EUCLID) / err
    res = {
        "range": [float(centres[sel].min() - config.BIN / 2), float(centres[sel].max() + config.BIN / 2)],
        "n_bins": int(sel.sum()),
        "slope": slope,
        "slope_err": err,
        "intercept": float(p[0]),
        "chi2": chi2,
        "dof": dof,
        "chi2_p": float(1 - stats.chi2.cdf(chi2, dof)) if dof > 0 else np.nan,
        "err_scaled_by": scale,
        "z_vs_euclidean": float(z),
        "p_vs_euclidean_two_sided": float(2 * stats.norm.sf(abs(z))),
    }
    print(
        f"  {label:7s} r {res['range'][0]:.1f} to {res['range'][1]:.1f}: slope {slope:.3f} +- {err:.3f} "
        f"(chi2/dof {chi2:.1f}/{dof}); vs 0.6: {z:+.1f} sigma, p = {res['p_vs_euclidean_two_sided']:.2g}"
    )
    return res


def compare(centres, n, sn, ref_m, ref_n, ref_N, label, shift=0.0, shift_err=0.0):
    ref_mag = ref_m + shift
    ok = (centres >= ref_mag.min()) & (centres <= ref_mag.max()) & np.isfinite(n) & (n > 0)
    if ok.sum() == 0:
        return None
    log_ref = np.interp(centres[ok], ref_mag, np.log10(ref_n))
    rel_ref = np.interp(centres[ok], ref_mag, 1 / np.sqrt(ref_N))
    slope_ref = np.gradient(np.log10(ref_n), ref_mag)
    rel_shift = np.log(10) * np.interp(centres[ok], ref_mag, slope_ref) * shift_err
    ratio = n[ok] / 10**log_ref
    s_ratio = ratio * np.sqrt((sn[ok] / n[ok]) ** 2 + rel_ref**2 + rel_shift**2)
    chi2 = float(np.sum(((ratio - 1) / s_ratio) ** 2))
    w = 1 / s_ratio**2
    mean_ratio = float(np.sum(w * ratio) / np.sum(w))
    print(
        f"  vs {label}: {ok.sum()} bins, weighted mean ratio {mean_ratio:.2f} +- "
        f"{1 / np.sqrt(np.sum(w)):.2f}, chi2 {chi2:.1f}/{ok.sum()} "
        f"(p = {1 - stats.chi2.cdf(chi2, ok.sum()):.2g})"
    )
    return {
        "centres": centres[ok].tolist(),
        "ratio": ratio.tolist(),
        "ratio_err": s_ratio.tolist(),
        "mean_ratio": mean_ratio,
        "mean_ratio_err": float(1 / np.sqrt(np.sum(w))),
        "chi2": chi2,
        "n_bins": int(ok.sum()),
        "p": float(1 - stats.chi2.cdf(chi2, ok.sum())),
    }