"""Cross-correlation WCS correction between lab and archive mosaics."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from pipeline.bands.config import PATCH_HALF, SEARCH_RADIUS
from pipeline.bands.resample import ShiftModel, design_matrix


def normalise_patch(values, valid):
    """High-pass and compress dynamic range for robust matching."""
    v = values.copy()
    v[~valid] = np.median(v[valid])
    v -= gaussian_filter(v, 12)
    noise = 1.4826 * np.median(np.abs(v[valid] - np.median(v[valid])))
    v = np.arcsinh(np.clip(v / max(noise, 1e-9), -5, 100))
    return v - v.mean()


def measure_offsets(lab, image, lab_bad, step=300):
    """Return patch centres and (dx, dy) shifts: lab(x) ~ image(x - shift)."""
    ny, nx = lab.shape
    h = PATCH_HALF
    r = SEARCH_RADIUS
    positions, shifts = [], []

    for yc in range(h + 22, ny - h - 22, step):
        for xc in range(h + 22, nx - h - 22, step):
            window = np.s_[yc - h : yc + h, xc - h : xc + h]
            a = lab[window]
            b = image[window]
            valid = np.isfinite(a) & np.isfinite(b) & ~lab_bad[window]
            if valid.mean() < 0.99:
                continue

            a = normalise_patch(a.astype(float), valid)
            b = normalise_patch(b.astype(float), valid)
            corr = fftconvolve(a, b[::-1, ::-1], mode="same")
            cy, cx = np.array(corr.shape) // 2
            small = corr[cy - r : cy + r + 1, cx - r : cx + r + 1]
            py, px = np.unravel_index(np.argmax(small), small.shape)
            if py in (0, 2 * r) or px in (0, 2 * r):
                continue
            score = small[py, px] / np.sqrt(np.sum(a * a) * np.sum(b * b))
            if score < 0.3:
                continue

            def parabola_peak(left, mid, right):
                denominator = left - 2 * mid + right
                return 0.5 * (left - right) / denominator if denominator else 0.0

            dx = px - r + parabola_peak(small[py, px - 1], small[py, px], small[py, px + 1])
            dy = py - r + parabola_peak(small[py - 1, px], small[py, px], small[py + 1, px])
            positions.append([xc, yc])
            shifts.append([dx, dy])
    return np.array(positions, dtype=float), np.array(shifts, dtype=float)


def fit_shift_model(positions, shifts, shape, max_degree=3):
    """Polynomial degree via 5-fold cross-validation, then robust fit."""
    if len(positions) == 0:
        from pipeline.bands.resample import identity_shift_model

        return identity_shift_model(shape), 0, float("nan")

    def robust_fit(points, values, degree):
        terms = design_matrix(points, degree, shape)
        keep = np.ones(len(points), bool)
        for _ in range(5):
            coefficients = np.linalg.lstsq(terms[keep], values[keep], rcond=None)[0]
            residual = np.linalg.norm(values - terms @ coefficients, axis=1)
            centre = np.median(residual[keep])
            spread = 1.4826 * np.median(np.abs(residual[keep] - centre))
            new_keep = residual < max(1.0, centre + 3 * spread)
            if new_keep.sum() < terms.shape[1] + 3:
                break
            keep = new_keep
        return coefficients

    folds = np.random.default_rng(42).permutation(len(positions)) % 5
    trials = []
    for degree in range(1, max_degree + 1):
        n_terms = (degree + 1) * (degree + 2) // 2
        if len(positions) < 2 * n_terms + 5:
            break
        held_out = []
        for k in range(5):
            train = folds != k
            coefficients = robust_fit(positions[train], shifts[train], degree)
            prediction = design_matrix(positions[~train], degree, shape) @ coefficients
            held_out.extend(np.linalg.norm(shifts[~train] - prediction, axis=1))
        trials.append((float(np.median(held_out)), degree))
        print(f"  degree {degree}: held-out median residual {np.median(held_out):.2f} px")

    if not trials:
        from pipeline.bands.resample import identity_shift_model

        return identity_shift_model(shape), 0, float("nan")

    best_residual, best_degree = min(trials)
    model = ShiftModel(robust_fit(positions, shifts, best_degree), best_degree, shape)
    return model, best_degree, best_residual


def offset_check(lab, image, lab_bad):
    """Median residual offset (px) between lab and registered image."""
    _, residual_shifts = measure_offsets(lab, image, lab_bad, step=450)
    if len(residual_shifts) == 0:
        return np.nan, 0
    return float(np.median(np.linalg.norm(residual_shifts, axis=1))), len(residual_shifts)