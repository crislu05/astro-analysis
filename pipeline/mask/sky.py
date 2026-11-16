"""Global sky level from a histogram Gaussian fit."""

import numpy as np
from scipy.optimize import curve_fit


def gauss(x: np.ndarray, a: float, mu: float, sigma: float) -> np.ndarray:
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma**2))


def fit_sky_histogram(
    values: np.ndarray,
    lo: float = 3300,
    hi: float = 3600,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Return sky, sigma, bin centres, counts, and Gaussian fit parameters."""
    counts, edges = np.histogram(values, bins=np.arange(lo, hi, 1))
    centres = 0.5 * (edges[:-1] + edges[1:])
    mode = centres[np.argmax(counts)]
    use = centres <= mode + 5
    p, _ = curve_fit(gauss, centres[use], counts[use], p0=[counts.max(), mode, 15])
    return float(p[1]), float(abs(p[2])), centres, counts, p


def fit_sky(
    values: np.ndarray,
    lo: float = 3300,
    hi: float = 3600,
) -> tuple[float, float]:
    sky, sigma, _, _, _ = fit_sky_histogram(values, lo, hi)
    return sky, sigma