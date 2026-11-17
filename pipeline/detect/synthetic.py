"""Synthetic images and source injection for testing detection."""

import numpy as np


def gaussian_stamp(flux: float, fwhm: float, dx: float = 0.0, dy: float = 0.0, half: int | None = None):
    sigma = fwhm / 2.3548
    half = int(np.ceil(4 * sigma)) if half is None else half
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    g = np.exp(-((xx - dx) ** 2 + (yy - dy) ** 2) / (2 * sigma**2))
    return flux * g / g.sum(), half


def add_gaussian(image: np.ndarray, x: float, y: float, flux: float, fwhm: float) -> None:
    xi, yi = int(round(x)), int(round(y))
    stamp, half = gaussian_stamp(flux, fwhm, x - xi, y - yi)
    ny, nx = image.shape
    y0, y1 = yi - half, yi + half + 1
    x0, x1 = xi - half, xi + half + 1
    sy0, sx0 = max(0, -y0), max(0, -x0)
    sy1 = stamp.shape[0] - max(0, y1 - ny)
    sx1 = stamp.shape[1] - max(0, x1 - nx)
    image[max(0, y0) : min(ny, y1), max(0, x0) : min(nx, x1)] += stamp[sy0:sy1, sx0:sx1]


def noise_image(shape, sky: float = 3420.0, sigma: float = 10.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(sky, sigma, shape)


def random_positions(n: int, allowed: np.ndarray, min_sep: float, seed: int = 0, margin: int = 20):
    rng = np.random.default_rng(seed)
    ny, nx = allowed.shape
    pts: list[tuple[float, float]] = []
    tries = 0
    while len(pts) < n and tries < 200 * n:
        tries += 1
        x = rng.uniform(margin, nx - margin)
        y = rng.uniform(margin, ny - margin)
        if not allowed[int(y), int(x)]:
            continue
        if pts and np.min(np.hypot(*(np.array(pts) - (x, y)).T)) < min_sep:
            continue
        pts.append((x, y))
    return np.array(pts)


def match(xt, yt, xd, yd, radius: float):
    idx = np.full(len(xt), -1)
    if len(xd) == 0:
        return idx
    for i, (x, y) in enumerate(zip(xt, yt)):
        d = np.hypot(xd - x, yd - y)
        j = int(np.argmin(d))
        if d[j] <= radius:
            idx[i] = j
    return idx