"""Shared mask geometry on the image grid."""

import numpy as np


def pixel_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    return xx, yy


def mask_circle(
    layer: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    xc: float,
    yc: float,
    r: float,
) -> None:
    ny, nx = layer.shape
    x0, x1 = max(0, int(xc - r - 1)), min(nx, int(xc + r + 2))
    y0, y1 = max(0, int(yc - r - 1)), min(ny, int(yc + r + 2))
    sub_xx = xx[y0:y1, x0:x1]
    sub_yy = yy[y0:y1, x0:x1]
    layer[y0:y1, x0:x1] |= (sub_xx - xc) ** 2 + (sub_yy - yc) ** 2 <= r**2


def mask_line(
    layer: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: float,
) -> None:
    dx, dy = x1 - x0, y1 - y0
    denom = dx * dx + dy * dy
    t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / denom, 0, 1)
    dist = np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy))
    layer |= dist <= width / 2