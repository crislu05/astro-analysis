"""Gaussian helpers for photometry self-tests."""

import numpy as np

from pipeline.photometry import config


def add_gaussian(image, x, y, flux, fwhm):
    sigma = fwhm / 2.3548
    half = int(np.ceil(5 * sigma))
    xi, yi = int(round(x)), int(round(y))
    yy, xx = np.mgrid[yi - half : yi + half + 1, xi - half : xi + half + 1]
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    g *= flux / (2 * np.pi * sigma**2)
    ny, nx = image.shape
    ok = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
    np.add.at(image, (yy[ok], xx[ok]), g[ok])


def gaussian_fraction_within(r, fwhm):
    sigma = fwhm / 2.3548
    return 1 - np.exp(-(r**2) / (2 * sigma**2))