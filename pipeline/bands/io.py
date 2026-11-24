"""FITS I/O for archive tiles and lab mosaics."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

warnings.simplefilter("ignore", FITSFixedWarning)


def open_image(path: Path, *, memmap: bool = True):
    """Return (hdul, hdu, merged header, celestial WCS) for the first 2-D image HDU."""
    hdul = fits.open(path, memmap=memmap)
    for hdu in hdul:
        if hdu.header.get("NAXIS") == 2:
            header = hdul[0].header.copy()
            header.extend(hdu.header, update=True)
            wcs = WCS(header).celestial
            if wcs.has_celestial:
                return hdul, hdu, header, wcs
    hdul.close()
    raise ValueError(f"No 2-D celestial image in {path}")


def load_lab_mosaic(path: Path):
    """Load lab mosaic data, header, and celestial WCS."""
    hdul, hdu, _, wcs = open_image(path, memmap=False)
    try:
        data = np.asarray(hdu.data, dtype=float)
        header = hdu.header.copy()
    finally:
        hdul.close()
    return data, header, wcs, data.shape