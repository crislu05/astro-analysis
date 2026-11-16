"""Load FITS and persist mask products."""

from pathlib import Path

import numpy as np
from astropy.io import fits

from pipeline.mask.pipeline import build_mask_state
from pipeline.mask.state import MaskState


def load_mosaic(path: str | Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        hdu = hdul[0]
        return hdu.data.astype(float), hdu.header


def run_mosaic_mask(path: str | Path) -> tuple[np.ndarray, MaskState]:
    data, header = load_mosaic(path)
    state = build_mask_state(data, header)
    return state.mask, state