"""Load mosaic and detection products for photometry."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table


def load_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(bool)
    return fits.getdata(path).astype(bool)


def load_detection_inputs(
    mosaic: Path,
    catalogue: Path,
    segmentation: Path,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, fits.Header, Table, np.ndarray, np.ndarray, float]:
    with fits.open(mosaic) as hdul:
        hdu = hdul[0]
        data = hdu.data.astype(float)
        header = hdu.header
    cat = Table.read(catalogue)
    segm = fits.getdata(segmentation).astype(np.int64)
    gain = float(header.get("GAIN", 3.1))
    if mask is None:
        from pipeline.mask import build_mask_state

        mask = build_mask_state(data, header).mask
    else:
        mask = np.asarray(mask, dtype=bool)
    return data, header, cat, segm, mask, gain