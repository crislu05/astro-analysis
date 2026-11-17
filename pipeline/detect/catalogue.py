"""Per-source measurements from the segmentation image."""

import numpy as np
from astropy.table import Table
from photutils.segmentation import SegmentationImage, SourceCatalog
from photutils.utils import calc_total_error
from scipy import ndimage

from pipeline.detect import config

ATTRIBUTES = {
    "label": ("label",),
    "x": ("x_centroid", "xcentroid"),
    "y": ("y_centroid", "ycentroid"),
    "area": ("area",),
    "flux": ("segment_flux",),
    "flux_err": ("segment_flux_err", "segment_fluxerr"),
    "peak": ("max_value",),
    "fwhm": ("fwhm",),
    "ellipticity": ("ellipticity",),
    "a": ("semimajor_axis", "semimajor_sigma"),
    "b": ("semiminor_axis", "semiminor_sigma"),
    "theta": ("orientation",),
}
EXTRA = ["snr", "near_mask", "local_rms"]


def _get(cat: SourceCatalog, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in dir(type(cat)):
            values = getattr(cat, name)
            return np.asarray(getattr(values, "value", values), dtype=float)
    raise AttributeError(f"SourceCatalog has none of {names}")


def empty_catalogue() -> Table:
    return Table(names=list(ATTRIBUTES) + EXTRA)


def build_catalogue(
    data_sub: np.ndarray,
    segm: SegmentationImage | None,
    rms: np.ndarray,
    mask: np.ndarray,
    convolved: np.ndarray | None = None,
    gain: float | None = None,
) -> Table:
    if segm is None or segm.nlabels == 0:
        return empty_catalogue()
    gain = config.DEFAULT_GAIN if gain is None else gain
    error = calc_total_error(data_sub, rms, gain)
    cat = SourceCatalog(data_sub, segm, error=error, mask=mask, convolved_data=convolved)

    tbl = Table({col: _get(cat, names) for col, names in ATTRIBUTES.items()})
    tbl["label"] = tbl["label"].astype(int)
    tbl["snr"] = tbl["flux"] / tbl["flux_err"]

    grown_mask = ndimage.binary_dilation(mask, iterations=config.NEAR_MASK_PX)
    touched = np.unique(segm.data[grown_mask & (segm.data > 0)])
    tbl["near_mask"] = np.isin(tbl["label"], touched)

    xi = np.clip(np.round(tbl["x"]).astype(int), 0, rms.shape[1] - 1)
    yi = np.clip(np.round(tbl["y"]).astype(int), 0, rms.shape[0] - 1)
    tbl["local_rms"] = rms[yi, xi]
    return tbl