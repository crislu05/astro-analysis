"""Magnitude calibration (lab eq. 5)."""

import numpy as np


def counts_to_mag(flux, zp, exptime=None):
    f = np.asarray(flux, float)
    if exptime is not None:
        f = f / exptime
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f > 0, zp - 2.5 * np.log10(f), np.nan)


def mag_error(flux, flux_err):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            np.asarray(flux) > 0,
            2.5 / np.log(10) * np.asarray(flux_err) / np.asarray(flux),
            np.nan,
        )