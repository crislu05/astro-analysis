"""Correlated noise factor from empty apertures."""

import numpy as np
from astropy.stats import sigma_clip
from scipy import ndimage

from pipeline.photometry import config
from pipeline.photometry.aperture import aperture_sum, local_sky
from pipeline.photometry.inputs import PhotometryInputs


def empty_aperture_factor(
    inp: PhotometryInputs,
    r_ap: float = config.R_AP,
    n: int = config.N_EMPTY_APERTURES,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    ny, nx = inp.data.shape
    clear = ~ndimage.binary_dilation(inp.not_sky, iterations=int(np.ceil(r_ap)) + 1)
    sums, pix_rms = [], []
    tries = 0
    while len(sums) < n and tries < 50 * n:
        tries += 1
        x = rng.uniform(config.R_OUT, nx - config.R_OUT)
        y = rng.uniform(config.R_OUT, ny - config.R_OUT)
        if not clear[int(y), int(x)]:
            continue
        sky, sky_std, n_sky, bad = local_sky(inp, x, y)
        if bad or not np.isfinite(sky):
            continue
        f, area, _, _, _ = aperture_sum(inp, x, y, r_ap, sky, own_label=-1)
        sums.append(f)
        pix_rms.append(sky_std * np.sqrt(area))
    sums = np.asarray(sums)
    if config.CORR_ESTIMATOR == "robust":
        measured = float(1.4826 * np.median(np.abs(sums - np.median(sums))))
    else:
        measured = float(np.std(sigma_clip(sums, sigma=4, maxiters=10, masked=False)))
    expected = float(np.median(pix_rms))
    return measured / expected, measured, expected, sums