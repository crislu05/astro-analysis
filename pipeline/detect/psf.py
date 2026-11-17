"""PSF size from isolated stars in the catalogue."""

import numpy as np
from astropy.modeling import fitting, models
from astropy.table import Table


def measure_stellar_fwhm(data: np.ndarray, cat: Table, n_stars: int = 40, half: int = 12):
    sel = cat[
        (cat["snr"] > 80)
        & ~cat["near_mask"]
        & (cat["ellipticity"] < 0.1)
        & (cat["peak"] < 20000)
    ]
    sel.sort("fwhm")
    fitter = fitting.LevMarLSQFitter()
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    fw = []
    for row in sel[:n_stars]:
        x, y = int(round(row["x"])), int(round(row["y"]))
        cut = data[y - half : y + half + 1, x - half : x + half + 1]
        if cut.shape != xx.shape:
            continue
        model = (
            models.Gaussian2D(cut.max() - np.median(cut), 0, 0, 2, 2)
            + models.Const2D(np.median(cut))
        )
        p = fitter(model, xx, yy, cut)
        fw.append(2.3548 * np.sqrt(abs(p.x_stddev_0 * p.y_stddev_0)))
    fw = np.array(fw)
    stellar = float(np.median(fw)) if len(fw) else float("nan")
    return stellar, fw