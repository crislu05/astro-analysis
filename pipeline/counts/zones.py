"""Inner/edge zone pixel counts for area weighting."""

from __future__ import annotations

import numpy as np

from pipeline.calibrate.config import EDGE_ZONE_PX


def zone_areas(mask) -> dict[str, int]:
    ny, nx = mask.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    dist = np.minimum.reduce([xx, nx - 1 - xx, yy, ny - 1 - yy])
    good = ~mask
    return {
        "edge": int((good & (dist < EDGE_ZONE_PX)).sum()),
        "inner": int((good & (dist >= EDGE_ZONE_PX)).sum()),
    }