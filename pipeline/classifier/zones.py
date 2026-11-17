"""Spatial zones (inner vs edge) — shared with calibrate/counts."""

from __future__ import annotations

import numpy as np

from pipeline.calibrate.config import EDGE_ZONE_PX


def zone_of(x, y, shape) -> np.ndarray:
    ny, nx = shape
    dist = np.minimum.reduce([x, nx - 1 - x, y, ny - 1 - y])
    return np.where(dist < EDGE_ZONE_PX, "edge", "inner")