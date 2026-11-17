"""Robust location and scale estimators."""

from __future__ import annotations

import numpy as np


def robust(values) -> tuple[float, float, float]:
    v = np.asarray(values, float)
    med = float(np.median(v))
    sig = float(1.4826 * np.median(np.abs(v - med)))
    return med, sig, float(1.2533 * sig / np.sqrt(len(v)))