"""Shared helpers for detection CLI and validation."""

import numpy as np


def instrumental_mag(flux, zp: float) -> np.ndarray:
    """Lab script eq. (5): m = ZP - 2.5 log10(counts)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return zp - 2.5 * np.log10(flux)


def expected_white_noise_ratio() -> float:
    from pipeline.detect.segment import make_kernel

    kernel = make_kernel().array
    return float(np.sqrt(np.sum(np.asarray(kernel) ** 2)))