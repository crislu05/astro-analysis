"""Orchestrate detection: background -> matched filter -> segmentation -> catalogue."""

from dataclasses import dataclass

import numpy as np
from astropy.table import Table

from pipeline.detect.background import BackgroundResult, estimate_background
from pipeline.detect.catalogue import build_catalogue
from pipeline.detect.segment import SegmentResult, segment


@dataclass
class DetectionResult:
    catalogue: Table
    background: BackgroundResult
    segmentation: SegmentResult
    mask: np.ndarray


def detect(
    data: np.ndarray,
    mask: np.ndarray,
    gain: float | None = None,
    background: BackgroundResult | None = None,
    **segment_kwargs,
) -> DetectionResult:
    data = np.asarray(data, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    bkg = estimate_background(data, mask) if background is None else background
    data_sub = data - bkg.background
    seg = segment(data_sub, bkg.rms, mask, **segment_kwargs)
    cat = build_catalogue(
        data_sub, seg.segm, bkg.rms, mask, convolved=seg.convolved, gain=gain
    )
    return DetectionResult(cat, bkg, seg, mask)


def detect_from_mask_state(state, gain: float | None = None, **segment_kwargs):
    return detect(state.data, state.mask, gain=gain, **segment_kwargs)