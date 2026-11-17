"""2D background and rms maps, estimated with the artefact mask applied."""

from dataclasses import dataclass

import numpy as np
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground, StdBackgroundRMS

from pipeline.detect import config


@dataclass
class BackgroundResult:
    background: np.ndarray
    rms: np.ndarray
    data_sub: np.ndarray


def estimate_background(data: np.ndarray, mask: np.ndarray) -> BackgroundResult:
    """Smooth sky and local noise maps. Masked pixels are ignored."""
    bkg = Background2D(
        data,
        box_size=config.BKG_BOX,
        filter_size=config.BKG_FILTER,
        mask=mask,
        sigma_clip=SigmaClip(sigma=config.BKG_SIGMA_CLIP),
        bkg_estimator=MedianBackground(),
        bkgrms_estimator=StdBackgroundRMS(),
    )
    return BackgroundResult(
        background=bkg.background,
        rms=bkg.background_rms,
        data_sub=data - bkg.background,
    )