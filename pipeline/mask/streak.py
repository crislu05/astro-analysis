"""Diffuse streak lines via background subtraction and Hough transform."""

from dataclasses import dataclass

import numpy as np
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from scipy import ndimage
from skimage.transform import probabilistic_hough_line

from pipeline.mask import config
from pipeline.mask.geometry import mask_line
from pipeline.mask.layers import MaskLayers


@dataclass
class StreakLine:
    x0: float
    y0: float
    x1: float
    y1: float
    length: float


def subtract_background_for_streaks(
    data: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    bkg = Background2D(
        data,
        box_size=config.BKG_BOX,
        filter_size=config.BKG_FILTER,
        mask=mask,
        sigma_clip=SigmaClip(sigma=config.BKG_SIGMA_CLIP),
        bkg_estimator=MedianBackground(),
    )
    return data - bkg.background


def detect_streaks(
    data_sub: np.ndarray,
    layers: MaskLayers,
    xx: np.ndarray,
    yy: np.ndarray,
    sigma: float,
) -> list[StreakLine]:
    s = config.STREAK_SMOOTH_SIGMA
    smooth = ndimage.gaussian_filter(data_sub, s)
    sigma_smooth = sigma / (2 * np.sqrt(np.pi) * s)
    combined = layers.combined()
    binary = (smooth > config.STREAK_THRESHOLD_NSIGMA * sigma_smooth) & ~combined
    binary = ndimage.binary_dilation(binary, iterations=1)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 1800, endpoint=False)
    segments = probabilistic_hough_line(
        binary,
        threshold=config.STREAK_HOUGH_THRESHOLD,
        line_length=config.STREAK_LINE_LENGTH,
        line_gap=config.STREAK_LINE_GAP,
        theta=theta,
        rng=0,
    )
    lines: list[StreakLine] = []
    for (x0, y0), (x1, y1) in segments:
        length = float(np.hypot(x1 - x0, y1 - y0))
        if length < config.STREAK_MIN_LENGTH:
            continue
        mask_line(
            layers.streak,
            xx,
            yy,
            x0,
            y0,
            x1,
            y1,
            config.STREAK_WIDTH,
        )
        lines.append(StreakLine(x0, y0, x1, y1, length))
    return lines


def apply_streaks(
    data: np.ndarray,
    layers: MaskLayers,
    xx: np.ndarray,
    yy: np.ndarray,
    sigma: float,
) -> list[StreakLine]:
    mask = layers.combined()
    data_sub = subtract_background_for_streaks(data, mask)
    return detect_streaks(data_sub, layers, xx, yy, sigma)