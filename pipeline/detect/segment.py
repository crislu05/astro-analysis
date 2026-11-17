"""Matched-filter thresholding, segmentation and deblending."""

from dataclasses import dataclass

import numpy as np
from astropy.convolution import convolve_fft
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import (
    SegmentationImage,
    deblend_sources,
    detect_sources,
    make_2dgaussian_kernel,
)

from pipeline.detect import config


@dataclass
class SegmentResult:
    segm: SegmentationImage | None
    convolved: np.ndarray
    threshold: np.ndarray
    filter_noise_ratio: float


def make_kernel(fwhm: float | None = None, size: int | None = None):
    fwhm = config.KERNEL_FWHM_PX if fwhm is None else fwhm
    size = config.KERNEL_SIZE if size is None else size
    return make_2dgaussian_kernel(fwhm, size=size)


def filter_image(data_sub: np.ndarray, mask: np.ndarray, kernel) -> np.ndarray:
    clean = np.where(mask, 0.0, data_sub)
    return convolve_fft(clean, kernel, normalize_kernel=True, allow_huge=True)


def measure_filter_noise_ratio(
    convolved: np.ndarray, rms: np.ndarray, mask: np.ndarray
) -> float:
    good = ~mask & (rms > 0)
    if good.sum() < 100:
        return 1.0
    _, _, std = sigma_clipped_stats(convolved[good] / rms[good], sigma=3.0, maxiters=10)
    return float(std) if np.isfinite(std) and std > 0 else 1.0


def segment(
    data_sub: np.ndarray,
    rms: np.ndarray,
    mask: np.ndarray,
    nsigma: float | None = None,
    npixels: int | None = None,
    kernel_fwhm: float | None = None,
    deblend: bool | None = None,
) -> SegmentResult:
    nsigma = config.DETECT_NSIGMA if nsigma is None else nsigma
    npixels = config.DETECT_NPIXELS if npixels is None else npixels
    deblend = config.DEBLEND if deblend is None else deblend

    kernel = make_kernel(kernel_fwhm)
    convolved = filter_image(data_sub, mask, kernel)
    ratio = measure_filter_noise_ratio(convolved, rms, mask)
    threshold = nsigma * ratio * rms

    segm = detect_sources(convolved, threshold, npixels=npixels, mask=mask)
    if segm is not None and deblend:
        segm = deblend_sources(
            convolved,
            segm,
            npixels=npixels,
            nlevels=config.DEBLEND_NLEVELS,
            contrast=config.DEBLEND_CONTRAST,
            progress_bar=False,
        )
    return SegmentResult(segm, convolved, threshold, ratio)