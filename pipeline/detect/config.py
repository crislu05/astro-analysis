"""Default parameters for source detection on the KPNO mosaic."""

# Background and noise maps
BKG_BOX = (100, 100)
BKG_FILTER = (5, 5)
BKG_SIGMA_CLIP = 3.0

KERNEL_FWHM_PX = 4.0
KERNEL_SIZE = 9

DETECT_NSIGMA = 4.0
DETECT_NPIXELS = 8

DEBLEND = True
DEBLEND_NLEVELS = 32
DEBLEND_CONTRAST = 0.005

NEAR_MASK_PX = 3
DEFAULT_GAIN = 3.1