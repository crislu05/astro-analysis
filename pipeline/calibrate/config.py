"""Default parameters for photometric calibration (B2 lab)."""

import numpy as np

SKY_LEVEL = 3418.5
SAT_FRAC = 0.8
ZP_STAR_RANGE = (17.0, 20.0)
KRON_STAR_RANGE = (19.0, 21.5)
EDGE_ZONE_PX = 450
INJ_MAGS = np.arange(21.0, 26.01, 0.5)
INJ_MIN_SEP = 60.0
MATCH_RADIUS = 3.0
SIZE_SAMPLE_RANGE = (21.0, 23.5)
SERSIC_N = 1.0
OVERSAMPLE = 5
BIAS_MIN_COMPLETENESS = 0.8
AP_CORR_FACTOR = 1.88