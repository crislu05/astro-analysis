"""Parameters and published reference counts for number counts."""

import numpy as np

BIN = 0.5
MAG_EDGES = np.arange(18.0, 26.01, BIN)
SDSS_STAR_LIMIT = 21.5
C_MIN = 0.5
C_PLOT_MIN = 0.3
FIT_C_MIN = 0.7
MIN_GAL = 5
BREAK_MAG = 22.0
EUCLID = 0.6

YASUDA = np.array(
    [
        (18.25, 113.2, 16890),
        (18.75, 197.4, 29443),
        (19.25, 336.5, 50206),
        (19.75, 538.4, 80324),
        (20.25, 825.9, 123213),
        (20.75, 1216.0, 181428),
        (21.25, 1627.0, 242743),
    ]
)

METCALFE = np.array(
    [
        (21.25, 1825, 23),
        (21.75, 3275, 41),
        (22.25, 4700, 59),
        (22.75, 8050, 101),
        (23.25, 12275, 154),
        (23.75, 20400, 224),
        (24.25, 34700, 373),
        (24.75, 50100, 493),
        (25.25, 79400, 696),
        (25.75, 109600, 833),
    ]
)
R_TO_R_SHIFT = 0.25
R_TO_R_SHIFT_ERR = 0.05