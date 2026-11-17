"""Completeness tables with chance-match floor correction."""

from __future__ import annotations

import numpy as np


def completeness_at(comp, zone, centres, floor):
    c = comp[comp["zone"] == zone]
    m, c_val, err = np.asarray(c["mag"]), np.asarray(c["C"]), np.asarray(c["C_err"])
    corrected = np.clip((c_val - floor) / (1 - floor), 0, 1)
    err_corr = err / (1 - floor)
    return np.interp(centres, m, corrected), np.interp(centres, m, err_corr)