"""Assign star / galaxy / artefact / unclassified labels."""

from __future__ import annotations

import numpy as np

from pipeline.classifier import config


def classify(mag, metric, flag, locus, loc_tab):
    edges_lo = np.asarray(loc_tab["mag_lo"])
    sig = np.asarray(loc_tab["sigma_used"])
    sep = np.asarray(loc_tab["separation"])
    ok_bins = np.isfinite(sep) & (sep >= config.SEPARATION_SIGMA)
    fail = np.where(~ok_bins & (edges_lo >= config.STAR_FIT_RANGE_MIN))[0]
    mag_limit = float(edges_lo[fail[0]]) if len(fail) else float(edges_lo[-1] + config.BIN)
    sig_at = np.interp(mag, edges_lo + config.BIN / 2, sig)
    dist = (metric - locus(mag)) / sig_at
    cls = np.full(len(mag), "unclassified", dtype="<U12")
    known = np.isfinite(mag) & np.isfinite(metric) & (mag < mag_limit)
    cls[known & (np.abs(dist) <= config.N_SIGMA)] = "star"
    cls[known & (dist > config.N_SIGMA)] = "galaxy"
    cls[known & (dist < -config.N_SIGMA)] = "artefact"
    return cls, dist, mag_limit