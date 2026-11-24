"""BC95 filter library (csp.f FILTER0 / FILTERBIN.RES)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from metcalfe.stella_population.photometry import Filter

MAX_FILTERS = 100
MAX_POINTS = 250


@dataclass(frozen=True)
class FilterLibrary:
    """Fixed-size tables as written by BC95 subroutine FILTER0."""

    number_of_filters: int
    points_per_filter: np.ndarray
    wavelengths_aa: np.ndarray
    transmission: np.ndarray

    def filter(self, filter_id: int) -> Filter:
        """Return filter by BC95 index (1-based, matches filters.log)."""
        if not 1 <= filter_id <= MAX_FILTERS:
            raise ValueError(f"filter_id must be in [1, {MAX_FILTERS}], got {filter_id}")
        index = filter_id - 1
        count = int(self.points_per_filter[index])
        if count <= 0:
            raise ValueError(f"filter {filter_id} has no points in FILTERBIN.RES")
        wavelengths = self.wavelengths_aa[index, :count].copy()
        transmission = self.transmission[index, :count].copy()
        return Filter(f"bc95_{filter_id}", wavelengths, transmission)


def read_filterbin(path: str | Path) -> FilterLibrary:
    """Read FILTERBIN.RES (big-endian Fortran unformatted, FILTER0 layout)."""
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError(f"FILTERBIN.RES too small: {path}")
    record_length = struct.unpack(">i", data[:4])[0]
    payload = data[4 : 4 + record_length]
    if len(payload) != record_length:
        raise ValueError(f"corrupt FILTERBIN.RES record length at {path}")

    number_of_filters = struct.unpack(">i", payload[:4])[0]
    points_per_filter = np.frombuffer(payload[4:404], dtype=">i4").copy()
    table = np.frombuffer(payload[404:], dtype=">f4")
    expected = 2 * MAX_FILTERS * MAX_POINTS
    if table.size < expected:
        raise ValueError(f"FILTERBIN.RES payload too short: {table.size} floats, need {expected}")
    # Fortran column order: r(250, 100, 2) -> index (point, filter, lambda/T)
    reshaped = table[:expected].reshape(2, MAX_FILTERS, MAX_POINTS)
    wavelengths = reshaped[0].copy()
    transmission = reshaped[1].copy()
    return FilterLibrary(number_of_filters, points_per_filter, wavelengths, transmission)