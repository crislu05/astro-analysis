"""C3: identify BC95 filters for photographic b, r, and auxiliary bands."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from metcalfe.stella_population.component_c.config import EXPECTED_WINDOWS_AA, PHOTON_COUNTING
from metcalfe.stella_population.photometry import Filter
from metcalfe.stella_population.report import Report


def effective_wavelength(band: Filter):
    weight = band.transmission * (band.wavelengths_aa if PHOTON_COUNTING else 1.0)
    return float(
        np.trapezoid(weight * band.wavelengths_aa, band.wavelengths_aa)
        / np.trapezoid(weight, band.wavelengths_aa)
    )


def half_power_range(band: Filter):
    above = band.wavelengths_aa[band.transmission >= 0.5 * band.transmission.max()]
    return float(above.min()), float(above.max())


def filters_log_entries(path, filter_ids):
    entries = {filter_id: [] for filter_id in filter_ids}
    if path is None or not Path(path).is_file():
        return entries
    for line in Path(path).read_text(errors="replace").splitlines():
        tokens = line.replace(")", " ").replace(".", " ", 1).split()
        if tokens and tokens[0].isdigit() and int(tokens[0]) in entries:
            entries[int(tokens[0])].append(line.strip())
    return entries


def identify_filters(library, filters_log_path=None, report: Report | None = None):
    filter_ids = list(EXPECTED_WINDOWS_AA)
    log_entries = filters_log_entries(filters_log_path, filter_ids)
    print("\nC3. Filters used for the paper's b and r (and B, V, K)")
    print(f"  {'id':>3} {'lambda_eff [A]':>15} {'half-power range [A]':>22}  filters.log")
    rows = {}
    for filter_id in filter_ids:
        band = library.filter(filter_id)
        centre = effective_wavelength(band)
        low, high = half_power_range(band)
        entry = "; ".join(log_entries[filter_id]) or "(no entry found)"
        print(f"  {filter_id:>3} {centre:>15.0f} {low:>10.0f} to {high:<9.0f}  {entry}")
        rows[filter_id] = dict(effective_aa=centre, half_power_aa=(low, high), log=entry)
        if report is not None:
            window = EXPECTED_WINDOWS_AA[filter_id]
            report.check_true(
                f"filter {filter_id}: lambda_eff inside the expected window {window[0]:.0f} to {window[1]:.0f} A",
                window[0] <= centre <= window[1],
            )
    print("  Confirm against the paper's definition of b and r before trusting the fitted tau.")
    return rows