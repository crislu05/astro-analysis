"""Real BC95 run: C3–C6 on disk data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from metcalfe.cosmology.config import LEGACY_Q005, UPDATED, UPDATED_TGAL_12P5
from metcalfe.cosmology.core import Cosmology
from metcalfe.stella_population.component_c.config import (
    B_PHOTOGRAPHIC_ID,
    GALAXY_TYPES,
    JOHNSON_B_ID,
    JOHNSON_K_ID,
    JOHNSON_V_ID,
    LEGACY_TAU_TOLERANCE,
    R_PHOTOGRAPHIC_ID,
    TAU_MAX_GYR,
    TAU_MIN_GYR,
    Bands,
)
from metcalfe.stella_population.component_c.filters_c3 import identify_filters
from metcalfe.stella_population.component_c.ke_grid import k_plus_e_grid
from metcalfe.stella_population.component_c.mass_light import mass_to_light_b, vega_units_look_physical
from metcalfe.stella_population.component_c.plots import save_evolution_figure
from metcalfe.stella_population.component_c.tau import fit_all_types, print_fits
from metcalfe.stella_population.data_paths import BC95Paths, default_bc95_paths
from metcalfe.stella_population.filters import read_filterbin
from metcalfe.stella_population.ised import read_ised
from metcalfe.stella_population.photometry import read_vega_spectrum, vega_colour
from metcalfe.stella_population.report import Report
from metcalfe.stella_population.sfr import exponential_sfr_spectrum


def run_on_bc95(output_dir, paths: BC95Paths | None = None):
    paths = paths or default_bc95_paths()
    missing = [line for line in paths.missing() if "tau1" not in line and "color" not in line]
    if missing:
        print("\nBC95 data not found, skipping the real run:")
        for line in missing:
            print(f"  {line}")
        return None

    report = Report()
    ssp = read_ised(paths.ssp_ised, "bc95_ssp_sp_1")
    library = read_filterbin(paths.filterbin)
    vega = read_vega_spectrum(paths.vega_spectrum)
    filters_log = paths.filterbin.parent / "filters.log"
    filter_rows = identify_filters(library, filters_log, report)
    bands = Bands(library.filter(B_PHOTOGRAPHIC_ID), library.filter(R_PHOTOGRAPHIC_ID), vega)
    print(
        f"  IMF of the SSP used for every type: x={ssp.imf_segments[0, 0]:.2f}, "
        f"{ssp.imf_segments[0, 1]:.2f} to {ssp.imf_segments[0, 2]:.0f} Msun "
        "(E/S0 cut IMF not available: forced deviation)"
    )

    configs = {
        "legacy": (LEGACY_Q005, 16.0),
        "updated": (UPDATED, 13.0),
        "updated_tgal_12p5": (UPDATED_TGAL_12P5, 12.5),
    }
    fits = {key: fit_all_types(ssp, bands, t_gal) for key, (_, t_gal) in configs.items()}
    print_fits("C2/C4. Legacy fit, t_gal = 16 Gyr", fits["legacy"])
    print_fits("C5. Updated fit, t_gal = 13.0 Gyr", fits["updated"])
    print_fits("C5. Sensitivity fit, t_gal = 12.5 Gyr", fits["updated_tgal_12p5"])

    print(
        "\nC4. Legacy regression against the paper (science checks, tolerance "
        f"{LEGACY_TAU_TOLERANCE:.0%} in tau). These are findings: they do not set the exit code."
    )
    code_report, report = report, Report()
    for galaxy_type in GALAXY_TYPES:
        fit = fits["legacy"][galaxy_type.name]
        if fit.ok:
            report.check(
                f"{galaxy_type.name}: tau at 16 Gyr vs paper",
                fit.tau_gyr / galaxy_type.legacy_tau_gyr,
                1.0,
                LEGACY_TAU_TOLERANCE,
                " (ratio)",
            )
        else:
            report.check_true(f"{galaxy_type.name}: target reachable at 16 Gyr ({fit.status})", False)
    early = fits["legacy"]["E/S0"]
    early_tau = early.tau_gyr if early.ok else GALAXY_TYPES[0].legacy_tau_gyr
    early_spectrum = exponential_sfr_spectrum(ssp, early_tau, 16.0)
    v_minus_k = vega_colour(
        ssp.wavelengths_aa,
        early_spectrum,
        library.filter(JOHNSON_V_ID),
        library.filter(JOHNSON_K_ID),
        vega,
        photon_counting=False,
    )
    print(f"  [INFO] E/S0 rest V - K = {v_minus_k:.2f} (paper 3.27 with the cut IMF; Salpeter here)")
    if vega_units_look_physical(vega):
        ml_b = mass_to_light_b(ssp, early_tau, 16.0, library.filter(JOHNSON_B_ID), vega)
        print(f"  [INFO] E/S0 M/L_B = {ml_b:.2f} (paper about 5 with the cut IMF; Salpeter here)")
    else:
        print("  [INFO] Vega file is not in erg/s/cm^2/A at Earth; M/L_B skipped")

    print("  Science regression:", end="")
    report.summary()
    report = code_report

    print("\nC5. Legacy and updated tau side by side [Gyr]")
    print(f"  {'type':<5} {'paper':>6} {'fit 16':>8} {'fit 13.0':>9} {'fit 12.5':>9}")
    for galaxy_type in GALAXY_TYPES:
        cells = []
        for key in ("legacy", "updated", "updated_tgal_12p5"):
            fit = fits[key][galaxy_type.name]
            cells.append(f"{fit.tau_gyr:.3f}" if fit.ok else fit.status)
        print(f"  {galaxy_type.name:<5} {galaxy_type.legacy_tau_gyr:>6.1f} {cells[0]:>8} {cells[1]:>9} {cells[2]:>9}")

    print("\nC6. [K + E] grids (bands: b, r, B)")
    band_list = [("b", bands.b), ("r", bands.r), ("B", library.filter(JOHNSON_B_ID))]
    grids = {}
    for key, (params, t_gal) in configs.items():
        taus = {}
        for galaxy_type in GALAXY_TYPES:
            fit = fits[key][galaxy_type.name]
            taus[galaxy_type.name] = (
                fit.tau_gyr if fit.ok else (TAU_MIN_GYR if fit.status == "too_red" else TAU_MAX_GYR)
            )
        grid = k_plus_e_grid(ssp, Cosmology(params), t_gal, taus, band_list)
        grid.config_name = key
        grids[key] = grid
        one = np.argmin(np.abs(grid.redshifts - 1.0))
        print(
            f"  {key}: z_f = {grid.redshifts[-1]:.2f};  E_B(z=1) per type: "
            + ", ".join(f"{name} {grid.evolution[i, 2, one]:+.2f}" for i, name in enumerate(grid.type_names))
        )
        evolution_b = grid.evolution[:, 2, 1 : one + 1]
        report.check_true(f"{key}: E_B(z) < 0 for 0 < z <= 1, every type", np.all(evolution_b < 0))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tau_table = {
        key: {
            name: dict(
                status=fit.status,
                tau_gyr=fit.tau_gyr,
                target=fit.target,
                model=fit.colour_at_tau,
                reachable=fit.colour_range,
            )
            for name, fit in config_fits.items()
        }
        for key, config_fits in fits.items()
    }
    tau_table["filters"] = {str(k): v for k, v in filter_rows.items()}
    (output_dir / "component_c_tau.json").write_text(json.dumps(tau_table, indent=2, default=float))
    np.savez(
        output_dir / "component_c_kegrid.npz",
        **{
            f"{key}_{field}": getattr(grid, field)
            for key, grid in grids.items()
            for field in ("redshifts", "k_plus_e", "k_only", "evolution")
        },
        band_names=np.array([name for name, _ in band_list]),
        type_names=np.array([t.name for t in GALAXY_TYPES]),
    )
    save_evolution_figure(grids, output_dir / "component_c_evolution.png")
    print(
        f"\n  wrote {output_dir / 'component_c_tau.json'}, "
        "component_c_kegrid.npz, component_c_evolution.png"
    )
    print("  Code checks on BC95 data:", end="")
    return report.summary()