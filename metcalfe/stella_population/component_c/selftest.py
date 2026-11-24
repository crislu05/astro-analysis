"""Synthetic self-tests (S1–S8) for Component C; no BC95 data required."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from metcalfe.cosmology.config import UPDATED
from metcalfe.cosmology.core import Cosmology
from metcalfe.stella_population.component_c.config import (
    GALAXY_TYPES,
    TAU_MAX_GYR,
    TAU_MIN_GYR,
)
from metcalfe.stella_population.component_c.filters_c3 import effective_wavelength, half_power_range
from metcalfe.stella_population.component_c.ke_grid import k_plus_e_grid
from metcalfe.stella_population.component_c.mass_light import mass_to_light_b, vega_units_look_physical
from metcalfe.stella_population.component_c.spectrum import galaxy_spectrum, stellar_mass
from metcalfe.stella_population.component_c.synthetic import synthetic_bands, synthetic_ssp
from metcalfe.stella_population.component_c.tau import fit_all_types, fit_tau, model_colour
from metcalfe.stella_population.ised import StellarPopulation
from metcalfe.stella_population.photometry import vega_colour
from metcalfe.stella_population.report import Report


def run_selftests() -> bool:
    report = Report()
    ssp = synthetic_ssp()
    bands, band_b = synthetic_bands()
    early, late = GALAXY_TYPES[0], GALAXY_TYPES[3]

    print("\nS1. Colour against tau is monotonic (synthetic SSP)")
    taus = np.geomspace(TAU_MIN_GYR, TAU_MAX_GYR, 40)
    for galaxy_type in (early, late):
        colours = np.array([model_colour(ssp, bands, galaxy_type, tau, 16.0) for tau in taus])
        report.check_true(f"{galaxy_type.name}: b - r falls as tau grows", np.all(np.diff(colours) < 0))

    print("\nS2. C2 recovers tau from a colour made with a known tau")
    for galaxy_type in (early, late):
        for t_gal in (16.0, 13.0):
            for true_tau in (0.5, 2.5, 9.0, 40.0):
                target = model_colour(ssp, bands, galaxy_type, true_tau, t_gal)
                fit = fit_tau(ssp, bands, galaxy_type, t_gal, target=target)
                report.check(
                    f"{galaxy_type.name}, t_gal={t_gal}: tau={true_tau}",
                    fit.tau_gyr / true_tau,
                    1.0,
                    1e-6,
                )

    print("\nS3. C2 reports unreachable targets instead of returning a grid edge")
    reddest = model_colour(ssp, bands, early, TAU_MIN_GYR, 13.0)
    bluest = model_colour(ssp, bands, early, TAU_MAX_GYR, 13.0)
    too_red = fit_tau(ssp, bands, early, 13.0, target=reddest + 0.1)
    too_blue = fit_tau(ssp, bands, early, 13.0, target=bluest - 0.1)
    report.check_true("target redder than tau -> 0 flagged 'too_red'", too_red.status == "too_red" and not too_red.ok)
    report.check_true("target bluer than tau -> inf flagged 'too_blue'", too_blue.status == "too_blue")
    blip = synthetic_ssp(red_supergiant_blip=True)
    blip_age = 0.032
    blip_colours = np.array([model_colour(blip, bands, late, tau, blip_age) for tau in taus])
    target = 0.5 * (max(blip_colours[0], blip_colours[-1]) + blip_colours.max())
    blip_fit = fit_tau(blip, bands, late, blip_age, target=target)
    report.check_true(
        f"non-monotonic colour (red supergiant blip, age 32 Myr): status '{blip_fit.status}', "
        f"roots {', '.join(f'{root:.4f}' for root in blip_fit.all_roots)} Gyr",
        blip_fit.status == "multiple_roots" and len(blip_fit.all_roots) == 2,
    )
    report.check_true(
        "each reported root reproduces the target colour",
        all(abs(model_colour(blip, bands, late, root, blip_age) - target) < 1e-6 for root in blip_fit.all_roots),
    )
    report.check_true("the blip population is flagged as not monotonic in tau", not blip_fit.monotonic)

    report.check_true(
        "same tau, older galaxy is redder (so 13 Gyr cannot always reach 16 Gyr colours)",
        model_colour(ssp, bands, early, 2.5, 16.0) > model_colour(ssp, bands, early, 2.5, 13.0),
    )

    print("\nS4. Dust only for Sbc, Scd, Sdm, and it reddens")
    spectrum_clean = galaxy_spectrum(ssp, late, 9.0, 13.0, with_dust=False)
    spectrum_dusty = galaxy_spectrum(ssp, late, 9.0, 13.0, with_dust=True)
    report.check_true(
        "Scd with dust is redder by about 0.2 A_B (B-V law, broad bands)",
        0.0
        < vega_colour(ssp.wavelengths_aa, spectrum_dusty, bands.b, bands.r, bands.vega, photon_counting=False)
        - vega_colour(ssp.wavelengths_aa, spectrum_clean, bands.b, bands.r, bands.vega, photon_counting=False)
        < 0.3 * late.a_b,
    )
    report.check_true(
        "E/S0 unchanged by the dust switch",
        np.array_equal(galaxy_spectrum(ssp, early, 2.5, 13.0, True), galaxy_spectrum(ssp, early, 2.5, 13.0, False)),
    )

    print("\nS5. Stellar mass convolution")
    constant = replace(ssp, extras={"stellar_mass": np.ones_like(ssp.ages_yr)})
    for tau, age in ((2.5, 16.0), (9.0, 13.0)):
        report.check(
            f"mass track = 1: mass(t) = 1 - exp(-t/tau), tau={tau}, t={age}",
            stellar_mass(constant, tau, age),
            1.0 - np.exp(-age / tau),
            1e-10,
        )

    print("\nS6. C3 helpers on top-hat filters")
    report.check("top-hat effective wavelength = centre", effective_wavelength(bands.b), 4600.0, 1e-6, " A")
    low, high = half_power_range(bands.r)
    report.check("top-hat half-power width", high - low, 1000.0, 1e-6, " A")

    print("\nS7. C6 [K + E] on spectra with known answers")
    cosmology = Cosmology(UPDATED)
    band_list = [("b", bands.b), ("r", bands.r)]
    power_law = 1.0 / ssp.wavelengths_aa
    shape_fixed = replace(ssp, spectra=np.outer(np.ones_like(ssp.ages_yr), power_law))
    grid = k_plus_e_grid(
        shape_fixed, cosmology, 13.0, {t.name: 2.5 for t in GALAXY_TYPES}, band_list, (early,), dz=0.25
    )
    finite = np.isfinite(grid.k_plus_e)
    report.check(
        "constant SSP with f_lambda ~ 1/lambda: K = 0 (max |K|)", np.nanmax(np.abs(grid.k_only)), 0.0, 1e-6, " mag"
    )
    expected_e = -2.5 * np.log10(
        (1.0 - np.exp(-(13.0 - np.array([float(cosmology.lookback_time(z)) for z in grid.redshifts])) / 2.5))
        / (1.0 - np.exp(-13.0 / 2.5))
    )
    report.check(
        "same SSP: E(z) = -2.5 log[(1 - e^{-t(z)/tau}) / (1 - e^{-t_gal/tau})] (max error)",
        np.max(np.abs(grid.evolution[0, 0][finite[0, 0]] - expected_e[finite[0, 0]])),
        0.0,
        1e-9,
        " mag",
    )
    report.check_true("grid starts at [K+E](0) = 0", np.allclose(grid.k_plus_e[0, :, 0], 0.0, atol=1e-12))

    full = k_plus_e_grid(ssp, cosmology, 13.0, {t.name: t.legacy_tau_gyr for t in GALAXY_TYPES}, band_list, dz=0.1)
    report.check_true(
        "realistic synthetic SSP: E(z) < 0 for 0 < z <= 3 in every type and band",
        np.all(full.evolution[:, :, 1:31] < 0),
    )
    report.check_true(
        "E/S0 (tau=2.5) brightens more than Scd (tau=9) at z=1",
        full.evolution[0, 0, 10] < full.evolution[3, 0, 10],
    )
    report.check_true(
        "beyond the age floor [K+E] = +inf (galaxy not yet formed)", np.isinf(full.k_plus_e[0, 0, -1])
    )
    report.check_true(
        "the grid stops at z_f",
        abs(full.redshifts[-1] - cosmology.z_formation(13.0)) < 0.1 + 1e-9,
    )

    print("\nS8. C4/C5 end to end on synthetic data (targets made at t_gal = 16 Gyr)")
    synthetic_types = tuple(
        replace(t, target_br=model_colour(ssp, bands, t, t.legacy_tau_gyr, 16.0)) for t in GALAXY_TYPES
    )
    legacy_fits = fit_all_types(ssp, bands, 16.0, synthetic_types)
    for t in synthetic_types:
        report.check(f"{t.name}: legacy tau recovered", legacy_fits[t.name].tau_gyr, t.legacy_tau_gyr, 1e-5, " Gyr")
    updated_fits = fit_all_types(ssp, bands, 13.0, synthetic_types)
    for t in synthetic_types:
        fit = updated_fits[t.name]
        report.check_true(
            f"{t.name}: at 13 Gyr either a shorter tau or flagged unreachable ({fit.status})",
            (fit.ok and fit.tau_gyr < t.legacy_tau_gyr) or fit.status == "too_red",
        )
    report.check_true(
        "M/L_B is positive and finite on the synthetic SSP",
        np.isfinite(mass_to_light_b(ssp, 2.5, 16.0, band_b, bands.vega)),
    )
    report.check_true("synthetic Vega passes the units check", vega_units_look_physical(bands.vega))
    return report.summary()