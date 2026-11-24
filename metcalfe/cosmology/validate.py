"""Validation suite for tabulated cosmology."""

import numpy as np
from scipy.integrate import quad

from metcalfe.cosmology.config import (
    ALL_CONFIGS,
    CosmologyParams,
    LEGACY_LAMBDA,
    LEGACY_Q005,
    LEGACY_Q05,
    UPDATED,
    UPDATED_TGAL_12P5,
)
from metcalfe.cosmology.core import Cosmology
from metcalfe.cosmology.reference import (
    mattig_luminosity_distance,
    non_evolving_counts,
    open_universe_comoving_volume,
)


class Report:
    def __init__(self):
        self.number_passed = 0
        self.number_failed = 0

    def check(self, label, value, target, tolerance, unit=""):
        passed = abs(value - target) <= tolerance
        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}] {label}: got {value:.6g}{unit}, "
            f"expected {target:.6g}{unit} +/- {tolerance:.2g}"
        )
        if passed:
            self.number_passed += 1
        else:
            self.number_failed += 1
        return passed


def run_validation() -> bool:
    report = Report()
    cosmologies = {params.name: Cosmology(params) for params in ALL_CONFIGS}
    legacy_q005 = cosmologies[LEGACY_Q005.name]
    legacy_lambda = cosmologies[LEGACY_LAMBDA.name]
    legacy_q05 = cosmologies[LEGACY_Q05.name]
    updated = cosmologies[UPDATED.name]
    updated_young = cosmologies[UPDATED_TGAL_12P5.name]

    test_redshifts = np.array(
        [1e-6, 1e-4, 0.0023, 0.0123, 0.1037, 0.5371, 1.0029, 2.2221, 3.0013, 4.7107, 6.3029, 9.9017, 14.9973]
    )
    spline_tolerance = 1e-7

    print("\n1. Tabulated functions against direct quadrature (max relative error)")
    for cosmology in cosmologies.values():
        worst_distance = 0.0
        worst_lookback = 0.0
        for z in test_redshifts:
            direct_distance = cosmology.hubble_distance_mpc * quad(
                cosmology._inverse_E, 0, z, epsrel=1e-12
            )[0]
            direct_lookback = cosmology.hubble_time_gyr * quad(
                cosmology._lookback_integrand, 0, z, epsrel=1e-12
            )[0]
            worst_distance = max(
                worst_distance, abs(cosmology.comoving_distance(z) / direct_distance - 1)
            )
            worst_lookback = max(
                worst_lookback, abs(cosmology.lookback_time(z) / direct_lookback - 1)
            )
        report.check(f"{cosmology.params.name}: d_C", worst_distance, 0.0, spline_tolerance)
        report.check(f"{cosmology.params.name}: t_lb", worst_lookback, 0.0, spline_tolerance)

    print("\n2. Mattig closed form for Lambda = 0 (max relative error in d_L)")
    for cosmology in [legacy_q005, legacy_q05]:
        numerical = cosmology.luminosity_distance(test_redshifts)
        analytic = mattig_luminosity_distance(test_redshifts, cosmology.params.q0, cosmology.h0)
        report.check(
            f"{cosmology.params.name}",
            np.max(np.abs(numerical / analytic - 1)),
            0.0,
            spline_tolerance,
        )

    print("\n3. Low-redshift limit of d_L")
    for cosmology in cosmologies.values():
        z_small = 1e-5
        hubble_law = cosmology.hubble_distance_mpc * z_small
        ratio = float(cosmology.luminosity_distance(z_small)) / hubble_law
        second_order = 1.0 + 0.5 * (1.0 - cosmology.params.q0) * z_small
        report.check(
            f"{cosmology.params.name}: d_L / (cz/H0) at z=1e-5", ratio, second_order, 1e-9
        )

    print("\n4. Integrated dV/dz against Hogg (1999) eq. 29, open legacy model")
    for z in [0.5, 2.0, 6.0]:
        integrated = legacy_q005.comoving_volume_all_sky(z)
        analytic = open_universe_comoving_volume(legacy_q005, z)
        report.check(f"V_c(z={z}) relative error", abs(integrated / analytic - 1), 0.0, 1e-8)

    print("\n5. Cross-check against astropy LambdaCDM (max relative error)")
    try:
        from astropy.cosmology import LambdaCDM

        for cosmology in cosmologies.values():
            reference = LambdaCDM(
                H0=cosmology.h0,
                Om0=cosmology.omega_m,
                Ode0=cosmology.omega_lambda,
                Tcmb0=0,
            )
            z = test_redshifts[1:]
            error_dl = np.max(
                np.abs(
                    cosmology.luminosity_distance(z) / reference.luminosity_distance(z).value - 1
                )
            )
            error_dv = np.max(
                np.abs(
                    cosmology.dvdz(z) / reference.differential_comoving_volume(z).value - 1
                )
            )
            error_tl = np.max(
                np.abs(cosmology.lookback_time(z) / reference.lookback_time(z).value - 1)
            )
            error_age = abs(cosmology.age_of_universe() / reference.age(0).value - 1)
            name = cosmology.params.name
            report.check(f"{name}: d_L", error_dl, 0.0, 2 * spline_tolerance)
            report.check(f"{name}: dV/dz", error_dv, 0.0, 2 * spline_tolerance)
            report.check(f"{name}: t_lb", error_tl, 0.0, 2 * spline_tolerance)
            report.check(f"{name}: age", error_age, 0.0, 1e-6)
    except ImportError:
        print("  [SKIP] astropy not installed")

    print("\n6. Ages and formation redshifts quoted in the proposal (rounded values)")
    report.check("Age, legacy q0=0.05", legacy_q005.age_of_universe(), 17.6, 0.05, " Gyr")
    report.check("Age, legacy Lambda", legacy_lambda.age_of_universe(), 18.9, 0.05, " Gyr")
    report.check("Age, Planck 2018", updated.age_of_universe(), 13.8, 0.05, " Gyr")
    report.check(
        "t_lb(z=6.3), legacy q0=0.05 (paper: t_gal=16)",
        float(legacy_q005.lookback_time(6.3)),
        16.0,
        0.05,
        " Gyr",
    )
    report.check(
        "t_lb(z=7.9), legacy Lambda (paper: t_gal=18)",
        float(legacy_lambda.lookback_time(7.9)),
        18.0,
        0.05,
        " Gyr",
    )
    print(
        f"  [INFO] exact z_f for t_gal=16 (q0=0.05): {legacy_q005.z_formation():.3f}; "
        f"for t_gal=18 (Lambda): {legacy_lambda.z_formation():.3f}"
    )
    report.check("z_f, updated, t_gal=13.0", updated.z_formation(), 6.8, 0.05)
    report.check("z_f, updated, t_gal=12.5", updated_young.z_formation(), 4.6, 0.05)

    for cosmology in [legacy_q005, legacy_lambda, updated, updated_young]:
        z_f = cosmology.z_formation()
        report.check(
            f"t_lb(z_f) = t_gal, {cosmology.params.name}",
            float(cosmology.lookback_time(z_f)),
            cosmology.params.t_gal_gyr,
            1e-8,
            " Gyr",
        )

    shoes = Cosmology(CosmologyParams("SH0ES H0=73", 0.315, 0.685, 73.0, 13.0))
    report.check("Age with H0=73 (proposal: 12.7 Gyr)", shoes.age_of_universe(), 12.7, 0.05, " Gyr")
    try:
        shoes.z_formation()
        print("  [FAIL] z_formation should refuse t_gal above the age of the universe")
        report.number_failed += 1
    except ValueError:
        print("  [PASS] z_formation refuses t_gal = 13.0 Gyr when the universe is 12.7 Gyr old")
        report.number_passed += 1

    print("\n7. Figure 2 statements")
    h_legacy = LEGACY_Q005.h
    h_updated = UPDATED.h
    volume_legacy_h_scaled = float(legacy_q005.dvdz(3.0)) * h_legacy**3
    volume_updated_h_scaled = float(updated.dvdz(3.0)) * h_updated**3
    report.check(
        "Planck / legacy q0=0.05 volume per unit z at z=3 (h-scaled)",
        volume_updated_h_scaled / volume_legacy_h_scaled,
        0.75,
        0.01,
    )
    lookback_difference = float(legacy_q005.lookback_time(2.0) - updated.lookback_time(2.0))
    report.check(
        "Legacy q0=0.05 minus Planck look-back time to z=2",
        lookback_difference,
        2.1,
        0.05,
        " Gyr",
    )

    print("\n8. Euclidean limit of the count slope (toy Schechter, no K/E/A)")
    toy_lf = dict(phi_star=1e-3, m_star=-20.5, alpha=-1.0)
    for cosmology in [legacy_q005, updated]:
        bright = non_evolving_counts(cosmology, 6.0, **toy_lf)
        brighter = non_evolving_counts(cosmology, 5.0, **toy_lf)
        slope = np.log10(bright / brighter)
        report.check(f"{cosmology.params.name}: slope at m = 5 to 6", slope, 0.6, 1e-3)

    print("\n9. H0 invariance of a non-evolving model after rescaling the LF")
    same_geometry_h50 = Cosmology(
        CosmologyParams("Planck geometry, H0=50", 0.315, 0.685, 50.0, None)
    )
    h_ratio = UPDATED.h0 / same_geometry_h50.h0
    legacy_lf = dict(phi_star=9.27e-4, m_star=-20.85, alpha=-0.7)
    rescaled_lf = dict(
        phi_star=legacy_lf["phi_star"] * h_ratio**3,
        m_star=legacy_lf["m_star"] + 5.0 * np.log10(h_ratio),
        alpha=legacy_lf["alpha"],
    )
    report.check("phi* factor (0.674/0.5)^3", h_ratio**3, 2.449, 0.001)
    report.check("M* shift 5 log10(0.674/0.5)", 5.0 * np.log10(h_ratio), 0.648, 0.001)
    for magnitude in [16.0, 20.0, 24.0]:
        counts_h50 = non_evolving_counts(same_geometry_h50, magnitude, **legacy_lf)
        counts_h674 = non_evolving_counts(updated, magnitude, **rescaled_lf)
        report.check(
            f"n(m={magnitude:.0f}) ratio H0=67.4 / H0=50", counts_h674 / counts_h50, 1.0, 1e-6
        )

    print(f"\nSummary: {report.number_passed} passed, {report.number_failed} failed")
    return report.number_failed == 0