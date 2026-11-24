"""Validation suite for Schechter LFs (Component B)."""

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma, gammaincc
from scipy.stats import kstest

from metcalfe.cosmology.config import UPDATED
from metcalfe.cosmology.core import Cosmology
from metcalfe.luminosity import config
from metcalfe.luminosity.counts import non_evolving_counts_per_deg2
from metcalfe.luminosity.fit import fit_normalisation
from metcalfe.luminosity.gamma import upper_incomplete_gamma
from metcalfe.luminosity.presets import LEGACY_TYPES, updated_types


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

    def check_true(self, label, condition):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {label}")
        if condition:
            self.number_passed += 1
        else:
            self.number_failed += 1


def run_validation() -> bool:
    report = Report()
    rescaled = updated_types()

    print("\n1. Rescaling to h = 0.674 against the proposal table")
    report.check(
        "M* shift 5 log10(0.674/0.5)",
        5.0 * np.log10(config.UPDATED_H / config.LEGACY_H),
        0.648,
        5e-4,
        " mag",
    )
    report.check(
        "phi* factor (0.674/0.5)^3", (config.UPDATED_H / config.LEGACY_H) ** 3, 2.449, 5e-4
    )
    proposal_m_star = [-20.20, -20.23, -20.57, -20.72, -20.73]
    proposal_phi_star = [22.7e-4, 11.3e-4, 15.2e-4, 6.7e-4, 3.3e-4]
    for lf, m_star, phi_star in zip(rescaled, proposal_m_star, proposal_phi_star):
        report.check(f"{lf.name}: M*", lf.m_star, m_star, 0.005 + 1e-9)
        last_digit = 0.05e-4 if phi_star >= 1e-3 else 0.05e-4
        report.check(
            f"{lf.name}: phi* [1e-4 Mpc^-3]",
            lf.phi_star * 1e4,
            phi_star * 1e4,
            last_digit * 1e4 + 1e-9,
        )

    print("\n2. Rescaling is a change of units")
    for legacy, new in zip(LEGACY_TYPES, rescaled):
        report.check(
            f"{legacy.name}: phi*/h^3 unchanged",
            (new.phi_star / new.h**3) / (legacy.phi_star / legacy.h**3),
            1.0,
            1e-12,
        )
        report.check(
            f"{legacy.name}: M* - 5 log10 h unchanged",
            (new.m_star - 5 * np.log10(new.h)) - (legacy.m_star - 5 * np.log10(legacy.h)),
            0.0,
            1e-12,
        )
        back = new.rescale_h(config.LEGACY_H)
        report.check_true(
            f"{legacy.name}: rescale to 0.674 and back returns the legacy values",
            np.isclose(back.phi_star, legacy.phi_star, rtol=1e-13)
            and np.isclose(back.m_star, legacy.m_star, rtol=0, atol=1e-12),
        )
        fixed_legacy_magnitude = -20.0
        fixed_new_magnitude = fixed_legacy_magnitude + 5 * np.log10(new.h / legacy.h)
        density_legacy = legacy.number_density(m_faint=fixed_legacy_magnitude) / legacy.h**3
        density_new = new.number_density(m_faint=fixed_new_magnitude) / new.h**3
        report.check(
            f"{legacy.name}: n(L > L_fixed) in h^3 Mpc^-3 unchanged",
            density_new / density_legacy,
            1.0,
            1e-12,
        )

    print("\n3. Analytic number and luminosity densities against quadrature of phi(M)")
    for lf in LEGACY_TYPES:
        for m_faint in [-16.0, -12.0]:
            numerical_number = quad(
                lambda m: float(lf.phi(m)),
                config.M_BRIGHT_DEFAULT,
                m_faint,
                epsabs=0,
                epsrel=1e-12,
                limit=200,
            )[0]
            numerical_light = quad(
                lambda m: float(lf.phi(m) * lf.x(m)),
                config.M_BRIGHT_DEFAULT,
                m_faint,
                epsabs=0,
                epsrel=1e-12,
                limit=200,
            )[0]
            report.check(
                f"{lf.name}, M<{m_faint:.0f}: number density",
                lf.number_density(m_faint=m_faint) / numerical_number,
                1.0,
                1e-9,
            )
            report.check(
                f"{lf.name}, M<{m_faint:.0f}: luminosity density",
                lf.luminosity_density_in_l_star(m_faint=m_faint) / numerical_light,
                1.0,
                1e-9,
            )

    for a in [-0.5, -0.1, 0.0, 0.3]:
        x = 0.01
        numerical = quad(
            lambda t: t ** (a - 1.0) * np.exp(-t), x, np.inf, epsabs=0, epsrel=1e-12, limit=200
        )[0]
        report.check(
            f"Gamma({a}, {x}) against quadrature",
            upper_incomplete_gamma(a, x) / numerical,
            1.0,
            1e-9,
        )

    for lf in LEGACY_TYPES:
        for m_faint in [-12.0, 5.0]:
            expected_fraction = gammaincc(lf.alpha + 2.0, float(lf.x(m_faint)))
            report.check(
                f"{lf.name}: rho_L(M<{m_faint:+.0f}) / [phi* Gamma(alpha+2)] = Q(alpha+2, x_faint)",
                lf.luminosity_density_in_l_star(m_faint=m_faint)
                / (lf.phi_star * gamma(lf.alpha + 2.0)),
                expected_fraction,
                1e-9,
            )

    print("\n4. Normalisation factor A")
    factor = 0.8
    normalised = updated_types(normalisation_factor=factor)
    ratios_before = np.array([lf.phi_star for lf in rescaled]) / rescaled[0].phi_star
    ratios_after = np.array([lf.phi_star for lf in normalised]) / normalised[0].phi_star
    report.check(
        "type ratios unchanged by A (max difference)",
        np.max(np.abs(ratios_after - ratios_before)),
        0.0,
        1e-14,
    )
    report.check("A multiplies every phi*", normalised[2].phi_star / rescaled[2].phi_star, factor, 1e-14)

    rng = np.random.default_rng(2001)
    model = np.array([120.0, 190.0, 300.0, 470.0, 740.0, 1150.0])
    true_factor = 0.87
    noiseless = true_factor * model
    best, error, chi2, _ = fit_normalisation(model, noiseless, np.sqrt(noiseless))
    report.check("fit A on noiseless synthetic counts", best, true_factor, 1e-12)
    report.check("chi^2 on noiseless synthetic counts", chi2, 0.0, 1e-12)

    recovered, pulls = [], []
    for _ in range(4000):
        noisy = rng.poisson(true_factor * model * 10) / 10.0
        best, error, _, _ = fit_normalisation(
            model, noisy, np.sqrt(np.maximum(true_factor * model / 10.0, 1e-9))
        )
        recovered.append(best)
        pulls.append((best - true_factor) / error)
    report.check(
        "fit A with Poisson noise: mean over 4000 trials", np.mean(recovered), true_factor, 3e-4
    )
    report.check("pull distribution width (should be 1)", np.std(pulls), 1.0, 0.03)

    print("\n5. Sampling absolute magnitudes")
    for lf in [rescaled[0], rescaled[4]]:
        draws = lf.sample(200_000, rng, m_bright=-24.0, m_faint=config.M_FAINT_DEFAULT)
        total = lf.number_density(m_bright=-24.0, m_faint=config.M_FAINT_DEFAULT)

        def analytic_cdf(m, lf=lf, total=total):
            m = np.atleast_1d(m)
            return (
                np.array([lf.number_density(m_bright=-24.0, m_faint=v) for v in m]) / total
            )

        statistic, p_value = kstest(draws, analytic_cdf)
        report.check_true(
            f"{lf.name}: KS test against analytic CDF (D={statistic:.4f}, p={p_value:.2f})",
            p_value > 0.01,
        )
        report.check_true(
            f"{lf.name}: all draws inside the limits",
            draws.min() >= -24.0 and draws.max() <= config.M_FAINT_DEFAULT,
        )

    print("\n6. Faint integration limit M_faint = -12 (non-evolving counts, updated cosmology)")
    cosmology = Cosmology(UPDATED)
    for apparent in [18.0, 22.0, 26.0]:
        counts_12 = non_evolving_counts_per_deg2(
            cosmology, rescaled, apparent, m_faint=-12.0
        )
        counts_8 = non_evolving_counts_per_deg2(
            cosmology, rescaled, apparent, m_faint=-8.0
        )
        change = counts_8 / counts_12 - 1.0
        report.check(
            f"m={apparent:.0f}: change in n(m) from M_faint=-12 to -8", change, 0.0, 0.01
        )
        print(f"         n(m={apparent:.0f}) = {counts_12:.4g} per mag per deg^2 (M_faint = -12)")

    print(f"\nSummary: {report.number_passed} passed, {report.number_failed} failed")
    return report.number_failed == 0