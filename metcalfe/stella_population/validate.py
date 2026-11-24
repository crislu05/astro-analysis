"""Validation suite for BC95 reader and exponential SFR convolution."""

import numpy as np

from metcalfe.stella_population.data_paths import BC95Paths
from metcalfe.stella_population.ised import StellarPopulation, read_ised
from metcalfe.cosmology.config import UPDATED
from metcalfe.cosmology.core import Cosmology
from metcalfe.stella_population.filters import read_filterbin
from metcalfe.stella_population.photometry import (
    ab_colour,
    band_flux,
    fit_tau_to_colour,
    k_correction,
    powerlaw_spectrum_integral,
    read_1color_table,
    read_3color_table,
    read_vega_spectrum,
    top_hat,
    vega_colour,
    vega_magnitude,
)
from metcalfe.stella_population.sfr import (
    bc95_trapezoid_convolution,
    exponential_sfr_mass_formed,
    exponential_sfr_spectrum,
)

BC95_PHOTON_COUNTING = False


class Report:
    def __init__(self):
        self.number_passed = 0
        self.number_failed = 0

    def check(self, label, value, target, tolerance, unit=""):
        passed = abs(value - target) <= tolerance
        print(
            f"  [{'PASS' if passed else 'FAIL'}] {label}: got {value:.6g}{unit}, "
            f"expected {target:.6g}{unit} +/- {tolerance:.2g}"
        )
        self.number_passed += passed
        self.number_failed += not passed
        return passed

    def check_true(self, label, condition):
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
        self.number_passed += bool(condition)
        self.number_failed += not condition


def run_validation(paths: BC95Paths) -> bool:
    report = Report()
    ssp = read_ised(paths.ssp_ised, "bc95_ssp_sp_1")
    tau1 = read_ised(paths.tau1_ised, "bc95_tau1_sp_1")

    print("\n1. Reading the BC95 files")
    report.check("number of ages", len(ssp.ages_yr), 221, 0)
    report.check("number of wavelengths", len(ssp.wavelengths_aa), 1206, 0)
    report.check("first wavelength", ssp.wavelengths_aa[0], 5.0, 0, " A")
    report.check("last wavelength", ssp.wavelengths_aa[-1], 1.0e6, 0, " A")
    report.check("last age", ssp.ages_gyr[-1], 20.0, 1e-5, " Gyr")
    report.check("IMF slope x (Salpeter)", ssp.imf_segments[0, 0], 1.35, 1e-6)
    report.check("IMF lower mass", ssp.imf_segments[0, 1], 0.1, 1e-6, " Msun")
    report.check("IMF upper mass", ssp.imf_segments[0, 2], 125.0, 1e-4, " Msun")
    report.check("SSP file has SFR code 0", ssp.sfr_code, 0, 0)
    report.check("tau1 file has SFR code 4 (exponential, no recycling)", tau1.sfr_code, 4, 0)
    from metcalfe.stella_population import config

    report.check("tau1 file has tau = 1 Gyr", tau1.tau_yr / config.YEARS_PER_GYR, 1.0, 1e-6, " Gyr")
    report.check_true("ages of both files identical", np.array_equal(ssp.ages_yr, tau1.ages_yr))
    report.check_true(
        "all fluxes finite and non-negative",
        np.all(np.isfinite(ssp.spectra)) and np.all(ssp.spectra >= 0),
    )

    print("\n2. Units: spectrum integral / BC95 bolometric luminosity")
    wavelengths = ssp.wavelengths_aa
    for index in [5, 60, 100, 150, 200, 220]:
        spectrum = ssp.spectra[index]
        bolometric = ssp.extras["bolometric_flux"][index]
        linear_ratio = np.trapezoid(spectrum, wavelengths) / bolometric
        powerlaw_ratio = powerlaw_spectrum_integral(wavelengths, spectrum) / bolometric
        report.check_true(
            f"age {ssp.ages_gyr[index]:.4g} Gyr: ratios {linear_ratio:.3f} (linear), "
            f"{powerlaw_ratio:.3f} (power law) within 10% of 1",
            abs(linear_ratio - 1) < 0.10 and abs(powerlaw_ratio - 1) < 0.10,
        )

    print("\n3. Bolometric magnitude and stellar mass against bc95_ssp_sp_1.3color")
    table = read_3color_table(paths.ssp_3color)
    log_ages = np.log10(np.where(ssp.ages_yr > 0, ssp.ages_yr, 1.0))
    worst_mbol = 0.0
    worst_mass = 0.0
    for log_age, mbol, _, _, _, stellar_mass, _ in table[::10]:
        index = np.argmin(np.abs(log_ages - log_age))
        if abs(log_ages[index] - log_age) > 1e-4:
            continue
        model_mbol = 4.75 - 2.5 * np.log10(ssp.extras["bolometric_flux"][index])
        worst_mbol = max(worst_mbol, abs(model_mbol - mbol))
        worst_mass = max(worst_mass, abs(ssp.extras["stellar_mass"][index] - stellar_mass))
    report.check("max |Mbol difference|", worst_mbol, 0.0, 1e-3, " mag")
    report.check("max |stellar mass difference|", worst_mass, 0.0, 1e-5, " Msun")

    print("\n4. Reproducing BC95's tau = 1 Gyr model with their trapezoid method")
    replica = bc95_trapezoid_convolution(ssp, 1.0)
    for index in [50, 120, 180, 220]:
        positive = tau1.spectra[index] > 1e-30 * tau1.spectra[index].max()
        difference = np.max(np.abs(replica[index, positive] / tau1.spectra[index, positive] - 1))
        report.check(f"age {ssp.ages_gyr[index]:.4g} Gyr: max relative difference", difference, 0.0, 1e-5)

    print("\n5. Exact integral against BC95's trapezoid (old ages, 3000-25000 A)")
    optical = (ssp.wavelengths_aa > 3000) & (ssp.wavelengths_aa < 25000)
    for index in [180, 200, 215, 220]:
        exact = exponential_sfr_spectrum(ssp, 1.0, ssp.ages_gyr[index])
        difference = np.max(np.abs(exact[optical] / tau1.spectra[index, optical] - 1))
        report.check(f"age {ssp.ages_gyr[index]:.4g} Gyr: max relative difference", difference, 0.0, 0.01)
        colour_shift = 2.5 * np.log10(
            np.trapezoid(exact[optical], ssp.wavelengths_aa[optical])
            / np.trapezoid(tau1.spectra[index, optical], ssp.wavelengths_aa[optical])
        )
        report.check(
            f"age {ssp.ages_gyr[index]:.4g} Gyr: magnitude offset 3000-25000 A",
            colour_shift,
            0.0,
            0.01,
            " mag",
        )

    print("\n6. Analytic checks of the exponential integral")
    constant = StellarPopulation(
        "constant",
        ssp.ages_yr,
        ssp.wavelengths_aa,
        np.ones_like(ssp.spectra),
        ssp.imf_segments,
        0,
        0.0,
    )
    for tau_gyr, age_gyr in [(1.0, 13.0), (9.0, 16.0), (2.5, 0.37), (50.0, 7.123)]:
        result = exponential_sfr_spectrum(constant, tau_gyr, age_gyr)[0]
        report.check(
            f"constant SSP, tau={tau_gyr}, t={age_gyr}: F = 1 - exp(-t/tau)",
            result,
            exponential_sfr_mass_formed(tau_gyr, age_gyr),
            1e-12,
        )
    linear = StellarPopulation(
        "linear",
        ssp.ages_yr,
        ssp.wavelengths_aa,
        np.outer(ssp.ages_gyr, np.ones_like(ssp.wavelengths_aa)),
        ssp.imf_segments,
        0,
        0.0,
    )
    for tau_gyr, age_gyr in [(1.0, 13.0), (9.0, 16.0)]:
        result = exponential_sfr_spectrum(linear, tau_gyr, age_gyr)[0]
        expected = age_gyr - tau_gyr * exponential_sfr_mass_formed(tau_gyr, age_gyr)
        report.check(f"SSP linear in age, tau={tau_gyr}, t={age_gyr}", result, expected, 1e-9)
    on_grid = exponential_sfr_spectrum(ssp, 2.5, ssp.ages_gyr[200])
    just_off = exponential_sfr_spectrum(ssp, 2.5, ssp.ages_gyr[200] * (1 + 1e-9))
    report.check(
        "continuous across a grid age",
        np.max(np.abs(just_off[optical] / on_grid[optical] - 1)),
        0.0,
        1e-6,
    )

    print("\n7. Photometry machinery")
    vega = read_vega_spectrum(paths.vega_spectrum)
    blue, red = top_hat("blue", 4400, 900), top_hat("red", 6500, 1300)
    report.check("Vega has Vega colour 0", vega_colour(*vega, blue, red, vega), 0.0, 1e-12)
    wavelengths = np.geomspace(100, 1e5, 5000)
    report.check(
        "flat f_nu has AB colour 0", ab_colour(wavelengths, 1.0 / wavelengths**2, blue, red), 0.0, 1e-3
    )
    flat_flux = band_flux(
        wavelengths, np.ones_like(wavelengths), blue, redshift=1.0, photon_counting=False
    )
    report.check(
        "flat f_lambda at z=1: band flux divided by (1+z)",
        flat_flux / np.trapezoid(blue.transmission, blue.wavelengths_aa),
        0.5,
        1e-9,
    )
    old = exponential_sfr_spectrum(ssp, 1.0, 13.0)
    young = exponential_sfr_spectrum(ssp, 1.0, 1.0)
    report.check_true(
        "older population is redder (blue - red colour larger)",
        vega_colour(ssp.wavelengths_aa, old, blue, red, vega)
        > vega_colour(ssp.wavelengths_aa, young, blue, red, vega),
    )

    print("\n8. BC95 colour tables (B-V, V-K) with FILTERBIN bands 14, 15, 35")
    library = read_filterbin(paths.filterbin)
    band_b = library.filter(14)
    band_v = library.filter(15)
    band_k = library.filter(35)
    colour_table = read_1color_table(paths.ssp_1color)
    log_ages = np.log10(np.where(ssp.ages_yr > 0, ssp.ages_yr, 1.0))
    worst_bv = 0.0
    worst_vk = 0.0
    for log_age, _, _, table_bmag, table_vmag, table_kmag in colour_table[::8]:
        index = np.argmin(np.abs(log_ages - log_age))
        if abs(log_ages[index] - log_age) > 2e-3:
            continue
        spectrum = ssp.spectra[index]
        wavelengths = ssp.wavelengths_aa
        model_bv = vega_colour(
            wavelengths, spectrum, band_b, band_v, vega, photon_counting=BC95_PHOTON_COUNTING
        )
        model_vk = vega_colour(
            wavelengths, spectrum, band_v, band_k, vega, photon_counting=BC95_PHOTON_COUNTING
        )
        worst_bv = max(worst_bv, abs(model_bv - (table_bmag - table_vmag)))
        worst_vk = max(worst_vk, abs(model_vk - (table_vmag - table_kmag)))
    report.check("max |B-V difference| vs bc95_ssp_sp_1.1color", worst_bv, 0.0, 0.02, " mag")
    report.check("max |V-K difference| vs bc95_ssp_sp_1.1color", worst_vk, 0.0, 0.05, " mag")

    print("\n9. tau fits to (b-r) at t_gal and K + E grid (filters 6, 7, 14)")
    band_bj = library.filter(6)
    band_rf = library.filter(7)
    t_gal_gyr = float(UPDATED.t_gal_gyr or 13.0)
    wavelengths = ssp.wavelengths_aa

    def spectrum_for_tau(tau_gyr):
        return exponential_sfr_spectrum(ssp, tau_gyr, t_gal_gyr)

    def photographic_br(spectrum):
        return vega_colour(
            wavelengths, spectrum, band_bj, band_rf, vega, photon_counting=BC95_PHOTON_COUNTING
        )

    for tau_gyr in (1.0, 2.5, 9.0):
        target_colour = photographic_br(spectrum_for_tau(tau_gyr))
        fitted_tau = fit_tau_to_colour(spectrum_for_tau, photographic_br, target_colour)
        report.check(f"recover tau={tau_gyr} Gyr from (b-r) at t={t_gal_gyr}", fitted_tau, tau_gyr, 0.05, " Gyr")

    cosmo = Cosmology(UPDATED)
    reference_spectrum = exponential_sfr_spectrum(ssp, 1.0, t_gal_gyr)
    redshifts = (0.5, 1.0, 1.5, 2.0)
    k_values = [k_correction(wavelengths, reference_spectrum, band_b, vega, z, BC95_PHOTON_COUNTING) for z in redshifts]
    report.check_true("K_B increases from z=0.5 to z=1.0", k_values[1] > k_values[0])
    report.check_true("all K_B > 0 for z in {0.5, 1.0, 1.5, 2.0}", all(k > 0 for k in k_values))
    for z in (0.5, 1.0):
        age_gyr = t_gal_gyr - float(cosmo.lookback_time(z))
        evolved = exponential_sfr_spectrum(ssp, 1.0, age_gyr)
        evolution_b = vega_magnitude(
            wavelengths, evolved, band_b, vega, 0.0, BC95_PHOTON_COUNTING
        ) - vega_magnitude(wavelengths, reference_spectrum, band_b, vega, 0.0, BC95_PHOTON_COUNTING)
        report.check_true(f"evolution E_B < 0 at z={z} (younger population brighter)", evolution_b < 0)

    print(f"\nSummary: {report.number_passed} passed, {report.number_failed} failed")
    return report.number_failed == 0