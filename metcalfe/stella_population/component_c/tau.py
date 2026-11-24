"""C2: fit exponential SFR tau from photographic (b - r)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from metcalfe.stella_population.component_c.config import (
    GALAXY_TYPES,
    TAU_MAX_GYR,
    TAU_MIN_GYR,
    TAU_SCAN_POINTS,
    Bands,
    GalaxyType,
    PHOTON_COUNTING,
)
from metcalfe.stella_population.component_c.spectrum import galaxy_spectrum
from metcalfe.stella_population.photometry import vega_colour


@dataclass
class TauFit:
    type_name: str
    t_gal_gyr: float
    target: float
    status: str
    tau_gyr: float
    colour_at_tau: float
    colour_range: tuple
    all_roots: tuple = ()
    monotonic: bool = True

    @property
    def ok(self):
        return self.status in ("ok", "multiple_roots")


def model_colour(ssp, bands: Bands, galaxy_type: GalaxyType, tau_gyr, age_gyr):
    spectrum = galaxy_spectrum(ssp, galaxy_type, tau_gyr, age_gyr)
    return vega_colour(
        ssp.wavelengths_aa,
        spectrum,
        bands.b,
        bands.r,
        bands.vega,
        photon_counting=PHOTON_COUNTING,
    )


def fit_tau(ssp, bands: Bands, galaxy_type: GalaxyType, t_gal_gyr, target=None):
    target = galaxy_type.target_br if target is None else target

    def residual(log_tau):
        return model_colour(ssp, bands, galaxy_type, np.exp(log_tau), t_gal_gyr) - target

    log_taus = np.linspace(np.log(TAU_MIN_GYR), np.log(TAU_MAX_GYR), TAU_SCAN_POINTS)
    residuals = np.array([residual(value) for value in log_taus])
    colours = residuals + target
    colour_range = (float(colours.min()), float(colours.max()))
    monotonic = bool(np.all(np.diff(colours) < 0))

    roots = []
    for index in range(len(log_taus) - 1):
        low_value, high_value = residuals[index], residuals[index + 1]
        if low_value == 0.0:
            roots.append(float(np.exp(log_taus[index])))
        elif low_value * high_value < 0.0:
            roots.append(
                float(np.exp(brentq(residual, log_taus[index], log_taus[index + 1], xtol=1e-10)))
            )
    if residuals[-1] == 0.0:
        roots.append(float(np.exp(log_taus[-1])))

    if not roots:
        status = "too_red" if target > colour_range[1] else "too_blue"
        edge = TAU_MIN_GYR if status == "too_red" else TAU_MAX_GYR
        return TauFit(
            galaxy_type.name,
            t_gal_gyr,
            target,
            status,
            edge,
            model_colour(ssp, bands, galaxy_type, edge, t_gal_gyr),
            colour_range,
            (),
            monotonic,
        )
    tau = roots[0]
    status = "ok" if len(roots) == 1 else "multiple_roots"
    return TauFit(
        galaxy_type.name,
        t_gal_gyr,
        target,
        status,
        tau,
        model_colour(ssp, bands, galaxy_type, tau, t_gal_gyr),
        colour_range,
        tuple(roots),
        monotonic,
    )


def fit_all_types(ssp, bands, t_gal_gyr, galaxy_types=GALAXY_TYPES):
    return {galaxy_type.name: fit_tau(ssp, bands, galaxy_type, t_gal_gyr) for galaxy_type in galaxy_types}


def print_fits(title, fits):
    print(f"\n{title}")
    print(f"  {'type':<5} {'target':>7} {'tau [Gyr]':>10} {'model':>7} {'reachable b-r':>16}  status")
    for fit in fits.values():
        tau_text = f"{fit.tau_gyr:>10.3f}" if fit.ok else f"{'none':>10}"
        extra = "" if fit.monotonic else ", colour not monotonic in tau"
        if fit.status == "multiple_roots":
            extra += ", roots " + ", ".join(f"{root:.3f}" for root in fit.all_roots)
        print(
            f"  {fit.type_name:<5} {fit.target:>7.2f} {tau_text} {fit.colour_at_tau:>7.3f} "
            f"{fit.colour_range[0]:>7.3f} to {fit.colour_range[1]:<6.3f}  {fit.status}{extra}"
        )