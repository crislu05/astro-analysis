"""Constants for Component C tau fit and K+E evolution."""

from __future__ import annotations

from dataclasses import dataclass

from metcalfe.stella_population.photometry import Filter

PHOTON_COUNTING = False  # BC95 tables use energy counting

B_PHOTOGRAPHIC_ID = 6
R_PHOTOGRAPHIC_ID = 7
JOHNSON_B_ID = 14
JOHNSON_V_ID = 15
JOHNSON_K_ID = 35

EXPECTED_WINDOWS_AA = {
    B_PHOTOGRAPHIC_ID: (4200.0, 4900.0),
    R_PHOTOGRAPHIC_ID: (5900.0, 7000.0),
    JOHNSON_B_ID: (4200.0, 4600.0),
    JOHNSON_V_ID: (5300.0, 5700.0),
    JOHNSON_K_ID: (20000.0, 23000.0),
}

TAU_MIN_GYR = 0.01
TAU_MAX_GYR = 200.0
TAU_SCAN_POINTS = 61

DUST_REFERENCE_AA = 4400.0
APPLY_DUST_TO_TARGET_COLOUR = True

KE_DZ = 0.02
AGE_FLOOR_GYR = 0.01

SUN_ABSOLUTE_B_VEGA = 5.48
PARSEC_CM = 3.0857e18
VEGA_FLUX_5556_CGS = 3.44e-9

LEGACY_TAU_TOLERANCE = 0.25


@dataclass(frozen=True)
class GalaxyType:
    name: str
    target_br: float
    legacy_tau_gyr: float
    has_dust: bool
    a_b: float = 0.3


GALAXY_TYPES = (
    GalaxyType("E/S0", 1.59, 2.5, False),
    GalaxyType("Sab", 1.38, 2.5, False),
    GalaxyType("Sbc", 1.16, 9.0, True),
    GalaxyType("Scd", 0.84, 9.0, True),
    GalaxyType("Sdm", 0.75, 9.0, True),
)


@dataclass(frozen=True)
class Bands:
    b: Filter
    r: Filter
    vega: tuple