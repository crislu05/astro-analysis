"""Legacy Table 13 parameters and updated (h-scaled) sets."""

from metcalfe.luminosity import config
from metcalfe.luminosity.schechter import SchechterLF

LEGACY_TYPES = [
    SchechterLF("E/S0", 9.27e-4, -20.85, -0.7, config.LEGACY_H),
    SchechterLF("Sab", 4.63e-4, -20.88, -0.7, config.LEGACY_H),
    SchechterLF("Sbc", 6.20e-4, -21.22, -1.1, config.LEGACY_H),
    SchechterLF("Scd", 2.73e-4, -21.37, -1.5, config.LEGACY_H),
    SchechterLF("Sdm", 1.36e-4, -21.38, -1.5, config.LEGACY_H),
]


def updated_types(normalisation_factor=1.0, h_new=config.UPDATED_H):
    return [
        lf.rescale_h(h_new).scale_normalisation(normalisation_factor) for lf in LEGACY_TYPES
    ]