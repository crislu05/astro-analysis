"""Printed LF parameter table."""

import numpy as np

from metcalfe.luminosity.presets import LEGACY_TYPES, updated_types


def print_tables() -> None:
    print("\nLuminosity functions (rest b band)")
    print(
        f"  {'Type':<5} {'alpha':>6} {'M* h=0.5':>9} {'M* h=0.674':>11} "
        f"{'phi* h=0.5':>11} {'phi* h=0.674':>13} {'n(M<-12)':>10} {'j_b share':>10}"
    )
    rescaled = updated_types()
    light = np.array(
        [lf.luminosity_density_in_l_star() * 10 ** (-0.4 * lf.m_star) for lf in rescaled]
    )
    shares = light / light.sum()
    for legacy, new, share in zip(LEGACY_TYPES, rescaled, shares):
        print(
            f"  {legacy.name:<5} {legacy.alpha:>6.1f} {legacy.m_star:>9.2f} {new.m_star:>11.2f} "
            f"{legacy.phi_star * 1e4:>9.2f}e-4 {new.phi_star * 1e4:>11.2f}e-4 "
            f"{new.number_density():>10.4f} {share:>10.3f}"
        )
    print(
        "  phi* in Mpc^-3; n(M<-12) in Mpc^-3 at h = 0.674, before A; "
        "j_b share = fraction of b-band light"
    )