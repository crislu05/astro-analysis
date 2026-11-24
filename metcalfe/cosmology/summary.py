"""Printed summary table for all preset cosmologies."""

from metcalfe.cosmology.config import ALL_CONFIGS
from metcalfe.cosmology.core import Cosmology


def print_summary_table() -> None:
    print("\nComponent A summary")
    print(
        f"  {'Configuration':<22} {'Om':>6} {'OL':>6} {'H0':>6} {'q0':>7} "
        f"{'Age [Gyr]':>10} {'t_gal':>6} {'z_f':>6}"
    )
    for params in ALL_CONFIGS:
        cosmology = Cosmology(params)
        age = cosmology.age_of_universe()
        if params.t_gal_gyr is None:
            t_gal_text, z_f_text = "-", "-"
        else:
            t_gal_text = f"{params.t_gal_gyr:.1f}"
            z_f_text = f"{cosmology.z_formation():.2f}"
        print(
            f"  {params.name:<22} {params.omega_m:>6.3f} {params.omega_lambda:>6.3f} "
            f"{params.h0:>6.1f} {params.q0:>7.3f} {age:>10.2f} {t_gal_text:>6} {z_f_text:>6}"
        )