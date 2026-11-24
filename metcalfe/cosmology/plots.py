"""Component A figure (proposal Figure 2)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metcalfe.cosmology.config import LEGACY_Q005, LEGACY_Q05, UPDATED
from metcalfe.cosmology.core import Cosmology


def save_component_a_figure(output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    redshift = np.linspace(0.0, 6.0, 601)
    curves = [
        (LEGACY_Q005, "Legacy: $q_0=0.05$, $H_0=50$", "-"),
        (LEGACY_Q05, "Legacy: $q_0=0.5$, $H_0=50$", "--"),
        (UPDATED, "Modern: Planck 2018, $H_0=67.4$", "-."),
    ]

    figure, (axis_volume, axis_time) = plt.subplots(1, 2, figsize=(11, 4.2))

    for params, label, style in curves:
        cosmology = Cosmology(params)
        volume_h_scaled = cosmology.dvdz_per_deg2(redshift) * params.h**3 / 1e6
        axis_volume.plot(redshift, volume_h_scaled, style, label=label)
        axis_time.plot(redshift, cosmology.lookback_time(redshift), style, label=label)

    planck_age = Cosmology(UPDATED).age_of_universe()
    axis_time.axhline(planck_age, color="grey", lw=0.8, ls=":")
    axis_time.text(0.1, planck_age + 0.3, "Planck age 13.8 Gyr", fontsize=9)

    axis_volume.set_xlabel("Redshift $z$")
    axis_volume.set_ylabel(
        r"$dV_c/dz\,d\Omega$ [$10^6\,h^{-3}\,$Mpc$^3\,$deg$^{-2}$]"
    )
    axis_volume.set_title("(a) Geometry ($h$-scaled)")
    axis_volume.legend(fontsize=8)

    axis_time.set_xlabel("Redshift $z$")
    axis_time.set_ylabel("Look-back time [Gyr]")
    axis_time.set_title("(b) Time scale (true $H_0$)")

    for axis in (axis_volume, axis_time):
        axis.set_xlim(0, 6)
        axis.set_ylim(bottom=0)
        axis.minorticks_on()
        axis.tick_params(direction="in", which="both", top=True, right=True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"\nFigure written to {output_path}")
    return output_path