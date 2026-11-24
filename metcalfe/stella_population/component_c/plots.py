"""Figure: E(z) per galaxy type for Component C."""

from __future__ import annotations

import numpy as np


def save_evolution_figure(grids, path, band_index=2):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from metcalfe.stella_population.component_c.config import GALAXY_TYPES

    figure, axes = plt.subplots(1, len(GALAXY_TYPES), figsize=(15, 3.2), sharey=True)
    styles = {"legacy": "-", "updated": "--", "updated_tgal_12p5": ":"}
    for type_index, axis in enumerate(axes):
        for key, grid in grids.items():
            finite = np.isfinite(grid.evolution[type_index, band_index])
            axis.plot(
                grid.redshifts[finite],
                grid.evolution[type_index, band_index][finite],
                styles[key],
                color="k",
                label=key,
            )
        axis.set_title(GALAXY_TYPES[type_index].name)
        axis.set_xlabel("z")
        axis.set_xlim(0, 4)
    shown = [grid.evolution[:, band_index][:, (grid.redshifts <= 4.0)] for grid in grids.values()]
    lowest = min(float(np.nanmin(np.where(np.isfinite(values), values, np.nan))) for values in shown)
    axes[0].set_ylim(lowest - 0.3, 0.3)
    axes[0].set_ylabel(f"E(z) in {list(grids.values())[0].band_names[band_index]} [mag]")
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)