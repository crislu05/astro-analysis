"""Metcalfe Component C §C2–C6: tau fits, legacy regression, [K+E] grids.

Run:
    python metcalfe_component_c_fit.py --selftest
    python metcalfe_component_c_fit.py

Outputs: result/metcalfe/component_c_tau.json, component_c_kegrid.npz, component_c_evolution.png.
"""

from metcalfe.stella_population.component_c.cli import main

if __name__ == "__main__":
    main()