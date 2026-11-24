"""Expected Bruzual & Charlot GISSEL 1995 (BC95) inputs."""

from dataclasses import dataclass
from pathlib import Path

from paths import DATA_DIR, REPO_DATA_DIR, REPO_ROOT

BC95_DIR_NAMES = ("bc95", "BC95")

SSP_NAMES = ("bc95_ssp_sp_1.ised",)
TAU1_NAMES = ("bc95_tau1_sp_1.ised",)
VEGA_NAMES = (
    "alpha_lyrae_kurucz.txt",
    "vega_kurucz.txt",
    "attachment_vega.txt",
    "alpha_lyrae.txt",
)
COLOR_NAMES = ("bc95_ssp_sp_1.3color", "bc95_ssp_sp_1_3color.txt", "bc95_ssp_sp_1.3color.txt")
COLOR1_NAMES = ("bc95_ssp_sp_1.1color",)
FILTERBIN_NAMES = ("FILTERBIN.RES",)
FILTERFRM_NAMES = ("filterfrm.res",)


@dataclass(frozen=True)
class BC95Paths:
    ssp_ised: Path
    tau1_ised: Path
    vega_spectrum: Path
    ssp_3color: Path
    ssp_1color: Path
    filterbin: Path

    def missing(self) -> list[str]:
        out = []
        for label, path in (
            ("SSP .ised", self.ssp_ised),
            ("tau1 .ised", self.tau1_ised),
            ("Vega spectrum", self.vega_spectrum),
            ("SSP .3color table", self.ssp_3color),
            ("SSP .1color table", self.ssp_1color),
            ("FILTERBIN.RES", self.filterbin),
        ):
            if not path.is_file():
                out.append(f"{label}: {path}")
        return out


def _find_named_file(names: tuple[str, ...], search_dirs: list[Path]) -> Path | None:
    for directory in search_dirs:
        if not directory.is_dir() and directory != REPO_ROOT:
            continue
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def _find_bc95_dir() -> Path:
    for root in (REPO_DATA_DIR, DATA_DIR):
        for name in BC95_DIR_NAMES:
            candidate = root / name
            if candidate.is_dir():
                return candidate
    return REPO_DATA_DIR / "bc95"


def default_bc95_paths() -> BC95Paths:
    base = _find_bc95_dir()
    search = [base, REPO_DATA_DIR / "bc95", DATA_DIR / "bc95"]

    ssp = _find_named_file(SSP_NAMES, search) or (base / SSP_NAMES[0])
    tau1 = _find_named_file(TAU1_NAMES, search) or (base / TAU1_NAMES[0])
    vega = _find_named_file(VEGA_NAMES, search) or (base / VEGA_NAMES[0])
    color = _find_named_file(COLOR_NAMES, search) or (base / COLOR_NAMES[0])
    color1 = _find_named_file(COLOR1_NAMES, search) or (base / COLOR1_NAMES[0])
    filterbin = _find_named_file(FILTERBIN_NAMES, search) or (base / FILTERBIN_NAMES[0])
    return BC95Paths(
        ssp_ised=ssp,
        tau1_ised=tau1,
        vega_spectrum=vega,
        ssp_3color=color,
        ssp_1color=color1,
        filterbin=filterbin,
    )


DATA_README = """
BC95 files (not in git) — place under data/bc95/ or the repo root:

  bc95_ssp_sp_1.ised      SSP, Salpeter IMF, solar Z (STScI cdbs/grid/bc95/binary)
  bc95_tau1_sp_1.ised     BC95 exponential tau=1 Gyr reference model
  alpha_lyrae_kurucz.txt  Kurucz Vega spectrum shipped with BC95 (two-column A, flux)
  bc95_ssp_sp_1.3color    Printed colour/mass table for the SSP file
  bc95_ssp_sp_1.1color    U,B,V,K magnitudes vs age (filter ids 14, 15, 35)
  FILTERBIN.RES           Filter transmission tables (csp.f FILTER0 layout)

Often bundled with the BC95 download as text attachments next to the .ised files.

Source: Bruzual & Charlot 1995 GISSEL atlas on STScI CDBS (binary .ised format).
""".strip()