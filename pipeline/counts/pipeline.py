"""Build differential galaxy number counts from classifier + calibrate outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from pipeline.calibrate.config import MATCH_RADIUS
from pipeline.classifier.zones import zone_of
from pipeline.counts import config
from pipeline.counts.completeness import completeness_at
from pipeline.counts.fitting import compare, fit_range
from pipeline.counts.plots import save_number_counts_plot
from pipeline.counts.zones import zone_areas
from pipeline.photometry.io import load_mask


def run_number_counts(
    mosaic: Path,
    catalogue: Path,
    completeness: Path,
    star_model: Path,
    out_dir: Path,
    *,
    mask_path: Path | None = None,
    transfer: Path | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header = fits.getheader(mosaic)
    pix_deg = float(header.get("PIXSCAL1", 0.258)) / 3600
    if mask_path is None:
        from pipeline.mask import build_mask_state
        from pipeline.mask.io import load_mosaic

        data, hdr = load_mosaic(mosaic)
        mask = build_mask_state(data, hdr).mask
    else:
        mask = load_mask(mask_path)
    cat = Table.read(catalogue)
    comp = Table.read(completeness)
    stars = Table.read(star_model)

    areas_px = zone_areas(mask)
    area = {z: v * pix_deg**2 for z, v in areas_px.items()}
    total_area = sum(area.values())
    floor = len(cat) / sum(areas_px.values()) * np.pi * MATCH_RADIUS**2
    print(
        f"area: inner {area['inner']:.4f}, edge {area['edge']:.4f}, total {total_area:.4f} deg^2; "
        f"chance-match floor {100 * floor:.1f}%"
    )

    centres = config.MAG_EDGES[:-1] + config.BIN / 2
    r = np.asarray(cat["r_gal"], float)
    zones = zone_of(np.asarray(cat["x"]), np.asarray(cat["y"]), mask.shape)
    finite = np.isfinite(r)
    print(f"sources with a finite r_gal: {finite.sum()} of {len(cat)}")

    s_lo = np.asarray(stars["mag_lo"])
    S = np.zeros(len(centres))
    var_S = np.zeros(len(centres))
    for i, lo in enumerate(config.MAG_EDGES[:-1]):
        j = np.where(np.isclose(s_lo, lo))[0]
        if len(j) == 0:
            continue
        j = j[0]
        if lo + config.BIN <= config.SDSS_STAR_LIMIT:
            S[i] = stars["n_star_sdss"][j]
            var_S[i] = S[i]
        else:
            S[i] = stars["n_star_model"][j] if np.isfinite(stars["n_star_model"][j]) else 0.0
            var_S[i] = S[i] + (0.5 * S[i]) ** 2

    rows = []
    for i, (lo, c) in enumerate(zip(config.MAG_EDGES[:-1], centres)):
        inb = finite & (r >= lo) & (r < lo + config.BIN)
        num, var = 0.0, 0.0
        nz, cz = {}, {}
        for z in ("inner", "edge"):
            n_pix = int(np.sum(inb & (zones == z)))
            c_val, sc = completeness_at(comp, z, np.array([c]), floor)
            c_val, sc = float(c_val[0]), float(sc[0])
            sz = S[i] * area[z] / total_area
            nz[z], cz[z] = n_pix, c_val
            if c_val <= 0:
                continue
            num += (n_pix - sz) / c_val
            var += (
                n_pix / c_val**2
                + var_S[i] * (area[z] / total_area) ** 2 / c_val**2
                + ((n_pix - sz) * sc / c_val**2) ** 2
            )
        c_all = (cz["inner"] * area["inner"] + cz["edge"] * area["edge"]) / total_area
        n_val = num / total_area
        sn = np.sqrt(var) / total_area
        n_class_gal = (
            int(np.sum(inb & (np.asarray(cat["class"]) == "galaxy"))) if "class" in cat.colnames else -1
        )
        rows.append(
            {
                "mag_lo": lo,
                "mag_hi": lo + config.BIN,
                "mag": c,
                "N_inner": nz["inner"],
                "N_edge": nz["edge"],
                "N_all": nz["inner"] + nz["edge"],
                "stars_subtracted": S[i],
                "C_inner": cz["inner"],
                "C_edge": cz["edge"],
                "C_area_weighted": c_all,
                "n_per_deg2": n_val,
                "n_err": sn,
                "reliable": bool(
                    c_all >= config.C_MIN and (nz["inner"] + nz["edge"] - S[i]) >= config.MIN_GAL
                ),
                "N_classified_galaxies": n_class_gal,
            }
        )
    tab = Table(rows=rows)
    tab.meta.update(
        {"units": "galaxies per 0.5 mag per deg^2", "area_deg2": total_area, "chance_floor": floor}
    )
    tab.write(out_dir / "number_counts.ecsv", overwrite=True)

    print("\n bin         N_all  stars   C_in  C_edge    n (deg^-2)        log n")
    for row in tab:
        if row["N_all"] == 0:
            continue
        mark = "" if row["reliable"] else "  (unreliable)"
        logn = np.log10(row["n_per_deg2"]) if row["n_per_deg2"] > 0 else np.nan
        print(
            f" {row['mag_lo']:.1f}-{row['mag_hi']:.1f}  {row['N_all']:5d}  {row['stars_subtracted']:5.1f}  "
            f"{row['C_inner']:.2f}  {row['C_edge']:.2f}  {row['n_per_deg2']:9.0f} +- {row['n_err']:6.0f}  "
            f"{logn:.3f}{mark}"
        )

    n = np.asarray(tab["n_per_deg2"], float)
    sn = np.asarray(tab["n_err"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logn = np.log10(n)
        slog = sn / (n * np.log(10))
    base = np.asarray(tab["C_area_weighted"]) >= config.FIT_C_MIN
    base &= (np.asarray(tab["N_all"]) - S >= config.MIN_GAL) & np.isfinite(logn) & (n > 0)

    print("\nslope fits (log10 n = a + b m):")
    fits_ = {
        "all": fit_range(centres, logn, slog, base, "all"),
        "bright": fit_range(centres, logn, slog, base & (centres < config.BREAK_MAG), "bright"),
        "faint": fit_range(centres, logn, slog, base & (centres >= config.BREAK_MAG), "faint"),
    }

    print("\ncomparison with published counts:")
    comps = {
        "yasuda2001": compare(
            centres,
            np.where(base, n, np.nan),
            sn,
            config.YASUDA[:, 0],
            config.YASUDA[:, 1],
            config.YASUDA[:, 2],
            "Yasuda et al. 2001 (r*)",
        ),
        "metcalfe2001": compare(
            centres,
            np.where(base, n, np.nan),
            sn,
            config.METCALFE[:, 0],
            config.METCALFE[:, 1],
            config.METCALFE[:, 2],
            f"Metcalfe et al. 2001 (R + {config.R_TO_R_SHIFT})",
            shift=config.R_TO_R_SHIFT,
            shift_err=config.R_TO_R_SHIFT_ERR,
        ),
    }

    edd = None
    if transfer is not None and fits_.get("all"):
        tr = Table.read(transfer)
        b = fits_["all"]["slope"]
        sig = np.interp(centres, np.asarray(tr["mag_in"]), np.asarray(tr["m_kron_scatter"]))
        boost = np.exp((b * np.log(10) * sig) ** 2 / 2)
        tab["eddington_boost"] = boost
        tab.write(out_dir / "number_counts.ecsv", overwrite=True)
        sel = base
        edd = {"boost_range_in_fit": [float(boost[sel].min()), float(boost[sel].max())]}
        print(
            f"\nEddington boost (not applied), bins in fit: {100 * (boost[sel].min() - 1):.1f}% to "
            f"{100 * (boost[sel].max() - 1):.1f}%"
        )

    for key, c in comps.items():
        if c and fits_.get("all"):
            dm = np.log10(c["mean_ratio"]) / fits_["all"]["slope"]
            c["equivalent_mag_shift"] = float(dm)
            print(
                f"  {key}: mean ratio {c['mean_ratio']:.2f} is equivalent to a magnitude offset of {dm:+.2f} mag"
            )

    summary = {
        "eddington": edd,
        "area_deg2": total_area,
        "area_inner_deg2": area["inner"],
        "area_edge_deg2": area["edge"],
        "chance_floor": floor,
        "fit_completeness_min": config.FIT_C_MIN,
        "fits": fits_,
        "comparison": comps,
    }
    (out_dir / "number_counts_fit.json").write_text(
        json.dumps(summary, indent=1, default=float) + "\n",
        encoding="utf-8",
    )
    save_number_counts_plot(out_dir, tab, centres, n, sn, base, fits_, comps)
    print(f"\nwrote {out_dir}")