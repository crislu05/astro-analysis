"""Run star/galaxy separation on a calibrated catalogue."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.table import Table

from pipeline.classifier.classify import classify
from pipeline.classifier.log import log
from pipeline.classifier.locus import stellar_locus
from pipeline.classifier.measure import match_reference, measure_shapes
from pipeline.classifier import config
from pipeline.classifier.plots import save_star_galaxy_plot
from pipeline.classifier.star_counts import sdss_counts_unmasked, star_count_model
from pipeline.detect.background import estimate_background
from pipeline.photometry.io import load_detection_inputs, load_mask


def run_classifier(
    mosaic: Path,
    catalogue: Path,
    segmentation: Path,
    sdss_stars: Path,
    out_dir: Path,
    *,
    mask_path: Path | None = None,
    sdss_galaxies: Path | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_override = load_mask(mask_path) if mask_path is not None else None
    data, header, cat, segm, mask, _gain = load_detection_inputs(
        mosaic, catalogue, segmentation, mask=mask_override
    )
    sdss = Table.read(sdss_stars, format="ascii.csv", comment="#")
    area_deg2 = float((~mask).sum()) * (float(header.get("PIXSCAL1", 0.258)) / 3600) ** 2

    bkg = estimate_background(data, mask)
    r50, conc = measure_shapes(bkg.data_sub, segm, mask, cat)
    mag = np.asarray(cat["r_point"], float)
    flag = np.asarray(cat["flag"])

    star_idx = match_reference(data, header, sdss, cat)
    log(f"SDSS stars matched to catalogue: {(star_idx >= 0).sum()} of {len(sdss)}")
    locus, sig_sdss, loc_tab = stellar_locus(mag, conc, star_idx, flag)

    cls, dist, mag_limit = classify(mag, conc, flag, locus, loc_tab)
    for row in loc_tab:
        log(
            f"  r {row['mag_lo']:.1f}: n {row['n']:5d}, locus width {row['sigma_used']:.3f}, "
            f"galaxy offset {row['galaxy_median_offset']:.3f} -> separation {row['separation']:.1f}"
        )
    log(
        f"classification possible to r = {mag_limit:.1f} "
        f"(galaxies > {config.SEPARATION_SIGMA:g} locus widths above the locus)"
    )
    for name in ("star", "galaxy", "artefact", "unclassified"):
        log(f"  {name:12s} {np.sum(cls == name):5d}")

    m_idx = star_idx[star_idx >= 0]
    inrange = mag[m_idx] < mag_limit
    star_completeness = float(np.mean(cls[m_idx][inrange] == "star")) if inrange.any() else np.nan
    log(
        f"SDSS stars inside the classified range classified as stars: "
        f"{np.sum(cls[m_idx][inrange] == 'star')} of {inrange.sum()} ({100 * star_completeness:.0f}%)"
    )
    missed = m_idx[inrange & (cls[m_idx] != "star")]
    for j in missed:
        log(f"    missed: r = {mag[j]:.2f}, c = {conc[j]:.3f}, distance {dist[j]:+.1f} sigma, flag {flag[j]}")
    purity = None
    if sdss_galaxies is not None:
        gal = Table.read(sdss_galaxies, format="ascii.csv", comment="#")
        g_idx = match_reference(data, header, gal, cat)
        g = g_idx[g_idx >= 0]
        g = g[mag[g] < mag_limit]
        purity = float(np.mean(cls[g] == "star")) if len(g) else np.nan
        log(f"SDSS galaxies classified as stars: {np.sum(cls[g] == 'star')} of {len(g)}")

    n_ref = sdss_counts_unmasked(header, mask, sdss)
    model, slope, slope_err = star_count_model(mag, cls, area_deg2, mag_limit, n_ref)
    log(f"star counts (fit to SDSS stars on unmasked pixels): slope dlog10N/dm = {slope:.3f} +- {slope_err:.3f}")
    ov = (np.asarray(model["mag_lo"]) >= config.STAR_FIT_RANGE_MIN) & (
        np.asarray(model["mag_hi"]) <= config.SDSS_STAR_LIMIT
    )
    n_cls_ov, n_ref_ov = int(np.sum(model["n_star_classified"][ov])), int(np.sum(model["n_star_sdss"][ov]))
    contamination = (n_cls_ov - n_ref_ov) / n_cls_ov if n_cls_ov else np.nan
    log(
        f"  classified stars vs SDSS stars for r {config.STAR_FIT_RANGE_MIN}-{config.SDSS_STAR_LIMIT}: "
        f"{n_cls_ov} vs {n_ref_ov} -> about {100 * contamination:.0f}% of our 'stars' are compact galaxies"
    )
    for row in model:
        if row["mag_lo"] >= config.STAR_FIT_RANGE_MIN:
            n_all = int(np.sum((mag >= row["mag_lo"]) & (mag < row["mag_hi"])))
            frac = row["n_star_model"] / n_all if n_all else np.nan
            log(
                f"  r {row['mag_lo']:.1f}-{row['mag_hi']:.1f}: all {n_all:5d}, "
                f"SDSS stars {row['n_star_sdss']:3d}, classified {row['n_star_classified']:3d}, "
                f"model {row['n_star_model']:6.1f} ({100 * frac:4.1f}% of all)"
            )

    out_cat = cat.copy()
    out_cat["r50"] = r50
    out_cat["concentration"] = conc
    out_cat["locus_distance_sigma"] = dist
    out_cat["class"] = cls
    out_cat.meta.update(
        {
            "locus_coeffs": loc_tab.meta["locus_coeffs"],
            "classification_limit": mag_limit,
            "N_SIGMA": config.N_SIGMA,
            "SEPARATION_SIGMA": config.SEPARATION_SIGMA,
        }
    )
    out_cat.write(out_dir / "classified_catalogue.ecsv", overwrite=True)
    model.write(out_dir / "star_count_model.ecsv", overwrite=True)
    loc_tab.write(out_dir / "stellar_locus.ecsv", overwrite=True)
    summary = {
        "locus_coeffs": loc_tab.meta["locus_coeffs"],
        "sdss_star_scatter": sig_sdss,
        "classification_limit_mag": mag_limit,
        "counts": {n: int(np.sum(cls == n)) for n in ("star", "galaxy", "artefact", "unclassified")},
        "sdss_star_completeness": star_completeness,
        "sdss_galaxy_as_star_fraction": purity,
        "star_count_slope": slope,
        "star_count_slope_err": slope_err,
        "star_class_contamination": contamination,
        "area_deg2": area_deg2,
    }
    (out_dir / "star_galaxy_summary.json").write_text(
        json.dumps(summary, indent=1, default=float) + "\n",
        encoding="utf-8",
    )
    save_star_galaxy_plot(out_dir, mag, r50, conc, cls, star_idx, locus, loc_tab, mag_limit, model)
    log(f"wrote {out_dir}")