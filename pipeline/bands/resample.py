"""Resample archive tiles onto the lab mosaic grid."""

from __future__ import annotations

import numpy as np
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

from pipeline.bands.config import BLOCK_ROWS


class ShiftModel:
    """Polynomial offset field (dx, dy) in lab pixels, as a function of lab (x, y)."""

    def __init__(self, coefficients, degree, shape):
        self.coefficients = np.asarray(coefficients)
        self.degree = degree
        self.shape = shape

    def __call__(self, x, y):
        terms = design_matrix(np.column_stack([x.ravel(), y.ravel()]), self.degree, self.shape)
        shift = terms @ self.coefficients
        return shift[:, 0].reshape(x.shape), shift[:, 1].reshape(x.shape)


def identity_shift_model(shape: tuple[int, int]) -> ShiftModel:
    """Zero offset everywhere (degree-0 polynomial)."""
    return ShiftModel(np.zeros((1, 2)), 0, shape)


def design_matrix(points, degree, shape):
    """Monomials x^a y^b with a + b <= degree, coordinates scaled to [-1, 1]."""
    ny, nx = shape
    x = (points[:, 0] - nx / 2) / (nx / 2)
    y = (points[:, 1] - ny / 2) / (ny / 2)
    columns = []
    for total in range(degree + 1):
        for a in range(total + 1):
            columns.append(x**a * y ** (total - a))
    return np.column_stack(columns)


def resample_tile(hdu, tile_wcs: WCS, lab_wcs: WCS, shape, shift_model=None):
    """Resample one archive tile onto the lab grid (flux-conserving bilinear)."""
    ny, nx = shape
    result = np.full(shape, np.nan, dtype=np.float32)
    tile_ny, tile_nx = hdu.shape

    for start in range(0, ny, BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, ny)
        yy, xx = np.mgrid[start : stop + 1, 0 : nx + 1].astype(float)
        if shift_model is not None:
            dx, dy = shift_model(xx, yy)
            xx = xx - dx
            yy = yy - dy
        sky = lab_wcs.pixel_to_world(xx, yy)
        sx, sy = tile_wcs.world_to_pixel(sky)

        area = np.abs(
            (sx[:-1, 1:] - sx[:-1, :-1]) * (sy[1:, :-1] - sy[:-1, :-1])
            - (sx[1:, :-1] - sx[:-1, :-1]) * (sy[:-1, 1:] - sy[:-1, :-1])
        )
        sx = sx[:-1, :-1]
        sy = sy[:-1, :-1]

        inside = (
            np.isfinite(sx)
            & np.isfinite(sy)
            & (sx >= 0)
            & (sy >= 0)
            & (sx <= tile_nx - 1)
            & (sy <= tile_ny - 1)
        )
        if not inside.any():
            continue

        x0 = max(0, int(np.floor(sx[inside].min())))
        x1 = min(tile_nx, int(np.ceil(sx[inside].max())) + 2)
        y0 = max(0, int(np.floor(sy[inside].min())))
        y1 = min(tile_ny, int(np.ceil(sy[inside].max())) + 2)
        cut = np.asarray(hdu.section[y0:y1, x0:x1], dtype=float)
        valid = np.isfinite(cut) & (cut != 0)

        coords = [np.where(inside, sy - y0, -100), np.where(inside, sx - x0, -100)]
        weight = map_coordinates(valid.astype(float), coords, order=1, cval=0, prefilter=False)
        value = map_coordinates(np.where(valid, cut, 0), coords, order=1, cval=0, prefilter=False)

        use = inside & (weight > 0.999999)
        block = result[start:stop]
        block[use] = (value[use] / weight[use]) * area[use]
    return result


def build_band(band: str, tile_dir, tiles, tile_path_fn, lab_wcs, shape, shift_model):
    """Resample tiles for one band; combine with zero-point scaling."""
    combined = np.full(shape, np.nan, dtype=np.float32)
    reference_header = None
    tiles_used = []

    for tile in tiles:
        path = tile_path_fn(tile, band)
        if not path.is_file():
            print(f"  {band}: {path.name} not found, skipped")
            continue
        from pipeline.bands.io import open_image

        hdul, hdu, header, tile_wcs = open_image(path)
        try:
            patch = resample_tile(hdu, tile_wcs, lab_wcs, shape, shift_model)
        finally:
            hdul.close()

        if reference_header is None:
            reference_header = header
            scale = 1.0
        else:
            zp_ref = reference_header.get("MAGZPT")
            zp_tile = header.get("MAGZPT")
            scale = 10 ** (0.4 * (zp_ref - zp_tile)) if zp_ref is not None and zp_tile is not None else 1.0
            overlap = np.isfinite(combined) & np.isfinite(patch)
            if overlap.sum() > 1000:
                a = combined[overlap]
                b = scale * patch[overlap]
                a = a - np.median(a)
                b = b - np.median(b)
                noise = 1.4826 * np.median(np.abs(b))
                bright = b > 30 * noise
                if bright.sum() > 50:
                    ratio = np.median(a[bright] / b[bright])
                    print(f"  {band}: tile overlap flux ratio after ZP scaling = {ratio:.3f} (expect ~1)")

        fill = np.isnan(combined) & np.isfinite(patch)
        combined[fill] = scale * patch[fill]
        tiles_used.append(
            {"file": path.name, "zp_scale": float(scale), "pixels_contributed": int(fill.sum())}
        )

    return combined, reference_header, tiles_used