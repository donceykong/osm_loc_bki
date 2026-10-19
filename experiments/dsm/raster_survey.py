"""Build a site's survey model (data/surface/<map>_epsg<zone>.npz, the format src/survey.cpp reads)
from DSM and DEM rasters, for sites whose survey comes as GeoTIFFs rather than LAS/LAZ tiles:

    # needs rasterio (not in environment.yaml; e.g. a separate env with rasterio, pyproj, pyyaml)
    python experiments/dsm/raster_survey.py --site kitti360 \\
        --dsm $DATASET_PATH/datasets/kitti360/geospatial/kitti360_DGM025-2026/kitti360_DOM1/merged_dsm.tif \\
        --dem $DATASET_PATH/datasets/kitti360/geospatial/kitti360_DGM025-2026/kitti360_DGM025/merged_dem.tif

The rasters are cropped to the site's map_bbox (osm_localization's config/sites.yaml; a site without
one takes that of a site sharing its map) plus a margin and resampled onto
one grid (`res`): the DSM by nearest cell (keeps roof edges sharp), the DEM by the mean of its cells.
The surveyed points are the DSM's surface, one point every `spacing` m (finer than the raster, so
they fill the map's voxels), plus wall points down each edge where a building-like cell (`solid`:
over `solid_height` above the terrain with a smooth surface; tree crowns are rough) steps down by
more than `wall_step`, as the LAS models synthesise them at roof edges. Heights stay in the rasters'
datum (the pipeline fits the offset, survey: z_offset auto).

The rasters' frame (e.g. ETRS89 / UTM, EPSG:258xx) is taken as the WGS84 UTM zone of the same number;
the two drift apart by ~2.5 cm a year (~0.6 m by 2013): `--shift DX DY` moves the survey onto the
site's frame.
"""

import argparse
import math
import os

import numpy as np
import pyproj
import rasterio
import rasterio.windows
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CELL = 25.0  # m: the crop bins src/survey.cpp expects (Survey::kCell)


def site_box(site, margin):
    """The site's map_bbox in its UTM zone, grown by margin: (epsg, x0, y0, x1, y1)."""
    with open(os.path.join(ROOT, "third_party", "osm_localization", "config", "sites.yaml")) as f:
        sites = yaml.safe_load(f)
    s = sites[site]
    box = s.get("map_bbox") or next((o["map_bbox"] for o in sites.values() if o.get("map") == s["map"] and o.get("map_bbox")),
                                    None)
    if box is None:
        raise SystemExit(f"site {site}: no map_bbox, nor a site with map {s['map']} that has one")
    (lat0, lon0), (lat1, lon1) = box
    zone = int(math.floor(((lon0 + lon1) / 2 + 180) / 6)) + 1
    epsg = (32600 if lat0 >= 0 else 32700) + zone
    utm = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
    xs, ys = utm.transform([lon0, lon1, lon0, lon1], [lat0, lat0, lat1, lat1])
    return s["map"], epsg, min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin


def read(path, x0, y0, x1, y1):
    """The raster over [x0, x1] x [y0, y1] -> (array indexed [x, y] with y north, its corner, res); NaN = none."""
    with rasterio.open(path) as r:
        win = rasterio.windows.from_bounds(x0, y0, x1, y1, r.transform).round_offsets().round_lengths()
        a = r.read(1, window=win).astype(np.float32)
        if r.nodata is not None:
            a[a == r.nodata] = np.nan
        t = r.window_transform(win)
        assert t.b == 0 and t.d == 0 and t.e < 0, "north-up rasters only"
        res = t.a
        corner = np.array([t.c, t.f + t.e * a.shape[0]])  # lower-left
    return np.ascontiguousarray(a[::-1].T), corner, res  # rows north-down -> [x, y]


def resample(a, corner, res, origin, out_res, shape, how):
    """a (res m cells from corner) onto the out_res grid at origin: nearest, or the mean of the cells within."""
    if how == "nearest" or res >= out_res:
        c = (origin + (np.indices(shape).reshape(2, -1).T + 0.5) * out_res - corner) / res
        ij = np.floor(c).astype(int)
        ok = (ij >= 0).all(1) & (ij[:, 0] < a.shape[0]) & (ij[:, 1] < a.shape[1])
        out = np.full(len(ij), np.nan, np.float32)
        out[ok] = a[ij[ok, 0], ij[ok, 1]]
        return out.reshape(shape)
    k = int(round(out_res / res))  # whole cells per output cell
    off = np.round((origin - corner) / res).astype(int)
    sub = a[off[0]: off[0] + shape[0] * k, off[1]: off[1] + shape[1] * k]
    pad = np.full((shape[0] * k, shape[1] * k), np.nan, np.float32)
    pad[: sub.shape[0], : sub.shape[1]] = sub
    return np.nanmean(pad.reshape(shape[0], k, shape[1], k), axis=(1, 3)).astype(np.float32)


def fill(a, fallback):
    """NaN cells -> fallback (or the median when the fallback has none there)."""
    a = np.where(np.isfinite(a), a, fallback)
    return np.where(np.isfinite(a), a, np.nanmedian(a)).astype(np.float32)


def roughness(z):
    """The spread of each cell's 3x3 neighbourhood about their mean (m)."""
    p = np.pad(z, 1, mode="edge")
    stack = np.stack([p[1 + dx: p.shape[0] - 1 + dx, 1 + dy: p.shape[1] - 1 + dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    return stack.std(0)


def surface_points(dsm, res, spacing):
    """One point every `spacing` m over the DSM (nearest cell): (N, 3) relative to the grid's corner."""
    k = max(1, int(round(res / spacing)))
    sub = (np.arange(k) + 0.5) * res / k
    nx, ny = dsm.shape
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    out = []
    for sx in sub:
        for sy in sub:
            out.append(np.c_[(ix * res + sx).ravel(), (iy * res + sy).ravel(), dsm.ravel()])
    return np.concatenate(out).astype(np.float32)


def wall_points(dsm, solid, res, spacing, step):
    """Points down each edge where a solid cell is over `step` m above its 4-neighbour, `spacing` m apart
    along the edge and down it, from the roof to the neighbour's height."""
    out = []
    along = (np.arange(max(1, int(round(res / spacing)))) + 0.5) * res / max(1, int(round(res / spacing)))
    for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
        nb = np.roll(dsm, -sign, axis=axis)
        drop = dsm - nb
        if axis == 0:
            drop[-1 if sign > 0 else 0, :] = 0
        else:
            drop[:, -1 if sign > 0 else 0] = 0
        i, j = np.nonzero(solid & (drop > step))
        top, low = dsm[i, j], nb[i, j]
        levels = np.floor((top - low) / spacing).astype(int)
        rep = np.repeat(np.arange(len(i)), levels)
        z = top[rep] - (np.arange(len(rep)) - np.repeat(np.cumsum(levels) - levels, levels) + 1) * spacing
        for a in along:
            if axis == 0:
                x = (i[rep] + (1 if sign > 0 else 0)) * res
                y = j[rep] * res + a
            else:
                x = i[rep] * res + a
                y = (j[rep] + (1 if sign > 0 else 0)) * res
            out.append(np.c_[x, y, z])
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 3), np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True, help="config/sites.yaml site: its map_bbox and map name")
    ap.add_argument("--dsm", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--margin", type=float, default=100.0, help="m around the map's bbox")
    ap.add_argument("--res", type=float, default=0.5, help="grid of dsm / dem, m")
    ap.add_argument("--spacing", type=float, default=0.25, help="surveyed points, m apart")
    ap.add_argument("--solid-height", type=float, default=2.5, help="building-like: this far above the terrain...")
    ap.add_argument("--solid-roughness", type=float, default=0.35, help="...with a 3x3 height spread under this, m")
    ap.add_argument("--wall-step", type=float, default=1.0, help="walls where a solid cell drops more than this, m")
    ap.add_argument("--shift", type=float, nargs=2, default=[0.0, 0.0], metavar=("DX", "DY"),
                    help="added to the rasters' x, y to reach the site's frame, m")
    ap.add_argument("--out", help="default: data/surface/<map>_epsg<zone>.npz")
    a = ap.parse_args()

    name, epsg, x0, y0, x1, y1 = site_box(a.site, a.margin)
    origin = np.floor(np.array([x0, y0]) / a.res) * a.res
    shape = tuple(int(v) for v in np.ceil((np.array([x1, y1]) - origin) / a.res))
    src = origin - a.shift  # the grid in the rasters' frame
    box = (src[0] - 2, src[1] - 2, src[0] + shape[0] * a.res + 2, src[1] + shape[1] * a.res + 2)
    dsm_r, dsm_c, dsm_res = read(a.dsm, *box)
    dem_r, dem_c, dem_res = read(a.dem, *box)
    print(f"{name}: EPSG:{epsg}, {shape[0] * a.res:.0f} x {shape[1] * a.res:.0f} m at {a.res} m; "
          f"DSM {dsm_res} m, DEM {dem_res} m; shift {a.shift}")
    dem = resample(dem_r, dem_c, dem_res, src, a.res, shape, "mean")
    dsm = resample(dsm_r, dsm_c, dsm_res, src, a.res, shape, "nearest")
    dem = fill(dem, dsm)
    dsm = fill(dsm, dem)
    dsm = np.maximum(dsm, dem)
    solid = ((dsm - dem) > a.solid_height) & (roughness(dsm) < a.solid_roughness)

    pts = np.concatenate([surface_points(dsm, a.res, a.spacing),
                          wall_points(dsm, solid, a.res, a.spacing, a.wall_step)])
    bins = np.ceil(np.array(shape) * a.res / CELL).astype(np.int64)
    b = np.clip(np.floor(pts[:, :2] / CELL).astype(np.int64), 0, bins - 1)
    key = b[:, 0] * bins[1] + b[:, 1]
    order = np.argsort(key, kind="stable")
    pts, key = pts[order], key[order]
    starts = np.searchsorted(key, np.arange(bins[0] * bins[1] + 1)).astype(np.int64)

    out = a.out or os.path.join(ROOT, "data", "surface", f"{name}_epsg{epsg}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, origin=origin.astype(np.float64), res=np.float64(a.res), epsg=np.int64(epsg), dsm=dsm, dem=dem,
             solid=solid, pts=pts, bins=bins, starts=starts)
    print(f"  {len(pts)} points ({solid.mean():.1%} of cells solid), heights {np.percentile(dsm, 1):.1f}.."
          f"{np.percentile(dsm, 99):.1f} m -> {out} ({os.path.getsize(out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
