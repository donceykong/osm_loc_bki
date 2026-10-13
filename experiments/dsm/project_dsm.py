"""Experiment: project an airborne survey (DSM / DEM and its points) into the semantic map's voxel grid.

Runs the pipeline on a sequence, then takes the survey within `--radius` m of the mapped poses,
voxelizes it at the map's resolution and gives every survey voxel

  a ground verdict, three ways:
    dem        its height above the survey's terrain model
    patchwork  Patchwork++ on the survey points around each mapped pose (sensor at the pose)
    lidar      the ground role of the class of the nearest LiDAR-mapped voxel, where one reaches it
  a class, three ways:
    lidar      the nearest LiDAR-mapped voxel's class, where one reaches it
    osm        OSM's support gated by the voxel's ground verdict (no LiDAR at all)
    combined   lidar where it reaches, else osm

and scores the LiDAR-free options against the LiDAR map where the two overlap.

    PYTHONPATH=python python experiments/dsm/project_dsm.py --site kth --seq kth_day_09 --model lidarrv_mcd \\
        --surface data/surface/mcd_kth_epsg32634.npz

The survey model (dsm, dem at `res` m from `origin`, UTM, the map's zone; `pts` airborne points plus
wall points synthesised at roof edges, relative to `origin`, heights in the survey's datum) is the one
osm_localization builds from LAS/LAZ tiles.
"""

import argparse
import os
import time

import matplotlib
import numpy as np
import yaml

import osmlocbki
from osmlocbki.view import bev

GROUND_CLASSES = ("road", "sidewalk", "parking", "terrain")


class Survey:
    def __init__(self, path):
        z = np.load(path)
        self.origin, self.res = z["origin"], float(z["res"])
        self.dsm, self.dem = z["dsm"], z["dem"]
        self.pts = z["pts"]  # (N, 3): x, y relative to origin, z absolute

    def cell(self, xy_utm):
        ij = np.floor((xy_utm - self.origin) / self.res).astype(int)
        return np.clip(ij, 0, np.array(self.dem.shape) - 1)

    def terrain(self, xy_utm):
        i, j = self.cell(xy_utm).T
        return self.dem[i, j]


def keys(xyz, res):
    """Voxel keys of the map's leaf grid (leaf centres at res * (k + 0.5)) packed into int64."""
    k = np.floor(xyz / res).astype(np.int64) + (1 << 20)
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def unpack(key, res):
    k = np.stack([(key >> 42) & 0x1FFFFF, (key >> 21) & 0x1FFFFF, key & 0x1FFFFF], 1) - (1 << 20)
    return ((k + 0.5) * res).astype(np.float32)


def nearest_class(query, ref_keys, ref_cls, reach):
    """For each query key, the class of the nearest reference voxel within `reach` voxels (0: none)."""
    order = np.argsort(ref_keys)
    rk, rc = ref_keys[order], ref_cls[order]
    out = np.zeros(len(query), dtype=int)
    best = np.full(len(query), np.inf)
    r = np.arange(-reach, reach + 1)
    for dx in r:
        for dy in r:
            for dz in r:
                d = dx * dx + dy * dy + dz * dz
                if d > reach * reach:
                    continue
                q = query + (dx << 42) + (dy << 21) + dz
                i = np.clip(np.searchsorted(rk, q), 0, len(rk) - 1)
                hit = (rk[i] == q) & (d < best)
                out[hit], best[hit] = rc[i[hit]], d
    return out


def votes_to_verdict(ground, nonground):
    return np.where(ground + nonground == 0, -1, (ground >= nonground).astype(int))


def agreement(a, b, mask):
    m = mask & (a >= 0) & (b >= 0)
    return (a[m] == b[m]).mean() if m.any() else np.nan, int(m.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--surface", required=True, help="survey model .npz (dsm, dem, pts)")
    ap.add_argument("--pose", default="lag", choices=["lag", "filtered", "gt"])
    ap.add_argument("--strength", type=float, default=-1)
    ap.add_argument("--max-kf", type=int, default=0)
    ap.add_argument("--radius", type=float, default=50.0, help="survey kept within this of the mapped poses")
    ap.add_argument("--reach", type=int, default=2, help="voxels a LiDAR voxel's class/verdict reaches")
    ap.add_argument("--ground-above-dem", type=float, default=0.4, help="dem verdict: ground below this height")
    ap.add_argument("--sensor-height", type=float, default=1.73, help="Patchwork++ on the survey: virtual sensor")
    ap.add_argument("--out", default="experiments/dsm/out")
    ap.add_argument("--reuse", action="store_true", help="reuse the last run's map and poses (from --out)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # the run (with --reuse, the pipeline is only built: OSM support needs its OSM prior)
    t0 = time.time()
    p = osmlocbki.Pipeline(a.site, a.seq, model=a.model, pose=a.pose, strength=a.strength, max_kf=a.max_kf)
    run_cache = os.path.join(a.out, f"{a.site}_{a.seq}_{a.pose}_run.npz")
    if a.reuse:
        r = np.load(run_cache)
        vox, poses = {"xyz": r["xyz"], "cls": r["cls"]}, r["poses"]
    else:
        while p.step():
            pass
        p.finish()
        print(p.status())
        vox = p.voxels()
        poses = p.gt[p.mapped] if a.pose == "gt" else p.trajectory(a.pose)[p.mapped]  # where scans were mapped
        np.savez(run_cache, xyz=vox["xyz"], cls=vox["cls"], poses=poses)
    tax = p.taxonomy()
    names, colors = tax["names"], tax["colors"] / 255.0
    roles = {c["name"]: c["ground"] for c in yaml.safe_load(open(os.path.join(osmlocbki.config_dir(), "labels.yaml")))["classes"]}
    is_ground = np.array([roles[n] == "ground" for n in names])
    is_nonground = np.array([roles[n] == "nonground" for n in names])
    res = 0.3  # config/mapping.yaml map.resolution
    origin = np.asarray(p.map_origin)
    lidar_keys, lidar_cls = keys(vox["xyz"], res), vox["cls"]
    print(f"run: {len(poses)} mapped keyframes, {len(lidar_cls)} LiDAR voxels ({time.time() - t0:.0f} s)")

    # the survey's height datum -> the map's: LiDAR ground-class voxels against the terrain model
    s = Survey(a.surface)
    g = np.isin(lidar_cls, [names.index(n) for n in GROUND_CLASSES])
    dz = np.median(vox["xyz"][g, 2] - s.terrain(vox["xyz"][g, :2] + origin))
    spread = np.percentile(vox["xyz"][g, 2] - s.terrain(vox["xyz"][g, :2] + origin) - dz, [10, 90])
    print(f"datum: map z = survey z {dz:+.2f} m (LiDAR ground voxels vs DEM: 10-90% spread {spread[0]:+.2f} {spread[1]:+.2f} m)")

    # survey points near the mapped poses, in the map frame
    xy = s.pts[:, :2].astype(np.float64) + s.origin
    near = np.zeros(len(xy), bool)
    for q in poses:
        box = np.all(np.abs(xy - q) < a.radius, 1)
        near[box] |= np.sum((xy[box] - q) ** 2, 1) < a.radius ** 2
    pts = np.c_[xy[near] - origin, s.pts[near, 2] + dz].astype(np.float32)
    terrain = s.terrain(xy[near]) + dz
    print(f"survey: {near.sum()} of {len(xy)} points within {a.radius:.0f} m of the mapped poses")

    # voxelize; per voxel: its height above terrain (max over its points) and Patchwork++ votes
    pk = keys(pts, res)
    vk, inv = np.unique(pk, return_inverse=True)
    above = np.full(len(vk), -np.inf)
    np.maximum.at(above, inv, pts[:, 2] - terrain)
    centers = unpack(vk, res)
    v_dem = (above > a.ground_above_dem).astype(int) ^ 1  # 1 ground, 0 non-ground

    t1 = time.time()
    gv, nv = np.zeros(len(vk), int), np.zeros(len(vk), int)
    for q in poses:
        sel = np.nonzero(np.sum((pts[:, :2] - (q - origin)) ** 2, 1) < a.radius ** 2)[0]
        if len(sel) < 100:
            continue
        sensor = np.r_[q - origin, s.terrain(q[None])[0] + dz + a.sensor_height]
        verdict = np.asarray(osmlocbki.segment_ground(pts[sel] - sensor.astype(np.float32), a.sensor_height))
        np.add.at(gv, inv[sel], verdict == 1)
        np.add.at(nv, inv[sel], verdict == 0)
    v_pw = votes_to_verdict(gv, nv)
    print(f"Patchwork++ on the survey: {len(poses)} crops ({time.time() - t1:.0f} s)")

    # what the LiDAR map says at each survey voxel: its own voxel (overlap) and within reach
    own = nearest_class(vk, lidar_keys, lidar_cls, 0)
    reached = nearest_class(vk, lidar_keys, lidar_cls, a.reach)
    v_lidar = np.where(reached > 0, np.where(is_ground[reached], 1, np.where(is_nonground[reached], 0, -1)), -1)
    v_ref = np.where(own > 0, np.where(is_ground[own], 1, np.where(is_nonground[own], 0, -1)), -1)
    overlap = own > 0

    print(f"\n{len(vk)} survey voxels; {overlap.mean():.1%} overlap LiDAR voxels, {(reached > 0).mean():.1%} within "
          f"{a.reach} voxels of one")
    print("ground verdicts vs the overlapping LiDAR voxel's class role (agreement, voxels):")
    for name, v in (("dem", v_dem), ("patchwork", v_pw)):
        acc, n = agreement(v, v_ref, overlap)
        print(f"  {name:10s} {acc:6.1%}  ({n} voxels; {np.mean(v == 1):.0%} ground, {np.mean(v < 0):.0%} no verdict)")

    # classes without LiDAR: OSM support gated by each ground verdict
    utm = centers[:, :2].astype(np.float64) + origin
    results = {}
    for name, v in (("dem", v_dem), ("patchwork", v_pw)):
        sup = np.asarray(p.osm_support(utm, v.astype(np.int8)))
        cls = np.where(sup.max(1) > 0, sup.argmax(1), 0)
        results[name] = cls
        m = overlap & (cls > 0)
        print(f"OSM + {name} ground: labels {np.mean(cls > 0):.0%} of survey voxels; where it overlaps the LiDAR map "
              f"it agrees on {np.mean(cls[m] == own[m]):.1%} ({m.sum()} voxels)")
        per = []
        for c in range(1, len(names)):
            k = m & (own == c)
            if k.sum() > 200:
                per.append(f"{names[c]} {np.mean(cls[k] == c):.0%}")
        print("    recall of the LiDAR class: " + ", ".join(per))
    combined = np.where(reached > 0, reached, results["dem"])

    # coverage: 1 m columns with any labelled voxel within radius of the mapped poses, and how many of
    # those hold something above 3 m over the terrain (roofs, canopy) that only the survey sees from above
    def columns(xyz, cls):
        xy = xyz[cls > 0, :2].astype(np.float64)
        inside = np.zeros(len(xy), bool)
        for q in poses - origin:
            box = np.all(np.abs(xy - q) < a.radius, 1)
            inside[box] |= np.sum((xy[box] - q) ** 2, 1) < a.radius ** 2
        return set(map(tuple, np.floor(xy[inside]).astype(np.int64)))
    lidar_cols, survey_cols = columns(vox["xyz"], lidar_cls), columns(centers, combined)
    high = set(map(tuple, np.floor(centers[(above > 3) & (combined > 0), :2]).astype(np.int64)))
    print(f"\ncoverage within {a.radius:.0f} m of the mapped poses (1 m columns): LiDAR {len(lidar_cols) / 1e4:.2f} ha, "
          f"survey {len(survey_cols) / 1e4:.2f} ha, both {len(lidar_cols | survey_cols) / 1e4:.2f} ha; survey columns "
          f"above 3 m not in the LiDAR map: {len(high - lidar_cols) / 1e4:.2f} ha")

    np.savez_compressed(os.path.join(a.out, f"{a.site}_{a.seq}_survey_voxels.npz"), xyz=centers, origin=origin,
                        ground_dem=v_dem, ground_patchwork=v_pw, ground_lidar=v_lidar, cls_lidar=reached,
                        cls_osm_dem=results["dem"], cls_osm_patchwork=results["patchwork"], cls_combined=combined)

    # figure: top-down views over the run
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    lo = np.r_[poses.min(0) - a.radius - origin]
    hi = np.r_[poses.max(0) + a.radius - origin]
    box = (lo[0], lo[1], hi[0], hi[1])
    cell = 0.6

    def raster(xyz, cls):
        img, ext = bev({"xyz": xyz, "cls": cls}, cell, box)
        rgb = np.ones(img.shape + (3,))
        rgb[img > 0] = colors[img[img > 0]]
        return rgb, ext

    gcol = np.array([[0.85, 0.85, 0.85], [0.55, 0.55, 0.55], [0.35, 0.65, 0.35]])  # none, non-ground, ground
    def verdict_raster(v):
        img, ext = bev({"xyz": centers, "cls": v + 2}, cell, box)  # -1 0 1 -> 1 2 3
        rgb = np.ones(img.shape + (3,))
        rgb[img > 0] = gcol[img[img > 0] - 1]
        return rgb, ext

    panels = [("LiDAR map (OSM-BKI)", raster(vox["xyz"], lidar_cls)),
              ("Survey ground verdict: DEM", verdict_raster(v_dem)),
              ("Survey ground verdict: Patchwork++", verdict_raster(v_pw)),
              ("Survey labelled by OSM + DEM ground (no LiDAR)", raster(centers, results["dem"])),
              ("Survey labelled by LiDAR where it reaches", raster(centers, reached)),
              ("Combined: LiDAR, else OSM + DEM ground", raster(centers, combined))]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, (title, (rgb, ext)) in zip(axes.flat, panels):
        ax.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")
        ax.plot(*(poses - origin).T, color="#222222", lw=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7, colors="#666666")
    axes[0, 0].legend(handles=[Patch(color=colors[c], label=names[c]) for c in range(1, len(names))],
                      loc="upper left", fontsize=7, ncol=2, framealpha=0.9)
    axes[0, 1].legend(handles=[Patch(color=gcol[2], label="ground"), Patch(color=gcol[1], label="non-ground"),
                               Patch(color=gcol[0], label="no verdict")], loc="upper left", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(a.out, f"{a.site}_{a.seq}_survey.png")
    fig.savefig(path, dpi=110)
    print("saved", path)


if __name__ == "__main__":
    main()
