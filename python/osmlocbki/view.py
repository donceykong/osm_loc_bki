"""The live view of a run (matplotlib), and the map's evaluation against ground-truth labels.

Left, localization: the OSM map (buildings, and the other OSM classes faintly), ground truth, the
estimate, the newest keyframe's front-end hits, and the last mapped scan at the pose it was mapped at.
Right, the OSM-BKI semantic map seen from above (each column's highest voxel), around that scan.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Patch

# recessive OSM backdrop, one tint per OSM class family
OSM_TINT = {"roads": "#c8c8c8", "sidewalks": "#d9c8e6", "cycleways": "#c8dce6", "parking": "#e1e1f0",
            "grasslands": "#d8ecd0", "trees": "#c4e0bc", "forest": "#b8d8b0", "fences": "#e6c8b4"}
TRACK = {"gt": "#222222", "estimate": "#2a6fdb", "hits": "#d6453d"}


def bev(vox, cell, box=None):
    """Voxels -> top-down class raster (each column's highest voxel): (image (H, W) int, extent)."""
    xyz, cls = vox["xyz"], vox["cls"]
    if len(cls) == 0:
        return None, None
    lo = xyz[:, :2].min(0) if box is None else np.array(box[:2])
    hi = xyz[:, :2].max(0) if box is None else np.array(box[2:])
    ij = np.floor((xyz[:, :2] - lo) / cell).astype(int)
    shape = np.floor((hi - lo) / cell).astype(int) + 1
    ok = (ij >= 0).all(1) & (ij < shape).all(1)
    ij, z, c = ij[ok], xyz[ok, 2], cls[ok]
    order = np.argsort(z)  # later writes win: the highest voxel
    img = np.zeros((shape[1], shape[0]), dtype=int)
    img[ij[order, 1], ij[order, 0]] = c[order]
    return img, (lo[0], lo[0] + shape[0] * cell, lo[1], lo[1] + shape[1] * cell)


class LiveView:
    def __init__(self, pipe, output="lag", window=70.0, follow=True, show=True):
        self.p, self.output, self.window, self.follow = pipe, output, window, follow
        tax = pipe.taxonomy()
        self.names, self.colors = tax["names"], tax["colors"] / 255.0
        self.origin = None
        if not show:
            plt.switch_backend("Agg")
        self.fig, (self.ax, self.axm) = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [1, 1]})
        self.fig.patch.set_facecolor("white")
        self._draw_osm()
        gt = pipe.gt
        self.ax.plot(gt[:, 0], gt[:, 1], color=TRACK["gt"], lw=0.8, alpha=0.5, label="ground truth")
        (self.est,) = self.ax.plot([], [], color=TRACK["estimate"], lw=2, label=f"estimate ({output})")
        self.hits = self.ax.scatter([], [], s=10, color=TRACK["hits"], label="front-end hits", zorder=4)
        self.scan = self.ax.scatter([], [], s=0.3, c=[], zorder=3)
        self.ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        self.ax.set_aspect("equal")
        self.ax.set_title("Global localization against OSM", fontsize=11)
        self.img = None
        self.axm.set_aspect("equal")
        self.axm.set_title("OSM-BKI semantic map (top-down)", fontsize=11)
        handles = [Patch(color=self.colors[c], label=self.names[c]) for c in range(1, len(self.names))]
        handles.append(Patch(facecolor="none", edgecolor="#555555", label="OSM buildings"))
        self.axm.legend(handles=handles, loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
        for a in (self.ax, self.axm):
            a.ticklabel_format(useOffset=False, style="plain")
            a.tick_params(labelsize=7, colors="#666666")
            for s in a.spines.values():
                s.set_color("#bbbbbb")
        self.text = self.fig.text(0.01, 0.01, "", fontsize=8, family="monospace", color="#333333")
        self.fig.tight_layout(rect=(0, 0.04, 1, 1))
        if show:
            plt.ion()
            plt.show(block=False)

    def _draw_osm(self):
        m = self.p.osm()
        polys = {}
        lines = {}
        for cls, kind, xy in m["features"]:
            (polys if kind == "area" else lines).setdefault(cls, []).append(xy)
        for cls, ps in polys.items():
            self.ax.add_collection(PolyCollection(ps, facecolor=OSM_TINT[cls], edgecolor="none", zorder=0))
        for cls, ls in lines.items():
            pts = [xy for xy in ls if len(xy) > 1]
            if pts:
                self.ax.add_collection(LineCollection(pts, colors=OSM_TINT[cls], linewidths=1.5, zorder=0))
        self.ax.add_collection(PolyCollection(m["buildings"], facecolor="#9a9a9a", edgecolor="#7a7a7a", lw=0.3, zorder=1))
        # the same outlines over the semantic map, to show how it sits on OSM
        self.axm.add_collection(PolyCollection(m["buildings"], facecolor="none", edgecolor="#555555", lw=0.6, zorder=3))
        gt = self.p.gt
        pad = 100
        self.ax.set_xlim(gt[:, 0].min() - pad, gt[:, 0].max() + pad)
        self.ax.set_ylim(gt[:, 1].min() - pad, gt[:, 1].max() + pad)

    def _class_rgb(self, cls):
        return self.colors[np.clip(cls, 0, len(self.colors) - 1)]

    def update(self, final=False):
        p, t = self.p, self.p.t
        if t < 0:
            return
        xy = p.trajectory(self.output)
        ok = np.isfinite(xy[:, 0])
        self.est.set_data(xy[ok, 0], xy[ok, 1])
        self.hits.set_offsets(p.hits(t))
        here = xy[ok][-1] if ok.any() else p.trajectory("filtered")[-1]
        if p.mapped:
            s = p.last_mapped()
            vox_origin = p.map_origin
            pts = s["points"][:, :2] + vox_origin
            keep = s["cls"] > 0
            self.scan.set_offsets(pts[keep])
            self.scan.set_color(self._class_rgb(s["cls"][keep]))
            c = s["xy"] - vox_origin
            w = self.window
            if final:
                vox = p.voxels()
                img, ext = bev(vox, 0.5)
            else:
                box = (c[0] - w, c[1] - w, c[0] + w, c[1] + w)
                img, ext = bev(p.voxels(box=box), 0.3, box)
            if img is not None:
                ext = (ext[0] + vox_origin[0], ext[1] + vox_origin[0], ext[2] + vox_origin[1], ext[3] + vox_origin[1])
                rgb = np.ones(img.shape + (3,))
                rgb[img > 0] = self._class_rgb(img[img > 0])
                if self.img is None:
                    self.img = self.axm.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")
                else:
                    self.img.set_data(rgb)
                    self.img.set_extent(ext)
                self.axm.set_xlim(ext[0], ext[1])
                self.axm.set_ylim(ext[2], ext[3])
        if self.follow and not final:
            w = 2.5 * self.window
            self.ax.set_xlim(here[0] - w, here[0] + w)
            self.ax.set_ylim(here[1] - w, here[1] + w)
        elif final:
            gt = p.gt
            self.ax.set_xlim(gt[:, 0].min() - 50, gt[:, 0].max() + 50)
            self.ax.set_ylim(gt[:, 1].min() - 50, gt[:, 1].max() + 50)
        self.text.set_text(p.status())
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save(self, path):
        self.fig.savefig(path, dpi=130)


def evaluate(pipe, labels_dir, label_set="mcd", frames=None):
    """The map against ground-truth labels: each mapped scan at its ground-truth pose,
    classified by the map, against <seq>/<labels_dir>/<scan>.bin (int32, in `label_set`).
    -> (per-class IoU over the common classes 1.., mIoU, accuracy); points the map left unlabeled
    count as misses; scans without a label file (KITTI-360 leaves a few out) are skipped."""
    from . import Site, label_mapping
    site = Site(pipe.site)
    lut = np.array(label_mapping(label_set))
    n = len(pipe.taxonomy()["names"])
    conf = np.zeros((n, n), dtype=np.int64)
    sids = pipe.frame_scan_ids
    for f in pipe.mapped if frames is None else frames:
        sid = sids[f]
        path = os.path.join(site.seq_dir(pipe.seq), labels_dir, f"{sid:010d}.bin")
        if not os.path.exists(path):
            continue
        gt = np.fromfile(path, dtype=np.int32)
        gt = np.where((gt >= 0) & (gt < len(lut)), lut[np.clip(gt, 0, len(lut) - 1)], 0)
        pred = np.asarray(pipe.classify_frame(f))
        k = gt > 0
        np.add.at(conf, (gt[k], pred[k]), 1)
    tp = np.diag(conf)[1:].astype(float)
    denom = conf[1:, :].sum(1) + conf[:, 1:].sum(0) - np.diag(conf)[1:]
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    return iou, np.nanmean(iou), tp.sum() / max(conf[1:].sum(), 1)
