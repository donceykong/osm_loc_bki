"""Localize a sequence against OSM and map it with OSM-BKI, live.

    python demos/run.py --site kth --seq kth_day_09 --model lidarrv_mcd          # network labels
    python demos/run.py --site tuhh --seq tuhh_day_03                            # ground-truth labels
    python demos/run.py ... --no-view --eval gt_labels --save run.png            # headless, scored

Needs $DATASET_PATH (the disk holding datasets/...). Keys in the view: close the window to stop.
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

import osmlocbki
from osmlocbki.view import LiveView, evaluate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--model", default="", help="segmentation model, for sites with inferred labels")
    ap.add_argument("--pose", default="lag", choices=["lag", "filtered", "gt"], help="pose scans are mapped at")
    ap.add_argument("--strength", type=float, default=-1, help="OSM strength (default: config/mapping.yaml)")
    ap.add_argument("--odometry", default="near_exact", choices=["near_exact", "drifting"])
    ap.add_argument("--scale-bias", type=float, default=0.0, help="simulated odometry scale drift")
    ap.add_argument("--yaw-rw", type=float, default=0.0, help="simulated heading random walk, deg/sqrt(m)")
    ap.add_argument("--max-kf", type=int, default=0)
    ap.add_argument("--map-spacing", type=float, default=5.0)
    ap.add_argument("--building-ids", type=int, nargs="*", default=[],
                    help="override the building label ids (default: the label set's, config/labels.yaml)")
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--draw-every", type=int, default=10, help="keyframes between redraws")
    ap.add_argument("--eval", metavar="LABELS_DIR", help="score the map against <seq>/LABELS_DIR (e.g. gt_labels)")
    ap.add_argument("--label-set", default="mcd", help="label set of --eval's labels")
    ap.add_argument("--save", help="save the final view (png)")
    a = ap.parse_args()

    p = osmlocbki.Pipeline(a.site, a.seq, model=a.model, pose=a.pose, strength=a.strength, odometry=a.odometry,
                           scale_bias=a.scale_bias, yaw_rw=a.yaw_rw, max_kf=a.max_kf, map_spacing=a.map_spacing,
                           building_ids=a.building_ids)
    view = LiveView(p, output="filtered" if a.pose == "filtered" else "lag" if a.pose == "lag" else "graph",
                    show=not a.no_view) if (not a.no_view or a.save) else None
    t0, n_mapped = time.time(), 0
    while p.step():
        if p.t % 100 == 0:
            print(p.status(), flush=True)
        if view and not a.no_view and (p.t % a.draw_every == 0 or len(p.mapped) != n_mapped):
            n_mapped = len(p.mapped)
            view.update()
            if not plt.fignum_exists(view.fig.number):  # the window was closed
                break
    p.finish()
    print(p.status())
    print(f"wall {time.time() - t0:.0f} s")
    gt = p.gt[: p.t + 1]
    for out in ("filtered", "graph", "lag"):
        e = np.linalg.norm(p.trajectory(out) - gt, axis=1)
        print(f"  {out:9s} within 5 m {np.mean(e < 5):6.1%}   median {np.nanmedian(e):5.2f} m")
    if a.eval:
        iou, miou, acc = evaluate(p, a.eval, a.label_set)
        names = p.taxonomy()["names"][1:]
        print("  map vs " + a.eval + ": mIoU %.3f, accuracy %.3f" % (miou, acc))
        print("    " + ", ".join(f"{n} {v:.2f}" for n, v in zip(names, iou) if np.isfinite(v)))
    if view:
        view.update(final=True)
        if a.save:
            view.save(a.save)
            print("saved", a.save)
        if not a.no_view:
            plt.ioff()
            plt.show()


if __name__ == "__main__":
    main()
