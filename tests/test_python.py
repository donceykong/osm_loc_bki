"""Python side: the OSM converter's classes, and (with $DATASET_PATH holding MCD KTH) a short run
through the bindings and the live view, headless.

    PYTHONPATH=python python tests/test_python.py        # or: pytest tests/test_python.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from osmlocbki import osm  # noqa: E402

OSM = """<?xml version="1.0"?><osm>
<node id="1" lon="18.000" lat="59.000"/><node id="2" lon="18.001" lat="59.000"/>
<node id="3" lon="18.001" lat="59.001"/><node id="4" lon="18.000" lat="59.001"/>
<node id="5" lon="18.002" lat="59.000"><tag k="natural" v="tree"/></node>
<way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/><tag k="building" v="yes"/></way>
<way id="11"><nd ref="1"/><nd ref="2"/><tag k="highway" v="primary"/><tag k="sidewalk" v="both"/></way>
<way id="12"><nd ref="2"/><nd ref="3"/><tag k="highway" v="footway"/><tag k="width" v="3 m"/></way>
<way id="13"><nd ref="3"/><nd ref="4"/><tag k="barrier" v="fence"/></way>
<way id="14"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="1"/><tag k="landuse" v="grass"/></way>
<way id="15"><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="2"/><tag k="amenity" v="parking"/><tag k="highway" v="service"/></way>
</osm>"""


def test_osm_classes():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "toy.osm")
        with open(src, "w") as f:
            f.write(OSM)
        counts = osm.to_geojson(src, os.path.join(d, "toy"))
    assert counts == {"buildings": 1, "grasslands": 1, "parking": 1, "roads": 1, "sidewalks": 2, "fences": 1,
                      "trees": 1}, counts
    assert osm.line_classes({"highway": "primary"}) == [("roads", 7.0)]
    assert osm.line_classes({"highway": "residential", "width": "5.5"}) == [("roads", 5.5)]
    assert osm.line_classes({"highway": "path", "bicycle": "yes"}) == [("cycleways", None)]
    assert osm.area_class({"natural": "wood"}) == "forest"


def test_short_run_and_view():
    base = os.environ.get("DATASET_PATH")
    if not base or not os.path.isdir(os.path.join(base, "datasets/mcd/kth/kth_day_09/inferred_labels/lidarrv_mcd")):
        print("  skipped: needs $DATASET_PATH with MCD KTH and lidarrv_mcd labels")
        return
    import osmlocbki
    from osmlocbki.view import LiveView, evaluate
    p = osmlocbki.Pipeline("kth", "kth_day_09", model="lidarrv_mcd", pose="filtered", max_kf=250)
    view = LiveView(p, output="filtered", show=False)
    while p.step():
        if p.t % 50 == 0:
            view.update()
    p.finish()
    view.update(final=True)
    with tempfile.TemporaryDirectory() as d:
        view.save(os.path.join(d, "view.png"))
        assert os.path.getsize(os.path.join(d, "view.png")) > 10000
    err = np.linalg.norm(p.trajectory("filtered")[150:] - p.gt[150:250], axis=1)
    assert np.median(err) < 3, np.median(err)
    assert len(p.mapped) > 10
    vox = p.voxels()
    assert len(vox["cls"]) > 1000 and set(np.unique(vox["cls"])) <= set(range(1, 9))
    iou, miou, acc = evaluate(p, "gt_labels_terrain")
    print(f"  {len(p.mapped)} scans mapped, {len(vox['cls'])} voxels, mIoU {miou:.2f}, accuracy {acc:.2f}")
    assert acc > 0.5


def test_nvblox_run_and_mesh():
    import osmlocbki
    base = os.environ.get("DATASET_PATH")
    if not osmlocbki.have_nvblox():
        print("  skipped: built without nvblox")
        return
    if not base or not os.path.isdir(os.path.join(base, "datasets/mcd/kth/kth_day_09/inferred_labels/lidarrv_mcd")):
        print("  skipped: needs $DATASET_PATH with MCD KTH and lidarrv_mcd labels")
        return
    p = osmlocbki.Pipeline("kth", "kth_day_09", model="lidarrv_mcd", pose="filtered", max_kf=250, mapper="nvblox")
    while p.step():
        pass
    p.finish()
    assert p.mapper == "nvblox" and len(p.mapped) > 10
    vox = p.voxels()
    assert len(vox["cls"]) > 1000 and set(np.unique(vox["cls"])) <= set(range(1, 9))
    m = p.mesh(full=True)
    v, f, c = m["vertices"], m["triangles"], m["colors"]
    assert len(f) > 10000 and f.max() < len(v) and len(c) == len(v)
    assert np.isin(c[:, 0], list(p.taxonomy()["colors"][:, 0]) + [127]).mean() > 0.9  # class colours (127: grey, none)
    with tempfile.TemporaryDirectory() as d:
        p.save_mesh(os.path.join(d, "mesh.ply"))
        size = os.path.getsize(os.path.join(d, "mesh.ply"))
        try:
            import pyvista  # noqa: F401
        except ImportError:
            print("  3D view skipped: no pyvista")
        else:
            from osmlocbki.view3d import LiveView3D
            view = LiveView3D(p, show=False, voxels=True, mesh=True)
            view.update(final=True)
            view.save(os.path.join(d, "view3d.png"))
            assert os.path.getsize(os.path.join(d, "view3d.png")) > 10000
    print(f"  {len(p.mapped)} scans mapped, {len(vox['cls'])} voxels, mesh {len(f)} triangles ({size / 1e6:.1f} MB ply)")
    assert size > 100000


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
