"""OpenStreetMap -> the map GeoJSON the C++ core reads.

    python -m osmlocbki.osm fetch   --site kth          # Overpass extract of the site's map_bbox
    python -m osmlocbki.osm convert --site kth          # data/maps/<map>.osm -> <map>_{buildings,semantic}.geojson

or with explicit paths: `fetch --bbox LAT_LO LON_LO LAT_HI LON_HI -o x.osm`, `convert x.osm --out-prefix p`.

  <prefix>_buildings.geojson : closed ways tagged `building` / `building:part`, and multipolygon
                               relations tagged so (outer rings assembled from their member ways,
                               each with the inner rings inside it), one MultiPolygon per outer ring.
                               The localizer's descriptor uses each feature's first ring; footprints
                               and the mapper's building layer use them all.
  <prefix>_semantic.geojson  : the other OSM classes of OSM-BKI, property `class` (and `width` in m
                               where known): areas (parking, grasslands, trees, forest) as Polygons
                               with holes, lines (roads, sidewalks, cycleways, fences) and single
                               trees as points. Lines and points stand for bands of their width.

`convert` also writes osm_localization's own map GeoJSON (<map>_{buildings,roads}.geojson, by its
converter, third_party/osm_localization/python/osmloc/osm.py) to data/localization/maps/, which the
localization reads. Its database is built on first use and cached as
data/localization/db_<map>_free2_n360_r50.npz (delete it after changing a map).
"""

import argparse
import json
import os
import re
import urllib.parse
import urllib.request

from lxml import etree

OVERPASS = "https://overpass-api.de/api/interpreter"

# OSM-BKI's classes of highway: roads (with OSM2World's default widths for the major ones),
# separately mapped paths, cycleways
ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street",
         "road", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}
ROAD_WIDTH = {"motorway": 8.75, "motorway_link": 8.75, "trunk": 7.0, "trunk_link": 7.0, "primary": 7.0,
              "primary_link": 7.0, "secondary": 7.0, "secondary_link": 7.0}
PATHS = {"footway", "path", "pedestrian", "foot"}


def fetch(bbox, out_path, endpoint=OVERPASS):
    """Download every node/way/relation touching bbox = (lat_lo, lon_lo, lat_hi, lon_hi).

    Overpass rejects requests without a User-Agent (HTTP 406), so one is always sent.
    """
    la0, lo0, la1, lo1 = bbox
    query = f"[out:xml][timeout:180];(nwr({la0},{lo0},{la1},{lo1}););(._;>;);out meta;"
    req = urllib.request.Request(
        endpoint, data=urllib.parse.urlencode({"data": query}).encode(),
        headers={"User-Agent": "osmlocbki (python urllib)", "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=300) as r, open(out_path, "wb") as f:
        body = r.read()
        f.write(body)
    return len(body)


def _is_building(tags):
    return "building" in tags or "building:part" in tags


def _width(tags):
    """The `width` tag in metres, or None."""
    m = re.match(r"\s*([0-9]*\.?[0-9]+)", tags.get("width", ""))
    return float(m.group(1)) if m else None


def area_class(tags):
    """OSM-BKI's class of an area, or None."""
    am, lu, nat, lei = (tags.get(k) for k in ("amenity", "landuse", "natural", "leisure"))
    if am in ("parking", "parking_space"):
        return "parking"
    if lu in ("grass", "meadow", "greenfield", "recreation_ground") or nat in ("grassland", "heath", "scrub") \
            or lei in ("park", "garden"):
        return "grasslands"
    if lu in ("orchard", "vineyard") or tags.get("landcover") == "trees":
        return "trees"
    if lu == "forest" or nat in ("wood", "forest"):
        return "forest"
    return None


def line_classes(tags):
    """OSM-BKI's classes of a way drawn as a line -> [(class, width or None)]."""
    hw, w = tags.get("highway"), _width(tags)
    if tags.get("barrier") == "fence":
        return [("fences", w)]
    if hw == "cycleway" or (hw == "path" and tags.get("bicycle") == "yes"):
        return [("cycleways", w)]
    if hw in PATHS:
        return [("sidewalks", w)]
    if hw in ROADS:
        out = [("roads", w or ROAD_WIDTH.get(hw))]
        # an implied sidewalk: the road's width tag is not the sidewalk's
        if tags.get("sidewalk") in ("both", "left", "right", "yes"):
            out.append(("sidewalks", None))
        return out
    return []


def _assemble(segments):
    """Member ways (lists of node ids) -> closed rings (first id == last id); unclosed chains are dropped."""
    segs = [list(s) for s in segments if len(s) >= 2]
    rings = []
    while segs:
        ring = segs.pop(0)
        while ring[0] != ring[-1]:
            for i, s in enumerate(segs):
                if s[0] == ring[-1]:
                    ring += s[1:]
                elif s[-1] == ring[-1]:
                    ring += s[-2::-1]
                else:
                    continue
                segs.pop(i)
                break
            else:
                break   # cannot be closed
        if ring[0] == ring[-1] and len(ring) >= 4:
            rings.append(ring)
    return rings


def _inside(pt, ring):
    """Even-odd test of a lon/lat point against a ring of lon/lat points."""
    x, y = pt
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def parse(osm_path):
    """.osm -> (buildings, areas, lines, points), all lon/lat:
    buildings and areas as (outer ring, [inner rings], tags or class), lines as (points, class, width),
    points as (point, class)."""
    nodes, way_nodes, area_ways, relations = {}, {}, set(), []
    buildings, areas, lines, points = [], [], [], []
    for _, el in etree.iterparse(str(osm_path), events=("end",), tag=("node", "way", "relation")):
        tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
        if el.tag == "node":
            nodes[el.get("id")] = (float(el.get("lon")), float(el.get("lat")))
            if tags.get("natural") == "tree":
                points.append((nodes[el.get("id")], "trees"))
        elif el.tag == "way":
            ids = [nd.get("ref") for nd in el.findall("nd")]
            way_nodes[el.get("id")] = ids
            pts = [nodes[i] for i in ids if i in nodes]
            if len(pts) < 2:
                pass
            elif _is_building(tags):
                ring = pts[:-1] if pts[0] == pts[-1] and len(pts) > 3 else pts
                if len(ring) >= 3:
                    buildings.append((ring, [], tags))
                    area_ways.add(el.get("id"))
            else:
                # parking areas first, then lines (a highway wins over the area tags), then areas
                ac, lc = area_class(tags), line_classes(tags)
                if (ac == "parking" or (ac and not lc)) and len(pts) >= 3:
                    areas.append((pts[:-1] if pts[0] == pts[-1] else pts, [], ac))
                    area_ways.add(el.get("id"))
                elif lc:
                    lines.extend((pts, c, w) for c, w in lc)
        else:
            if tags.get("type") == "multipolygon" and (_is_building(tags) or area_class(tags)):
                members = [(m.get("ref"), m.get("role") or "outer") for m in el.findall("member") if m.get("type") == "way"]
                relations.append((members, tags))
        el.clear()
    for members, tags in relations:
        rings = {}
        for role in ("outer", "inner"):
            ways = [w for w, r in members if r == role and w in way_nodes]
            # an outer that is itself a tagged area is already in
            if role == "outer":
                ways = [w for w in ways if w not in area_ways]
            rings[role] = [[nodes[i] for i in ring[:-1] if i in nodes]
                           for ring in _assemble([way_nodes[w] for w in ways])]
        for outer in rings["outer"]:
            if len(outer) < 3:
                continue
            inners = [r for r in rings["inner"] if len(r) >= 3 and _inside(r[0], outer)]
            if _is_building(tags):
                buildings.append((outer, inners, tags))
            else:
                areas.append((outer, inners, area_class(tags)))
    return buildings, areas, lines, points


def _closed(ring):
    ring = [list(c) for c in ring]
    return ring if ring[0] == ring[-1] else ring + [ring[0]]


def _buildings_geojson(buildings):
    """Open rings, as the descriptor expects (the database caches were built from these)."""
    feats = [{"type": "Feature",
              "properties": {k: tags.get(k) for k in ("name", "building")},
              "geometry": {"type": "MultiPolygon", "coordinates": [[[list(c) for c in r] for r in [outer] + inners]]}}
             for outer, inners, tags in buildings]
    return {"type": "FeatureCollection", "features": feats}


def _semantic_geojson(areas, lines, points):
    feats = []
    for outer, inners, cls in areas:
        geom = {"type": "Polygon", "coordinates": [_closed(r) for r in [outer] + inners]}
        feats.append({"type": "Feature", "properties": {"class": cls}, "geometry": geom})
    for pts, cls, width in lines:
        props = {"class": cls} if width is None else {"class": cls, "width": width}
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "LineString", "coordinates": [list(c) for c in pts]}})
    for pt, cls in points:
        feats.append({"type": "Feature", "properties": {"class": cls},
                      "geometry": {"type": "Point", "coordinates": list(pt)}})
    return {"type": "FeatureCollection", "features": feats}


def to_geojson(osm_path, out_prefix):
    """Write <out_prefix>_buildings.geojson and <out_prefix>_semantic.geojson -> feature counts."""
    buildings, areas, lines, points = parse(osm_path)
    with open(f"{out_prefix}_buildings.geojson", "w") as f:
        json.dump(_buildings_geojson(buildings), f)
    semantic = _semantic_geojson(areas, lines, points)
    with open(f"{out_prefix}_semantic.geojson", "w") as f:
        json.dump(semantic, f)
    counts = {"buildings": len(buildings)}
    for feat in semantic["features"]:
        counts[feat["properties"]["class"]] = counts.get(feat["properties"]["class"], 0) + 1
    return counts


def _osm_localization():
    """osm_localization's converter module, loaded from its file (its package needs its own C++ module)."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "osm_localization",
                        "python", "osmloc", "osm.py")
    spec = importlib.util.spec_from_file_location("osm_localization_osm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(prog="python -m osmlocbki.osm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download an OSM extract (Overpass)")
    f.add_argument("--site", help="use the site's map_bbox and write data/maps/<map>.osm")
    f.add_argument("--bbox", type=float, nargs=4, metavar=("LAT_LO", "LON_LO", "LAT_HI", "LON_HI"))
    f.add_argument("-o", "--out")
    f.add_argument("--endpoint", default=OVERPASS)
    c = sub.add_parser("convert", help=".osm -> buildings / semantic GeoJSON")
    c.add_argument("osm", nargs="?")
    c.add_argument("--site", help="convert data/maps/<map>.osm in place")
    c.add_argument("--out-prefix")
    a = ap.parse_args()

    if a.site:
        from . import Site, data_dir
        site = Site(a.site)
        prefix = os.path.join(data_dir(), "maps", site.map)
    if a.cmd == "fetch":
        bbox = a.bbox or (site.map_bbox if a.site else None)
        out = a.out or (prefix + ".osm" if a.site else None)
        if not bbox or not out:
            ap.error("fetch needs --site, or --bbox and -o")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        print(f"wrote {out} ({fetch(bbox, out, a.endpoint) / 1e6:.1f} MB)")
    else:
        src = a.osm or (prefix + ".osm" if a.site else None)
        out = a.out_prefix or (prefix if a.site else None)
        if not src or not out:
            ap.error("convert needs --site, or an .osm file and --out-prefix")
        counts = to_geojson(src, out)
        print(", ".join(f"{n} {k}" for k, n in counts.items()) + f" -> {out}_{{buildings,semantic}}.geojson")
        if a.site or not a.out_prefix:
            from . import data_dir
            loc = os.path.join(data_dir(), "localization", "maps", os.path.basename(out))
            os.makedirs(os.path.dirname(loc), exist_ok=True)
            nb, nr = _osm_localization().to_geojson(src, loc)
            print(f"{nb} buildings, {nr} highway ways -> {loc}_{{buildings,roads}}.geojson (osm_localization)")


if __name__ == "__main__":
    main()
