"""Global LiDAR localization against OpenStreetMap, and OSM-BKI semantic mapping at the poses it
finds: the C++ core (osmlocbki._osmlocbki) and the Python tools around it.

    osmlocbki.osm    download an OSM extract and convert it to the map GeoJSON
    osmlocbki.view   the live view of a run
"""

import os

# idle OpenMP threads sleep instead of spinning while the view draws; read when the extension loads
# OpenMP, so set before importing it
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
# the nvblox backend logs through glog, read when it loads too: warnings and errors only
os.environ.setdefault("GLOG_minloglevel", "1")

# config/ and data/ belong to the checkout this package is in (python/osmlocbki/ -> the repo), not to
# wherever the C++ core happened to be built, so a moved or copied repo uses its own
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.isdir(os.path.join(_root, "config")):
    os.environ.setdefault("OSMLOCBKI_ROOT", _root)
    os.environ.setdefault("OSMLOC_ROOT", os.path.join(_root, "third_party", "osm_localization"))

from ._osmlocbki import (  # noqa: E402
    Pipeline, Site, config_dir, data_dir, have_nvblox, label_mapping, segment_ground, sites)

__all__ = ["Pipeline", "Site", "config_dir", "data_dir", "have_nvblox", "label_mapping", "segment_ground", "sites"]
