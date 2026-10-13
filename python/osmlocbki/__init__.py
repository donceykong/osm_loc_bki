"""Global LiDAR localization against OpenStreetMap, and OSM-BKI semantic mapping at the poses it
finds: the C++ core (osmlocbki._osmlocbki) and the Python tools around it.

    osmlocbki.osm    download an OSM extract and convert it to the map GeoJSON
    osmlocbki.view   the live view of a run
"""

import os

# idle OpenMP threads sleep instead of spinning while the view draws; read when the extension loads
# OpenMP, so set before importing it
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

from ._osmlocbki import Pipeline, Site, config_dir, data_dir, label_mapping, sites  # noqa: E402

__all__ = ["Pipeline", "Site", "config_dir", "data_dir", "label_mapping", "sites"]
