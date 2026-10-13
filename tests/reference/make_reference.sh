#!/usr/bin/env bash
# Rebuild tests/data/bki_ref.npz from the ORIGINAL OSM-BKI mapper. Needs a checkout of OSM-BKI
# (default: ../OSM-BKI next to this repository) and PCL headers; ROS is not needed (the mapper's
# one rclcpp include is stubbed out).
#
#   tests/reference/make_reference.sh [path/to/OSM-BKI]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
osmbki="$(cd "${1:-$repo/../OSM-BKI}" && pwd)"
sbki="$osmbki/osm_bki/sbki"
pcl="$(ls -d /usr/include/pcl-1.* 2>/dev/null | tail -1)"
build="$(mktemp -d)"
trap 'rm -rf "$build"' EXIT
mkdir -p "$build/stub/rclcpp" && touch "$build/stub/rclcpp/rclcpp.hpp"
g++ -std=c++17 -O2 -w \
  -I"$build/stub" -I"$sbki/include/osm_bki/mapping" -I"$sbki/include/osm_bki/common" -I"$pcl" \
  -I/usr/include/eigen3 -I"$repo/include" \
  "$here/make_reference.cpp" "$repo/src/npz.cpp" \
  "$sbki/src/mapping/bkioctomap.cpp" "$sbki/src/mapping/bkiblock.cpp" "$sbki/src/mapping/bkioctree.cpp" \
  "$sbki/src/mapping/bkioctree_node.cpp" "$sbki/src/common/point3f.cpp" \
  -lpcl_common -lpcl_filters -lz -o "$build/make_reference"
"$build/make_reference" "$repo/tests/data/bki_ref.npz"
