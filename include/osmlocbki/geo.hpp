// WGS84 <-> UTM, 2D shapes and a grid index over them, and the OSM map read from the GeoJSON that
// python/osmlocbki/osm.py writes: building footprints (the localizer's descriptor and an OSM-BKI
// layer) and the other OSM-BKI layers (roads, sidewalks, ... as areas, buffered lines or points).
#pragma once

#include <string>
#include <vector>

#include "osmlocbki/common.hpp"

namespace olb {

// UTM zone EPSG of a point: 326zz north, 327zz south
int utm_epsg(double lat, double lon);
// lat/lon (deg) -> UTM easting/northing in zone `epsg` (Krueger series, nm-level)
Vec2 to_utm(double lat, double lon, int epsg);
// UTM -> (lat, lon) deg
Vec2 from_utm(double x, double y, int epsg);

using Ring = std::vector<Vec2>;  // open ring (first vertex not repeated), UTM
using Box = Eigen::Vector4d;     // xmin, ymin, xmax, ymax

struct Polygon {
  Ring outer;
  std::vector<Ring> holes;  // courtyards
};

// even-odd containment, holes respected; on the boundary counts as outside
bool contains(const Polygon& poly, const Vec2& p);
Box bounds(const std::vector<Vec2>& pts, double margin = 0.0);

// The OSM classes of OSM-BKI, in the column order of its confusion matrices
enum OsmClass : int { ROADS, SIDEWALKS, CYCLEWAYS, PARKING, GRASSLANDS, TREES, FOREST, BUILDINGS, FENCES, kNumOsmClasses };
const char* osm_class_name(int c);
int osm_class_of(const std::string& name);  // -1 if unknown

// One feature of an OSM layer: an area, or a line / point that stands for a band of `width` m
struct Shape {
  int cls = 0;
  Polygon area;            // areas: outer ring and holes
  std::vector<Vec2> path;  // lines (2+ vertices) and points (1)
  double width = 0;        // lines / points: the OSM width tag, 0 = untagged (class default)
  bool is_area() const { return path.empty(); }
};

// signed distance to the shape's boundary: negative inside; lines and points are bands of
// half-width `radius` around their centreline
double signed_distance(const Shape& s, const Vec2& p, double radius = 0.0);

// Boxes bucketed on a uniform grid: visit(p, f) calls f(i) for every box i that contains p
class BoxGrid {
 public:
  BoxGrid() = default;
  BoxGrid(std::vector<Box> boxes, double cell);
  template <typename F>
  void visit(const Vec2& p, F&& f) const {
    if (box_.empty()) return;
    long i = static_cast<long>(std::floor((p[0] - lo_[0]) / cell_)), j = static_cast<long>(std::floor((p[1] - lo_[1]) / cell_));
    if (i < 0 || j < 0 || i >= nx_ || j >= ny_) return;
    for (int k : grid_[i * ny_ + j]) {
      const Box& b = box_[k];
      if (p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3]) f(k);
    }
  }

 private:
  std::vector<Box> box_;
  double cell_ = 1;
  Vec2 lo_ = Vec2::Zero();
  long nx_ = 0, ny_ = 0;
  std::vector<std::vector<int>> grid_;
};

// point-in-footprint over many polygons
class Footprints {
 public:
  explicit Footprints(std::vector<Polygon> p, double cell = 50.0);
  bool contains(const Vec2& p) const;

 private:
  std::vector<Polygon> p_;
  BoxGrid grid_;
};

struct OsmMap {
  int epsg = 0;
  std::vector<Polygon> buildings;  // one per outer ring
  // the first ring of each building feature: what the descriptor ray-casts against
  std::vector<Ring> descriptor_rings;
  std::vector<Shape> features;  // the other OSM-BKI layers (<prefix>_semantic.geojson, if present)
};

// <prefix>_buildings.geojson and <prefix>_semantic.geojson; epsg 0 = the zone of the first building
OsmMap load_map(const std::string& prefix, int epsg = 0);

}  // namespace olb
