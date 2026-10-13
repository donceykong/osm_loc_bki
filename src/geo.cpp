#include "osmlocbki/geo.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace olb {

// ---- UTM: Krueger series to 6th order (Karney 2011), WGS84 -------------------------------

namespace {

struct Tm {
  double a = 6378137.0, f = 1 / 298.257223563, k0 = 0.9996;
  double n, e, A, alpha[7], beta[7];
  Tm() {
    n = f / (2 - f);
    e = std::sqrt(f * (2 - f));
    double n2 = n * n, n3 = n2 * n, n4 = n3 * n, n5 = n4 * n, n6 = n5 * n;
    A = a / (1 + n) * (1 + n2 / 4 + n4 / 64 + n6 / 256);
    alpha[1] = n / 2 - 2 * n2 / 3 + 5 * n3 / 16 + 41 * n4 / 180 - 127 * n5 / 288 + 7891 * n6 / 37800;
    alpha[2] = 13 * n2 / 48 - 3 * n3 / 5 + 557 * n4 / 1440 + 281 * n5 / 630 - 1983433 * n6 / 1935360;
    alpha[3] = 61 * n3 / 240 - 103 * n4 / 140 + 15061 * n5 / 26880 + 167603 * n6 / 181440;
    alpha[4] = 49561 * n4 / 161280 - 179 * n5 / 168 + 6601661 * n6 / 7257600;
    alpha[5] = 34729 * n5 / 80640 - 3418889 * n6 / 1995840;
    alpha[6] = 212378941 * n6 / 319334400;
    beta[1] = n / 2 - 2 * n2 / 3 + 37 * n3 / 96 - n4 / 360 - 81 * n5 / 512 + 96199 * n6 / 604800;
    beta[2] = n2 / 48 + n3 / 15 - 437 * n4 / 1440 + 46 * n5 / 105 - 1118711 * n6 / 3870720;
    beta[3] = 17 * n3 / 480 - 37 * n4 / 840 - 209 * n5 / 4480 + 5569 * n6 / 90720;
    beta[4] = 4397 * n4 / 161280 - 11 * n5 / 504 - 830251 * n6 / 7257600;
    beta[5] = 4583 * n5 / 161280 - 108847 * n6 / 3991680;
    beta[6] = 20648693 * n6 / 638668800;
  }
};

const Tm& tm() {
  static const Tm t;
  return t;
}

void zone_of(int epsg, double& lon0, double& fn) {
  int zone = epsg % 100;
  bool north = epsg / 100 == 326, south = epsg / 100 == 327;
  if (!(north || south) || zone < 1 || zone > 60) throw std::invalid_argument("not a UTM EPSG: " + std::to_string(epsg));
  lon0 = radians(zone * 6.0 - 183.0);
  fn = south ? 1e7 : 0.0;
}

}  // namespace

int utm_epsg(double lat, double lon) {
  int zone = static_cast<int>((lon + 180.0) / 6.0) + 1;
  return (lat >= 0 ? 32600 : 32700) + zone;
}

Vec2 to_utm(double lat, double lon, int epsg) {
  const Tm& T = tm();
  double lon0, fn;
  zone_of(epsg, lon0, fn);
  double phi = radians(lat), lam = radians(lon) - lon0;
  double s = std::sin(phi);
  double t = std::sinh(std::atanh(s) - T.e * std::atanh(T.e * s));
  double xi_ = std::atan2(t, std::cos(lam)), eta_ = std::atanh(std::sin(lam) / std::sqrt(1 + t * t));
  double xi = xi_, eta = eta_;
  for (int j = 1; j <= 6; ++j) {
    xi += T.alpha[j] * std::sin(2 * j * xi_) * std::cosh(2 * j * eta_);
    eta += T.alpha[j] * std::cos(2 * j * xi_) * std::sinh(2 * j * eta_);
  }
  return Vec2(500000.0 + T.k0 * T.A * eta, fn + T.k0 * T.A * xi);
}

Vec2 from_utm(double x, double y, int epsg) {
  const Tm& T = tm();
  double lon0, fn;
  zone_of(epsg, lon0, fn);
  double xi = (y - fn) / (T.k0 * T.A), eta = (x - 500000.0) / (T.k0 * T.A);
  double xi_ = xi, eta_ = eta;
  for (int j = 1; j <= 6; ++j) {
    xi_ -= T.beta[j] * std::sin(2 * j * xi) * std::cosh(2 * j * eta);
    eta_ -= T.beta[j] * std::cos(2 * j * xi) * std::sinh(2 * j * eta);
  }
  double tau_ = std::sin(xi_) / std::sqrt(std::sinh(eta_) * std::sinh(eta_) + std::cos(xi_) * std::cos(xi_));
  double lam = std::atan2(std::sinh(eta_), std::cos(xi_));
  // conformal -> geodetic latitude (Newton on tau = tan(phi))
  const double e2 = T.e * T.e;
  double tau = tau_;
  for (int it = 0; it < 10; ++it) {
    double sig = std::sinh(T.e * std::atanh(T.e * tau / std::sqrt(1 + tau * tau)));
    double ti = tau * std::sqrt(1 + sig * sig) - sig * std::sqrt(1 + tau * tau);
    double d = (tau_ - ti) / std::sqrt(1 + ti * ti) * (1 + (1 - e2) * tau * tau) / ((1 - e2) * std::sqrt(1 + tau * tau));
    tau += d;
    if (std::abs(d) < 1e-14) break;
  }
  return Vec2(degrees(std::atan(tau)), degrees(lam + lon0));
}


// ---- shapes ---------------------------------------------------------------------------------

namespace {

bool in_ring(const Ring& r, const Vec2& p) {
  bool in = false;
  for (size_t i = 0, j = r.size() - 1; i < r.size(); j = i++) {
    const Vec2 &a = r[i], &b = r[j];
    if ((a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]) in = !in;
  }
  return in;
}

double segment_distance2(const Vec2& p, const Vec2& a, const Vec2& b) {
  Vec2 d = b - a;
  double l2 = d.squaredNorm();
  double t = l2 < 1e-24 ? 0.0 : std::clamp((p - a).dot(d) / l2, 0.0, 1.0);
  return (p - a - t * d).squaredNorm();
}

// squared distance to the nearest edge of a ring (closed) or a path (open)
double edges_distance2(const std::vector<Vec2>& v, const Vec2& p, bool closed) {
  if (v.size() == 1) return (p - v[0]).squaredNorm();
  double best = std::numeric_limits<double>::infinity();
  for (size_t i = closed ? 0 : 1; i < v.size(); ++i)
    best = std::min(best, segment_distance2(p, v[i == 0 ? v.size() - 1 : i - 1], v[i]));
  return best;
}

const char* kOsmClasses[kNumOsmClasses] = {"roads",      "sidewalks", "cycleways", "parking", "grasslands",
                                           "trees",      "forest",    "buildings", "fences"};

}  // namespace

const char* osm_class_name(int c) { return kOsmClasses[c]; }

int osm_class_of(const std::string& name) {
  for (int c = 0; c < kNumOsmClasses; ++c)
    if (name == kOsmClasses[c]) return c;
  return -1;
}

bool contains(const Polygon& poly, const Vec2& p) {
  if (!in_ring(poly.outer, p)) return false;
  for (const auto& h : poly.holes)
    if (in_ring(h, p)) return false;
  return true;
}

Box bounds(const std::vector<Vec2>& pts, double margin) {
  Box b(1e300, 1e300, -1e300, -1e300);
  for (const auto& p : pts) b = Box(std::min(b[0], p[0]), std::min(b[1], p[1]), std::max(b[2], p[0]), std::max(b[3], p[1]));
  return b + Box(-margin, -margin, margin, margin);
}

double signed_distance(const Shape& s, const Vec2& p, double radius) {
  if (!s.is_area()) return std::sqrt(edges_distance2(s.path, p, false)) - radius;
  double d2 = edges_distance2(s.area.outer, p, true);
  for (const auto& h : s.area.holes) d2 = std::min(d2, edges_distance2(h, p, true));
  return contains(s.area, p) ? -std::sqrt(d2) : std::sqrt(d2);
}

BoxGrid::BoxGrid(std::vector<Box> boxes, double cell) : box_(std::move(boxes)), cell_(cell) {
  if (box_.empty()) return;
  Vec2 lo = Vec2::Constant(1e300), hi = -lo;
  for (const auto& b : box_) lo = lo.cwiseMin(b.head<2>()), hi = hi.cwiseMax(b.tail<2>());
  lo_ = lo;
  nx_ = static_cast<long>((hi[0] - lo[0]) / cell) + 1, ny_ = static_cast<long>((hi[1] - lo[1]) / cell) + 1;
  grid_.resize(nx_ * ny_);
  for (size_t k = 0; k < box_.size(); ++k) {
    long i0 = static_cast<long>((box_[k][0] - lo[0]) / cell), i1 = static_cast<long>((box_[k][2] - lo[0]) / cell);
    long j0 = static_cast<long>((box_[k][1] - lo[1]) / cell), j1 = static_cast<long>((box_[k][3] - lo[1]) / cell);
    for (long i = i0; i <= i1; ++i)
      for (long j = j0; j <= j1; ++j) grid_[i * ny_ + j].push_back(static_cast<int>(k));
  }
}

Footprints::Footprints(std::vector<Polygon> p, double cell) : p_(std::move(p)) {
  std::vector<Box> boxes;
  for (const auto& x : p_) boxes.push_back(bounds(x.outer));
  grid_ = BoxGrid(std::move(boxes), cell);
}

bool Footprints::contains(const Vec2& p) const {
  bool in = false;
  grid_.visit(p, [&](int k) { in = in || olb::contains(p_[k], p); });
  return in;
}

// ---- GeoJSON ----------------------------------------------------------------------------------

namespace {

nlohmann::json read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error(path + " is missing: make it with python -m osmlocbki.osm convert");
  return nlohmann::json::parse(f);
}

std::vector<Vec2> lonlat(const nlohmann::json& pts) {
  std::vector<Vec2> out;
  for (const auto& c : pts) out.emplace_back(c[0].get<double>(), c[1].get<double>());
  return out;
}

// the parts of a Polygon / MultiPolygon, each a list of rings of (lon, lat)
std::vector<std::vector<std::vector<Vec2>>> polygon_parts(const nlohmann::json& g) {
  std::vector<std::vector<std::vector<Vec2>>> parts;
  auto part = [&](const nlohmann::json& p) {
    std::vector<std::vector<Vec2>> rings;
    for (const auto& r : p) rings.push_back(lonlat(r));
    return rings;
  };
  if (g["type"] == "MultiPolygon")
    for (const auto& p : g["coordinates"]) parts.push_back(part(p));
  else if (g["type"] == "Polygon")
    parts.push_back(part(g["coordinates"]));
  return parts;
}

}  // namespace

OsmMap load_map(const std::string& prefix, int epsg) {
  OsmMap m;
  auto buildings = read_json(prefix + "_buildings.geojson");
  if (!epsg) {
    for (const auto& f : buildings["features"]) {
      auto parts = polygon_parts(f["geometry"]);
      if (parts.empty() || parts[0].empty() || parts[0][0].empty()) continue;
      epsg = utm_epsg(parts[0][0][0][1], parts[0][0][0][0]);
      break;
    }
    if (!epsg) throw std::runtime_error(prefix + ": no buildings to choose a UTM zone from");
  }
  m.epsg = epsg;
  auto utm = [&](const std::vector<Vec2>& ll) {
    std::vector<Vec2> out;
    out.reserve(ll.size());
    for (const auto& p : ll) out.push_back(to_utm(p[1], p[0], epsg));
    return out;
  };
  // GeoJSON rings repeat their first vertex; ours are open
  auto ring = [&](const std::vector<Vec2>& ll) {
    Ring r = utm(ll);
    if (r.size() > 3 && r.front() == r.back()) r.pop_back();
    return r;
  };
  auto polygons = [&](const nlohmann::json& g) {
    std::vector<Polygon> out;
    for (const auto& part : polygon_parts(g)) {
      if (part.empty() || part[0].size() < 3) continue;
      Polygon p{ring(part[0]), {}};
      for (size_t k = 1; k < part.size(); ++k) p.holes.push_back(ring(part[k]));
      out.push_back(std::move(p));
    }
    return out;
  };
  for (const auto& f : buildings["features"]) {
    auto parts = polygon_parts(f["geometry"]);
    if (parts.empty() || parts[0].empty()) continue;
    m.descriptor_rings.push_back(utm(parts[0][0]));
    for (auto& p : polygons(f["geometry"])) m.buildings.push_back(std::move(p));
  }
  if (!std::ifstream(prefix + "_semantic.geojson")) return m;
  auto semantic = read_json(prefix + "_semantic.geojson");
  for (const auto& f : semantic["features"]) {
    const auto& props = f["properties"];
    int cls = osm_class_of(props.value("class", ""));
    if (cls < 0) continue;
    const auto& g = f["geometry"];
    Shape s;
    s.cls = cls;
    if (props.contains("width") && props["width"].is_number()) s.width = props["width"].get<double>();
    if (g["type"] == "LineString") {
      s.path = utm(lonlat(g["coordinates"]));
      if (s.path.size() > 1) m.features.push_back(std::move(s));
    } else if (g["type"] == "Point") {
      s.path = utm(lonlat(nlohmann::json::array({g["coordinates"]})));
      m.features.push_back(std::move(s));
    } else {
      for (auto& p : polygons(g)) {
        Shape a = s;
        a.area = std::move(p);
        m.features.push_back(std::move(a));
      }
    }
  }
  return m;
}

}  // namespace olb
