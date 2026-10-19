#include "osmlocbki/site.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "osmloc/site.hpp"
#include "osmlocbki/geo.hpp"

namespace fs = std::filesystem;

namespace olb {

#ifndef OSMLOCBKI_SOURCE_DIR
#define OSMLOCBKI_SOURCE_DIR "."
#endif

namespace {

std::string env(const char* k, const std::string& def = "") {
  const char* v = std::getenv(k);
  return v && *v ? v : def;
}

template <typename T>
std::vector<T> read_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::vector<T> v(static_cast<size_t>(f.tellg()) / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(T));
  return v;
}

std::string stem(long id) {
  char b[32];
  std::snprintf(b, sizeof b, "%010ld.bin", id);
  return b;
}

// osm_localization's files go to <data>/localization, apart from ours: its map GeoJSON (its own
// converter's, written by python -m osmlocbki.osm convert), OSM databases and scan descriptors; it
// reads the survey models from ours
void share_data_with_osm_localization() {
  if (!env("OSM_LOC_DATA").empty()) return;
  const fs::path d = fs::path(data_dir()) / "localization";
  fs::create_directories(d / "maps");
  fs::create_directories(fs::path(data_dir()) / "surface");
  if (!fs::exists(fs::symlink_status(d / "surface"))) fs::create_directory_symlink(fs::path("..") / "surface", d / "surface");
  setenv("OSM_LOC_DATA", d.c_str(), 1);
}

}  // namespace

std::string repo_root() { return env("OSMLOCBKI_ROOT", OSMLOCBKI_SOURCE_DIR); }
std::string data_dir() { return env("OSMLOCBKI_DATA", repo_root() + "/data"); }
std::string config_dir() { return repo_root() + "/config"; }

// ---- site ----------------------------------------------------------------------------------

std::vector<std::string> Site::names() { return osmloc::Site::names(); }

Site Site::load(const std::string& name) {
  share_data_with_osm_localization();
  Site s;
  s.loc_ = std::make_shared<const osmloc::Site>(osmloc::Site::load(name));
  s.name = name;
  s.map = s.loc_->map;
  YAML::Node all = YAML::LoadFile(config_dir() + "/sites.yaml");
  s.cfg_ = std::make_shared<YAML::Node>(all[name] ? all[name] : YAML::Node(YAML::NodeType::Map));
  return s;
}

std::string Site::map_prefix() const { return data_dir() + "/maps/" + map; }

std::string Site::label_set(const std::string& model) const {
  if ((*cfg_)["label_set"]) return (*cfg_)["label_set"].as<std::string>();
  if (model.empty())
    throw std::runtime_error("site " + name + ": the label set of its labels is not known (config/sites.yaml label_set)");
  YAML::Node set = YAML::LoadFile(config_dir() + "/labels.yaml")["models"][model];
  if (!set) throw std::runtime_error("model " + model + " has no label set in config/labels.yaml (models:)");
  return set.as<std::string>();
}

std::optional<LidarModel> Site::lidar() const {
  const YAML::Node& c = (*cfg_)["lidar"];
  if (!c) return std::nullopt;
  return LidarModel{c["azimuth"].as<int>(), c["elevation"].as<int>(), c["elevation_deg"][0].as<double>(),
                    c["elevation_deg"][1].as<double>()};
}

std::optional<Mat4> Site::lidar_to_body() const {
  const YAML::Node& c = (*cfg_)["lidar_to_body"];
  if (!c) return loc_->lidar_to_body();
  YAML::Node v = YAML::LoadFile(loc_->root() + "/" + c["file"].as<std::string>());
  std::istringstream keys(c["key"].as<std::string>());
  for (std::string k; std::getline(keys, k, '.');) v = v[k];
  Mat4 T;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) T(i, j) = v[i][j].as<double>();
  return T;
}

std::string Site::alpha_dir(const std::string& seq, const std::string& model) const {
  return loc_->seq_dir(seq) + "/inferred_labels/" + model + "/multiclass_alpha";
}

std::optional<std::set<long>> Site::alpha_scans(const std::string& seq, const std::string& model) const {
  if (model.empty()) return std::nullopt;
  std::string d = alpha_dir(seq, model);
  if (!fs::is_directory(d)) return std::nullopt;
  std::set<long> ids;
  for (const auto& e : fs::directory_iterator(d)) {
    std::string f = e.path().filename().string();
    if (f.size() > 4 && f.substr(f.size() - 4) == ".bin") ids.insert(std::stol(f.substr(0, f.size() - 4)));
  }
  return ids;
}

std::vector<float> Site::load_alpha(const std::string& seq, long id, const std::string& model) const {
  return read_bin<float>(alpha_dir(seq, model) + "/" + stem(id));
}

std::vector<float> Site::load_scan(const std::string& seq, long id) const { return loc_->load_scan(seq, id); }

std::vector<int32_t> Site::load_labels(const std::string& seq, long id, long n, const std::string& model) const {
  return loc_->load_labels(seq, id, n, model);
}

// ---- frames --------------------------------------------------------------------------------

Frames::Frames(const Site& site, const std::string& seq, const std::vector<long>& keyframe_ids,
               const std::set<long>& extra_ids, int epsg) {
  osmloc::Poses all = site.loc().read_poses(seq);
  std::map<long, int> kf;
  for (size_t k = 0; k < keyframe_ids.size(); ++k) kf[keyframe_ids[k]] = static_cast<int>(k);
  const auto X = site.lidar_to_body();
  of_keyframe.resize(keyframe_ids.size());
  std::vector<Vec3> at;  // the LiDAR's world x, y, heading (deg)
  int cur = -1;
  for (size_t i = 0; i < all.ids.size(); ++i) {
    const long id = all.ids[i];
    auto k = kf.find(id);
    if (k != kf.end()) cur = k->second;
    if (cur < 0 || (k == kf.end() && !extra_ids.count(id))) continue;
    Eigen::Matrix3d R = all.q[i].normalized().toRotationMatrix();
    Vec3 p = all.p.row(i).transpose();
    if (X) p += R * X->block<3, 1>(0, 3), R = R * X->block<3, 3>(0, 0);  // body -> lidar
    const double psi = degrees(std::atan2(R(1, 0), R(0, 0)));
    of_keyframe[cur].push_back(static_cast<long>(scan_id.size()));
    scan_id.push_back(id);
    anchor.push_back(cur);
    keyframe.push_back(k != kf.end() ? cur : -1);
    level.push_back(rz(-radians(psi)) * R);
    z.push_back(p[2]);
    at.emplace_back(p[0], p[1], psi);
  }
  if (static_cast<long>(std::count_if(keyframe.begin(), keyframe.end(), [](int k) { return k >= 0; })) !=
      static_cast<long>(keyframe_ids.size()))
    throw std::runtime_error(site.name + "/" + seq + ": a localization keyframe is missing from the poses");
  const long n = static_cast<long>(size());
  std::vector<int> first(keyframe_ids.size(), -1);
  for (long f = 0; f < n; ++f)
    if (keyframe[f] >= 0) first[keyframe[f]] = static_cast<int>(f);
  from_anchor.resize(n);
  for (long f = 0; f < n; ++f) {
    const Vec3& a = at[first[anchor[f]]];
    const double c = std::cos(radians(a[2])), s = std::sin(radians(a[2]));
    const double dx = at[f][0] - a[0], dy = at[f][1] - a[1];
    from_anchor[f] = Vec3(c * dx + s * dy, -s * dx + c * dy, wrap180(at[f][2] - a[2]));
  }
  // ground truth through the site's georeference; headings rotated by the grid's angle to the
  // world frame (a fit over the run: the georeference is rigid or near it at this scale)
  MatX2 xy(n, 2);
  for (long f = 0; f < n; ++f) xy.row(f) << at[f][0], at[f][1];
  MatX2 ll = site.loc().latlon(seq, scan_id, xy);
  gt_xy.resize(n, 2);
  for (long f = 0; f < n; ++f) gt_xy.row(f) = to_utm(ll(f, 0), ll(f, 1), epsg).transpose();
  double sc = 0, ss = 0;
  if (n > 1) {
    const Eigen::RowVector2d mw = xy.colwise().mean(), mu = gt_xy.colwise().mean();
    for (long f = 0; f < n; ++f) {
      const Eigen::RowVector2d w = xy.row(f) - mw, u = gt_xy.row(f) - mu;
      sc += w[0] * u[0] + w[1] * u[1], ss += w[0] * u[1] - w[1] * u[0];
    }
  }
  const double grid = n > 1 ? degrees(std::atan2(ss, sc)) : 0.0;
  for (long f = 0; f < n; ++f) gt_psi.push_back(pymod(at[f][2] + grid, 360.0));
}

void Frames::pose(long f, const Vec2& axy, double apsi, Vec2& xy, double& psi) const {
  const Vec3& d = from_anchor[f];
  const double c = std::cos(radians(apsi)), s = std::sin(radians(apsi));
  xy = axy + Vec2(c * d[0] - s * d[1], s * d[0] + c * d[1]);
  psi = apsi + d[2];
}

}  // namespace olb
