#include "osmlocbki/site.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <stdexcept>

#include "osmlocbki/frontend.hpp"
#include "osmlocbki/geo.hpp"
#include "osmlocbki/npz.hpp"

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

std::string expand(std::string p) {
  if (!p.empty() && p[0] == '~') p = env("HOME") + p.substr(1);
  for (size_t i; (i = p.find("${")) != std::string::npos;) {
    size_t j = p.find('}', i);
    p = p.substr(0, i) + env(p.substr(i + 2, j - i - 2).c_str()) + p.substr(j + 1);
  }
  return p;
}

std::string replace_seq(std::string s, const std::string& seq) {
  for (size_t i; (i = s.find("{seq}")) != std::string::npos;) s.replace(i, 5, seq);
  return s;
}

// Python's "{:g}" for the values used in cache names
std::string fmt_g(double v) {
  std::ostringstream o;
  o << v;
  return o.str();
}

std::vector<std::string> split(const std::string& line, char sep) {
  std::vector<std::string> out;
  std::string cur;
  std::istringstream is(line);
  while (std::getline(is, cur, sep)) {
    size_t a = cur.find_first_not_of(" \t\r"), b = cur.find_last_not_of(" \t\r");
    out.push_back(a == std::string::npos ? "" : cur.substr(a, b - a + 1));
  }
  return out;
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

constexpr double kEarth = 6378137.0;

}  // namespace

std::string repo_root() { return env("OSMLOCBKI_ROOT", OSMLOCBKI_SOURCE_DIR); }
std::string data_dir() { return env("OSMLOCBKI_DATA", repo_root() + "/data"); }
std::string config_dir() { return repo_root() + "/config"; }

// ---- site ----------------------------------------------------------------------------------

std::vector<std::string> Site::names() {
  YAML::Node all = YAML::LoadFile(config_dir() + "/sites.yaml");
  std::vector<std::string> out;
  for (const auto& kv : all) out.push_back(kv.first.as<std::string>());
  std::sort(out.begin(), out.end());
  return out;
}

Site Site::load(const std::string& name) {
  YAML::Node all = YAML::LoadFile(config_dir() + "/sites.yaml");
  if (!all[name]) {
    std::string known;
    for (const auto& n : names()) known += (known.empty() ? "" : ", ") + n;
    throw std::invalid_argument("unknown site '" + name + "'; config/sites.yaml has " + known);
  }
  Site s;
  s.cfg_ = std::make_shared<YAML::Node>(all[name]);
  const YAML::Node& c = *s.cfg_;
  s.name = name;
  s.map = c["map"].as<std::string>();
  if (c["map_bbox"]) {
    auto b = c["map_bbox"];
    s.map_bbox = std::array<double, 4>{b[0][0].as<double>(), b[0][1].as<double>(), b[1][0].as<double>(), b[1][1].as<double>()};
  }
  return s;
}

std::string Site::root() const {
  std::string rel = expand((*cfg_)["data_dir"].as<std::string>());
  if (fs::path(rel).is_absolute()) return rel;
  std::string base = env("DATASET_PATH");
  if (base.empty())
    throw std::runtime_error("DATASET_PATH is not set, so site '" + name + "' (" + rel +
                             ") cannot be found; set it to the disk holding the datasets");
  return expand(base) + "/" + rel;
}

std::string Site::seq_dir(const std::string& seq) const { return root() + "/" + seq; }
std::string Site::map_prefix() const { return data_dir() + "/maps/" + map; }
std::optional<LidarModel> Site::lidar() const {
  const YAML::Node& c = (*cfg_)["lidar"];
  if (!c) return std::nullopt;
  return LidarModel{c["azimuth"].as<int>(), c["elevation"].as<int>(), c["elevation_deg"][0].as<double>(),
                    c["elevation_deg"][1].as<double>()};
}

std::optional<Mat4> Site::lidar_to_body() const {
  const YAML::Node& c = (*cfg_)["lidar_to_body"];
  if (!c) return std::nullopt;
  YAML::Node v = YAML::LoadFile(root() + "/" + c["file"].as<std::string>());
  std::istringstream keys(c["key"].as<std::string>());
  for (std::string k; std::getline(keys, k, '.');) v = v[k];
  Mat4 T;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) T(i, j) = v[i][j].as<double>();
  return T;
}

Poses Site::read_poses(const std::string& seq) const {
  const YAML::Node& c = (*cfg_)["poses"];
  std::string path = seq_dir(seq) + "/" + replace_seq(c["file"].as<std::string>(), seq);
  std::string fmt = c["format"].as<std::string>();
  Poses P;
  std::vector<Vec3> p;
  if (fmt == "kitti") {  // KITTI-360: scan id, then the 4x4 lidar -> world matrix row-major, space separated
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    for (std::string line; std::getline(f, line);) {
      std::istringstream s(line);
      double id, m[16];
      if (!(s >> id)) continue;
      for (double& v : m) s >> v;
      Eigen::Matrix3d R;
      R << m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10];
      P.ids.push_back(static_cast<long>(id));
      p.emplace_back(m[3], m[7], m[11]);
      P.q.emplace_back(Eigen::Quaterniond(R).normalized());
    }
  } else {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<std::string> head;
    std::map<std::string, int> col;
    if (fmt == "utm_csv") {
      head = {"timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"};
    } else {
      std::string line;
      std::getline(f, line);
      head = split(line, ',');
    }
    for (size_t i = 0; i < head.size(); ++i) col[head[i] == "w" ? "qw" : head[i]] = static_cast<int>(i);
    std::string id_col = fmt == "csv" ? c["id_col"].as<std::string>() : "";
    long row = 0;
    for (std::string line; std::getline(f, line);) {
      if (line.empty() || line[0] == '#') continue;
      auto v = split(line, ',');
      auto num = [&](const std::string& k) { return std::stod(v.at(col.at(k))); };
      P.ids.push_back(fmt == "csv" ? static_cast<long>(num(id_col)) : row);
      p.emplace_back(num("x"), num("y"), num("z"));
      P.q.emplace_back(num("qw"), num("qx"), num("qy"), num("qz"));
      ++row;
    }
  }
  P.p.resize(p.size(), 3);
  for (size_t i = 0; i < p.size(); ++i) P.p.row(i) = p[i].transpose();
  return P;
}

MatX2 Site::latlon(const MatX2& xy) const {
  const YAML::Node& g = (*cfg_)["georef"];
  MatX2 out(xy.rows(), 2);
  if (g["utm_fit"]) {
    std::ifstream f(config_dir() + "/georef/" + g["utm_fit"].as<std::string>());
    auto fit = nlohmann::json::parse(f);
    double a = radians(fit["yaw_deg"].get<double>()), c = std::cos(a), s = std::sin(a);
    int epsg = fit["epsg"].get<int>();
    for (long i = 0; i < xy.rows(); ++i) {
      double ux = c * xy(i, 0) - s * xy(i, 1) + fit["tx"].get<double>();
      double uy = s * xy(i, 0) + c * xy(i, 1) + fit["ty"].get<double>();
      out.row(i) = from_utm(ux, uy, epsg).transpose();
    }
    return out;
  }
  if (g["mercator"]) {
    double lat0 = g["mercator"]["origin"][0].as<double>(), lon0 = g["mercator"]["origin"][1].as<double>();
    double rx = g["mercator"]["rel_pos"][0].as<double>(), ry = g["mercator"]["rel_pos"][1].as<double>();
    double scale = std::cos(radians(lat0));
    double ox = scale * radians(lon0) * kEarth, oy = scale * kEarth * std::log(std::tan(kPi / 4 + radians(lat0) / 2));
    for (long i = 0; i < xy.rows(); ++i) {
      double x = xy(i, 0) - rx, y = xy(i, 1) - ry;
      out(i, 1) = degrees((x + ox) / (scale * kEarth));
      out(i, 0) = degrees(2 * std::atan(std::exp((y + oy) / (scale * kEarth))) - kPi / 2);
    }
    return out;
  }
  throw std::runtime_error("site " + name + ": unknown georef");
}

std::string Site::labels_tag(const std::string& model) const {
  const YAML::Node& lab = (*cfg_)["labels"];
  if (lab["flat"]) return lab["flat"].as<std::string>() == "gt_labels" ? "gt_labels" : "dataset_labels";
  if (model.empty()) throw std::invalid_argument("site " + name + " uses inferred labels: pass a label model");
  return model;
}

std::string Site::label_set(const std::string& model) const {
  const YAML::Node& lab = (*cfg_)["labels"];
  if (lab["flat"]) {
    if (!lab["label_set"]) throw std::runtime_error("site " + name + ": its labels have no label_set in sites.yaml");
    return lab["label_set"].as<std::string>();
  }
  YAML::Node set = YAML::LoadFile(config_dir() + "/labels.yaml")["models"][labels_tag(model)];
  if (!set) throw std::runtime_error("model " + model + " has no label set in config/labels.yaml (models:)");
  return set.as<std::string>();
}

// <seq>/inferred_labels/<model>/<kind>, or under data/<root>/ when the site moves them there
std::string Site::inferred_dir(const std::string& seq, const std::string& model, const std::string& kind) const {
  const YAML::Node& lab = (*cfg_)["labels"];
  std::string base = lab["root"] ? data_dir() + "/" + lab["root"].as<std::string>() + "/" + seq : seq_dir(seq);
  return base + "/inferred_labels/" + model + "/" + kind;
}

namespace {

std::set<long> bin_ids(const std::string& dir) {
  std::set<long> ids;
  for (const auto& e : fs::directory_iterator(dir)) {
    std::string f = e.path().filename().string();
    if (f.size() > 4 && f.substr(f.size() - 4) == ".bin") ids.insert(std::stol(f.substr(0, f.size() - 4)));
  }
  return ids;
}

}  // namespace

std::optional<std::set<long>> Site::labelled_scans(const std::string& seq, const std::string& model) const {
  const YAML::Node& lab = (*cfg_)["labels"];
  if (lab["flat"]) return std::nullopt;
  std::string d = inferred_dir(seq, model, lab["model"].as<std::string>() == "int" ? "predicted_labels" : "multiclass_alpha");
  if (!fs::is_directory(d)) throw std::runtime_error("no labels for " + name + "/" + seq + " from model " + model + ": " + d);
  return bin_ids(d);
}

std::optional<std::set<long>> Site::alpha_scans(const std::string& seq, const std::string& model) const {
  if ((*cfg_)["labels"]["flat"]) return std::nullopt;
  std::string d = inferred_dir(seq, model, "multiclass_alpha");
  if (!fs::is_directory(d)) return std::nullopt;
  return bin_ids(d);
}

std::vector<float> Site::load_scan(const std::string& seq, long id) const {
  return read_bin<float>(seq_dir(seq) + "/" + (*cfg_)["scans"].as<std::string>() + "/" + stem(id));
}

std::vector<float> Site::load_alpha(const std::string& seq, long id, const std::string& model) const {
  return read_bin<float>(inferred_dir(seq, model, "multiclass_alpha") + "/" + stem(id));
}

std::vector<int32_t> Site::load_labels(const std::string& seq, long id, long n, const std::string& model) const {
  const YAML::Node& lab = (*cfg_)["labels"];
  if (lab["flat"]) return read_bin<int32_t>(seq_dir(seq) + "/" + replace_seq(lab["flat"].as<std::string>(), seq) + "/" + stem(id));
  if (lab["model"].as<std::string>() == "int") return read_bin<int32_t>(inferred_dir(seq, model, "predicted_labels") + "/" + stem(id));
  auto alpha = load_alpha(seq, id, model);
  long k = static_cast<long>(alpha.size()) / n;
  std::vector<int32_t> out(n);
  for (long i = 0; i < n; ++i)
    out[i] = static_cast<int32_t>(std::max_element(alpha.begin() + i * k, alpha.begin() + (i + 1) * k) - (alpha.begin() + i * k));
  return out;
}

// ---- sequence ------------------------------------------------------------------------------

Sequence::Sequence(const Site& s, const std::string& n, const std::string& m, double sp, const std::set<long>& include)
    : site(s), name(n), model(m), labels(s.labels_tag(m)), spacing(sp) {
  Poses all = site.read_poses(name);
  auto have = site.labelled_scans(name, model);
  std::vector<size_t> keep;
  Vec2 last;
  for (size_t i = 0; i < all.ids.size(); ++i) {
    if (have && !have->count(all.ids[i])) continue;
    Vec2 p = all.p.row(i).head<2>().transpose();
    if (keep.empty() || (p - last).norm() >= spacing || include.count(all.ids[i])) keep.push_back(i), last = p;
  }
  const auto X = site.lidar_to_body();
  const size_t T = keep.size();
  world_xy.resize(T, 2);
  lidar_world.resize(T, 3);
  for (size_t k = 0; k < T; ++k) {
    size_t i = keep[k];
    scan_ids.push_back(all.ids[i]);
    world_xy.row(k) = all.p.row(i).head<2>();
    Eigen::Matrix3d R = all.q[i].normalized().toRotationMatrix();
    world_psi.push_back(degrees(std::atan2(R(1, 0), R(0, 0))));
    Vec3 p = all.p.row(i).transpose();
    if (X) p += R * X->block<3, 1>(0, 3), R = R * X->block<3, 3>(0, 0);  // body -> lidar
    lidar_world.row(k) = p.transpose();
    lidar_psi.push_back(degrees(std::atan2(R(1, 0), R(0, 0))));
    level.push_back(rz(-radians(lidar_psi.back())) * R);
  }
  latlon = site.latlon(world_xy);
}

MatX2 Sequence::gt_xy(int epsg) const {
  MatX2 out(latlon.rows(), 2);
  for (long i = 0; i < latlon.rows(); ++i) out.row(i) = to_utm(latlon(i, 0), latlon(i, 1), epsg).transpose();
  return out;
}

void Sequence::gt_lidar(int epsg, MatX2& xy, std::vector<double>& psi) const {
  // the heading through the georeference: a point 1 m ahead of the LiDAR, mapped like the LiDAR
  const long T = static_cast<long>(size());
  MatX2 at = lidar_world.leftCols<2>(), ahead(T, 2);
  for (long k = 0; k < T; ++k) ahead.row(k) = at.row(k) + Eigen::RowVector2d(std::cos(radians(lidar_psi[k])), std::sin(radians(lidar_psi[k])));
  MatX2 a = site.latlon(at), b = site.latlon(ahead);
  xy.resize(T, 2);
  psi.resize(T);
  for (long k = 0; k < T; ++k) {
    Vec2 p = to_utm(a(k, 0), a(k, 1), epsg), q = to_utm(b(k, 0), b(k, 1), epsg);
    xy.row(k) = p.transpose();
    psi[k] = pymod(degrees(std::atan2(q[1] - p[1], q[0] - p[0])), 360.0);
  }
}

std::vector<OdometryStep> Sequence::odometry(const OdometryNoise& n) const {
  std::mt19937_64 rng(n.seed);
  auto normal = [&](double sigma) { return std::normal_distribution<double>(0.0, 1.0)(rng) * sigma; };
  std::vector<OdometryStep> out{{std::nullopt, 0.0}};
  for (size_t t = 1; t < size(); ++t) {
    double a = radians(world_psi[t - 1]);
    Vec2 dw = (world_xy.row(t) - world_xy.row(t - 1)).transpose();
    Vec2 db(std::cos(a) * dw[0] + std::sin(a) * dw[1], -std::sin(a) * dw[0] + std::cos(a) * dw[1]);
    double step = db.norm();
    // draws in a fixed order, so a seed gives the same drift whichever terms are on
    double scale = 1.0 + n.scale_bias + normal(n.trans_noise);
    Vec2 lat(normal(n.lat_noise), normal(n.lat_noise));
    db = db * scale + lat;
    double d_psi = wrap180(world_psi[t] - world_psi[t - 1]) + normal(n.yaw_noise);
    d_psi += normal(n.yaw_rw * std::sqrt(step));
    out.push_back({db, d_psi});
  }
  return out;
}

MatXd Sequence::descriptors(const std::vector<int>& ids_in, bool cache) const {
  std::vector<int> ids = ids_in;
  std::sort(ids.begin(), ids.end());
  std::string b;
  for (size_t i = 0; i < ids.size(); ++i) b += (i ? "-" : "") + std::to_string(ids[i]);
  std::string path = data_dir() + "/queries_" + site.name + "_" + name + "_" + labels + "_b" + b + "_q" + fmt_g(spacing) +
                     "_n" + std::to_string(kNumBin) + "_r" + fmt_g(kSensorRange) + ".npz";
  const long T = static_cast<long>(size());
  if (cache && fs::exists(path)) {
    Npz z = load_npz(path);
    auto sid = z.at("scan_ids").as<long>();
    if (sid == scan_ids) {
      auto d = z.at("desc").as<double>();
      return Eigen::Map<MatXd>(d.data(), T, kNumBin);
    }
  }
  std::fprintf(stderr, "scan descriptors: %s/%s, %ld keyframes, labels %s\n", site.name.c_str(), name.c_str(), T,
               labels.c_str());
  MatXd desc(T, kNumBin);
#pragma omp parallel for schedule(dynamic, 4)
  for (long k = 0; k < T; ++k) {
    auto pc = site.load_scan(name, scan_ids[k]);
    long n = static_cast<long>(pc.size() / 4);
    auto lab = site.load_labels(name, scan_ids[k], n, model);
    auto d = scan_descriptor(pc.data(), n, 4, lab.data(), ids);
    for (int j = 0; j < kNumBin; ++j) desc(k, j) = d[j];
  }
  if (cache) {
    fs::create_directories(data_dir());
    save_npz(path, {{"scan_ids", NpyArray::from(scan_ids, {T})}, {"desc", NpyArray::from(desc.data(), {T, kNumBin})}});
  }
  return desc;
}

}  // namespace olb
