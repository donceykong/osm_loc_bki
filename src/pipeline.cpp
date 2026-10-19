#include "osmlocbki/pipeline.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

#include "osmloc/database.hpp"
#include "osmloc/site.hpp"

namespace olb {

namespace {

template <typename T>
void read(const YAML::Node& n, const char* key, T& v) {
  if (n && n[key]) v = n[key].as<T>();
}

// osm_localization caches one OSM database per map. A site without a map_bbox would build it over the
// whole OSM extract (osm_localization's kth: out of memory), so when there is none yet, build it as
// a site with the same map and a box would (kth_gt)
void build_database_in_a_box(const osmloc::Site& site, const osmloc::DatabaseOptions& o) {
  if (site.map_bbox) return;
  for (const auto& e : std::filesystem::directory_iterator(osmloc::data_dir()))
    if (e.path().filename().string().rfind("db_" + site.map + "_", 0) == 0) return;
  for (const auto& name : osmloc::Site::names()) {
    osmloc::Site s = osmloc::Site::load(name);
    if (s.map == site.map && s.map_bbox) {
      std::fprintf(stderr, "site %s has no map_bbox: the OSM database of map %s is built in %s's\n", site.name.c_str(),
                   site.map.c_str(), name.c_str());
      osmloc::site_database(s, o);
      return;
    }
  }
}

// the map's origin: a round point near the middle of the route
Vec2 map_origin(const MatX2& route) {
  Vec2 mid = 0.5 * (route.colwise().minCoeff() + route.colwise().maxCoeff()).transpose();
  return (mid / 100.0).array().round() * 100.0;
}

}  // namespace

Pipeline::Pipeline(const PipelineConfig& c) : c_(c), site_(Site::load(c.loc.site)) {
  if (c_.pose != "lag" && c_.pose != "filtered" && c_.pose != "gt") throw std::invalid_argument("pose must be lag, filtered or gt");
  if (c_.pose == "lag" && c_.loc.lag_m <= 0) c_.pose = "filtered";
  const std::string& seq = c_.loc.seq;
  // the labels' own taxonomy says which ids are buildings, so any model's labels localize
  tax_ = Taxonomy::load();
  const std::string label_set = site_.label_set(c_.loc.model);
  to_common_ = tax_.mapping(label_set);
  if (c_.loc.building_ids.empty()) c_.loc.building_ids = tax_.building_ids(label_set);
  c_.loc.view = false;
  for (const char* layer : {"_buildings.geojson", "_roads.geojson"})
    if (!std::filesystem::exists(site_.loc().map_prefix() + layer))
      throw std::runtime_error("no " + site_.loc().map_prefix() + layer + " for osm_localization: python -m osmlocbki.osm convert --site " +
                               site_.name);
  build_database_in_a_box(site_.loc(), c_.loc.db);
  run_ = std::make_unique<osmloc::Run>(c_.loc);
  T_ = run_->size();
  // osm_localization's keyframes (its Run chooses them the same way from the same arguments)
  const auto all_kf = osmloc::Sequence(site_.loc(), seq, c_.loc.model, c_.loc.spacing).scan_ids;
  kf_ids_.assign(all_kf.begin(), all_kf.begin() + T_);
  osm_ = load_map(site_.map_prefix());
  // the scans the model exported class distributions for are the ones mapped; they need not be keyframes
  auto alpha = site_.alpha_scans(seq, c_.loc.model);
  alpha_ = alpha.has_value();
  frames_ = std::make_unique<Frames>(site_, seq, all_kf, alpha_ ? *alpha : std::set<long>{}, osm_.epsg);
  gt_.resize(T_, 2);
  gt_psi_.resize(T_);
  for (long f = 0; f < static_cast<long>(frames_->size()); ++f) {
    const int k = frames_->keyframe[f];
    if (k >= 0 && k < T_) gt_.row(k) = frames_->gt_xy.row(f), gt_psi_[k] = frames_->gt_psi[f];
  }
  std::string ids;
  for (int b : c_.loc.building_ids) ids += (ids.empty() ? "" : ",") + std::to_string(b);
  std::fprintf(stderr, "%s/%s: %ld keyframes, %zu buildings, %zu other OSM features; label set %s, building ids %s\n",
               c_.loc.site.c_str(), seq.c_str(), T_, osm_.buildings.size(), osm_.features.size(), label_set.c_str(), ids.c_str());
  if (!c_.map) return;

  // mapping: config/<mapping>
  YAML::Node y = YAML::LoadFile(config_dir() + "/" + c.mapping);
  MapParams mp;
  const YAML::Node m = y["map"], o = y["osm"], g = y["ground"];
  read(m, "resolution", mp.resolution), read(m, "block_depth", mp.block_depth), read(m, "ell", mp.ell);
  read(m, "sf2", mp.sf2), read(m, "prior", mp.prior), read(m, "ds_resolution", mp.ds_resolution);
  read(m, "max_range", mp.max_range);
  if (m && m["adaptive_kernel"]) {
    const YAML::Node a = m["adaptive_kernel"];
    read(a, "enabled", mp.adaptive), read(a, "a", mp.adapt_a), read(a, "b", mp.adapt_b);
  }
  read(o, "strength", mp.strength);
  if (c.strength >= 0) mp.strength = c.strength;
  read(m, "backend", mp.backend);
  if (!c.mapper.empty()) mp.backend = c.mapper;
  if (const YAML::Node v = y["nvblox"]) {
    NvbloxParams& nv = mp.nvblox;
    read(v, "azimuth", nv.azimuth), read(v, "elevation", nv.elevation), read(v, "max_range", nv.max_range);
    read(v, "truncation_vox", nv.truncation_vox), read(v, "free_space_mask", nv.free_space_mask);
    read(v, "free_distance", nv.free_distance), read(v, "interp_max_difference_vox", nv.interp_max_difference_vox);
    read(v, "nearest_max_dist_vox", nv.nearest_max_dist_vox), read(v, "every_keyframe", nv.every_keyframe);
    read(v, "unified_memory", nv.unified_memory);
    if (v["elevation_deg"]) nv.min_elevation_deg = v["elevation_deg"][0].as<double>(), nv.max_elevation_deg = v["elevation_deg"][1].as<double>();
  }
  if (auto l = site_.lidar()) {
    NvbloxParams& nv = mp.nvblox;
    nv.azimuth = l->azimuth, nv.elevation = l->elevation;
    nv.min_elevation_deg = l->min_elevation_deg, nv.max_elevation_deg = l->max_elevation_deg;
  }
  OsmParams op;
  read(o, "decay_m", op.decay_m);
  if (o && o["widths"])
    for (const auto& kv : o["widths"]) {
      int cls = osm_class_of(kv.first.as<std::string>());
      if (cls < 0) throw std::runtime_error(c.mapping + ": unknown OSM class " + kv.first.as<std::string>());
      op.width[cls] = kv.second.as<double>();
    }
  std::string cm = o && o["confusion"] ? o["confusion"].as<std::string>() : "";
  if (!cm.empty())
    prior_ = std::make_shared<OsmPrior>(osm_, op, OsmPrior::load_confusion(config_dir() + "/" + cm, tax_.size()), tax_.roles);
  map_ = std::make_unique<SemanticMap>(tax_.size(), mp, prior_, map_origin(gt_));
  if (g && g["enabled"] && g["enabled"].as<bool>()) {
    GroundParams gp;
    read(g, "sensor_height", gp.sensor_height), read(g, "min_range", gp.min_range), read(g, "max_range", gp.max_range);
    read(g, "reflected_noise_removal", gp.reflected_noise_removal);
    ground_ = std::make_unique<GroundSegmenter>(gp);
  }
  if (!c.surface.empty()) {
    std::string path = c.surface;
    if (path == "auto") path = data_dir() + "/surface/" + site_.map + "_epsg" + std::to_string(osm_.epsg) + ".npz";
    if (!std::filesystem::exists(path)) throw std::runtime_error("no survey model at " + path);
    if (!prior_) throw std::runtime_error("the survey is labelled by OSM: the mapping config needs a confusion matrix");
    const YAML::Node v = y["survey"];
    read(v, "weight", sp_.weight), read(v, "radius", sp_.radius), read(v, "ground_above_dem", sp_.ground_above_dem);
    if (v && v["z_offset"] && v["z_offset"].as<std::string>() != "auto") sp_.auto_offset = false, dz_ = v["z_offset"].as<double>();
    if (sp_.auto_offset && !ground_) throw std::runtime_error("survey z_offset auto needs ground segmentation (or set z_offset)");
    survey_ = std::make_unique<Survey>(path);
    if (survey_->epsg() != osm_.epsg)
      throw std::runtime_error(path + " is in EPSG:" + std::to_string(survey_->epsg()) + ", the map in EPSG:" + std::to_string(osm_.epsg));
    std::fprintf(stderr, "  survey %s: within %g m of each mapped pose, weight %g\n", path.c_str(), sp_.radius, sp_.weight);
  }

  // frames to map: map_spacing apart along the route, among the scans with class distributions
  const Frames& F = *frames_;
  is_map_f_.assign(F.size(), 0);
  Vec2 last;
  for (long f = 0; f < static_cast<long>(F.size()); ++f) {
    if (F.anchor[f] >= T_ || (alpha_ ? !alpha->count(F.scan_id[f]) : F.keyframe[f] < 0)) continue;
    Vec2 p = F.gt_xy.row(f).transpose();
    if (map_f_.empty() || (p - last).norm() >= c_.map_spacing) is_map_f_[f] = 1, map_f_.push_back(f), last = p;
  }
  std::fprintf(stderr, "  mapping %zu scans (%s, OSM strength %g), pose %s, %s backend\n", map_f_.size(),
               alpha_ ? "the model's class distributions" : "one-hot labels", mp.strength, c_.pose.c_str(), mp.backend.c_str());
}

Pipeline::~Pipeline() = default;

bool Pipeline::step() {
  if (t_ + 1 >= T_ || !run_->step()) return false;
  const long t = t_ = run_->t();
  const osmloc::Localizer& loc = run_->localizer();
  const bool agree = c_.lock_agree_m <= 0 ||
                     (loc.trajectory("rigid").row(t) - loc.trajectory("graph").row(t)).norm() <= c_.lock_agree_m;
  locked_ = loc.buf().dist[t] >= c_.lock_m && agree;
  if (!map_) return true;
  if (c_.pose == "gt") {
    place_from(t, gt_.row(t).transpose(), gt_psi_[t]);
  } else if (c_.pose == "filtered") {
    if (locked_) place_from(t, loc.trajectory("filtered").row(t).transpose(), loc.heading("filtered")[t]);
  } else {
    settle_lag();
  }
  return true;
}

// map off the keyframes whose lag pose settled since the last call (they settle in order)
void Pipeline::settle_lag() {
  const osmloc::Localizer& loc = run_->localizer();
  const MatX2 lag = loc.trajectory("lag");
  const auto& psi = loc.heading("lag");
  while (settled_ + 1 <= t_ && std::isfinite(lag(settled_ + 1, 0))) {
    const long u = ++settled_;
    if (locked_) place_from(u, lag.row(u).transpose(), psi[u]);
  }
}

void Pipeline::place_from(long u, const Vec2& kxy, double kpsi) {
  const Frames& F = *frames_;
  for (long f : F.of_keyframe[u]) {
    Vec2 xy;
    double psi;
    F.pose(f, kxy, kpsi, xy, psi);
    if (is_map_f_[f]) {
      insert(f, xy, psi);
    } else if (F.keyframe[f] >= 0 && map_->has_geometry() && map_->params().nvblox.every_keyframe) {
      Stopwatch sw;
      const Vec2 at = xy - map_->utm_origin();
      map_->observe(place(load(f, false).levelled, xy, psi, f), Vec3(at[0], at[1], F.z[f]));
      geo_s_ += sw.seconds();
    }
  }
}

void Pipeline::finish() {
  run_->finish();
  if (map_ && c_.pose == "lag") settle_lag();
}

Pipeline::Scan Pipeline::load(long f, bool semantics) const {
  const long id = frames_->scan_id[f];
  const std::string& seq = c_.loc.seq;
  const std::string& model = c_.loc.model;
  auto pc = site_.load_scan(seq, id);
  const long n = static_cast<long>(pc.size() / 4);
  Semantics all;
  if (!semantics) {
  } else if (alpha_) {
    auto a = site_.load_alpha(seq, id, model);
    all = from_alpha(a.data(), n, static_cast<int>(a.size() / n), to_common_, tax_.size());
  } else {
    auto lab = site_.load_labels(seq, id, n, model);
    all = from_labels(lab.data(), n, to_common_, tax_.size());
  }
  // drop the sensor's zero-filled non-returns; level the rest (gravity down, heading removed)
  Scan s;
  s.raw = n;
  for (long i = 0; i < n; ++i)
    if (std::hypot(pc[4 * i], pc[4 * i + 1]) > osmloc::kMinScanRange) s.index.push_back(i);
  const long m = static_cast<long>(s.index.size());
  const Eigen::Matrix3f R = frames_->level[f].cast<float>();
  s.levelled.resize(m, 3);
  s.sem.probs.resize(semantics ? m : 0, tax_.size());
  s.sem.vacuity.resize(semantics ? m : 0);
  s.intensity.resize(m);
  for (long k = 0; k < m; ++k) {
    long i = s.index[k];
    s.levelled.row(k) = (R * Eigen::Vector3f(pc[4 * i], pc[4 * i + 1], pc[4 * i + 2])).transpose();
    if (semantics) s.sem.probs.row(k) = all.probs.row(i), s.sem.vacuity[k] = all.vacuity[i];
    s.intensity[k] = pc[4 * i + 3];
  }
  return s;
}

MatX3f Pipeline::place(const MatX3f& levelled, const Vec2& xy, double psi, long f) const {
  const Eigen::Matrix3f R = rz(radians(psi)).cast<float>();
  const Vec2 at = xy - map_->utm_origin();
  const Eigen::RowVector3f o(static_cast<float>(at[0]), static_cast<float>(at[1]), static_cast<float>(frames_->z[f]));
  return (levelled * R.transpose()).rowwise() + o;
}

void Pipeline::insert(long f, const Vec2& xy, double psi) {
  Stopwatch sw;
  Scan s = load(f);
  std::vector<int8_t> ground;
  if (ground_) {
    MatXf in(s.levelled.rows(), 4);
    in.leftCols(3) = s.levelled;
    in.col(3) = Eigen::Map<const Eigen::VectorXf>(s.intensity.data(), s.intensity.size());
    ground = ground_->segment(in);
  }
  MatX3f pts = place(s.levelled, xy, psi, f);
  const Vec2 at = xy - map_->utm_origin();
  map_->insert(pts, s.sem, ground, Vec3(at[0], at[1], frames_->z[f]));
  if (survey_) project_survey(xy, pts, ground);
  mapped_.push_back(f);
  mapped_xy_.push_back(xy);
  last_.f = f, last_.scan_id = frames_->scan_id[f], last_.xy = xy, last_.psi = psi;
  last_.points = std::move(pts);
  last_.cls.resize(s.sem.probs.rows());
  for (long i = 0; i < s.sem.probs.rows(); ++i) {
    Eigen::Index c;
    last_.cls[i] = s.sem.probs.row(i).maxCoeff(&c) > 0 ? static_cast<int>(c) : 0;
  }
  map_s_ += sw.seconds();
}

void Pipeline::project_survey(const Vec2& xy, const MatX3f& pts, const std::vector<int8_t>& ground) {
  const Vec2 o = map_->utm_origin();
  // the survey's height datum: where this scan's ground points sit over the survey's terrain
  if (sp_.auto_offset) {
    std::vector<double> d;
    for (long i = 0; i < pts.rows(); ++i)
      if (ground[i] == 1) d.push_back(pts(i, 2) - survey_->terrain(Vec2(pts(i, 0), pts(i, 1)) + o));
    if (d.size() > 100) {
      std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
      dz_scans_.push_back(d[d.size() / 2]);
      std::vector<double> m = dz_scans_;
      std::nth_element(m.begin(), m.begin() + m.size() / 2, m.end());
      dz_ = m[m.size() / 2];
    }
  }
  if (std::isnan(dz_)) return;
  // the survey's points around the pose, snapped to voxels not projected yet
  const MatX3 sv = survey_->crop(xy, sp_.radius);
  const double res = map_->params().resolution;
  std::vector<long> fresh;
  for (long i = 0; i < sv.rows(); ++i) {
    Vec3 x(sv(i, 0) - o[0], sv(i, 1) - o[1], sv(i, 2) + dz_);
    const int64_t k[3] = {static_cast<int64_t>(std::floor(x[0] / res)), static_cast<int64_t>(std::floor(x[1] / res)),
                          static_cast<int64_t>(std::floor(x[2] / res))};
    if (survey_done_.insert(((k[0] + (1 << 20)) << 42) | ((k[1] + (1 << 20)) << 21) | (k[2] + (1 << 20))).second)
      fresh.push_back(i);
  }
  MatX3f at(fresh.size(), 3);
  MatXf rows(fresh.size(), tax_.size());
  for (size_t j = 0; j < fresh.size(); ++j) {
    const long i = fresh[j];
    const Vec2 u(sv(i, 0), sv(i, 1));
    at.row(j) << static_cast<float>(u[0] - o[0]), static_cast<float>(u[1] - o[1]), static_cast<float>(sv(i, 2) + dz_);
    const int g = sv(i, 2) - survey_->terrain(u) < sp_.ground_above_dem ? 1 : 0;
    prior_->project(u, g, rows.row(j).data());
  }
  map_->deposit(at, static_cast<float>(sp_.weight) * rows);
}

std::string Pipeline::survey_status() const {
  if (!survey_) return "";
  char b[96];
  std::snprintf(b, sizeof b, ", survey %ld voxels (datum %+.2f m)", survey_voxels(), dz_);
  return b;
}

std::vector<int> Pipeline::classify_frame(long f) const {
  if (!map_) throw std::logic_error("no map: the pipeline runs with map = false");
  Scan s = load(f);
  auto cls = map_->classify(place(s.levelled, frames_->gt_xy.row(f).transpose(), frames_->gt_psi[f], f));
  std::vector<int> out(s.raw, 0);
  for (size_t k = 0; k < s.index.size(); ++k) out[s.index[k]] = cls[k];
  return out;
}

std::string Pipeline::status() const {
  char b[512];
  if (t_ < 0) return "starting";
  const osmloc::Localizer& loc = run_->localizer();
  const auto& d = loc.buf().dist;
  auto err = [&](const char* o) { return (loc.trajectory(o).row(t_) - gt_.row(t_)).norm(); };
  const auto& tm = loc.timing;
  char geo[48] = "";
  if (geo_s_ > 0) std::snprintf(geo, sizeof geo, ", geometry %.0f", geo_s_ / (t_ + 1) * 1e3);
  std::snprintf(b, sizeof b,
                "keyframe %ld / %ld, %.0f m%s | error: filtered %.1f m, graph %.1f m, rigid %.1f m | "
                "mapped %zu / %zu scans (pose %s)%s | ms per keyframe: match %.0f, back-end %.0f%s; per mapped scan %.0f",
                t_ + 1, T_, d[t_], locked_ ? ", locked" : ", not locked", err("filtered"), err("graph"), err("rigid"),
                mapped_.size(), map_f_.size(), c_.pose.c_str(), survey_status().c_str(), tm.at("match") / (t_ + 1) * 1e3,
                (tm.at("rigid") + tm.at("graph")) / (t_ + 1) * 1e3, geo, mapped_.empty() ? 0.0 : map_s_ / mapped_.size() * 1e3);
  return b;
}

}  // namespace olb
