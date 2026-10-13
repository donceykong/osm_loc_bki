#include "osmlocbki/pipeline.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace olb {

namespace {

template <typename T>
void read(const YAML::Node& n, const char* key, T& v) {
  if (n && n[key]) v = n[key].as<T>();
}

// the map's origin: a round point near the middle of the OSM database
Vec2 map_origin(const Database& db) {
  Vec2 mid = 0.5 * (db.xy.colwise().minCoeff() + db.xy.colwise().maxCoeff()).transpose();
  return (mid / 100.0).array().round() * 100.0;
}

}  // namespace

Pipeline::Pipeline(const PipelineConfig& c) : c_(c), site_(Site::load(c.site)) {
  if (c_.pose != "lag" && c_.pose != "filtered" && c_.pose != "gt") throw std::invalid_argument("pose must be lag, filtered or gt");
  if (c_.pose == "lag" && c_.lag_m <= 0) c_.pose = "filtered";
  // the scans the model exported class distributions for are mapped, so they must be keyframes
  auto alpha = site_.alpha_scans(c.seq, c.model);
  alpha_ = alpha.has_value();
  seq_ = std::make_unique<Sequence>(site_, c.seq, c.model, c.spacing, alpha_ ? *alpha : std::set<long>{});
  db_ = site_database(site_, c.db_spacing);
  // the labels' own taxonomy says which ids are buildings, so any model's labels localize
  tax_ = Taxonomy::load();
  const std::string label_set = site_.label_set(c.model);
  to_common_ = tax_.mapping(label_set);
  const auto building = c.building_ids.empty() ? tax_.building_ids(label_set) : c.building_ids;
  q_ = seq_->descriptors(building);
  odo_ = seq_->odometry(c.odometry);
  seq_->gt_lidar(db_->epsg, gt_, gt_psi_);
  T_ = c.max_kf > 0 ? std::min<long>(c.max_kf, static_cast<long>(seq_->size())) : static_cast<long>(seq_->size());
  loc_ = std::make_unique<Localizer>(std::make_shared<Matcher>(db_, c.top_n), c.k, c.graph, c.agree_m, c.lag_m, c.filter);
  osm_ = load_map(site_.map_prefix(), db_->epsg);
  std::string ids;
  for (int b : building) ids += (ids.empty() ? "" : ",") + std::to_string(b);
  std::fprintf(stderr, "%s/%s: %zu keyframes, %ld OSM candidates, %zu buildings, %zu other OSM features; label set %s, building ids %s\n",
               c.site.c_str(), c.seq.c_str(), seq_->size(), db_->size(), osm_.buildings.size(), osm_.features.size(),
               label_set.c_str(), ids.c_str());
  if (!c.map) return;

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
  map_ = std::make_unique<SemanticMap>(tax_.size(), mp, prior_, map_origin(*db_));
  if (g && g["enabled"] && g["enabled"].as<bool>()) {
    GroundParams gp;
    read(g, "sensor_height", gp.sensor_height), read(g, "min_range", gp.min_range), read(g, "max_range", gp.max_range);
    read(g, "reflected_noise_removal", gp.reflected_noise_removal);
    ground_ = std::make_unique<GroundSegmenter>(gp);
  }

  // mapping keyframes: map_spacing apart (3D), among the scans with class distributions
  is_map_kf_.assign(T_, 0);
  Vec3 last;
  for (long t = 0; t < T_; ++t) {
    if (alpha_ && !alpha->count(seq_->scan_ids[t])) continue;
    Vec3 p = seq_->lidar_world.row(t).transpose();
    if (map_kf_.empty() || (p - last).norm() >= c.map_spacing) is_map_kf_[t] = 1, map_kf_.push_back(t), last = p;
  }
  std::fprintf(stderr, "  mapping %zu keyframes (%s, OSM strength %g), pose %s\n", map_kf_.size(),
               alpha_ ? "the model's class distributions" : "one-hot labels", mp.strength, c_.pose.c_str());
}

Pipeline::~Pipeline() = default;

bool Pipeline::step() {
  if (t_ + 1 >= T_) return false;
  const long t = t_ + 1;
  Estimate e = loc_->step(q_.row(t).data(), odo_[t].d_body, odo_[t].d_psi);
  locked_ = loc_->buf().dist[t] >= c_.lock_m && e.agree;
  t_ = t;
  if (map_) map_ready(e);
  return true;
}

void Pipeline::map_ready(const Estimate& e) {
  if (c_.pose == "gt") {
    if (is_map_kf_[e.t]) insert(e.t, gt_.row(e.t).transpose(), gt_psi_[e.t]);
  } else if (c_.pose == "filtered") {
    if (is_map_kf_[e.t] && locked_) insert(e.t, e.xy[FILTERED], e.psi[FILTERED]);
  } else {
    for (int u : e.lagged)
      if (is_map_kf_[u] && locked_) insert(u, loc_->xy(LAG, u), loc_->heading(LAG)[u]);
  }
}

void Pipeline::finish() {
  if (c_.lag_m <= 0) return;
  for (int u : loc_->finish_lag())
    if (map_ && c_.pose == "lag" && is_map_kf_[u] && locked_) insert(u, loc_->xy(LAG, u), loc_->heading(LAG)[u]);
}

Pipeline::Scan Pipeline::load(long t) const {
  const long id = seq_->scan_ids[t];
  auto pc = site_.load_scan(c_.seq, id);
  const long n = static_cast<long>(pc.size() / 4);
  Semantics all;
  if (alpha_) {
    auto a = site_.load_alpha(c_.seq, id, c_.model);
    all = from_alpha(a.data(), n, static_cast<int>(a.size() / n), to_common_, tax_.size());
  } else {
    auto lab = site_.load_labels(c_.seq, id, n, c_.model);
    all = from_labels(lab.data(), n, to_common_, tax_.size());
  }
  // drop the sensor's zero-filled non-returns; level the rest (gravity down, heading removed)
  Scan s;
  s.raw = n;
  for (long i = 0; i < n; ++i)
    if (std::hypot(pc[4 * i], pc[4 * i + 1]) > kMinScanRange) s.index.push_back(i);
  const long m = static_cast<long>(s.index.size());
  const Eigen::Matrix3f R = seq_->level[t].cast<float>();
  s.levelled.resize(m, 3);
  s.sem.probs.resize(m, tax_.size());
  s.sem.vacuity.resize(m);
  s.intensity.resize(m);
  for (long k = 0; k < m; ++k) {
    long i = s.index[k];
    s.levelled.row(k) = (R * Eigen::Vector3f(pc[4 * i], pc[4 * i + 1], pc[4 * i + 2])).transpose();
    s.sem.probs.row(k) = all.probs.row(i);
    s.sem.vacuity[k] = all.vacuity[i];
    s.intensity[k] = pc[4 * i + 3];
  }
  return s;
}

MatX3f Pipeline::place(const MatX3f& levelled, const Vec2& xy, double psi, long t) const {
  const Eigen::Matrix3f R = rz(radians(psi)).cast<float>();
  const Vec2 at = xy - map_->utm_origin();
  const Eigen::RowVector3f o(static_cast<float>(at[0]), static_cast<float>(at[1]), static_cast<float>(seq_->lidar_world(t, 2)));
  return (levelled * R.transpose()).rowwise() + o;
}

void Pipeline::insert(long t, const Vec2& xy, double psi) {
  Stopwatch sw;
  Scan s = load(t);
  std::vector<int8_t> ground;
  if (ground_) {
    MatXf in(s.levelled.rows(), 4);
    in.leftCols(3) = s.levelled;
    in.col(3) = Eigen::Map<const Eigen::VectorXf>(s.intensity.data(), s.intensity.size());
    ground = ground_->segment(in);
  }
  MatX3f pts = place(s.levelled, xy, psi, t);
  const Vec2 at = xy - map_->utm_origin();
  map_->insert(pts, s.sem, ground, Vec3(at[0], at[1], seq_->lidar_world(t, 2)));
  mapped_.push_back(t);
  last_.t = t, last_.scan_id = seq_->scan_ids[t], last_.xy = xy, last_.psi = psi;
  last_.points = std::move(pts);
  last_.cls.resize(s.sem.probs.rows());
  for (long i = 0; i < s.sem.probs.rows(); ++i) {
    Eigen::Index c;
    last_.cls[i] = s.sem.probs.row(i).maxCoeff(&c) > 0 ? static_cast<int>(c) : 0;
  }
  map_s_ += sw.seconds();
}

std::vector<int> Pipeline::classify_keyframe(long t) const {
  if (!map_) throw std::logic_error("no map: the pipeline runs with map = false");
  Scan s = load(t);
  auto cls = map_->classify(place(s.levelled, gt_.row(t).transpose(), gt_psi_[t], t));
  std::vector<int> out(s.raw, 0);
  for (size_t k = 0; k < s.index.size(); ++k) out[s.index[k]] = cls[k];
  return out;
}

std::string Pipeline::status() const {
  char b[512];
  if (t_ < 0) return "starting";
  const auto& d = loc_->buf().dist;
  auto err = [&](Output o) { return (loc_->xy(o, static_cast<int>(t_)) - gt_.row(t_).transpose()).norm(); };
  const auto& tm = loc_->timing;
  std::snprintf(b, sizeof b,
                "keyframe %ld / %ld, %.0f m%s | error: filtered %.1f m, graph %.1f m, rigid %.1f m | "
                "mapped %zu / %zu keyframes (pose %s) | ms per keyframe: match %.0f, back-end %.0f; per mapped scan %.0f",
                t_ + 1, T_, d[t_], locked_ ? ", locked" : ", not locked", err(FILTERED), err(GRAPH), err(RIGID),
                mapped_.size(), map_kf_.size(), c_.pose.c_str(), tm.match / (t_ + 1) * 1e3,
                (tm.rigid + tm.graph) / (t_ + 1) * 1e3, mapped_.empty() ? 0.0 : map_s_ / mapped_.size() * 1e3);
  return b;
}

}  // namespace olb
