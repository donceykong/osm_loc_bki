#include "osmlocbki/semantic_map.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace olb {

namespace {

constexpr int64_t kBias = int64_t(1) << 20;  // 21 bits per axis
int64_t pack(int64_t i, int64_t j, int64_t k) { return ((i + kBias) << 42) | ((j + kBias) << 21) | (k + kBias); }

}  // namespace

SemanticMap::SemanticMap(int num_class, const MapParams& p, std::shared_ptr<const OsmPrior> osm, const Vec2& utm_origin)
    : nc_(num_class), p_(p), osm_(std::move(osm)), origin_utm_(utm_origin) {
  if (osm_ && osm_->num_class() != nc_) throw std::invalid_argument("the OSM prior has another number of classes");
  StoreParams sp;
  sp.resolution = p_.resolution, sp.block_depth = p_.block_depth, sp.sf2 = p_.sf2, sp.prior = p_.prior;
  if (p_.backend == "cpu")
    store_ = make_cpu_store(nc_, sp);
  else if (p_.backend == "nvblox")
    store_ = make_nvblox_store(nc_, sp, p_.nvblox);
  else
    throw std::invalid_argument("unknown map backend " + p_.backend + " (cpu or nvblox)");
}

void SemanticMap::insert(const MatX3f& pts, const Semantics& sem, const std::vector<int8_t>& ground, const Vec3& origin) {
  const long n = pts.rows();
  if (sem.probs.rows() != n || sem.probs.cols() != nc_ || static_cast<long>(sem.vacuity.size()) != n ||
      !(ground.empty() || static_cast<long>(ground.size()) == n))
    throw std::invalid_argument("insert: points, semantics and ground verdicts disagree in size");

  // training points: the mean of the points in each ds_resolution voxel, with the mean class
  // distribution and vacuity and the majority ground verdict
  struct Acc {
    Eigen::Vector3d x = Eigen::Vector3d::Zero();
    std::vector<double> p;
    double vac = 0;
    int n = 0, ground = 0, nonground = 0;
  };
  const float inv_ds = p_.ds_resolution > 0 ? 1.0f / static_cast<float>(p_.ds_resolution) : 0.0f;
  std::unordered_map<int64_t, size_t> slot;
  std::vector<Acc> acc;
  const Eigen::Vector3f o = origin.cast<float>();
  for (long i = 0; i < n; ++i) {
    Eigen::Vector3f x = pts.row(i).transpose();
    if (!x.allFinite() || (p_.max_range > 0 && (x - o).norm() > p_.max_range)) continue;
    int64_t key = inv_ds > 0 ? pack(static_cast<int64_t>(std::floor(x[0] * inv_ds)), static_cast<int64_t>(std::floor(x[1] * inv_ds)),
                                    static_cast<int64_t>(std::floor(x[2] * inv_ds)))
                             : i;
    auto [it, fresh] = slot.emplace(key, acc.size());
    if (fresh) acc.push_back(Acc{Eigen::Vector3d::Zero(), std::vector<double>(nc_, 0.0)});
    Acc& a = acc[it->second];
    a.x += x.cast<double>();
    for (int c = 0; c < nc_; ++c) a.p[c] += sem.probs(i, c);
    a.vac += std::clamp(sem.vacuity[i], 0.f, 1.f);
    a.n += 1;
    if (!ground.empty()) a.ground += ground[i] == 1, a.nonground += ground[i] == 0;
  }

  // what each training point deposits (OSM-boosted), and how far it reaches
  std::vector<KernelPoint> train;
  MatXf rows(acc.size(), nc_);
  std::vector<float> s(nc_);
  const bool use_osm = osm_ && p_.strength > 0;
  for (const Acc& a : acc) {
    Eigen::Vector3f x = (a.x / a.n).cast<float>();
    Eigen::VectorXf pm(nc_);
    for (int c = 0; c < nc_; ++c) pm[c] = static_cast<float>(a.p[c] / a.n);
    if (pm.sum() <= 1e-9f) continue;  // all its mass was on classes the map does not represent
    if (use_osm) {
      int g = a.ground || a.nonground ? (a.ground >= a.nonground ? 1 : 0) : -1;
      osm_->project(origin_utm_ + x.head<2>().cast<double>(), g, s.data());
      for (int c = 0; c < nc_; ++c) pm[c] = std::max(0.0f, pm[c] + static_cast<float>(p_.strength) * s[c] * (1.0f - pm[c]));
    }
    float len = static_cast<float>(p_.ell);
    if (p_.adaptive) {
      float u = static_cast<float>(a.vac / a.n);
      len *= std::max(1e-6f, static_cast<float>(p_.adapt_a) * std::exp(1.0f + static_cast<float>(p_.adapt_b) * u));
    }
    rows.row(static_cast<long>(train.size())) = pm.transpose();
    train.push_back({x, 1.0f / len});
  }
  if (store_->has_geometry()) store_->observe(pts, o);
  if (train.empty()) return;
  store_->add_kernel(train, rows.topRows(static_cast<long>(train.size())));
}

void SemanticMap::deposit(const MatX3f& pts, const MatXf& rows) {
  if (rows.rows() != pts.rows() || rows.cols() != nc_) throw std::invalid_argument("deposit: one row of classes per point");
  store_->add_at(pts, rows);
}

std::vector<int> SemanticMap::classify(const MatX3f& pts) const { return store_->classify(pts); }

void SemanticMap::voxels(MatX3f& xyz, std::vector<int>& cls, std::vector<float>& prob, std::vector<float>& mass,
                         const Box* box) const {
  store_->voxels(xyz, cls, prob, mass, box);
}

void SemanticMap::save_mesh(const std::string& path, const std::vector<std::array<uint8_t, 3>>& colors) const {
  store_->save_mesh(path, colors);
}

}  // namespace olb
