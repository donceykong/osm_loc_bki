#include "osmlocbki/semantic_map.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace olb {

namespace {

constexpr int64_t kBias = int64_t(1) << 20;  // 21 bits per axis: +-1M blocks

int64_t pack(int64_t i, int64_t j, int64_t k) { return ((i + kBias) << 42) | ((j + kBias) << 21) | (k + kBias); }
int64_t unpack(int64_t key, int axis) { return ((key >> (42 - 21 * axis)) & ((int64_t(1) << 21) - 1)) - kBias; }

// the sparse kernel of Melkumyan & Ramos (2009) at r = distance / length scale, without sf2
inline float sparse_kernel(float r) {
  constexpr float two_pi = 2.0f * 3.1415926f;
  return (2.0f + std::cos(two_pi * r)) * (1.0f - r) / 3.0f + std::sin(two_pi * r) / two_pi;
}

}  // namespace

SemanticMap::SemanticMap(int num_class, const MapParams& p, std::shared_ptr<const OsmPrior> osm, const Vec2& utm_origin)
    : nc_(num_class), p_(p), osm_(std::move(osm)), origin_utm_(utm_origin) {
  if (p_.block_depth < 1) throw std::invalid_argument("block_depth must be at least 1");
  if (osm_ && osm_->num_class() != nc_) throw std::invalid_argument("the OSM prior has another number of classes");
  cells_ = 1 << (p_.block_depth - 1);
  block_size_ = cells_ * p_.resolution;
  for (int k = 0; k < cells_; ++k)
    for (int j = 0; j < cells_; ++j)
      for (int i = 0; i < cells_; ++i)
        leaf_offset_.push_back(Eigen::Vector3f(i - (cells_ - 1) / 2.0, j - (cells_ - 1) / 2.0, k - (cells_ - 1) / 2.0) *
                               static_cast<float>(p_.resolution));
}

int64_t SemanticMap::key_of(const Eigen::Vector3f& x) const {
  auto b = [&](float v) { return static_cast<int64_t>(std::floor(v / block_size_ + 0.5)); };
  return pack(b(x[0]), b(x[1]), b(x[2]));
}

Eigen::Vector3f SemanticMap::center_of(int64_t key) const {
  return Eigen::Vector3f(unpack(key, 0), unpack(key, 1), unpack(key, 2)) * static_cast<float>(block_size_);
}

int SemanticMap::leaf_of(const Eigen::Vector3f& x, int64_t key) const {
  Eigen::Vector3f lo = center_of(key) - Eigen::Vector3f::Constant(0.5f * block_size_);
  int idx[3];
  for (int a = 0; a < 3; ++a) idx[a] = std::clamp(static_cast<int>(std::floor((x[a] - lo[a]) / p_.resolution)), 0, cells_ - 1);
  return idx[0] + cells_ * (idx[1] + cells_ * idx[2]);
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
  std::vector<Train> train;
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
    train.push_back({x, 1.0f / len, static_cast<int>(train.size())});
  }
  if (train.empty()) return;

  // points by block; the blocks they reach: their own and its six face neighbours
  std::unordered_map<int64_t, std::vector<int>> by_block;
  for (size_t j = 0; j < train.size(); ++j) by_block[key_of(train[j].x)].push_back(static_cast<int>(j));
  auto neighbours = [&](int64_t key) {
    int64_t i = unpack(key, 0), j = unpack(key, 1), k = unpack(key, 2);
    return std::array<int64_t, 7>{key, pack(i + 1, j, k), pack(i - 1, j, k), pack(i, j + 1, k), pack(i, j - 1, k),
                                  pack(i, j, k + 1), pack(i, j, k - 1)};
  };
  std::vector<std::pair<int64_t, std::vector<float>*>> test;
  {
    std::unordered_map<int64_t, char> seen;
    for (const auto& kv : by_block)
      for (int64_t b : neighbours(kv.first))
        if (seen.emplace(b, 1).second) {
          auto& ev = blocks_[b];
          if (ev.empty()) ev.assign(leaf_offset_.size() * nc_, 0.0f);
          test.emplace_back(b, &ev);
        }
  }

  const float sf2 = static_cast<float>(p_.sf2);
#pragma omp parallel for schedule(dynamic, 64)
  for (long t = 0; t < static_cast<long>(test.size()); ++t) {
    const int64_t key = test[t].first;
    std::vector<float>& ev = *test[t].second;
    const Eigen::Vector3f c = center_of(key);
    std::vector<double> ybar(nc_);
    for (size_t l = 0; l < leaf_offset_.size(); ++l) {
      const Eigen::Vector3f leaf = c + leaf_offset_[l];
      std::fill(ybar.begin(), ybar.end(), 0.0);
      bool any = false;
      for (int64_t b : neighbours(key)) {
        auto it = by_block.find(b);
        if (it == by_block.end()) continue;
        for (int j : it->second) {
          const Train& tp = train[j];
          float r = (leaf - tp.x).norm() * tp.inv_len;
          if (r >= 1.0f) continue;  // the kernel's compact support
          float k = sf2 * sparse_kernel(r);
          if (k <= 0) continue;
          for (int cc = 0; cc < nc_; ++cc) ybar[cc] += k * rows(tp.row, cc);
          any = true;
        }
      }
      if (any)
        for (int cc = 0; cc < nc_; ++cc) ev[l * nc_ + cc] += static_cast<float>(ybar[cc]);
    }
  }
}

std::vector<int> SemanticMap::classify(const MatX3f& pts) const {
  std::vector<int> out(pts.rows(), 0);
  for (long i = 0; i < pts.rows(); ++i) {
    Eigen::Vector3f x = pts.row(i).transpose();
    int64_t key = key_of(x);
    auto it = blocks_.find(key);
    if (it == blocks_.end()) continue;
    const float* e = it->second.data() + leaf_of(x, key) * nc_;
    out[i] = static_cast<int>(std::max_element(e, e + nc_) - e);  // the prior is uniform: argmax of the evidence
    if (e[out[i]] <= 0) out[i] = 0;
  }
  return out;
}

void SemanticMap::voxels(MatX3f& xyz, std::vector<int>& cls, std::vector<float>& prob, std::vector<float>& mass,
                         const Box* box) const {
  std::vector<Eigen::Vector3f> at;
  cls.clear(), prob.clear(), mass.clear();
  const float prior = static_cast<float>(p_.prior);
  for (const auto& [key, ev] : blocks_) {
    const Eigen::Vector3f c = center_of(key);
    if (box && (c[0] < (*box)[0] || c[1] < (*box)[1] || c[0] > (*box)[2] || c[1] > (*box)[3])) continue;
    for (size_t l = 0; l < leaf_offset_.size(); ++l) {
      const float* e = ev.data() + l * nc_;
      float m = 0;
      for (int cc = 0; cc < nc_; ++cc) m += e[cc];
      if (!(m > 0)) continue;
      int best = static_cast<int>(std::max_element(e, e + nc_) - e);
      at.push_back(c + leaf_offset_[l]);
      cls.push_back(best);
      prob.push_back((prior + e[best]) / (nc_ * prior + m));
      mass.push_back(m);
    }
  }
  xyz.resize(at.size(), 3);
  for (size_t i = 0; i < at.size(); ++i) xyz.row(i) = at[i].transpose();
}

}  // namespace olb
