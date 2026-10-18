// The CPU store: OSM-BKI's blocks in a hash map, each block's voxels gathering evidence from the
// points of its extended block (itself and its six face neighbours), as SemanticBKIOctoMap does.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

#include "osmlocbki/voxel_store.hpp"

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

class CpuStore : public VoxelStore {
 public:
  CpuStore(int nc, const StoreParams& p) : nc_(nc), p_(p) {
    if (p_.block_depth < 1) throw std::invalid_argument("block_depth must be at least 1");
    cells_ = 1 << (p_.block_depth - 1);
    block_size_ = cells_ * p_.resolution;
    for (int k = 0; k < cells_; ++k)
      for (int j = 0; j < cells_; ++j)
        for (int i = 0; i < cells_; ++i)
          leaf_offset_.push_back(Eigen::Vector3f(i - (cells_ - 1) / 2.0, j - (cells_ - 1) / 2.0, k - (cells_ - 1) / 2.0) *
                                 static_cast<float>(p_.resolution));
  }

  void add_kernel(const std::vector<KernelPoint>& train, const MatXf& rows) override {
    // points by block; the blocks they reach: their own and its six face neighbours
    std::unordered_map<int64_t, std::vector<int>> by_block;
    for (size_t j = 0; j < train.size(); ++j) by_block[key_of(train[j].x)].push_back(static_cast<int>(j));
    std::vector<std::pair<int64_t, std::vector<float>*>> test;
    {
      std::unordered_map<int64_t, char> seen;
      for (const auto& kv : by_block)
        for (int64_t b : neighbours(kv.first))
          if (seen.emplace(b, 1).second) test.emplace_back(b, &block(b));
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
            float r = (leaf - train[j].x).norm() * train[j].inv_len;
            if (r >= 1.0f) continue;  // the kernel's compact support
            float k = sf2 * sparse_kernel(r);
            if (k <= 0) continue;
            for (int cc = 0; cc < nc_; ++cc) ybar[cc] += k * rows(j, cc);
            any = true;
          }
        }
        if (any)
          for (int cc = 0; cc < nc_; ++cc) ev[l * nc_ + cc] += static_cast<float>(ybar[cc]);
      }
    }
  }

  void add_at(const MatX3f& pts, const MatXf& rows) override {
    for (long i = 0; i < pts.rows(); ++i) {
      Eigen::Vector3f x = pts.row(i).transpose();
      if (!x.allFinite() || !(rows.row(i).sum() > 0)) continue;
      int64_t key = key_of(x);
      float* e = block(key).data() + leaf_of(x, key) * nc_;
      for (int c = 0; c < nc_; ++c) e[c] += rows(i, c);
    }
  }

  std::vector<int> classify(const MatX3f& pts) const override {
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

  void voxels(MatX3f& xyz, std::vector<int>& cls, std::vector<float>& prob, std::vector<float>& mass,
              const Box* box) const override {
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

 private:
  int64_t key_of(const Eigen::Vector3f& x) const {
    auto b = [&](float v) { return static_cast<int64_t>(std::floor(v / block_size_ + 0.5)); };
    return pack(b(x[0]), b(x[1]), b(x[2]));
  }
  Eigen::Vector3f center_of(int64_t key) const {
    return Eigen::Vector3f(unpack(key, 0), unpack(key, 1), unpack(key, 2)) * static_cast<float>(block_size_);
  }
  int leaf_of(const Eigen::Vector3f& x, int64_t key) const {
    Eigen::Vector3f lo = center_of(key) - Eigen::Vector3f::Constant(0.5f * block_size_);
    int idx[3];
    for (int a = 0; a < 3; ++a) idx[a] = std::clamp(static_cast<int>(std::floor((x[a] - lo[a]) / p_.resolution)), 0, cells_ - 1);
    return idx[0] + cells_ * (idx[1] + cells_ * idx[2]);
  }
  static std::array<int64_t, 7> neighbours(int64_t key) {
    int64_t i = unpack(key, 0), j = unpack(key, 1), k = unpack(key, 2);
    return {key, pack(i + 1, j, k), pack(i - 1, j, k), pack(i, j + 1, k), pack(i, j - 1, k), pack(i, j, k + 1), pack(i, j, k - 1)};
  }
  // a block's evidence (the prior excluded): (cells^3, nc) row-major, zero when new
  std::vector<float>& block(int64_t key) {
    auto& ev = blocks_[key];
    if (ev.empty()) ev.assign(leaf_offset_.size() * nc_, 0.0f);
    return ev;
  }

  int nc_;
  StoreParams p_;
  int cells_;
  double block_size_;
  std::vector<Eigen::Vector3f> leaf_offset_;
  std::unordered_map<int64_t, std::vector<float>> blocks_;
};

}  // namespace

void VoxelStore::save_mesh(const std::string&, const std::vector<std::array<uint8_t, 3>>&) const {
  throw std::logic_error("this map has no geometry to mesh: map with nvblox (--mapper nvblox)");
}

void VoxelStore::mesh(MatX3f&, MatX3i&, MatX3u8&, const std::vector<std::array<uint8_t, 3>>&, bool) const {
  throw std::logic_error("this map has no geometry to mesh: map with nvblox (--mapper nvblox)");
}

std::unique_ptr<VoxelStore> make_cpu_store(int num_class, const StoreParams& p) { return std::make_unique<CpuStore>(num_class, p); }

}  // namespace olb
