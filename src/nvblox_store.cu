// The GPU store: BKI class evidence in an nvblox voxel-block layer, next to an nvblox Mapper that
// integrates the same scans into a TSDF (free space, the mesh).
//
// The semantic layer shares the TSDF's voxel grid (same voxel size, same 8^3 blocks), so a semantic
// voxel and its TSDF voxel have the same index. BKI's update runs point-parallel: each training
// point scatters its kernel into the voxels of its BKI block and the six face neighbours' (the
// voxels the CPU store's extended blocks would give it), with atomic adds, so the sums match the
// CPU store's up to float rounding order.
//
// nvblox frame = map frame + `shift`: BKI leaves of an odd block (block_depth 1) sit on voxel
// corners of nvblox's grid, so they are moved half a voxel onto its centres.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

#include "nvblox/core/types.h"
#include "nvblox/map/blox.h"

#ifndef OSMLOCBKI_MAX_CLASSES
#define OSMLOCBKI_MAX_CLASSES 9
#endif

namespace olb {

struct SemanticVoxel {
  static constexpr int kMaxClasses = OSMLOCBKI_MAX_CLASSES;
  __host__ __device__ SemanticVoxel() {
    for (int c = 0; c < kMaxClasses; ++c) ev[c] = 0.f;
  }
  float ev[kMaxClasses];  // class evidence, the prior excluded
};
using SemanticBlock = nvblox::VoxelBlock<SemanticVoxel>;

}  // namespace olb

namespace nvblox {

// recycled blocks start empty again (nvblox defines this per voxel type in its own library)
template <>
void initializeBlocksAsync<olb::SemanticBlock>(host_vector<olb::SemanticBlock*>& blocks, const CudaStream& cuda_stream,
                                              const MemoryType memory_type) {
  for (auto* b : blocks) olb::SemanticBlock::initAsync(b, memory_type, cuda_stream);
}

}  // namespace nvblox

#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh"
#include "nvblox/gpu_hash/internal/cuda/impl/gpu_layer_view_impl.cuh"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/mapper/mapper.h"
#include "nvblox/sensors/lidar.h"
#include "nvblox/sensors/pointcloud.h"
#include "osmlocbki/voxel_store.hpp"

namespace nvblox {
template class GPULayerView<olb::SemanticBlock>;
}

namespace olb {

namespace {

using SemanticLayer = nvblox::VoxelBlockLayer<SemanticVoxel>;
using SemanticHash = nvblox::Index3DDeviceHashMapType<SemanticBlock>;
constexpr int kSide = SemanticBlock::kVoxelsPerSide;
constexpr int kThreads = 256;

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + what + ": " + cudaGetErrorString(e));
}

// a device array, refilled per call
template <typename T>
class Buffer {
 public:
  ~Buffer() { cudaFree(p_); }
  T* upload(const T* host, size_t n, cudaStream_t s) {
    reserve(n);
    if (n) check(cudaMemcpyAsync(p_, host, n * sizeof(T), cudaMemcpyHostToDevice, s), "upload");
    return p_;
  }
  T* reserve(size_t n) {
    if (n > cap_) {
      cudaFree(p_);
      check(cudaMalloc(&p_, n * sizeof(T)), "cudaMalloc");
      cap_ = n;
    }
    return p_;
  }
  void download(T* host, size_t n, cudaStream_t s) const {
    if (n) check(cudaMemcpyAsync(host, p_, n * sizeof(T), cudaMemcpyDeviceToHost, s), "download");
  }

 private:
  T* p_ = nullptr;
  size_t cap_ = 0;
};

// the BKI block geometry, rounded as the CPU store rounds it
constexpr int kMaxCells = 8;  // block_depth <= 4
struct Grid {
  float res;            // voxel size
  double bki;           // BKI block size (blocks are picked in double, as on the CPU)
  float shift;          // map -> nvblox frame
  float block;          // nvblox block size
  int cells;            // voxels per BKI block side
  float off[kMaxCells]; // leaf offsets from a block's centre
};

__host__ __device__ inline int bki_block(float v, double bki) { return static_cast<int>(floor(v / bki + 0.5)); }

__device__ inline float sparse_kernel(float r) {
  constexpr float two_pi = 2.0f * 3.1415926f;
  return (2.0f + cosf(two_pi * r)) * (1.0f - r) / 3.0f + sinf(two_pi * r) / two_pi;
}

__constant__ int kFace[7][3] = {{0, 0, 0}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

__global__ void scatter_kernel(SemanticHash hash, const float4* pts, const float* rows, int n, int nc, float sf2, Grid g) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float4 q = pts[i];
  const nvblox::Vector3f x(q.x, q.y, q.z);
  const int b[3] = {bki_block(q.x, g.bki), bki_block(q.y, g.bki), bki_block(q.z, g.bki)};
  const float bs = static_cast<float>(g.bki);
  const float* row = rows + static_cast<long>(i) * nc;
  for (int f = 0; f < 7; ++f) {
    const nvblox::Vector3f c(static_cast<float>(b[0] + kFace[f][0]) * bs, static_cast<float>(b[1] + kFace[f][1]) * bs,
                             static_cast<float>(b[2] + kFace[f][2]) * bs);
    for (int k = 0; k < g.cells; ++k)
      for (int j = 0; j < g.cells; ++j)
        for (int l = 0; l < g.cells; ++l) {
          const nvblox::Vector3f leaf = c + nvblox::Vector3f(g.off[l], g.off[j], g.off[k]);
          const float r = (leaf - x).norm() * q.w;
          if (r >= 1.0f) continue;  // the kernel's compact support
          const float kv = sf2 * sparse_kernel(r);
          if (kv <= 0) continue;
          SemanticVoxel* v;
          if (!nvblox::getVoxelAtPosition(hash, (leaf.array() + g.shift).matrix(), g.block, &v)) continue;
          for (int cc = 0; cc < nc; ++cc) atomicAdd(&v->ev[cc], kv * row[cc]);
        }
  }
}

__global__ void deposit_kernel(SemanticHash hash, const float4* pts, const float* rows, int n, int nc, Grid g) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  SemanticVoxel* v;
  if (!nvblox::getVoxelAtPosition(hash, nvblox::Vector3f(pts[i].x + g.shift, pts[i].y + g.shift, pts[i].z + g.shift), g.block, &v))
    return;
  for (int cc = 0; cc < nc; ++cc) atomicAdd(&v->ev[cc], rows[static_cast<long>(i) * nc + cc]);
}

__device__ inline int best_of(const SemanticVoxel& v, int nc, float* mass) {
  int best = 0;
  float m = 0;
  for (int cc = 0; cc < nc; ++cc) {
    m += v.ev[cc];
    if (v.ev[cc] > v.ev[best]) best = cc;
  }
  *mass = m;
  return best;
}

__global__ void classify_kernel(SemanticHash hash, const float4* pts, int n, int nc, Grid g, int* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = 0;
  SemanticVoxel* v;
  if (!nvblox::getVoxelAtPosition(hash, nvblox::Vector3f(pts[i].x + g.shift, pts[i].y + g.shift, pts[i].z + g.shift), g.block, &v))
    return;
  float m;
  const int best = best_of(*v, nc, &m);
  if (v->ev[best] > 0) out[i] = best;
}

// one thread per voxel of the listed blocks: class, its probability and the mass (0: none, or
// masked as free space by the TSDF block alongside, when there is one)
__global__ void readout_kernel(SemanticBlock* const* blocks, const nvblox::TsdfBlock* const* tsdf, int nc, float prior,
                               float free_distance, int* cls, float* prob, float* mass) {
  const int b = blockIdx.x;
  const int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;
  const long o = static_cast<long>(b) * kSide * kSide * kSide + (x * kSide + y) * kSide + z;
  const SemanticVoxel& v = blocks[b]->voxels[x][y][z];
  float m;
  const int best = best_of(v, nc, &m);
  if (tsdf && tsdf[b]) {
    const nvblox::TsdfVoxel& t = tsdf[b]->voxels[x][y][z];
    if (t.weight > 0 && t.distance > free_distance) m = 0;
  }
  cls[o] = best;
  mass[o] = m;
  prob[o] = m > 0 ? (prior + v.ev[best]) / (nc * prior + m) : 0.f;
}

// the class colour into the colour layer, which nvblox's mesher paints the TSDF mesh with
__global__ void paint_kernel(SemanticBlock* const* blocks, nvblox::ColorBlock* const* color, int nc, const uchar3* palette) {
  const int b = blockIdx.x;
  const int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;
  float m;
  const int best = best_of(blocks[b]->voxels[x][y][z], nc, &m);
  if (!(m > 0)) return;
  nvblox::ColorVoxel& c = color[b]->voxels[x][y][z];
  c.color = nvblox::Color(palette[best].x, palette[best].y, palette[best].z);
  c.weight = 1.f;
}

class NvbloxStore : public VoxelStore {
 public:
  NvbloxStore(int nc, const StoreParams& p, const NvbloxParams& g)
      : nc_(nc),
        p_(p),
        g_(g),
        stream_(std::make_shared<nvblox::CudaStreamOwning>()),
        mapper_(static_cast<float>(p.resolution), nvblox::BlockMemoryPoolParams(), nvblox::ProjectiveLayerType::kTsdf, stream_),
        layer_(static_cast<float>(p.resolution), nvblox::BlockMemoryPoolParams()),
        lidar_(g.azimuth, g.elevation, 0.5f, static_cast<float>(radians(-g.min_elevation_deg)),
               static_cast<float>(radians(g.max_elevation_deg))) {
    if (nc_ > SemanticVoxel::kMaxClasses)
      throw std::invalid_argument("the nvblox store holds at most " + std::to_string(SemanticVoxel::kMaxClasses) +
                                  " classes (cmake -DOSMLOCBKI_MAX_CLASSES=N)");
    if (p_.block_depth < 1) throw std::invalid_argument("block_depth must be at least 1");
    if (p_.block_depth > 4) throw std::invalid_argument("the nvblox store takes block_depth <= 4");
    grid_.cells = 1 << (p_.block_depth - 1);
    grid_.res = static_cast<float>(p_.resolution);
    grid_.bki = grid_.cells * p_.resolution;
    for (int l = 0; l < grid_.cells; ++l) grid_.off[l] = static_cast<float>((l - (grid_.cells - 1) / 2.0) * p_.resolution);
    grid_.shift = grid_.cells % 2 ? 0.5f * grid_.res : 0.f;
    grid_.block = layer_.block_size();
    nvblox::MapperParams mp;
    mp.projective_integrator_params.lidar_projective_integrator_max_integration_distance_m = static_cast<float>(g_.max_range);
    mp.projective_integrator_params.projective_integrator_truncation_distance_vox = static_cast<float>(g_.truncation_vox);
    mapper_.setMapperParams(mp);
    lidar_.linear_interpolation_max_allowable_difference_vox(static_cast<float>(g_.interp_max_difference_vox));
    lidar_.nearest_interpolation_max_allowable_dist_to_ray_vox(static_cast<float>(g_.nearest_max_dist_vox));
  }

  bool has_geometry() const override { return true; }

  std::string memory() const override {
    auto& m = mapper();
    auto& sem = const_cast<SemanticLayer&>(layer_);
    auto mb = [](size_t b) { return static_cast<double>(b) / (1 << 20); };
    char b[256];
    std::snprintf(b, sizeof b, "GPU MB: semantic %.0f (%zu blocks), tsdf %.0f, color %.0f, mesh %.0f", mb(sem.numAllocatedBytes()),
                  sem.numAllocatedBlocks(), mb(m.tsdf_layer().numAllocatedBytes()), mb(m.color_layer().numAllocatedBytes()),
                  mb(m.color_mesh_layer().numAllocatedBytes()));
    return b;
  }

  void observe(const MatX3f& pts, const Eigen::Vector3f& origin) override {
    std::vector<nvblox::Vector3f> local;
    local.reserve(pts.rows());
    for (long i = 0; i < pts.rows(); ++i) {
      nvblox::Vector3f v = pts.row(i).transpose() - origin;
      if (v.allFinite()) local.push_back(v);
    }
    nvblox::Pointcloud cloud(nvblox::MemoryType::kDevice);
    cloud.copyPointsFromAsync(local, *stream_);
    nvblox::Transform T = nvblox::Transform::Identity();
    T.translation() = (origin.array() + grid_.shift).matrix();
    mapper_.integrateDepth(cloud, T, lidar_, false);
    // keep the mesh current a little at a time: meshing every pending block at once can take more
    // GPU memory than the map itself
    if (++unmeshed_ >= 10) mapper_.updateColorMesh(nvblox::UpdateFullLayer::kNo), unmeshed_ = 0;
  }

  void add_kernel(const std::vector<KernelPoint>& train, const MatXf& rows) override {
    const int n = static_cast<int>(train.size());
    if (!n) return;
    // allocate the nvblox blocks each point's extended BKI block touches
    std::vector<float4> pts(n);
    std::unordered_set<int64_t> need;
    for (int i = 0; i < n; ++i) {
      const auto& t = train[i];
      pts[i] = make_float4(t.x[0], t.x[1], t.x[2], t.inv_len);
      int lo[3], hi[3];
      for (int a = 0; a < 3; ++a) {
        const int b = bki_block(t.x[a], grid_.bki);
        const float bs = static_cast<float>(grid_.bki);
        lo[a] = block_of_voxel(voxel_of(static_cast<float>(b - 1) * bs + grid_.off[0]));
        hi[a] = block_of_voxel(voxel_of(static_cast<float>(b + 1) * bs + grid_.off[grid_.cells - 1]));
      }
      for (int i0 = lo[0]; i0 <= hi[0]; ++i0)
        for (int i1 = lo[1]; i1 <= hi[1]; ++i1)
          for (int i2 = lo[2]; i2 <= hi[2]; ++i2) need.insert(pack(i0, i1, i2));
    }
    allocate(need);
    const cudaStream_t s = *stream_;
    float4* dp = pts_.upload(pts.data(), n, s);
    float* dr = rows_.upload(rows.data(), static_cast<size_t>(n) * nc_, s);
    scatter_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, s>>>(hash(), dp, dr, n, nc_, static_cast<float>(p_.sf2), grid_);
    check(cudaGetLastError(), "scatter");
    stream_->synchronize();
  }

  void add_at(const MatX3f& at, const MatXf& rows) override {
    std::vector<float4> pts;
    std::vector<float> keep;
    std::unordered_set<int64_t> need;
    for (long i = 0; i < at.rows(); ++i) {
      if (!at.row(i).allFinite() || !(rows.row(i).sum() > 0)) continue;
      pts.push_back(make_float4(at(i, 0), at(i, 1), at(i, 2), 0.f));
      keep.insert(keep.end(), rows.row(i).data(), rows.row(i).data() + nc_);
      need.insert(pack(block_of_voxel(voxel_of(at(i, 0))), block_of_voxel(voxel_of(at(i, 1))), block_of_voxel(voxel_of(at(i, 2)))));
    }
    const int n = static_cast<int>(pts.size());
    if (!n) return;
    allocate(need);
    const cudaStream_t s = *stream_;
    float4* dp = pts_.upload(pts.data(), n, s);
    float* dr = rows_.upload(keep.data(), keep.size(), s);
    deposit_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, s>>>(hash(), dp, dr, n, nc_, grid_);
    check(cudaGetLastError(), "deposit");
    stream_->synchronize();
  }

  std::vector<int> classify(const MatX3f& at) const override {
    const int n = static_cast<int>(at.rows());
    std::vector<int> out(n, 0);
    if (!n || !layer_.numBlocks()) return out;
    std::vector<float4> pts(n);
    for (int i = 0; i < n; ++i) pts[i] = make_float4(at(i, 0), at(i, 1), at(i, 2), 0.f);
    const cudaStream_t s = *stream_;
    float4* dp = pts_.upload(pts.data(), n, s);
    int* dc = cls_.reserve(n);
    classify_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, s>>>(hash(), dp, n, nc_, grid_, dc);
    check(cudaGetLastError(), "classify");
    cls_.download(out.data(), n, s);
    stream_->synchronize();
    return out;
  }

  void voxels(MatX3f& xyz, std::vector<int>& cls, std::vector<float>& prob, std::vector<float>& mass,
              const Box* box) const override {
    std::vector<nvblox::Index3D> idx;
    for (const auto& b : layer_.getAllBlockIndices()) {
      if (box) {
        const float cx = (b.x() + 0.5f) * grid_.block - grid_.shift, cy = (b.y() + 0.5f) * grid_.block - grid_.shift;
        const float h = 0.5f * grid_.block;
        if (cx + h < (*box)[0] || cy + h < (*box)[1] || cx - h > (*box)[2] || cy - h > (*box)[3]) continue;
      }
      idx.push_back(b);
    }
    // a chunk of blocks at a time: the readout's buffers stay small however large the map
    constexpr int kChunk = 4096;
    constexpr long kPer = kSide * kSide * kSide;
    std::vector<Eigen::Vector3f> at;
    cls.clear(), prob.clear(), mass.clear();
    std::vector<SemanticBlock*> bp;
    std::vector<const nvblox::TsdfBlock*> tp;
    std::vector<int> c(kChunk * kPer);
    std::vector<float> pr(kChunk * kPer), m(kChunk * kPer);
    const cudaStream_t s = *stream_;
    for (size_t first = 0; first < idx.size(); first += kChunk) {
      const int nb = static_cast<int>(std::min<size_t>(kChunk, idx.size() - first));
      const long nv = nb * kPer;
      bp.assign(nb, nullptr), tp.assign(nb, nullptr);
      for (int i = 0; i < nb; ++i) {
        bp[i] = const_cast<SemanticBlock*>(layer_.getBlockAtIndex(idx[first + i]).get());
        if (g_.free_space_mask) tp[i] = mapper_.tsdf_layer().getBlockAtIndex(idx[first + i]).get();
      }
      SemanticBlock** db = blocks_.upload(bp.data(), nb, s);
      const nvblox::TsdfBlock** dt = g_.free_space_mask ? tsdf_.upload(tp.data(), nb, s) : nullptr;
      int* dc = cls_.reserve(nv);
      float* dpr = prob_.reserve(nv);
      float* dm = mass_.reserve(nv);
      readout_kernel<<<nb, dim3(kSide, kSide, kSide), 0, s>>>(db, dt, nc_, static_cast<float>(p_.prior),
                                                              static_cast<float>(g_.free_distance), dc, dpr, dm);
      check(cudaGetLastError(), "readout");
      cls_.download(c.data(), nv, s), prob_.download(pr.data(), nv, s), mass_.download(m.data(), nv, s);
      stream_->synchronize();
      for (int i = 0; i < nb; ++i) {
        const nvblox::Index3D& b = idx[first + i];
        for (int x = 0; x < kSide; ++x)
          for (int y = 0; y < kSide; ++y)
            for (int z = 0; z < kSide; ++z) {
              const long o = i * kPer + (x * kSide + y) * kSide + z;
              if (!(m[o] > 0)) continue;
              const Eigen::Vector3f centre =
                  (Eigen::Vector3f(b.x() * kSide + x, b.y() * kSide + y, b.z() * kSide + z).array() + 0.5f) * grid_.res - grid_.shift;
              if (box && (centre[0] < (*box)[0] || centre[1] < (*box)[1] || centre[0] > (*box)[2] || centre[1] > (*box)[3]))
                continue;
              at.push_back(centre);
              cls.push_back(c[o]), prob.push_back(pr[o]), mass.push_back(m[o]);
            }
      }
    }
    xyz.resize(at.size(), 3);
    for (size_t i = 0; i < at.size(); ++i) xyz.row(i) = at[i].transpose();
  }

  void save_mesh(const std::string& path, const std::vector<std::array<uint8_t, 3>>& colors) const override {
    remesh(colors, true);
    if (!mapper().saveColorMeshAsPly(path)) throw std::runtime_error("could not write " + path);
  }

  void mesh(MatX3f& vertices, MatX3i& triangles, MatX3u8& rgb, const std::vector<std::array<uint8_t, 3>>& colors,
            bool full) const override {
    remesh(colors, full);
    const auto cm = mapper().color_mesh_layer().getMesh(*stream_);
    const auto v = cm->vertices.toVectorAsync(*stream_);
    const auto c = cm->vertex_appearances.toVectorAsync(*stream_);
    const auto f = cm->triangles.toVectorAsync(*stream_);
    stream_->synchronize();
    vertices.resize(static_cast<long>(v.size()), 3);
    rgb.resize(static_cast<long>(v.size()), 3);
    for (size_t i = 0; i < v.size(); ++i) {
      vertices.row(i) = (v[i].array() - grid_.shift).matrix().transpose();
      if (i < c.size()) rgb.row(i) << c[i].r(), c[i].g(), c[i].b();
    }
    triangles = Eigen::Map<const MatX3i>(f.data(), static_cast<long>(f.size() / 3), 3);
  }


 private:
  nvblox::Mapper& mapper() const { return const_cast<nvblox::Mapper&>(mapper_); }

  // the class colour into nvblox's colour layer where the TSDF and the semantics both have a block,
  // then the mesh: of the blocks the TSDF changed since the last call, or of all of them
  void remesh(const std::vector<std::array<uint8_t, 3>>& colors, bool full) const {
    if (static_cast<int>(colors.size()) < nc_) throw std::invalid_argument("mesh: one colour per class");
    auto& m = mapper();
    std::vector<SemanticBlock*> bp;
    std::vector<nvblox::ColorBlock*> cp;
    for (const auto& b : layer_.getAllBlockIndices()) {
      if (!m.tsdf_layer().isBlockAllocated(b)) continue;
      bp.push_back(const_cast<SemanticBlock*>(layer_.getBlockAtIndex(b).get()));
      cp.push_back(m.color_layer().allocateBlockAtIndexAsync(b, *stream_).get());
    }
    std::vector<uchar3> pal(nc_);
    for (int c = 0; c < nc_; ++c) pal[c] = make_uchar3(colors[c][0], colors[c][1], colors[c][2]);
    const cudaStream_t s = *stream_;
    if (!bp.empty()) {
      SemanticBlock** db = blocks_.upload(bp.data(), bp.size(), s);
      paint_kernel<<<static_cast<int>(bp.size()), dim3(kSide, kSide, kSide), 0, s>>>(db, color_.upload(cp.data(), cp.size(), s),
                                                                                     nc_, palette_.upload(pal.data(), nc_, s));
      check(cudaGetLastError(), "paint");
      stream_->synchronize();
    }
    m.updateColorMesh(nvblox::UpdateFullLayer::kNo);  // what the scans changed since
    unmeshed_ = 0;
    if (!full) return;
    // all of it, a batch of blocks at a time (meshing every block at once runs out of GPU memory)
    constexpr size_t kBatch = 2048;
    const auto all = m.tsdf_layer().getAllBlockIndices();
    for (size_t i = 0; i < all.size(); i += kBatch) {
      m.markBlocksForUpdate(std::vector<nvblox::Index3D>(all.begin() + i, all.begin() + std::min(all.size(), i + kBatch)));
      m.updateColorMesh(nvblox::UpdateFullLayer::kNo);
    }
  }

  static int64_t pack(int64_t i, int64_t j, int64_t k) {
    constexpr int64_t bias = int64_t(1) << 20;
    return ((i + bias) << 42) | ((j + bias) << 21) | (k + bias);
  }
  static nvblox::Index3D unpack(int64_t key) {
    constexpr int64_t bias = int64_t(1) << 20, mask = (int64_t(1) << 21) - 1;
    return nvblox::Index3D(static_cast<int>(((key >> 42) & mask) - bias), static_cast<int>(((key >> 21) & mask) - bias),
                           static_cast<int>((key & mask) - bias));
  }
  int voxel_of(float x) const { return static_cast<int>(std::floor((x + grid_.shift) / grid_.res)); }
  static int block_of_voxel(int v) { return v >= 0 ? v / kSide : -((-v + kSide - 1) / kSide); }

  void allocate(const std::unordered_set<int64_t>& keys) {
    std::vector<nvblox::Index3D> idx;
    idx.reserve(keys.size());
    for (int64_t k : keys) idx.push_back(unpack(k));
    layer_.allocateBlocksAtIndices(idx, *stream_);
  }
  SemanticHash hash() const { return layer_.getGpuLayerView(*stream_).getHash().impl_; }

  int nc_;
  StoreParams p_;
  NvbloxParams g_;
  Grid grid_;
  std::shared_ptr<nvblox::CudaStream> stream_;
  nvblox::Mapper mapper_;
  SemanticLayer layer_;
  nvblox::Lidar lidar_;
  mutable Buffer<float4> pts_;
  mutable Buffer<float> rows_, prob_, mass_;
  mutable Buffer<int> cls_;
  mutable Buffer<SemanticBlock*> blocks_;
  mutable Buffer<const nvblox::TsdfBlock*> tsdf_;
  mutable Buffer<nvblox::ColorBlock*> color_;
  mutable Buffer<uchar3> palette_;
  mutable int unmeshed_ = 0;  // scans integrated since the mesh was last updated
};

}  // namespace

bool have_nvblox() { return true; }

std::unique_ptr<VoxelStore> make_nvblox_store(int num_class, const StoreParams& p, const NvbloxParams& g) {
  return std::make_unique<NvbloxStore>(num_class, p, g);
}

}  // namespace olb
