// The OSM-BKI semantic map: Bayesian kernel inference over voxels, each a Dirichlet over the
// common classes. Every voxel starts at `prior` per class; a scan's points (voxel-downsampled)
// deposit their class probabilities p through the sparse kernel, with OSM boosting each point's
// count where it supports a class:
//
//     y_c = p_c + strength * s_c * (1 - p_c)        s = OsmPrior::project at the point
//     alpha_c(voxel) += sum_points sf2 * k(|voxel - x| / l) * y_c
//
// with the per-point adaptive length l = ell * a * exp(1 + b * u) (u: the point's vacuity) when
// enabled. Voxels live in blocks of 2^(block_depth - 1) cells a side, and a block takes evidence
// from the points in itself and its six face neighbours only (the extended block of BKI octomaps),
// which bounds the kernel's reach. The evidence is kept by a VoxelStore: on the CPU, or on the GPU
// in nvblox next to a TSDF of the same scans (voxel_store.hpp).
//
// The map frame is local (float precision): map = UTM - utm_origin.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "osmlocbki/labels.hpp"
#include "osmlocbki/osm_prior.hpp"
#include "osmlocbki/voxel_store.hpp"

namespace olb {

struct MapParams {
  double resolution = 0.3;
  int block_depth = 2;
  double ell = 1.0, sf2 = 1.0, prior = 1.0;
  bool adaptive = true;  // ell * a * exp(1 + b * u) per point
  double adapt_a = 1.0, adapt_b = -3.0;
  double ds_resolution = 0.1;  // training points: mean of each such voxel (<= 0: every point)
  double max_range = 200.0;    // from the sensor origin
  double strength = 0.1;       // OSM strength (lambda); 0 = no OSM
  std::string backend = "cpu";  // where the voxels live: cpu, or nvblox (GPU; adds a TSDF)
  NvbloxParams nvblox;
};

class SemanticMap {
 public:
  SemanticMap(int num_class, const MapParams& p, std::shared_ptr<const OsmPrior> osm = nullptr,
              const Vec2& utm_origin = Vec2::Zero());
  // one scan in the map frame: points (N, 3), their semantics, Patchwork++ verdicts (N: 1 ground,
  // 0 non-ground, -1 unknown; empty = none) and the sensor origin (all points: the geometry)
  void insert(const MatX3f& pts, const Semantics& sem, const std::vector<int8_t>& ground, const Vec3& origin);
  // add evidence rows (N, C) straight to the voxels holding the points (N, 3), no kernel: how
  // evidence from outside the scans (an airborne survey) enters the map
  void deposit(const MatX3f& pts, const MatXf& rows);
  // the class of the voxel holding each point (0: free or never observed)
  std::vector<int> classify(const MatX3f& pts) const;
  // every voxel with evidence (within `box`, map frame, if given): centre, class, that class's
  // probability, evidence mass
  void voxels(MatX3f& xyz, std::vector<int>& cls, std::vector<float>& prob, std::vector<float>& mass,
              const Box* box = nullptr) const;
  // the surface mesh coloured by class (nvblox backend), as PLY
  void save_mesh(const std::string& path, const std::vector<std::array<uint8_t, 3>>& colors) const;
  void mesh(MatX3f& vertices, MatX3i& triangles, MatX3u8& rgb, const std::vector<std::array<uint8_t, 3>>& colors,
            bool full) const {
    store_->mesh(vertices, triangles, rgb, colors, full);
  }
  bool has_geometry() const { return store_->has_geometry(); }
  std::string memory() const { return store_->memory(); }
  // a scan's geometry only (map frame, sensor origin): the nvblox backend's TSDF
  void observe(const MatX3f& pts, const Vec3& origin) { store_->observe(pts, origin.cast<float>()); }
  const Vec2& utm_origin() const { return origin_utm_; }
  const MapParams& params() const { return p_; }

 private:
  int nc_;
  MapParams p_;
  std::shared_ptr<const OsmPrior> osm_;
  Vec2 origin_utm_;
  std::unique_ptr<VoxelStore> store_;
};

}  // namespace olb
