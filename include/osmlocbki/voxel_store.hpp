// Where a semantic map's class evidence lives, and how BKI's kernel spreads it there: on the CPU in
// OSM-BKI's own block layout, or on the GPU in nvblox voxel blocks next to nvblox's TSDF of the same
// scans (free space, meshes).
//
// Both keep the BKI support rule of OSM-BKI: voxels are grouped into blocks of `cells` a side
// (2^(block_depth - 1)), and a point reaches the voxels of its own block and of its six face
// neighbours only, where its kernel is non-zero.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "osmlocbki/common.hpp"
#include "osmlocbki/geo.hpp"

namespace olb {

struct StoreParams {
  double resolution = 0.3;  // voxel size, m
  int block_depth = 2;      // BKI blocks of 2^(block_depth - 1) voxels a side
  double sf2 = 1.0;         // kernel scale
  double prior = 1.0;       // Dirichlet pseudo-count per class
};

// a training point: where it is and the inverse of its kernel length
struct KernelPoint {
  Eigen::Vector3f x;
  float inv_len;
};

// nvblox's geometry alongside the semantics (config/mapping.yaml nvblox:)
struct NvbloxParams {
  int azimuth = 512, elevation = 128;               // the LiDAR's range image
  double min_elevation_deg = -22.5, max_elevation_deg = 22.5;
  double max_range = 60.0;                          // TSDF integration distance, m
  double truncation_vox = 4.0;                      // TSDF truncation, voxels
  // report a semantic voxel only where the TSDF does not see free space: no BKI smear into the air
  // around surfaces, and no trails where objects moved
  bool free_space_mask = true;
  double free_distance = 0.45;                      // m: a TSDF distance beyond this is free space
};

class VoxelStore {
 public:
  virtual ~VoxelStore() = default;
  // BKI: each point adds sf2 * k(|v - x| * inv_len) * row to each voxel v it reaches (rows: N, C)
  virtual void add_kernel(const std::vector<KernelPoint>& pts, const MatXf& rows) = 0;
  // rows (N, C) straight into the voxels holding the points (N, 3)
  virtual void add_at(const MatX3f& pts, const MatXf& rows) = 0;
  // the most likely class of the voxel holding each point (0: no evidence)
  virtual std::vector<int> classify(const MatX3f& pts) const = 0;
  // every voxel with evidence (centre within `box` if given): centre, class, its probability, mass
  virtual void voxels(MatX3f& xyz, std::vector<int>& cls, std::vector<float>& prob, std::vector<float>& mass,
                      const Box* box) const = 0;
  // a scan's geometry: points in the map frame and the sensor's position (no-op without geometry)
  virtual void observe(const MatX3f& /*pts*/, const Eigen::Vector3f& /*origin*/) {}
  virtual bool has_geometry() const { return false; }
  // the surface mesh, coloured by class (colors: one RGB per class), as PLY
  virtual void save_mesh(const std::string& path, const std::vector<std::array<uint8_t, 3>>& colors) const;
};

std::unique_ptr<VoxelStore> make_cpu_store(int num_class, const StoreParams& p);
// throws when built without nvblox (cmake -DOSMLOCBKI_NVBLOX=ON, see the README)
std::unique_ptr<VoxelStore> make_nvblox_store(int num_class, const StoreParams& p, const NvbloxParams& g);
bool have_nvblox();

}  // namespace olb
