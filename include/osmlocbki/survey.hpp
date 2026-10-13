// An airborne survey of the site (built by osm_localization from LAS/LAZ tiles or DSM/DEM rasters):
// a terrain model (DEM) and the surveyed points (airborne returns plus wall points synthesised at
// roof edges), in the map's UTM zone, heights in the survey's datum. The pipeline projects the
// survey around each mapped pose into the semantic map, so roofs, canopy and gaps the vehicle's
// LiDAR never sees get voxels too.
#pragma once

#include <string>
#include <vector>

#include "osmlocbki/common.hpp"

namespace olb {

class Survey {
 public:
  // data/surface/<map>_epsg<zone>.npz: origin, res, dem (nx, ny), pts (N, 3) relative to origin,
  // sorted into kCell bins (bins, starts)
  explicit Survey(const std::string& path);
  // terrain height at a UTM position (the nearest cell, clamped to the model)
  float terrain(const Vec2& utm) const;
  // the surveyed points within `radius` of a UTM position: (N, 3) UTM xy, survey height
  MatX3 crop(const Vec2& utm, double radius) const;
  int epsg() const { return epsg_; }

 private:
  static constexpr double kCell = 25.0;  // crop bins of the model, m
  Vec2 origin_;
  double res_;
  int epsg_;
  long nx_, ny_, bins_[2];
  std::vector<float> dem_, pts_;
  std::vector<int64_t> starts_;
};

}  // namespace olb
