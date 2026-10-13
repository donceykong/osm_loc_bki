// Per-point ground / non-ground verdicts from Patchwork++ (a piecewise ground surface per scan),
// the gate OSM-BKI puts on OSM support. One segmenter per sequence: Patchwork++ adapts its
// elevation and flatness thresholds across scans.
#pragma once

#include <memory>
#include <vector>

#include "osmlocbki/common.hpp"

namespace patchwork {
class PatchWorkpp;
}

namespace olb {

struct GroundParams {
  double sensor_height = 1.73;
  double min_range = 0.0, max_range = 200.0;  // outside (min, max] a point gets no verdict
  bool reflected_noise_removal = false;       // needs a calibrated intensity channel
};

class GroundSegmenter {
 public:
  explicit GroundSegmenter(const GroundParams& p);
  ~GroundSegmenter();
  // a levelled scan (sensor at the origin, z up; (N, 3), or (N, 4) with intensity) -> per point
  // 1 ground, 0 non-ground, -1 unknown (outside the range annulus, or removed as noise)
  std::vector<int8_t> segment(const MatXf& pts);

 private:
  GroundParams p_;
  std::unique_ptr<patchwork::PatchWorkpp> pw_;
};

}  // namespace olb
