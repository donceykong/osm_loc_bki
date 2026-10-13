// The common taxonomy the semantic map is built in (config/labels.yaml): class names and colours,
// where each class may sit relative to the ground, and how each segmentation's raw label ids map
// onto it. Class 0 (unlabeled) doubles as free space and never receives evidence.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "osmlocbki/common.hpp"

namespace olb {

// Where a class may occur relative to the ground surface (drives the OSM ground gate)
enum GroundRole : int8_t { ROLE_BOTH = 0, ROLE_GROUND = 1, ROLE_NONGROUND = 2 };

struct Taxonomy {
  std::vector<std::string> names;
  std::vector<std::array<uint8_t, 3>> colors;
  std::vector<int8_t> roles;  // GroundRole per class
  int size() const { return static_cast<int>(names.size()); }
  static Taxonomy load(const std::string& path);  // default: config/labels.yaml
  static Taxonomy load();
  // raw label id -> common class for one label set (mcd, kitti360, semkitti); unmapped ids -> 0
  std::vector<int> mapping(const std::string& label_set) const;
  // the localizer's building label ids for a label set: localization_buildings, else every id the
  // set maps to the common class "building"
  std::vector<int> building_ids(const std::string& label_set) const;

 private:
  std::string path_;
};

// One scan's semantic input to the map, per point: common-class probabilities (N, C) and EDL
// vacuity. A class's probability is the sum over the raw classes merged into it; raw classes
// with no counterpart (-> class 0) deposit nothing, so rows may sum to less than 1.
struct Semantics {
  MatXf probs;
  std::vector<float> vacuity;
};

// Dirichlet outputs (N, K) alphas: p = alpha / sum(alpha), vacuity = K / sum(alpha)
Semantics from_alpha(const float* alpha, long n, int k, const std::vector<int>& to_common, int num_class);
// hard labels: one-hot rows, vacuity 0
Semantics from_labels(const int32_t* labels, long n, const std::vector<int>& to_common, int num_class);

}  // namespace olb
