// What OSM says about a point (OSM-BKI): its distance d to each OSM class becomes a support
// w = max(0, 1 - d / decay_m) (1 inside), plus "none" = 1 - max(w); the confusion matrix M
// (P(common class | OSM class), C x (O + 1)) projects w onto the common classes, s = M w; and the
// Patchwork++ ground gate drops every class whose ground role contradicts the point's verdict.
// Lines and points (roads, fences, single trees, ...) are bands of their width.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "osmlocbki/geo.hpp"

namespace olb {

constexpr int kNumOsmColumns = kNumOsmClasses + 1;  // the classes and "none"

struct OsmParams {
  double decay_m = 2.0;  // falloff past a shape's edge
  // band width of lines and points without a width tag, per OSM class (m)
  std::array<double, kNumOsmClasses> width{};
};

class OsmPrior {
 public:
  // `confusion`: (C, O + 1) with row 0 (unlabeled / free) zero; `roles`: GroundRole per class
  OsmPrior(const OsmMap& map, const OsmParams& p, const MatXf& confusion, std::vector<int8_t> roles);
  // (O + 1) supports at p
  void support(const Vec2& p, float* w) const;
  // (C) gated class support at p; ground: 1 ground, 0 non-ground, -1 unknown (no gating)
  void project(const Vec2& p, int ground, float* s) const;
  int num_class() const { return static_cast<int>(m_.rows()); }

  // a confusion matrix file: {confusion_matrix: {class id: [P(class | roads), ... fences(, none)]}}
  static MatXf load_confusion(const std::string& path, int num_class);

 private:
  OsmParams p_;
  std::vector<Shape> shapes_;
  std::vector<double> radius_;  // band half-width per shape (0 for areas)
  BoxGrid grid_;                // shapes' boxes grown by their reach
  MatXf m_;
  std::vector<int8_t> roles_;
};

}  // namespace olb
