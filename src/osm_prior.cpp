#include "osmlocbki/osm_prior.hpp"

#include <yaml-cpp/yaml.h>

#include <limits>
#include <stdexcept>

#include "osmlocbki/labels.hpp"

namespace olb {

OsmPrior::OsmPrior(const OsmMap& map, const OsmParams& p, const MatXf& confusion, std::vector<int8_t> roles)
    : p_(p), m_(confusion), roles_(std::move(roles)) {
  if (m_.cols() != kNumOsmColumns || static_cast<long>(roles_.size()) != m_.rows())
    throw std::invalid_argument("OsmPrior: confusion must be (classes, OSM classes + 1), one ground role per class");
  shapes_ = map.features;
  for (const auto& b : map.buildings) shapes_.push_back(Shape{BUILDINGS, b, {}, 0.0});
  std::vector<Box> boxes;
  for (const auto& s : shapes_) {
    double r = s.is_area() ? 0.0 : 0.5 * (s.width > 0 ? s.width : p_.width[s.cls]);
    radius_.push_back(r);
    boxes.push_back(bounds(s.is_area() ? s.area.outer : s.path, r + p_.decay_m));
  }
  grid_ = BoxGrid(std::move(boxes), 50.0);
}

void OsmPrior::support(const Vec2& p, float* w) const {
  std::array<double, kNumOsmClasses> d;
  d.fill(std::numeric_limits<double>::infinity());
  grid_.visit(p, [&](int k) { d[shapes_[k].cls] = std::min(d[shapes_[k].cls], signed_distance(shapes_[k], p, radius_[k])); });
  float most = 0;
  for (int c = 0; c < kNumOsmClasses; ++c) {
    if (d[c] <= 0)
      w[c] = 1.f;
    else
      w[c] = p_.decay_m > 0 ? static_cast<float>(std::max(0.0, 1.0 - d[c] / p_.decay_m)) : 0.f;
    most = std::max(most, w[c]);
  }
  w[kNumOsmClasses] = 1.f - most;
}

void OsmPrior::project(const Vec2& p, int ground, float* s) const {
  Eigen::Matrix<float, kNumOsmColumns, 1> w;
  support(p, w.data());
  Eigen::Map<Eigen::VectorXf>(s, m_.rows()) = m_ * w;
  if (ground < 0) return;
  for (long c = 0; c < m_.rows(); ++c)
    if ((roles_[c] == ROLE_GROUND && ground != 1) || (roles_[c] == ROLE_NONGROUND && ground == 1)) s[c] = 0.f;
}

MatXf OsmPrior::load_confusion(const std::string& path, int num_class) {
  YAML::Node cm = YAML::LoadFile(path)["confusion_matrix"];
  if (!cm) throw std::runtime_error(path + ": no confusion_matrix");
  MatXf m = MatXf::Zero(num_class, kNumOsmColumns);
  for (const auto& kv : cm) {
    int c = kv.first.as<int>();
    auto row = kv.second.as<std::vector<float>>();
    if (c <= 0 || c >= num_class) continue;  // class 0 (unlabeled / free) takes no OSM support
    if (row.size() > static_cast<size_t>(kNumOsmColumns)) throw std::runtime_error(path + ": a row is too long");
    for (size_t o = 0; o < row.size(); ++o) m(c, static_cast<long>(o)) = row[o];
  }
  return m;
}

}  // namespace olb
