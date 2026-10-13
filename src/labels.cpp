#include "osmlocbki/labels.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <stdexcept>

#include "osmlocbki/site.hpp"

namespace olb {

Taxonomy Taxonomy::load() { return load(config_dir() + "/labels.yaml"); }

Taxonomy Taxonomy::load(const std::string& path) {
  YAML::Node y = YAML::LoadFile(path);
  Taxonomy t;
  t.path_ = path;
  for (const auto& c : y["classes"]) {
    t.names.push_back(c["name"].as<std::string>());
    auto rgb = c["color"].as<std::vector<int>>();
    t.colors.push_back({static_cast<uint8_t>(rgb.at(0)), static_cast<uint8_t>(rgb.at(1)), static_cast<uint8_t>(rgb.at(2))});
    std::string r = c["ground"] ? c["ground"].as<std::string>() : "both";
    if (r != "both" && r != "ground" && r != "nonground") throw std::runtime_error(path + ": ground must be both, ground or nonground");
    t.roles.push_back(r == "ground" ? ROLE_GROUND : r == "nonground" ? ROLE_NONGROUND : ROLE_BOTH);
  }
  return t;
}

std::vector<int> Taxonomy::mapping(const std::string& label_set) const {
  YAML::Node m = YAML::LoadFile(path_)["label_sets"][label_set];
  if (!m) throw std::runtime_error(path_ + ": no label set '" + label_set + "'");
  std::vector<int> out;
  for (const auto& kv : m) {
    int raw = kv.first.as<int>(), c = kv.second.as<int>();
    if (c < 0 || c >= size()) throw std::runtime_error(path_ + ": label set " + label_set + " maps to class " + std::to_string(c));
    if (c == 0) continue;  // unmapped is the default
    if (raw >= static_cast<int>(out.size())) out.resize(raw + 1, 0);
    out[raw] = c;
  }
  return out;
}

std::vector<int> Taxonomy::building_ids(const std::string& label_set) const {
  YAML::Node listed = YAML::LoadFile(path_)["localization_buildings"][label_set];
  if (listed) return listed.as<std::vector<int>>();
  auto c = std::find(names.begin(), names.end(), "building");
  if (c == names.end()) throw std::runtime_error(path_ + ": no class 'building'");
  auto m = mapping(label_set);
  std::vector<int> out;
  for (int raw = 0; raw < static_cast<int>(m.size()); ++raw)
    if (m[raw] == c - names.begin()) out.push_back(raw);
  if (out.empty()) throw std::runtime_error(path_ + ": label set " + label_set + " has no building class");
  return out;
}

Semantics from_alpha(const float* alpha, long n, int k, const std::vector<int>& to_common, int num_class) {
  Semantics s{MatXf::Zero(n, num_class), std::vector<float>(n)};
  for (long i = 0; i < n; ++i) {
    const float* a = alpha + i * k;
    float sum = 0;
    for (int j = 0; j < k; ++j) sum += a[j];
    if (sum < 1e-10f) sum = 1.f;
    s.vacuity[i] = std::min(1.0f, static_cast<float>(k) / sum);
    for (int j = 0; j < k && j < static_cast<int>(to_common.size()); ++j)
      if (to_common[j] > 0) s.probs(i, to_common[j]) += a[j] / sum;
  }
  return s;
}

Semantics from_labels(const int32_t* labels, long n, const std::vector<int>& to_common, int num_class) {
  Semantics s{MatXf::Zero(n, num_class), std::vector<float>(n, 0.f)};
  for (long i = 0; i < n; ++i) {
    int l = labels[i];
    if (l >= 0 && l < static_cast<int>(to_common.size()) && to_common[l] > 0) s.probs(i, to_common[l]) = 1.f;
  }
  return s;
}

}  // namespace olb
