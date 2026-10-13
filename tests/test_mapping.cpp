// Mapping: OSM support, its projection and ground gate, the semantic map's update, and the map
// against the original OSM-BKI implementation (tests/data/bki_ref.npz, made by tests/reference/).
#include <filesystem>
#include <map>

#include "check.hpp"
#include "osmlocbki/npz.hpp"
#include "osmlocbki/semantic_map.hpp"

using namespace olb;
using olb_test::data;

namespace {

constexpr int kRoad = 1, kBuilding = 4, kVegetation = 6, C = 9;

// a 10 m square building with a 4 m courtyard, a road along y = 20 (4 m wide), a single tree at (30, 0)
OsmMap toy_map() {
  OsmMap m;
  m.buildings.push_back({{Vec2(0, 0), Vec2(10, 0), Vec2(10, 10), Vec2(0, 10)}, {{Vec2(3, 3), Vec2(7, 3), Vec2(7, 7), Vec2(3, 7)}}});
  Shape road;
  road.cls = ROADS, road.path = {Vec2(0, 20), Vec2(10, 20)};
  Shape tree;
  tree.cls = TREES, tree.path = {Vec2(30, 0)};
  m.features = {road, tree};
  return m;
}

OsmParams toy_params() {
  OsmParams p;
  p.decay_m = 2.0;
  p.width[ROADS] = 4.0, p.width[TREES] = 10.0;
  return p;
}

// P(road | roads) = P(building | buildings) = P(vegetation | trees) = 1
MatXf toy_confusion() {
  MatXf m = MatXf::Zero(C, kNumOsmColumns);
  m(kRoad, ROADS) = m(kBuilding, BUILDINGS) = m(kVegetation, TREES) = 1;
  return m;
}

std::vector<int8_t> roles() {
  std::vector<int8_t> r(C, ROLE_BOTH);
  r[kRoad] = ROLE_GROUND, r[kBuilding] = ROLE_NONGROUND, r[kVegetation] = ROLE_NONGROUND;
  return r;
}

std::array<float, kNumOsmColumns> support(const OsmPrior& p, double x, double y) {
  std::array<float, kNumOsmColumns> w;
  p.support(Vec2(x, y), w.data());
  return w;
}

}  // namespace

TEST(osm_support_falls_off_past_each_shape) {
  OsmPrior p(toy_map(), toy_params(), toy_confusion(), roles());
  CHECK_NEAR(support(p, 1, 1)[BUILDINGS], 1.0, 1e-6);    // inside
  CHECK_NEAR(support(p, 1, 1)[kNumOsmClasses], 0.0, 1e-6);  // "none"
  CHECK_NEAR(support(p, 11, 5)[BUILDINGS], 0.5, 1e-6);  // 1 m outside: half
  CHECK_NEAR(support(p, 5, 5)[BUILDINGS], 0.0, 1e-6);   // courtyard centre, 2 m from the walls
  CHECK_NEAR(support(p, 5, 4)[BUILDINGS], 0.5, 1e-6);   // courtyard, 1 m from a wall
  CHECK_NEAR(support(p, 5, 21)[ROADS], 1.0, 1e-6);      // within the road's half-width
  CHECK_NEAR(support(p, 5, 23)[ROADS], 0.5, 1e-6);      // 1 m past its edge
  CHECK_NEAR(support(p, 36, 0)[TREES], 0.5, 1e-6);      // 1 m past the tree's 5 m radius
  auto far = support(p, 100, 100);
  for (int c = 0; c < kNumOsmClasses; ++c) CHECK(far[c] == 0.f);
  CHECK_NEAR(far[kNumOsmClasses], 1.0, 1e-6);
}

TEST(osm_projection_is_gated_by_ground) {
  OsmPrior p(toy_map(), toy_params(), toy_confusion(), roles());
  std::vector<float> s(C);
  p.project(Vec2(1, 1), -1, s.data());  // unknown: no gate
  CHECK_NEAR(s[kBuilding], 1.0, 1e-6);
  p.project(Vec2(1, 1), 0, s.data());   // non-ground: building stays
  CHECK_NEAR(s[kBuilding], 1.0, 1e-6);
  p.project(Vec2(1, 1), 1, s.data());   // ground: building is dropped
  CHECK_NEAR(s[kBuilding], 0.0, 1e-6);
  p.project(Vec2(5, 21), 0, s.data());  // a road at a non-ground point: dropped
  CHECK_NEAR(s[kRoad], 0.0, 1e-6);
  p.project(Vec2(5, 21), 1, s.data());
  CHECK_NEAR(s[kRoad], 1.0, 1e-6);
  CHECK(s[0] == 0.f);
}

TEST(map_deposits_through_the_sparse_kernel) {
  MapParams mp;
  mp.adaptive = false, mp.ds_resolution = -1, mp.ell = 1.0, mp.strength = 0;
  SemanticMap map(C, mp);
  MatX3f pts(1, 3);
  pts << 0.15f, 0.15f, 0.15f;  // a leaf centre (resolution 0.3, block depth 2)
  Semantics sem{MatXf::Zero(1, C), {0.f}};
  sem.probs(0, kRoad) = 1.f;
  map.insert(pts, sem, {}, Vec3::Zero());
  MatX3f xyz;
  std::vector<int> cls;
  std::vector<float> prob, mass;
  map.voxels(xyz, cls, prob, mass);
  auto k = [](double r) { return (2 + std::cos(2 * kPi * r)) * (1 - r) / 3 + std::sin(2 * kPi * r) / (2 * kPi); };
  bool found = false;
  for (long i = 0; i < xyz.rows(); ++i) {
    double r = (xyz.row(i) - pts.row(0)).norm();
    CHECK(r < 1.0);       // nothing past the kernel's support
    CHECK(cls[i] == kRoad);
    CHECK_NEAR(mass[i], k(r), 1e-5);
    if (r < 1e-6) {
      found = true;
      CHECK_NEAR(prob[i], 2.0 / 10.0, 1e-6);  // (prior 1 + 1) / (9 + 1)
    }
  }
  CHECK(found);
  CHECK(map.classify(pts)[0] == kRoad);
  MatX3f far(1, 3);
  far << 50.f, 0.f, 0.f;
  CHECK(map.classify(far)[0] == 0);
}

TEST(osm_boosts_the_classes_it_supports) {
  // a point inside the building, unsure between building and vegetation: OSM closes part of the
  // building's gap, y = p + strength * s * (1 - p)
  MapParams mp;
  mp.adaptive = false, mp.ds_resolution = -1, mp.strength = 0.5;
  auto prior = std::make_shared<OsmPrior>(toy_map(), toy_params(), toy_confusion(), roles());
  SemanticMap with(C, mp, prior), without(C, [&] { MapParams q = mp; q.strength = 0; return q; }(), prior);
  MatX3f pts(1, 3);
  pts << 1.05f, 1.05f, 1.05f;
  Semantics sem{MatXf::Zero(1, C), {0.f}};
  sem.probs(0, kBuilding) = 0.4f, sem.probs(0, kVegetation) = 0.6f;
  with.insert(pts, sem, {0}, Vec3::Zero());
  without.insert(pts, sem, {0}, Vec3::Zero());
  CHECK(without.classify(pts)[0] == kVegetation);
  CHECK(with.classify(pts)[0] == kBuilding);  // 0.4 + 0.5 * 0.6 = 0.7 > 0.6
  // ...but not on the ground, where the gate drops the building's support
  SemanticMap gated(C, mp, prior);
  gated.insert(pts, sem, {1}, Vec3::Zero());
  CHECK(gated.classify(pts)[0] == kVegetation);
}

// The map against the original OSM-BKI code (SemanticBKIOctoMap::insert_pointcloud, soft labels,
// OSM evidence, adaptive kernel, Patchwork++ verdicts) on a synthetic scan: tests/reference/
TEST(map_matches_original_osm_bki) {
  if (!std::filesystem::exists(data("bki_ref.npz"))) {
    std::printf("  skipped: tests/data/bki_ref.npz missing (tests/reference/make_reference.sh)\n");
    return;
  }
  Npz z = load_npz(data("bki_ref.npz"));
  auto P = z.at("points").as<float>(), probs = z.at("probs").as<float>(), vac = z.at("vacuity").as<float>();
  auto gl = z.at("ground").as<int8_t>();
  auto prm = z.at("params").as<double>();  // resolution, depth, ell, sf2, prior, a, b, ds, max_range, strength, decay
  const long N = z.at("points").shape[0];
  MapParams mp;
  mp.resolution = prm[0], mp.block_depth = static_cast<int>(prm[1]), mp.ell = prm[2], mp.sf2 = prm[3], mp.prior = prm[4];
  mp.adaptive = true, mp.adapt_a = prm[5], mp.adapt_b = prm[6], mp.ds_resolution = prm[7], mp.max_range = prm[8];
  mp.strength = prm[9];
  OsmMap m;
  auto rings = [&](const std::string& key) {
    std::vector<Polygon> out;
    const auto& a = z.at(key);
    auto v = a.as<double>();
    for (long r = 0; r < a.shape[0]; ++r) {
      Polygon p;
      for (long k = 0; k < a.shape[1]; ++k) p.outer.emplace_back(v[(r * a.shape[1] + k) * 2], v[(r * a.shape[1] + k) * 2 + 1]);
      out.push_back(p);
    }
    return out;
  };
  m.buildings = rings("buildings");
  for (auto& r : rings("roads")) m.features.push_back(Shape{ROADS, r, {}, 0.0});
  OsmParams op;
  op.decay_m = prm[10];
  auto cm = z.at("confusion").as<float>();
  MatXf M = Eigen::Map<MatXf>(cm.data(), C, kNumOsmColumns);
  std::vector<int8_t> rl(C, ROLE_BOTH);
  auto rr = z.at("roles").as<int8_t>();
  std::copy(rr.begin(), rr.end(), rl.begin());
  SemanticMap map(C, mp, std::make_shared<OsmPrior>(m, op, M, rl));
  Semantics sem{Eigen::Map<MatXf>(probs.data(), N, C), vac};
  map.insert(Eigen::Map<MatX3f>(P.data(), N, 3), sem, std::vector<int8_t>(gl.begin(), gl.end()), Vec3::Zero());

  // the reference's leaves with evidence: centre, alpha (prior included)
  auto rx = z.at("ref_xyz").as<float>(), ra = z.at("ref_alpha").as<float>();
  const long L = z.at("ref_xyz").shape[0];
  MatX3f xyz;
  std::vector<int> cls;
  std::vector<float> prob, mass;
  map.voxels(xyz, cls, prob, mass);
  // match leaves by centre (those with more than a trace of evidence, as the reference exported),
  // then compare their Dirichlet parameters
  std::map<std::array<long, 3>, long> mine;
  auto key = [&](float x, float y, float zz) {
    return std::array<long, 3>{std::lround(x / mp.resolution * 2), std::lround(y / mp.resolution * 2), std::lround(zz / mp.resolution * 2)};
  };
  for (long i = 0; i < xyz.rows(); ++i)
    if (mass[i] > 1e-6f) mine[key(xyz(i, 0), xyz(i, 1), xyz(i, 2))] = i;
  CHECK(static_cast<long>(mine.size()) == L);
  long matched = 0, same_class = 0;
  double worst = 0;
  for (long l = 0; l < L; ++l) {
    auto it = mine.find(key(rx[3 * l], rx[3 * l + 1], rx[3 * l + 2]));
    if (it == mine.end()) continue;
    ++matched;
    const float* a = &ra[l * C];
    double sum = 0;
    for (int c = 0; c < C; ++c) sum += a[c] - mp.prior;
    worst = std::max(worst, std::abs(sum - mass[it->second]) / std::max(1.0, sum));
    // the same class, or a tie in the reference (broken by rounding)
    int ref_cls = static_cast<int>(std::max_element(a, a + C) - a);
    same_class += ref_cls == cls[it->second] || (a[ref_cls] - a[cls[it->second]]) <= 1e-5 * (sum + C * mp.prior);
  }
  std::printf("  %ld reference voxels, %ld matched, %ld same class, worst relative mass difference %.2e\n", L, matched,
              same_class, worst);
  CHECK(matched == L);
  CHECK(same_class == L);
  CHECK(worst < 1e-4);
}
