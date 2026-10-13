// Reference outputs of the ORIGINAL OSM-BKI mapper (SemanticBKIOctoMap from the OSM-BKI repository)
// on a synthetic scan, for tests/test_mapping.cpp (map_matches_original_osm_bki). Built and run by
// make_reference.sh; writes tests/data/bki_ref.npz.
//
// The scan: a ground plane, a wall and a box, with random soft labels (mass left on unmapped
// classes too), vacuities and Patchwork++-style verdicts; OSM: a building and a road polygon.
// The original runs its soft-label path: OSM evidence, adaptive kernel, ground gate, no count prior.
//
// Its downsampling keys voxels by (ix * 73856093) ^ (iy * 19349669) ^ (iz * 83492791), which often
// gives mirror-image voxels ((-x, -y, z) and (x, y, z)) one key, so their points are averaged into
// one training point between them. The port keys voxels exactly; to compare everything else, the
// scan leaves out the points of colliding voxels.
#include <bkioctomap.h>

#include <map>
#include <random>
#include <set>

#include "osmlocbki/npz.hpp"

using namespace osm_bki;

int main(int argc, char** argv) {
  const char* out = argc > 1 ? argv[1] : "bki_ref.npz";
  const float res = 0.3f, ell = 1.0f, sf2 = 1.0f, prior = 1.0f, a = 1.0f, b = -3.0f, ds = 0.1f, max_range = 200.0f;
  const float strength = 0.5f, decay = 2.0f;
  const int depth = 2, nc = 9;
  SemanticBKIOctoMap map(res, depth, nc, sf2, ell, prior, 0.09f, 0.4f, 0.6f);
  map.set_adaptive_kernel(true, a, b);
  map.set_osm_use_in_evidence(true);
  map.set_osm_evidence_strength(strength);
  map.set_osm_dirichlet_count_prior(0);
  map.set_osm_decay_meters(decay);
  // roles in config/labels.yaml order: both, ground x3, nonground x4, ground
  std::vector<uint8_t> roles{GROUND_ROLE_BOTH,      GROUND_ROLE_GROUND,    GROUND_ROLE_GROUND,
                             GROUND_ROLE_GROUND,    GROUND_ROLE_NONGROUND, GROUND_ROLE_NONGROUND,
                             GROUND_ROLE_NONGROUND, GROUND_ROLE_NONGROUND, GROUND_ROLE_GROUND};
  map.set_ground_class_roles(roles);
  // config/osm_confusion/mcd_tuned.yaml, with a "none" column added on two classes to exercise it
  std::vector<std::vector<float>> cm{
      {0.38f, 0.26f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.09f, 0.00f},  // road
      {0.34f, 0.47f, 0.00f, 0.00f, 0.20f, 0.01f, 0.00f, 0.00f, 0.11f, 0.00f},  // sidewalk
      {0.19f, 0.27f, 0.00f, 0.18f, 0.00f, 0.00f, 0.00f, 0.00f, 0.11f, 0.00f},  // parking
      {0.00f, 0.00f, 0.00f, 0.00f, 0.17f, 0.21f, 0.00f, 0.49f, 0.22f, 0.00f},  // building
      {0.00f, 0.00f, 0.00f, 0.20f, 0.00f, 0.20f, 0.00f, 0.42f, 0.18f, 0.00f},  // fence
      {0.00f, 0.00f, 0.00f, 0.31f, 0.28f, 0.48f, 0.50f, 0.00f, 0.29f, 0.30f},  // vegetation
      {0.00f, 0.00f, 0.00f, 0.31f, 0.00f, 0.10f, 0.00f, 0.09f, 0.00f, 0.00f},  // vehicle
      {0.09f, 0.00f, 0.00f, 0.00f, 0.35f, 0.00f, 0.50f, 0.00f, 0.00f, 0.20f},  // terrain
  };
  map.set_osm_confusion_matrix(cm, {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}});
  auto poly = [](std::vector<std::pair<float, float>> c) {
    Geometry2D g;
    g.coords = std::move(c);
    compute_bbox(g);
    return g;
  };
  Geometry2D building = poly({{0, 10}, {10, 10}, {10, 20}, {0, 20}});
  Geometry2D road = poly({{-15, -2}, {15, -2}, {15, 2}, {-15, 2}});
  map.set_osm_buildings({building});
  map.set_osm_roads({road});

  // the scan (map frame), sensor 1.7 m up at the origin
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> U(0, 1);
  PCLPointCloud cloud;
  std::vector<std::vector<float>> probs;
  std::vector<uint8_t> ground;
  std::vector<float> vac;
  auto add = [&](float x, float y, float z, bool is_ground) {
    PCLPointType p;
    p.x = x, p.y = y, p.z = z, p.label = 0;
    cloud.push_back(p);
    std::vector<float> row(nc, 0.f);
    float total = 0.6f + 0.4f * U(rng), s = 0;
    for (int c = 1; c < nc; ++c) row[c] = std::pow(U(rng), 3.f), s += row[c];
    for (int c = 1; c < nc; ++c) row[c] *= total / s;
    probs.push_back(row);
    float g = U(rng);
    ground.push_back(g < 0.1f ? GROUND_LABEL_UNKNOWN : is_ground ? GROUND_LABEL_GROUND : GROUND_LABEL_NONGROUND);
    vac.push_back(U(rng));
  };
  for (int i = 0; i < 4000; ++i) add(-12 + 24 * U(rng), -6 + 24 * U(rng), 0.05f * U(rng), true);   // ground
  for (int i = 0; i < 1500; ++i) add(1 + 8 * U(rng), 9.9f + 0.2f * U(rng), 4 * U(rng), false);     // wall
  for (int i = 0; i < 800; ++i) add(-8 + 2 * U(rng), -1 + 2 * U(rng), 0.2f + 1.3f * U(rng), false); // a box on the road
  {  // drop the points of voxels whose hash collides with another voxel's
    const float inv = 1.0f / ds;
    auto cell = [&](const PCLPointType& p) {
      return std::array<int64_t, 3>{static_cast<int64_t>(std::floor(p.x * inv)), static_cast<int64_t>(std::floor(p.y * inv)),
                                    static_cast<int64_t>(std::floor(p.z * inv))};
    };
    auto hash = [](const std::array<int64_t, 3>& c) { return (c[0] * 73856093) ^ (c[1] * 19349669) ^ (c[2] * 83492791); };
    std::map<int64_t, std::set<std::array<int64_t, 3>>> cells;
    for (const auto& p : cloud) cells[hash(cell(p))].insert(cell(p));
    PCLPointCloud kept;
    std::vector<std::vector<float>> kp;
    std::vector<uint8_t> kg;
    std::vector<float> kv;
    for (size_t i = 0; i < cloud.size(); ++i)
      if (cells[hash(cell(cloud[i]))].size() == 1)
        kept.push_back(cloud[i]), kp.push_back(probs[i]), kg.push_back(ground[i]), kv.push_back(vac[i]);
    std::printf("dropped %zu points in colliding voxels\n", cloud.size() - kept.size());
    cloud = kept, probs = kp, ground = kg, vac = kv;
  }
  map.insert_pointcloud(cloud, point3f(0, 0, 1.7f), ds, 1000.0f, max_range, &probs, &ground, &vac);

  // every leaf with evidence: centre and Dirichlet parameters (prior included)
  std::vector<float> xyz, alpha;
  for (auto it = map.begin_leaf(); it != map.end_leaf(); ++it) {
    const SemanticOcTreeNode& n = it.get_node();
    float m = n.get_evidence_mass();
    if (!(m > 1e-6f)) continue;
    std::vector<float> p(nc);
    n.get_probs(p);
    point3f c = it.get_loc();
    xyz.insert(xyz.end(), {c.x(), c.y(), c.z()});
    for (int k = 0; k < nc; ++k) alpha.push_back(p[k] * (nc * prior + m));
  }
  const long N = static_cast<long>(cloud.size()), L = static_cast<long>(xyz.size() / 3);
  std::vector<float> pts, pr;
  std::vector<int8_t> gl;
  for (long i = 0; i < N; ++i) {
    pts.insert(pts.end(), {cloud[i].x, cloud[i].y, cloud[i].z});
    pr.insert(pr.end(), probs[i].begin(), probs[i].end());
    gl.push_back(ground[i] == GROUND_LABEL_GROUND ? 1 : ground[i] == GROUND_LABEL_NONGROUND ? 0 : -1);
  }
  std::vector<double> buildings{0, 10, 10, 10, 10, 20, 0, 20}, roads{-15, -2, 15, -2, 15, 2, -15, 2};
  std::vector<float> confusion(nc * 10, 0.f);
  for (int r = 0; r < 8; ++r)
    for (int c = 0; c < 10; ++c) confusion[(r + 1) * 10 + c] = cm[r][c];
  std::vector<int8_t> rl(roles.begin(), roles.end());
  std::vector<double> params{res, depth, ell, sf2, prior, a, b, ds, max_range, strength, decay};
  olb::save_npz(out, {{"points", olb::NpyArray::from(pts, {N, 3})},
                      {"probs", olb::NpyArray::from(pr, {N, nc})},
                      {"vacuity", olb::NpyArray::from(vac, {N})},
                      {"ground", olb::NpyArray::from(gl, {N})},
                      {"buildings", olb::NpyArray::from(buildings, {1, 4, 2})},
                      {"roads", olb::NpyArray::from(roads, {1, 4, 2})},
                      {"confusion", olb::NpyArray::from(confusion, {nc, 10})},
                      {"roles", olb::NpyArray::from(rl, {nc})},
                      {"params", olb::NpyArray::from(params, {11})},
                      {"ref_xyz", olb::NpyArray::from(xyz, {L, 3})},
                      {"ref_alpha", olb::NpyArray::from(alpha, {L, nc})}});
  std::printf("%ld points -> %ld voxels with evidence: %s\n", N, L, out);
  return 0;
}
