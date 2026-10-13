// End to end on real sequences; needs $DATASET_PATH (and is skipped without the data).
//   MCD TUHH day_03, ground-truth labels, exact odometry: osmloc's README results row, then the
//   map built at the lag poses agrees with the map built at ground truth
#include <cstdlib>
#include <filesystem>

#include "check.hpp"
#include "osmlocbki/pipeline.hpp"

using namespace olb;

namespace {

bool have(const std::string& rel) {
  const char* base = std::getenv("DATASET_PATH");
  return base && std::filesystem::exists(std::string(base) + "/" + rel);
}

double median(std::vector<double> v) {
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

}  // namespace

TEST(tuhh_day_03_localizes) {
  if (!have("datasets/mcd/tuhh/tuhh_day_03")) {
    std::printf("  skipped: needs $DATASET_PATH with MCD TUHH\n");
    return;
  }
  PipelineConfig c;
  c.site = "tuhh", c.seq = "tuhh_day_03", c.map = false;
  Pipeline p(c);
  while (p.step()) {
  }
  p.finish();
  const Localizer& loc = p.localizer();
  CHECK(p.size() == 1043);
  auto within5 = [&](Output o) {
    MatX2 xy = loc.trajectory(o);
    long in = 0;
    for (long t = 0; t < p.size(); ++t) in += (xy.row(t) - p.gt().row(t)).norm() < 5;
    return double(in) / p.size();
  };
  auto med = [&](Output o) {
    MatX2 xy = loc.trajectory(o);
    std::vector<double> e;
    for (long t = 0; t < p.size(); ++t) e.push_back((xy.row(t) - p.gt().row(t)).norm());
    return median(e);
  };
  std::printf("  within 5 m: rigid %.3f, hybrid %.3f, lag %.3f; median hybrid %.2f m\n", within5(RIGID), within5(HYBRID),
              within5(LAG), med(HYBRID));
  CHECK(within5(RIGID) >= 0.93);
  CHECK(within5(HYBRID) >= 0.93);
  CHECK(within5(LAG) >= 0.99);
  CHECK(med(HYBRID) < 0.7);
}

TEST(tuhh_day_03_maps_at_the_found_poses) {
  if (!have("datasets/mcd/tuhh/tuhh_day_03")) {
    std::printf("  skipped: needs $DATASET_PATH with MCD TUHH\n");
    return;
  }
  // the first 400 m: mapped at the lag poses, and (the oracle) at ground truth
  auto run = [](const std::string& pose) {
    PipelineConfig c;
    c.site = "tuhh", c.seq = "tuhh_day_03", c.pose = pose, c.max_kf = 400, c.map_spacing = 10.0;
    auto p = std::make_unique<Pipeline>(c);
    while (p->step()) {
    }
    p->finish();
    return p;
  };
  auto lag = run("lag"), gt = run("gt");
  CHECK(lag->mapped().size() >= gt->mapped().size() / 2);
  // each map, queried with the mapped keyframes' scans at ground truth: how often do they agree
  long same = 0, n = 0;
  for (long t : lag->mapped()) {
    auto a = lag->classify_keyframe(t), b = gt->classify_keyframe(t);
    for (size_t i = 0; i < a.size(); ++i)
      if (b[i] > 0) n += 1, same += a[i] == b[i];
  }
  std::printf("  %zu of %zu keyframes mapped at lag poses; %.1f%% of the oracle map's labels reproduced\n",
              lag->mapped().size(), gt->mapped().size(), 100.0 * same / std::max(1L, n));
  CHECK(n > 0);
  CHECK(same > 0.8 * n);
}
