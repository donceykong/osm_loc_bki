// End to end on real sequences; needs $DATASET_PATH (and is skipped without the data).
//   MCD TUHH day_03, ground-truth labels, exact odometry: osm_localization localizes it through the
//   pipeline, and the map built at the lag poses agrees with the map built at ground truth
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
  c.loc.site = "tuhh", c.loc.seq = "tuhh_day_03", c.map = false;
  Pipeline p(c);
  while (p.step()) {
  }
  p.finish();
  const osmloc::Localizer& loc = p.localizer();
  CHECK(p.size() == 1043);
  auto within5 = [&](const char* o) {
    MatX2 xy = loc.trajectory(o);
    long in = 0;
    for (long t = 0; t < p.size(); ++t) in += (xy.row(t) - p.gt().row(t)).norm() < 5;
    return double(in) / p.size();
  };
  auto med = [&](const char* o) {
    MatX2 xy = loc.trajectory(o);
    std::vector<double> e;
    for (long t = 0; t < p.size(); ++t) e.push_back((xy.row(t) - p.gt().row(t)).norm());
    return median(e);
  };
  std::printf("  within 5 m: rigid %.3f, hybrid %.3f, lag %.3f; median hybrid %.2f m\n", within5("rigid"), within5("hybrid"),
              within5("lag"), med("hybrid"));
  CHECK(within5("rigid") >= 0.93);
  CHECK(within5("hybrid") >= 0.93);
  CHECK(within5("lag") >= 0.99);
  CHECK(med("hybrid") < 0.7);
}

TEST(tuhh_day_03_maps_at_the_found_poses) {
  if (!have("datasets/mcd/tuhh/tuhh_day_03")) {
    std::printf("  skipped: needs $DATASET_PATH with MCD TUHH\n");
    return;
  }
  // the first 400 m: mapped at the lag poses, and (the oracle) at ground truth
  auto run = [](const std::string& pose) {
    PipelineConfig c;
    c.loc.site = "tuhh", c.loc.seq = "tuhh_day_03", c.loc.max_kf = 400, c.pose = pose, c.map_spacing = 10.0;
    auto p = std::make_unique<Pipeline>(c);
    while (p->step()) {
    }
    p->finish();
    return p;
  };
  auto lag = run("lag"), gt = run("gt");
  CHECK(lag->mapped().size() >= gt->mapped().size() / 2);
  // each map, queried with the mapped scans at ground truth: how often do they agree
  long same = 0, n = 0;
  for (long f : lag->mapped()) {
    auto a = lag->classify_frame(f), b = gt->classify_frame(f);
    for (size_t i = 0; i < a.size(); ++i)
      if (b[i] > 0) n += 1, same += a[i] == b[i];
  }
  std::printf("  %zu of %zu scans mapped at lag poses; %.1f%% of the oracle map's labels reproduced\n",
              lag->mapped().size(), gt->mapped().size(), 100.0 * same / std::max(1L, n));
  CHECK(n > 0);
  CHECK(same > 0.8 * n);
}
