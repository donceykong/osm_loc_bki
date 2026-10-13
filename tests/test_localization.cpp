// Localization: .npz files, UTM, the OSM database, the scan descriptor and matching, and the
// back-end, against the original Python implementation's and GTSAM's outputs (tests/data).
#include <algorithm>
#include <cstdio>
#include <filesystem>

#include "check.hpp"
#include "osmlocbki/backend.hpp"
#include "osmlocbki/npz.hpp"

using namespace olb;
using olb_test::data;

TEST(npz_round_trip) {
  std::vector<double> a{1.5, -2, 3e9};
  std::vector<int32_t> b{1, 2, 3, 4, 5, 6};
  std::string path = (std::filesystem::temp_directory_path() / "osmlocbki_test.npz").string();
  save_npz(path, {{"a", NpyArray::from(a, {3})}, {"b", NpyArray::from(b, {2, 3})}, {"s", NpyArray::scalar_of<int64_t>(7)}});
  Npz z = load_npz(path);
  CHECK(z.at("a").as<double>() == a);
  CHECK(z.at("b").as<int32_t>() == b);
  CHECK((z.at("b").shape == std::vector<long>{2, 3}));
  CHECK(z.at("s").scalar<int64_t>() == 7);
  std::filesystem::remove(path);
}

TEST(utm_matches_pyproj) {
  auto rows = load_npz(data("utm.npz")).at("rows");
  auto v = rows.as<double>();
  for (long i = 0; i < rows.shape[0]; ++i) {
    double lat = v[5 * i], lon = v[5 * i + 1], x = v[5 * i + 3], y = v[5 * i + 4];
    int epsg = static_cast<int>(v[5 * i + 2]);
    CHECK(utm_epsg(lat, lon) == epsg);
    Vec2 p = to_utm(lat, lon, epsg);
    CHECK_NEAR(p[0], x, 1e-6);
    CHECK_NEAR(p[1], y, 1e-6);
    Vec2 q = from_utm(x, y, epsg);
    CHECK_NEAR(q[0], lat, 1e-10);
    CHECK_NEAR(q[1], lon, 1e-10);
  }
}

TEST(database_matches_python) {
  auto b = load_npz(data("mini_bbox.npz")).at("bbox").as<double>();
  OsmMap m = load_map(data("mini"));
  CHECK(m.buildings.size() == 3);
  CHECK(m.buildings[1].holes.size() == 1);
  Database db = build_database(m, std::array<double, 4>{b[0], b[1], b[2], b[3]});
  Database ref = load_database(data("mini_db_free.npz"));
  CHECK(db.size() == ref.size());
  CHECK(db.epsg == ref.epsg);
  if (db.size() != ref.size()) return;
  CHECK((db.xy - ref.xy).cwiseAbs().maxCoeff() < 1e-6);
  CHECK((db.ll - ref.ll).cwiseAbs().maxCoeff() < 1e-9);
  double worst = 0;
  for (size_t i = 0; i < db.desc.size(); ++i) worst = std::max(worst, double(std::abs(db.desc[i] - ref.desc[i])));
  CHECK_NEAR(worst, 0.0, 1e-4);
}

TEST(footprints_respect_courtyards) {
  Polygon b{{Vec2(0, 0), Vec2(10, 0), Vec2(10, 10), Vec2(0, 10)}, {{Vec2(3, 3), Vec2(7, 3), Vec2(7, 7), Vec2(3, 7)}}};
  Footprints f({b});
  CHECK(f.contains(Vec2(1, 1)));
  CHECK(!f.contains(Vec2(5, 5)));
  CHECK(!f.contains(Vec2(12, 5)));
}

TEST(scan_descriptor_matches_python) {
  Npz z = load_npz(data("scan_descriptor.npz"));
  auto scan = z.at("scan").as<float>();
  auto lab = z.at("labels").as<int32_t>();
  auto ref = z.at("desc").as<double>();
  int kept = 0;
  auto d = scan_descriptor(scan.data(), static_cast<long>(lab.size()), 4, lab.data(), {2}, &kept);
  CHECK(kept == z.at("kept").scalar<int64_t>());
  for (int j = 0; j < kNumBin; ++j) CHECK_NEAR(d[j], ref[j], 1e-12);
}

TEST(matcher_finds_a_rotated_candidate) {
  auto db = std::make_shared<Database>(load_database(data("mini_db_free.npz")));
  Matcher m(db, static_cast<int>(db->size()));
  for (long row : {100L, 2500L, 4000L}) {
    // the candidate's own descriptor rolled by 33 bins: found with shift 33
    std::vector<double> q(kNumBin);
    for (int b = 0; b < kNumBin; ++b) q[b] = db->row(row)[(b - 33 + kNumBin) % kNumBin];
    Hits h = m(q.data(), 5);
    CHECK((h.xy.row(0) - db->xy.row(row)).norm() < 1e-9);
    CHECK_NEAR(h.psi[0], psi_of(33), 1e-9);
  }
}

// ---- back-end, on a recorded synthetic drive (backend.npz: front-end hits, drifting odometry,
// and the outputs of the original Python localizer and of GTSAM's optimizer) -------------------

namespace {

double max_diff(const MatX2& a, const std::vector<double>& b) {
  double m = 0;
  for (long i = 0; i < a.rows(); ++i) m = std::max(m, (a.row(i) - Eigen::RowVector2d(b[2 * i], b[2 * i + 1])).norm());
  return m;
}

std::unique_ptr<Localizer> replay(const Npz& z) {
  auto hx = z.at("hits_xy").as<double>(), hp = z.at("hits_psi").as<double>();
  auto db = z.at("d_body").as<double>(), dp = z.at("d_psi").as<double>();
  const long T = z.at("hits_xy").shape[0], K = z.at("hits_xy").shape[1];
  auto loc = std::make_unique<Localizer>(nullptr, static_cast<int>(K), GraphParams::drifting(), 2.0, 60.0);
  for (long t = 0; t < T; ++t) {
    Hits h{Eigen::Map<MatX2>(hx.data() + t * K * 2, K, 2), Eigen::Map<VecXd>(hp.data() + t * K, K)};
    std::optional<Vec2> d;
    if (t) d = Vec2(db[2 * t], db[2 * t + 1]);
    loc->step_hits(h, d, dp[t]);
  }
  return loc;
}

}  // namespace

TEST(backend_matches_python_and_gtsam) {
  Npz z = load_npz(data("backend.npz"));
  auto p = replay(z);
  Localizer& loc = *p;
  // rigid tracker and history check: as the Python localizer (with stable sorts)
  CHECK(max_diff(loc.trajectory(RIGID), z.at("py_rigid").as<double>()) < 1e-6);
  CHECK(max_diff(loc.trajectory(HYBRID), z.at("py_hybrid").as<double>()) < 1e-3);
  // the chain solver: as GTSAM's Levenberg-Marquardt on the same graphs
  CHECK(max_diff(loc.trajectory(GRAPH), z.at("gtsam_graph").as<double>()) < 1e-5);
  loc.finish_lag();
  CHECK(max_diff(loc.trajectory(LAG), z.at("gtsam_lag").as<double>()) < 1e-5);
}

TEST(chain_solver_basics) {
  // three poses, exact odometry, one prior each on a shifted copy: the chain lands on the priors
  ChainGraph g;
  g.hit_isig = Vec3(1, 1, 10);
  for (int i = 0; i < 3; ++i) {
    if (i) g.odo.push_back(Pose2d::from(1, 0, 0)), g.odo_isig.push_back(Vec3(100, 100, 100));
    g.hits.push_back({Pose2d::from(5.0 + i, 2.0, 0.0)});
  }
  std::vector<Pose2d> x(3, Pose2d::from(0, 0, 0.2));
  int iters = 0;
  auto sol = optimize(g, x, LMParams(), &iters);
  for (int i = 0; i < 3; ++i) {
    CHECK_NEAR(sol[i].x, 5.0 + i, 1e-3);
    CHECK_NEAR(sol[i].y, 2.0, 1e-3);
    CHECK_NEAR(sol[i].theta(), 0.0, 1e-3);
  }
  CHECK(iters > 0);
}

TEST(filtered_is_smooth_and_lag_has_headings) {
  Npz z = load_npz(data("backend.npz"));
  auto p = replay(z);
  Localizer& loc = *p;
  loc.finish_lag();
  auto gt = z.at("gt_xy").as<double>();
  const long T = loc.size();
  // jitter: median of how far each step departs from the odometry step (in the output's own
  // heading); the median, as relocalization jumps before lock-on are meant to come through
  const auto& b = loc.buf();
  auto jitter = [&](Output o) {
    MatX2 xy = loc.trajectory(o);
    const auto& psi = loc.heading(o);
    std::vector<double> j;
    for (long t = 1; t < T; ++t) {
      Vec2 step = rot2(radians(psi[t - 1])) * (rot2(-b.o_psi_rad[t - 1]) * (b.o[t] - b.o[t - 1]));
      j.push_back((Vec2(xy.row(t) - xy.row(t - 1)) - step).norm());
    }
    std::nth_element(j.begin(), j.begin() + j.size() / 2, j.end());
    return j[j.size() / 2];
  };
  auto within5 = [&](Output o) {
    MatX2 xy = loc.trajectory(o);
    long in = 0;
    for (long t = 0; t < T; ++t) in += (xy.row(t) - Eigen::RowVector2d(gt[2 * t], gt[2 * t + 1])).norm() < 5;
    return double(in) / T;
  };
  CHECK(jitter(FILTERED) < 0.5 * jitter(GRAPH));
  CHECK(within5(FILTERED) >= within5(GRAPH) - 0.02);
  for (double p : loc.heading(LAG)) CHECK(std::isfinite(p));
}
