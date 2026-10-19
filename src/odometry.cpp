#include "osmlocbki/odometry.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <form/form.hpp>

#include "osmloc/site.hpp"
#include "osmlocbki/npz.hpp"

namespace fs = std::filesystem;

namespace olb {

namespace {

// a scan (N, 4) as FORM's row-major range image; points in the same cell: the nearest
std::vector<form::PointXYZf> range_image(const std::vector<float>& pc, const OdometryParams& p) {
  const long n = static_cast<long>(pc.size() / 4), cells = static_cast<long>(p.rows) * p.cols;
  std::vector<form::PointXYZf> img;
  if (n == cells) {  // already one: the Ousters' scans
    img.reserve(n);
    for (long i = 0; i < n; ++i) img.emplace_back(pc[4 * i], pc[4 * i + 1], pc[4 * i + 2]);
    return img;
  }
  // scan lines: a Velodyne's points come line by line, each a sweep in azimuth, so a line ends where
  // the azimuth jumps by over half a turn; else lines are elevation bands
  std::vector<int> line(n, 0);
  std::vector<float> el(n);
  int lines = 0;
  float lo = 1e9f, hi = -1e9f;
  for (long i = 0; i < n; ++i) {
    const float x = pc[4 * i], y = pc[4 * i + 1], z = pc[4 * i + 2];
    el[i] = std::atan2(z, std::hypot(x, y));
    lo = std::min(lo, el[i]), hi = std::max(hi, el[i]);
    if (i > 0) {
      const float a = std::atan2(y, x), b = std::atan2(pc[4 * i - 3], pc[4 * i - 4]);
      if (std::abs(a - b) > static_cast<float>(kPi)) ++lines;
    }
    line[i] = lines;
  }
  const bool by_sweep = std::abs(lines + 1 - p.rows) <= 2;
  img.assign(cells, form::PointXYZf(0, 0, 0));
  std::vector<float> best(cells, 1e30f);
  for (long i = 0; i < n; ++i) {
    const float x = pc[4 * i], y = pc[4 * i + 1], z = pc[4 * i + 2], r = x * x + y * y + z * z;
    int row = by_sweep ? line[i] : static_cast<int>(std::lround((hi - el[i]) / std::max(hi - lo, 1e-6f) * (p.rows - 1)));
    row = std::clamp(row, 0, p.rows - 1);
    int col = static_cast<int>(std::floor((kPi - std::atan2(y, x)) / (2 * kPi) * p.cols)) % p.cols;
    const long c = static_cast<long>(row) * p.cols + col;
    if (r < best[c]) best[c] = r, img[c] = form::PointXYZf(x, y, z);
  }
  return img;
}

}  // namespace

std::map<long, Eigen::Isometry3d> form_odometry(const Site& site, const std::string& seq, long last_id,
                                                const OdometryParams& p) {
  // every scan from the first posed one to last_id: pose files can skip scans inside a run
  // (KITTI-360's), which odometry should not jump
  const osmloc::Poses all = site.loc().read_poses(seq);
  std::vector<long> ids;
  for (long id = all.ids.front(); id <= last_id; ++id)
    if (fs::exists(site.scan_path(seq, id))) ids.push_back(id);
  char tag[64];
  std::snprintf(tag, sizeof tag, "_form_%dx%d.npz", p.rows, p.cols);
  const fs::path path = fs::path(data_dir()) / "odometry" / (site.name + "_" + seq + tag);
  std::map<long, Eigen::Isometry3d> out;
  auto put = [&](long id, const double* m) {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.matrix().topRows<3>() = Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(m);
    out[id] = T;
  };
  if (fs::exists(path)) {  // a run as long or longer, cached
    Npz z = load_npz(path.string());
    auto cid = z.at("ids").as<long>();
    auto pose = z.at("poses").as<double>();
    if (cid.size() >= ids.size() && std::equal(ids.begin(), ids.end(), cid.begin())) {
      for (size_t k = 0; k < ids.size(); ++k) put(ids[k], pose.data() + 12 * k);
      return out;
    }
  }
  std::fprintf(stderr, "FORM odometry: %s/%s, %zu scans (%dx%d range images)\n", site.name.c_str(), seq.c_str(), ids.size(),
               p.rows, p.cols);
  form::Estimator::Params fp;
  fp.extraction.num_rows = p.rows, fp.extraction.num_columns = p.cols;
  fp.extraction.min_norm_squared = p.min_range * p.min_range;
  fp.extraction.max_norm_squared = p.max_range * p.max_range;
  form::Estimator est(fp);
  std::vector<double> pose(12 * ids.size());
  Stopwatch sw;
  for (size_t k = 0; k < ids.size(); ++k) {
    est.register_scan(range_image(site.load_scan(seq, ids[k]), p));
    const gtsam::Pose3 T = est.current_lidar_estimate();
    Eigen::Matrix<double, 3, 4, Eigen::RowMajor> m;
    m << T.rotation().matrix(), T.translation();
    std::copy(m.data(), m.data() + 12, pose.data() + 12 * k);
    put(ids[k], m.data());
    if ((k + 1) % 1000 == 0)
      std::fprintf(stderr, "  %zu / %zu scans, %.0f ms per scan\n", k + 1, ids.size(), sw.seconds() / (k + 1) * 1e3);
  }
  fs::create_directories(path.parent_path());
  save_npz(path.string(), {{"ids", NpyArray::from(ids, {static_cast<long>(ids.size())})},
                           {"poses", NpyArray::from(pose.data(), {static_cast<long>(ids.size()), 12})}});
  return out;
}

}  // namespace olb
