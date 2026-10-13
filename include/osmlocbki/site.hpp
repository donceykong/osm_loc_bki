// Datasets: a site from config/sites.yaml (where its sequences live, how its world frame maps
// onto OSM, pose / scan / label formats) and a sequence's keyframes.
//
// Paths: the repository is $OSMLOCBKI_ROOT (default: where it was built from); generated files go
// to $OSMLOCBKI_DATA (default <root>/data); dataset folders resolve against $DATASET_PATH.
// Ground truth is used for choosing keyframes, simulating odometry, levelling scans (roll and
// pitch, as an IMU would) and scoring only.
#pragma once

#include <array>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "osmlocbki/common.hpp"

namespace YAML {
class Node;
}

namespace olb {

std::string repo_root();
std::string data_dir();
std::string config_dir();

struct Poses {
  std::vector<long> ids;
  MatX3 p;                        // world positions
  std::vector<Eigen::Quaterniond> q;
};

class Site {
 public:
  static std::vector<std::string> names();
  static Site load(const std::string& name);

  std::string name, map;
  std::optional<std::array<double, 4>> map_bbox;  // lat_lo, lon_lo, lat_hi, lon_hi

  std::string root() const;
  std::string seq_dir(const std::string& seq) const;
  std::string map_prefix() const;  // data/maps/<map>
  // lidar -> body extrinsic when the poses are of the body
  std::optional<Mat4> lidar_to_body() const;

  Poses read_poses(const std::string& seq) const;
  // (lat, lon) of world positions
  MatX2 latlon(const MatX2& world_xy) const;

  std::string labels_tag(const std::string& model = "") const;
  // the label set (config/labels.yaml) the scans' labels are in: the site's own for its labels,
  // the model's for inferred ones
  std::string label_set(const std::string& model = "") const;
  // scan ids with a label file, or nothing when every scan has one
  std::optional<std::set<long>> labelled_scans(const std::string& seq, const std::string& model = "") const;
  // scan ids the model exported Dirichlet outputs for (multiclass_alpha), or nothing if it did not
  std::optional<std::set<long>> alpha_scans(const std::string& seq, const std::string& model = "") const;
  std::vector<float> load_scan(const std::string& seq, long id) const;  // (N, 4) x y z intensity
  std::vector<int32_t> load_labels(const std::string& seq, long id, long n_points, const std::string& model = "") const;
  std::vector<float> load_alpha(const std::string& seq, long id, const std::string& model) const;  // (N, K)

 private:
  std::shared_ptr<YAML::Node> cfg_;
  std::string inferred_dir(const std::string& seq, const std::string& model, const std::string& kind) const;
};

struct OdometryNoise {
  double scale_bias = 0, yaw_rw = 0, trans_noise = 0, lat_noise = 0, yaw_noise = 0;
  unsigned seed = 0;
};

struct OdometryStep {
  std::optional<Vec2> d_body;  // none for the first keyframe
  double d_psi = 0;            // deg
};

class Sequence {
 public:
  // keyframes at least `spacing` m apart along the ground-truth path, plus every scan in `include`
  Sequence(const Site& site, const std::string& name, const std::string& model = "", double spacing = 1.0,
           const std::set<long>& include = {});
  size_t size() const { return scan_ids.size(); }
  MatX2 gt_xy(int epsg) const;
  // the LiDAR's ground-truth position (UTM) and the heading of its +x axis (deg, UTM grid)
  void gt_lidar(int epsg, MatX2& xy, std::vector<double>& psi) const;
  // per keyframe: ground-truth relative motion, with optional simulated drift
  std::vector<OdometryStep> odometry(const OdometryNoise& n = {}) const;
  // (T, num_bin) scan descriptors of the keyframes from their points with these labels, cached under data/
  MatXd descriptors(const std::vector<int>& building_ids, bool cache = true) const;

  Site site;
  std::string name, model, labels;
  double spacing;
  std::vector<long> scan_ids;
  MatX2 world_xy, latlon;
  std::vector<double> world_psi;  // deg
  // the LiDAR per keyframe: its rotation with the heading removed (x along its own heading,
  // gravity down), its world position and heading (deg)
  std::vector<Eigen::Matrix3d> level;
  MatX3 lidar_world;
  std::vector<double> lidar_psi;
};

}  // namespace olb
