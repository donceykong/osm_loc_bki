// Datasets. A site is one of osm_localization's (third_party/osm_localization/config/sites.yaml:
// where its sequences live, its georeference, pose / scan / label formats, its OSM map), plus what
// mapping adds to it (config/sites.yaml: the label set of the site's own labels, the LiDAR's mounting
// when osm_localization does not need it, the LiDAR's range image).
//
// Paths: the repository is $OSMLOCBKI_ROOT (default: where it was built from); generated files go
// to $OSMLOCBKI_DATA (default <root>/data), osm_localization's to <data>/localization; dataset
// folders resolve against $DATASET_PATH. Ground truth is used for levelling scans (roll and pitch,
// as an IMU would), placing scans between keyframes (their motion from the keyframe, as odometry
// would give it) and scoring only.
#pragma once

#include <array>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <map>

#include "osmlocbki/common.hpp"

namespace YAML {
class Node;
}
namespace osmloc {
class Site;
}

namespace olb {

std::string repo_root();
std::string data_dir();
std::string config_dir();

// the LiDAR's range image, for the TSDF (config/mapping.yaml nvblox: is MCD's Ouster)
struct LidarModel {
  int azimuth, elevation;
  double min_elevation_deg, max_elevation_deg;
};

class Site {
 public:
  static std::vector<std::string> names();  // osm_localization's sites
  static Site load(const std::string& name);

  std::string name, map;
  const osmloc::Site& loc() const { return *loc_; }
  std::string map_prefix() const;  // data/maps/<map>
  // the site's airborne survey model: <its data_dir>/surface/<map>_epsg<zone>.npz (on the dataset's disk)
  std::string survey_path(int epsg) const;

  // the label set (config/labels.yaml) the scans' labels are in: the model's for inferred labels,
  // else the site's own (config/sites.yaml label_set)
  std::string label_set(const std::string& model = "") const;
  std::optional<LidarModel> lidar() const;
  std::optional<Mat4> lidar_to_body() const;  // config/sites.yaml's, else osm_localization's
  // scan ids the model exported Dirichlet outputs for (multiclass_alpha), or nothing if it did not
  std::optional<std::set<long>> alpha_scans(const std::string& seq, const std::string& model) const;
  std::vector<float> load_alpha(const std::string& seq, long id, const std::string& model) const;  // (N, K)
  std::vector<float> load_scan(const std::string& seq, long id) const;  // (N, 4) x y z intensity
  std::string scan_path(const std::string& seq, long id) const;
  std::vector<int32_t> load_labels(const std::string& seq, long id, long n_points, const std::string& model = "") const;

 private:
  std::shared_ptr<const osmloc::Site> loc_;
  std::shared_ptr<YAML::Node> cfg_;  // config/sites.yaml's entry (its own, else the longest site name it starts with)
  std::string scans_;  // the scans' folder in a sequence (osm_localization's sites.yaml)
  std::string alpha_dir(const std::string& seq, const std::string& model) const;
};

// The scans a run maps or integrates: the localization keyframes, and the scans the model exported
// class distributions for (which need not be keyframes), in recording order. Each hangs off the
// keyframe at or before it (its anchor) and is placed at that keyframe's estimated pose plus the
// motion between the two (under a keyframe spacing) by odometry: `odometry`'s LiDAR poses (gravity
// aligned, world), else ground truth's. Their roll and pitch level the scans and their height places
// them; ground truth is kept for scoring. Scans the odometry has no pose for are left out.
class Frames {
 public:
  Frames(const Site& site, const std::string& seq, const std::vector<long>& keyframe_ids, const std::set<long>& extra_ids,
         int epsg, const std::map<long, Eigen::Isometry3d>* odometry = nullptr);
  size_t size() const { return scan_id.size(); }
  // frame f's pose from its anchor keyframe's (UTM, heading deg)
  void pose(long f, const Vec2& anchor_xy, double anchor_psi, Vec2& xy, double& psi) const;

  std::vector<long> scan_id;
  std::vector<int> anchor;    // its keyframe
  std::vector<int> keyframe;  // the keyframe it is, or -1
  std::vector<std::vector<long>> of_keyframe;  // the frames anchored at each keyframe, in order
  // the LiDAR per frame: its rotation with the heading removed (x along its own heading, gravity
  // down), its height (world), its motion from the anchor in the anchor's frame (x, y, heading deg),
  // and its ground truth (UTM, heading deg on the UTM grid)
  std::vector<Eigen::Matrix3d> level;
  std::vector<double> z;
  std::vector<Vec3> from_anchor;
  MatX2 gt_xy;
  std::vector<double> gt_psi;
};

}  // namespace olb
