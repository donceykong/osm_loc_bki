// LiDAR odometry: FORM (third_party/form, Potokar et al. 2025) over a sequence's scans in recording
// order. Each scan's pose is FORM's estimate when the scan arrived (FORM smooths over a fixed lag;
// what it reports at a scan's arrival uses no later scan), so a run reads them as an online
// odometry would give them. They are cached under data/odometry/.
#pragma once

#include <Eigen/Geometry>
#include <map>
#include <string>

#include "osmlocbki/site.hpp"

namespace olb {

// FORM takes a scan as a range image, row by row (rows = scan lines): MCD's and CU-Multi's Ousters
// store theirs so; other scans (KITTI-360's Velodyne) are sorted into one first
struct OdometryParams {
  int rows = 128, cols = 512;
  double min_range = 1.0, max_range = 100.0;
};

// id -> the LiDAR's pose in FORM's frame (the first scan's LiDAR frame), for the scans of the pose
// file up to and including last_id
std::map<long, Eigen::Isometry3d> form_odometry(const Site& site, const std::string& seq, long last_id,
                                                const OdometryParams& p);

}  // namespace olb
