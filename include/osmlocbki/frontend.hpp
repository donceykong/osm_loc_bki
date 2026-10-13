// Localization front-end: the building range descriptor of Cho et al. 2022, the OSM database it is
// matched against, and two-stage matching that turns one scan into its top-k hits (position and
// heading candidates on the map).
//
// Database candidates sit on a grid over the space outside building footprints (cells with no
// building in range are dropped: an all-zero descriptor matches any featureless scan); each is
// ray-cast against the buildings' outer rings, as make_descriptor.m does.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "osmlocbki/geo.hpp"

namespace olb {

class Site;

constexpr int kNumBin = 360;
constexpr double kSensorRange = 50.0;
constexpr double kBinSize = 5.0;
constexpr double kMinScanRange = 0.5;  // drops the sensor's zero-filled non-returns

// Building points, sensor frame (N rows of `stride` floats, x y first) -> nearest planar range per
// azimuth wedge (0 = nothing within range).
std::vector<double> lidar_descriptor(const float* pts, long n, int stride, int num_bin = kNumBin,
                                     double sensor_range = kSensorRange);

// A labelled scan -> descriptor of its building points, and how many there were.
std::vector<double> scan_descriptor(const float* pts, long n, int stride, const int32_t* labels,
                                    const std::vector<int>& building_ids, int* kept = nullptr,
                                    int num_bin = kNumBin, double sensor_range = kSensorRange);

// Distances -> count of rays per `bin_size` range bin (rotation invariant); zeros are skipped.
template <typename T>
void rotation_invariant(const T* d, int num_bin, float* out, double sensor_range = kSensorRange,
                        double bin_size = kBinSize);

inline double psi_of(long shift, int n_bin = kNumBin) { return pymod(1.0 - shift * 360.0 / n_bin, 360.0); }

struct Database {
  MatX2 xy;                 // (M, 2) UTM candidate positions
  MatX2 ll;                 // (M, 2) the same as (lon, lat)
  int epsg = 0;
  std::vector<float> desc;  // (M, num_bin) building range descriptors
  int num_bin = kNumBin;
  long size() const { return xy.rows(); }
  const float* row(long i) const { return desc.data() + i * num_bin; }
};

struct Hits {
  MatX2 xy;   // (k, 2)
  VecXd psi;  // (k,) heading of the scan's +x axis, deg
};

// one position -> (num_bin,) distance to the first building edge per ray (0 = none in range);
// rays at j * 360 / num_bin deg, j = 1..num_bin
class RayCaster {
 public:
  RayCaster(const std::vector<Ring>& rings, int num_bin = kNumBin, double sensor_range = kSensorRange);
  void cast(const Vec2& origin, float* out) const;

 private:
  std::vector<Ring> rings_;
  double range_;
  int n_;
  std::vector<double> cos_, sin_;
  // vertex grid with `range` cells
  std::unordered_map<int64_t, std::vector<int>> grid_;
  std::vector<int> owner_;
  std::vector<Vec2> vxy_;
};

// candidates every `spacing` m over the map box (lat_lo, lon_lo, lat_hi, lon_hi), or the buildings' extent
Database build_database(const OsmMap& map, const std::optional<std::array<double, 4>>& bbox_ll, double spacing = 2.0);
void save_database(const Database& db, const std::string& path);
Database load_database(const std::string& path);

// the site's database, built and cached under data/ on first use (one per map)
std::shared_ptr<Database> site_database(const Site& site, double spacing = 2.0, bool rebuild = false);

class Matcher {
 public:
  Matcher(std::shared_ptr<const Database> db, int top_n = 1000);
  // (num_bin,) scan descriptor -> top-k hits: the `top_n` nearest rotation-invariant descriptors,
  // re-ranked by the full descriptor over every shift (which also gives the heading)
  Hits operator()(const double* q, int k = 20) const;

 private:
  std::vector<long> stage1(const double* q) const;
  void stage2(const double* q, const std::vector<long>& cand, std::vector<long>& idx, std::vector<long>& shift) const;
  std::shared_ptr<const Database> db_;
  std::vector<float> ri_;
  int n_ri_, top_n_;
};

}  // namespace olb
