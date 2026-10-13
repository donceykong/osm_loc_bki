// End to end, online: global localization against OSM building outlines (no prior map, no GPS,
// no initial pose), and the OSM-BKI semantic map built from the scans at the poses it finds.
//
// Per keyframe (every `spacing` m) the localizer sees the scan's building descriptor and the
// odometry since the previous keyframe. Mapping keyframes (every `map_spacing` m, among the scans
// the segmentation model exported class distributions for) are inserted into the map once their
// pose is ready, while the localizer is locked:
//   pose = lag       the fixed-lag pose, settled lag_m of travel later (the default: accurate)
//          filtered  the filtered pose, at once (online, a little noisier)
//          gt        ground truth (an oracle, to compare against)
// Locked: at least lock_m of path behind, and the rigid and graph outputs agree.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "osmlocbki/backend.hpp"
#include "osmlocbki/ground.hpp"
#include "osmlocbki/semantic_map.hpp"
#include "osmlocbki/site.hpp"

namespace olb {

struct PipelineConfig {
  std::string site, seq, model;   // model: the segmentation model, for sites with inferred labels
  std::vector<int> building_ids;  // default: the label set's (Taxonomy::building_ids)
  double spacing = 1.0;           // localization keyframes, m
  long max_kf = 0;                // 0 = all
  // localization
  double db_spacing = 2.0;
  int k = 20, top_n = 1000;
  double agree_m = 2.0, lag_m = 60.0;
  GraphParams graph;              // GraphParams() or GraphParams::drifting()
  FilterParams filter;
  OdometryNoise odometry;
  // mapping
  bool map = true;
  std::string pose = "lag";       // lag | filtered | gt
  double map_spacing = 5.0;       // m between mapped keyframes
  double lock_m = 100.0;
  std::string mapping = "mapping.yaml";  // under config/
  double strength = -1;           // OSM strength; < 0: the mapping config's
};

// a mapped keyframe: its scan in the map frame and the classes its points came in with
struct MappedScan {
  long t = -1, scan_id = -1;
  Vec2 xy;    // UTM
  double psi; // deg
  MatX3f points;
  std::vector<int> cls;
};

class Pipeline {
 public:
  explicit Pipeline(const PipelineConfig& c);
  ~Pipeline();
  // localize the next keyframe and map what is ready -> false when the sequence is done
  bool step();
  // settle the fixed lag and map the keyframes it settles
  void finish();

  long t() const { return t_; }       // last keyframe localized
  long size() const { return T_; }
  bool locked() const { return locked_; }
  const PipelineConfig& config() const { return c_; }
  const Sequence& sequence() const { return *seq_; }
  const Localizer& localizer() const { return *loc_; }
  const OsmMap& osm() const { return osm_; }
  const Taxonomy& taxonomy() const { return tax_; }
  const SemanticMap* semantic_map() const { return map_.get(); }
  const MatX2& gt() const { return gt_; }  // the LiDAR's ground truth, UTM
  const std::vector<double>& gt_psi() const { return gt_psi_; }
  const std::vector<long>& map_keyframes() const { return map_kf_; }  // the keyframes to map
  const std::vector<long>& mapped() const { return mapped_; }         // and those mapped so far
  const MappedScan& last_mapped() const { return last_; }
  // keyframe t's scan at its ground-truth pose, classified by the map (0 for dropped points)
  std::vector<int> classify_keyframe(long t) const;
  std::string status() const;
  double map_seconds() const { return map_s_; }

 private:
  struct Scan {
    MatX3f levelled;
    Semantics sem;
    std::vector<float> intensity;
    std::vector<long> index;  // into the raw scan
    long raw = 0;             // points in the raw scan
  };
  Scan load(long t) const;
  MatX3f place(const MatX3f& levelled, const Vec2& xy, double psi, long t) const;
  void insert(long t, const Vec2& xy, double psi);
  void map_ready(const Estimate& e);

  PipelineConfig c_;
  Site site_;
  std::unique_ptr<Sequence> seq_;
  std::shared_ptr<Database> db_;
  std::unique_ptr<Localizer> loc_;
  MatXd q_;
  std::vector<OdometryStep> odo_;
  MatX2 gt_;
  std::vector<double> gt_psi_;
  long T_ = 0, t_ = -1;
  bool locked_ = false;
  // mapping
  OsmMap osm_;
  Taxonomy tax_;
  std::vector<int> to_common_;
  bool alpha_ = false;
  std::unique_ptr<SemanticMap> map_;
  std::unique_ptr<GroundSegmenter> ground_;
  std::vector<char> is_map_kf_;
  std::vector<long> map_kf_, mapped_;
  MappedScan last_;
  double map_s_ = 0;
};

}  // namespace olb
