// End to end, online: global localization against OSM building outlines (no prior map, no GPS,
// no initial pose) by osm_localization (third_party/osm_localization), and the OSM-BKI semantic map
// built from the scans at the poses it finds.
//
// Per keyframe (every `loc.spacing` m) osm_localization's run localizes the scan from its building
// descriptor and the odometry since the previous keyframe. The scans to map (every `map_spacing` m,
// among the scans the segmentation model exported class distributions for) are placed off the
// keyframe at or before them (Frames) once that keyframe's pose is ready, while the localizer is
// locked:
//   pose = lag       the fixed-lag pose, settled loc.lag_m of travel later (the default: accurate)
//          filtered  the filtered pose, at once (online, a little noisier)
//          gt        ground truth (an oracle, to compare against)
// Locked: at least lock_m of path behind, and (lock_agree_m > 0) the rigid and graph outputs within
// lock_agree_m of each other.
//
// With an airborne survey (`surface`), each mapped keyframe also projects the survey within
// survey.radius of its pose into the map, every voxel once: a voxel gets survey.weight times OSM's
// class support, gated by the voxel's ground verdict (its height over the survey's terrain), so
// LiDAR evidence outweighs it wherever the vehicle sees the same place.
#pragma once

#include <limits>
#include <memory>
#include <unordered_set>
#include <optional>
#include <string>
#include <vector>

#include "osmloc/run.hpp"
#include "osmlocbki/ground.hpp"
#include "osmlocbki/semantic_map.hpp"
#include "osmlocbki/site.hpp"
#include "osmlocbki/survey.hpp"

namespace olb {

struct PipelineConfig {
  // localization: osm_localization's run (site, seq, model, keyframe spacing, max_kf, odometry,
  // graph, lag, surface registration, ...); its building ids default to the label set's
  // (Taxonomy::building_ids), and it never opens its own view
  osmloc::RunConfig loc;
  // mapping
  bool map = true;
  std::string pose = "lag";       // lag | filtered | gt
  double map_spacing = 5.0;       // m between mapped keyframes
  double lock_m = 100.0, lock_agree_m = 2.0;
  std::string mapping = "mapping.yaml";  // under config/
  double strength = -1;           // OSM strength; < 0: the mapping config's
  std::string surface;            // airborne survey .npz ("auto": data/surface/<map>_epsg<zone>.npz); empty: none
  std::string mapper;             // map backend, cpu | nvblox; empty: the mapping config's
};

// a mapped scan (frame f): its scan in the map frame and the classes its points came in with
struct MappedScan {
  long f = -1, scan_id = -1;
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
  // settle the fixed lag (and osm_localization's registration in flight) and map what it settles
  void finish();

  long t() const { return t_; }       // last keyframe localized
  long size() const { return T_; }
  bool locked() const { return locked_; }
  const PipelineConfig& config() const { return c_; }
  const Frames& frames() const { return *frames_; }
  const osmloc::Run& run() const { return *run_; }
  const osmloc::Localizer& localizer() const { return run_->localizer(); }
  const std::vector<long>& keyframe_ids() const { return kf_ids_; }
  const OsmMap& osm() const { return osm_; }
  const Taxonomy& taxonomy() const { return tax_; }
  const SemanticMap* semantic_map() const { return map_.get(); }
  const OsmPrior* osm_prior() const { return prior_.get(); }  // null without a confusion matrix
  const MatX2& gt() const { return gt_; }  // the keyframes' LiDAR ground truth, UTM
  const std::vector<double>& gt_psi() const { return gt_psi_; }
  const std::vector<long>& map_frames() const { return map_f_; }  // the frames to map
  const std::vector<long>& mapped() const { return mapped_; }     // and those mapped so far
  const std::vector<Vec2>& mapped_xy() const { return mapped_xy_; }  // where each was mapped (UTM)
  const MappedScan& last_mapped() const { return last_; }
  // frame f's scan at its ground-truth pose, classified by the map (0 for dropped points)
  std::vector<int> classify_frame(long f) const;
  std::string status() const;
  double map_seconds() const { return map_s_; }
  long survey_voxels() const { return static_cast<long>(survey_done_.size()); }  // projected so far
  double survey_offset() const { return dz_; }  // survey height -> map z (NaN until estimated)

 private:
  struct Scan {
    MatX3f levelled;
    Semantics sem;
    std::vector<float> intensity;
    std::vector<long> index;  // into the raw scan
    long raw = 0;             // points in the raw scan
  };
  Scan load(long f, bool semantics = true) const;
  MatX3f place(const MatX3f& levelled, const Vec2& xy, double psi, long f) const;
  void insert(long f, const Vec2& xy, double psi);
  // keyframe u's pose is ready: the frames anchored at it are placed off it, each mapped if it is a
  // frame to map, else (a keyframe) its geometry alone when the map has geometry and takes every
  // keyframe's (nvblox every_keyframe)
  void place_from(long u, const Vec2& xy, double psi);
  void settle_lag();
  void project_survey(const Vec2& xy, const MatX3f& pts, const std::vector<int8_t>& ground);
  std::string survey_status() const;

  PipelineConfig c_;
  Site site_;
  std::unique_ptr<osmloc::Run> run_;
  std::vector<long> kf_ids_;
  std::unique_ptr<Frames> frames_;
  MatX2 gt_;
  std::vector<double> gt_psi_;
  long T_ = 0, t_ = -1, settled_ = -1;  // settled_: the newest keyframe whose lag pose is settled
  bool locked_ = false;
  // mapping
  OsmMap osm_;
  Taxonomy tax_;
  std::vector<int> to_common_;
  bool alpha_ = false;
  std::shared_ptr<const OsmPrior> prior_;
  std::unique_ptr<SemanticMap> map_;
  std::unique_ptr<GroundSegmenter> ground_;
  std::vector<char> is_map_f_;
  std::vector<long> map_f_, mapped_;
  std::vector<Vec2> mapped_xy_;
  MappedScan last_;
  double map_s_ = 0, geo_s_ = 0;  // mapping, and geometry-only keyframes
  // the airborne survey
  struct SurveyParams {
    double weight = 0.1, radius = 50.0, ground_above_dem = 0.4;
    bool auto_offset = true;
    double offset = 0.0;
  } sp_;
  std::unique_ptr<Survey> survey_;
  std::unordered_set<int64_t> survey_done_;  // voxels already projected
  std::vector<double> dz_scans_;              // per-scan offset estimates (auto)
  double dz_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace olb
