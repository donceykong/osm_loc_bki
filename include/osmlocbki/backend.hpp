// Localization back-end: the front-end's noisy per-scan hits plus odometry -> a trajectory, online.
//
//   KeyframeBuffer  dead-reckoned odometry and hits per keyframe
//   RigidTracker    hypotheses = rigid placements of the dead-reckoned trajectory
//   HistoryCheck    every candidate placement scored on the whole history
//   WindowGraph     sliding-window pose graph (odometry + robust hit priors), chain-solved
//   Localizer       the per-keyframe pipeline tying them together
//
// Outputs per keyframe, using nothing after it:
//   rigid     the history check's best placement
//   graph     the window graph's newest pose, seeded by rigid
//   hybrid    rigid when within agree_m of graph, else graph
//   lag       each pose from the window lag_m of travel later (NaN until settled)
//   filtered  odometry carries the pose; each keyframe it moves `alpha` of the way to graph
//             (gaps over snap_m, i.e. relocalizations, are taken at once)
#pragma once

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "osmlocbki/chain.hpp"
#include "osmlocbki/frontend.hpp"

namespace olb {

class KeyframeBuffer {
 public:
  explicit KeyframeBuffer(int k) : k(k) {}
  // dead-reckon one odometry step (body-frame translation, heading change deg; none for the
  // first keyframe) and store the keyframe's hits -> its index
  int append(const std::optional<Vec2>& d_body, double d_psi, const Hits& hits);
  int size() const { return static_cast<int>(o.size()); }
  Vec2 e_xy(int t, int j) const { return Vec2(e_xy_[(t * k + j) * 2], e_xy_[(t * k + j) * 2 + 1]); }
  double e_psi(int t, int j) const { return e_psi_[t * k + j]; }

  int k;
  std::vector<Vec2> o;
  std::vector<double> o_psi, o_psi_rad, dist;  // deg, rad, odometry path length
  std::vector<double> e_xy_, e_psi_;           // (T, k, 2), (T, k) deg
};

// a rigid placement of the dead-reckoned trajectory on the map
struct Placement {
  double theta;  // rad
  Vec2 t;        // map frame
};

// When a hit agrees with a placement, and what agreeing is worth (log-odds)
struct HitModel {
  double radius = 5.0, psi_tol = 15.0, p_hit = 0.25, p_false = 0.02, pos_sigma = 3.0, psi_sigma = 1.5;
  int heading_cap = 5;
  double hit_gain() const;
  double miss_cost() const;
  // heading prior weight (m^2 / rad^2) of the placement fit
  double lambda() const;
};

// Running sums of a 2D Procrustes fit (trajectory points p onto hits q), with a heading prior from
// the hits' heading residuals capped at `heading_cap` hits' worth
struct Procrustes {
  double n = 0, sdot = 0, scross = 0, hc = 0, hs = 0;
  Vec2 sp = Vec2::Zero(), sq = Vec2::Zero();
  void add(const Vec2& p, const Vec2& q, double psi_res_rad);
  Placement fit(const HitModel& m) const;
};

class RigidTracker {
 public:
  struct Params : HitModel {
    double drop = 15.0;
    int grace = 50, max_hyp = 2000;
  };
  RigidTracker() : RigidTracker(Params()) {}
  explicit RigidTracker(const Params& p) : p_(p) {}
  void step(const Hits& hits, const std::optional<Vec2>& d_body, double d_psi);
  // live hypotheses' placements, map frame (origin added)
  std::vector<Placement> placements() const;

 private:
  struct Hyp {
    double lo = 0;
    int age = 0;
    Procrustes fit;
    Placement pl{0.0, Vec2::Zero()};
  };
  void pose(const Hyp& h, Vec2& xy, double& psi) const;

  Params p_;
  Vec2 o_ = Vec2::Zero();
  double o_psi_ = 0;
  std::optional<Vec2> origin_;
  std::vector<Hyp> h_;
};

class HistoryCheck {
 public:
  struct Params : HitModel {
    int n_refine = 2, carry = 20;
  };
  HistoryCheck() : HistoryCheck(Params()) {}
  explicit HistoryCheck(const Params& p) : p_(p) {}
  // add keyframe t (its hits are buf's newest) and return the best placement's pose at t
  void step(const KeyframeBuffer& buf, const std::vector<Placement>& tracked, Vec2& xy, double& psi_deg);

 private:
  // log-odds of each placement on keyframes 0..t
  std::vector<double> check(const KeyframeBuffer& buf, const std::vector<Placement>& c, int t, double radius) const;
  // each placement re-fitted to the hits that agree with it
  std::vector<Placement> refine(const KeyframeBuffer& buf, const std::vector<Placement>& c, int t, double radius) const;
  Params p_;
  std::vector<Placement> carried_;
};

struct GraphParams {
  double window_m = 200.0, every_m = 2.0;
  double odo_scale = 0.0015, odo_yaw = 0.035, odo_floor_m = 0.0025, odo_floor_deg = 0.01;
  double hit_sigma = 1.5, hit_psi = 10.0, gm_c = 0.5;
  int iters = 20;
  double carry_margin = 0.0;
  static GraphParams drifting();  // for odometry that drifts (the default trusts it tightly)
};

// odometry from keyframe u to v
Pose2d relative(const KeyframeBuffer& b, int u, int v);

class WindowGraph {
 public:
  WindowGraph(const KeyframeBuffer& buf, const GraphParams& p) : buf_(buf), p_(p) {}
  // solve the window ending at keyframe t, seeded by the history check's pose at t (heading rad)
  void step(int t, const Vec2& rigid_xy, double rigid_psi, Vec2& xy, double& psi_deg);
  // the latest solution's pose of any keyframe (its nearest window pose, chained by odometry)
  Pose2d pose_of(int u) const;

 private:
  // one keyframe per every_m bucket of the window_m of path behind t, and t
  std::vector<int> window(int t) const;
  const KeyframeBuffer& buf_;
  GraphParams p_;
  std::map<int, Pose2d> prev_;
  std::vector<Pose2d> sol_;
  std::vector<int> sub_;
};

struct FilterParams {
  double alpha = 0.15, snap_m = 8.0;
};

enum Output : int { RIGID, GRAPH, HYBRID, LAG, FILTERED, kNumOutputs };
Output output_of(const std::string& name);  // throws on an unknown name

struct Estimate {
  int t;
  std::array<Vec2, kNumOutputs> xy;    // this keyframe's outputs (lag: NaN, it settles later)
  std::array<double, kNumOutputs> psi;  // deg
  bool agree;                           // rigid and graph within agree_m
  std::vector<int> lagged;              // keyframes whose lag pose settled at this step
};

class Localizer {
 public:
  Localizer(std::shared_ptr<const Matcher> matcher, int k = 20, const GraphParams& graph = GraphParams(),
            double agree_m = 2.0, double lag_m = 60.0, const FilterParams& filter = FilterParams());
  Localizer(const Localizer&) = delete;  // the window graph refers to the buffer
  Localizer& operator=(const Localizer&) = delete;
  // one keyframe: its scan descriptor and the odometry since the previous one
  Estimate step(const double* q_desc, const std::optional<Vec2>& d_body, double d_psi);
  // the same with the front-end's hits given (replays, tests)
  Estimate step_hits(const Hits& hits, const std::optional<Vec2>& d_body, double d_psi);
  // settle every lag pose still open (end of the run) -> keyframes settled now
  std::vector<int> finish_lag();
  int size() const { return buf_.size(); }
  // (n, 2) and (n,) deg of one output
  MatX2 trajectory(Output o) const;
  const std::vector<double>& heading(Output o) const { return psi_[o]; }
  Vec2 xy(Output o, int t) const { return xy_[o][t]; }
  const KeyframeBuffer& buf() const { return buf_; }
  struct Timing {
    double match = 0, rigid = 0, graph = 0;  // s, summed over keyframes
  } timing;

 private:
  void settle(int u, const Pose2d& q);
  std::shared_ptr<const Matcher> match_;
  int k_;
  double agree_m_, lag_m_;
  GraphParams params_;
  KeyframeBuffer buf_;
  RigidTracker rigid_;
  HistoryCheck history_;
  WindowGraph graph_;
  FilterParams filter_;
  std::array<std::vector<Vec2>, kNumOutputs> xy_;
  std::array<std::vector<double>, kNumOutputs> psi_;
};

}  // namespace olb
