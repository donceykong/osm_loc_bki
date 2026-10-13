// The online localizer: per keyframe t, using nothing after t, match -> rigid tracker + history
// check -> window graph -> hybrid, filtered and the fixed-lag output.
#include "osmlocbki/backend.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace olb {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

double angle_diff(double a, double b) { return std::abs(pymod(a - b + 180.0, 360.0) - 180.0); }

Vec2 rotate(double c, double s, const Vec2& p) { return Vec2(c * p[0] - s * p[1], s * p[0] + c * p[1]); }

}  // namespace

// ---- keyframes, hit model, placement fit ------------------------------------------------------

int KeyframeBuffer::append(const std::optional<Vec2>& d_body, double d_psi, const Hits& hits) {
  if (hits.xy.rows() != k) throw std::invalid_argument("expected k hits per keyframe");
  int t = size();
  if (t == 0) {
    o.push_back(Vec2::Zero()), o_psi.push_back(0.0), dist.push_back(0.0);
  } else {
    double a = radians(o_psi[t - 1]);
    o.push_back(o[t - 1] + rotate(std::cos(a), std::sin(a), *d_body));
    o_psi.push_back(o_psi[t - 1] + d_psi);
    dist.push_back(dist[t - 1] + (o[t] - o[t - 1]).norm());
  }
  o_psi_rad.push_back(radians(o_psi[t]));
  for (int j = 0; j < k; ++j) {
    e_xy_.push_back(hits.xy(j, 0)), e_xy_.push_back(hits.xy(j, 1));
    e_psi_.push_back(hits.psi[j]);
  }
  return t;
}

double HitModel::hit_gain() const { return std::log(p_hit / p_false); }
double HitModel::miss_cost() const { return -std::log((1.0 - p_hit) / (1.0 - p_false)); }
double HitModel::lambda() const { return std::pow(pos_sigma / radians(psi_sigma), 2); }

void Procrustes::add(const Vec2& p, const Vec2& q, double psi_res_rad) {
  n += 1;
  sp += p, sq += q;
  sdot += p[0] * q[0] + p[1] * q[1];
  scross += p[0] * q[1] - p[1] * q[0];
  hc += std::cos(psi_res_rad), hs += std::sin(psi_res_rad);
}

Placement Procrustes::fit(const HitModel& m) const {
  Vec2 pbar = sp / n, qbar = sq / n;
  double c_dot = sdot - n * (pbar[0] * qbar[0] + pbar[1] * qbar[1]);
  double c_cross = scross - n * (pbar[0] * qbar[1] - pbar[1] * qbar[0]);
  double w = m.lambda() * std::min(n, static_cast<double>(m.heading_cap)) / n;
  double theta = std::atan2(c_cross + w * hs, c_dot + w * hc);
  return {theta, qbar - rotate(std::cos(theta), std::sin(theta), pbar)};
}

// ---- rigid tracker: every hit either agrees with a hypothesis (a vote and a re-fit) or starts a
// new one; converged hypotheses merge and, after a grace period, ones far below the leader go ----

void RigidTracker::pose(const Hyp& h, Vec2& xy, double& psi) const {
  xy = rotate(std::cos(h.pl.theta), std::sin(h.pl.theta), o_) + h.pl.t + *origin_;
  psi = pymod(degrees(h.pl.theta) + o_psi_, 360.0);
}

void RigidTracker::step(const Hits& hits, const std::optional<Vec2>& d_body, double d_psi) {
  const int K = static_cast<int>(hits.xy.rows());
  if (!origin_) origin_ = Vec2(hits.xy.row(0).transpose());
  if (d_body) {
    double a = radians(o_psi_);
    o_ = o_ + rotate(std::cos(a), std::sin(a), *d_body);
    o_psi_ = o_psi_ + d_psi;
  }
  const double gain = p_.hit_gain(), miss = p_.miss_cost();
  auto vote = [&](Hyp& h, int j) {
    h.fit.add(o_, Vec2(hits.xy.row(j).transpose()) - *origin_, radians(hits.psi[j] - o_psi_));
    h.pl = h.fit.fit(p_);
  };
  std::vector<char> claimed(K, 0);
  for (auto& h : h_) {
    Vec2 xy;
    double psi;
    pose(h, xy, psi);
    int jbest = -1;
    double dbest = std::numeric_limits<double>::infinity();
    for (int j = 0; j < K; ++j) {
      double dist = (xy - Vec2(hits.xy.row(j).transpose())).norm();
      if (dist <= p_.radius && angle_diff(psi, hits.psi[j]) <= p_.psi_tol) {
        claimed[j] = 1;
        if (dist < dbest) dbest = dist, jbest = j;
      }
    }
    h.lo += jbest >= 0 ? gain : -miss;
    if (jbest >= 0) vote(h, jbest);
  }
  // births: unclaimed hits, one per `radius` neighbourhood
  std::vector<int> born;
  for (int j = 0; j < K; ++j) {
    if (claimed[j]) continue;
    bool near = false;
    for (int b : born) near |= (Vec2(hits.xy.row(j) - hits.xy.row(b))).norm() <= p_.radius;
    if (!near) born.push_back(j);
  }
  for (size_t i = 0; i < born.size(); ++i) {
    Hyp h;
    h.lo = -1e-3 * static_cast<double>(i);
    vote(h, born[i]);
    h_.push_back(h);
  }
  for (auto& h : h_) h.age += 1;
  // merge converged hypotheses (keep the strongest), prune after the grace period
  const size_t H = h_.size();
  std::vector<Vec2> xy(H);
  std::vector<double> psi(H);
  for (size_t i = 0; i < H; ++i) pose(h_[i], xy[i], psi[i]);
  std::vector<size_t> order(H);
  std::iota(order.begin(), order.end(), size_t(0));
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return -h_[a].lo < -h_[b].lo; });
  std::vector<size_t> keep;
  for (size_t i : order) {
    if (static_cast<int>(keep.size()) >= p_.max_hyp) break;
    if (h_[i].lo < h_[order[0]].lo - p_.drop && h_[i].age > p_.grace) continue;
    bool close = false;
    for (size_t k : keep)
      if ((xy[k] - xy[i]).norm() <= 0.5 * p_.radius && angle_diff(psi[k], psi[i]) <= p_.psi_tol) {
        close = true;
        break;
      }
    if (!close) keep.push_back(i);
  }
  std::vector<Hyp> kept;
  kept.reserve(keep.size());
  for (size_t i : keep) kept.push_back(h_[i]);
  h_.swap(kept);
}

std::vector<Placement> RigidTracker::placements() const {
  std::vector<Placement> out;
  out.reserve(h_.size());
  for (const auto& h : h_) out.push_back({h.pl.theta, h.pl.t + *origin_});
  return out;
}

// ---- history check: hit_gain where one of keyframe u's own hits agrees with where the placement
// puts the robot at u, -miss_cost where none does --------------------------------------------------

namespace {

// keyframe u under a placement -> index of the nearest agreeing hit and its distance (inf if none)
struct Agree {
  int j;
  double d;
};

inline Agree agree(const KeyframeBuffer& b, double c, double s, double th_deg, const Vec2& tv, int u, double psi_tol) {
  const Vec2& o = b.o[u];
  double x = c * o[0] - s * o[1] + tv[0], y = s * o[0] + c * o[1] + tv[1];
  double psi = pymod(th_deg + b.o_psi[u], 360.0);
  Agree a{0, std::numeric_limits<double>::infinity()};
  for (int j = 0; j < b.k; ++j) {
    const double* e = &b.e_xy_[(static_cast<size_t>(u) * b.k + j) * 2];
    if (angle_diff(psi, b.e_psi_[static_cast<size_t>(u) * b.k + j]) > psi_tol) continue;
    double d = std::sqrt((x - e[0]) * (x - e[0]) + (y - e[1]) * (y - e[1]));
    if (d < a.d) a = {j, d};
  }
  return a;
}

}  // namespace

std::vector<double> HistoryCheck::check(const KeyframeBuffer& b, const std::vector<Placement>& cand, int t,
                                        double radius) const {
  const double gain = p_.hit_gain(), miss = p_.miss_cost();
  std::vector<double> lo(cand.size());
#pragma omp parallel for schedule(dynamic, 8)
  for (long m = 0; m < static_cast<long>(cand.size()); ++m) {
    double c = std::cos(cand[m].theta), s = std::sin(cand[m].theta), th = degrees(cand[m].theta);
    long nh = 0;
    for (int u = 0; u <= t; ++u) nh += agree(b, c, s, th, cand[m].t, u, p_.psi_tol).d <= radius;
    lo[m] = nh * gain - (t + 1 - nh) * miss;
  }
  return lo;
}

std::vector<Placement> HistoryCheck::refine(const KeyframeBuffer& b, const std::vector<Placement>& cand, int t,
                                            double radius) const {
  std::vector<Placement> out(cand);
#pragma omp parallel for schedule(dynamic, 8)
  for (long m = 0; m < static_cast<long>(cand.size()); ++m) {
    double c = std::cos(cand[m].theta), s = std::sin(cand[m].theta), th = degrees(cand[m].theta);
    Procrustes fit;
    for (int u = 0; u <= t; ++u) {
      Agree a = agree(b, c, s, th, cand[m].t, u, p_.psi_tol);
      if (a.d <= radius) fit.add(b.o[u], b.e_xy(u, a.j), radians(b.e_psi(u, a.j) - b.o_psi[u]));
    }
    if (fit.n >= 2) out[m] = fit.fit(p_);
  }
  return out;
}

void HistoryCheck::step(const KeyframeBuffer& b, const std::vector<Placement>& tracked, Vec2& xy, double& psi_deg) {
  const int t = b.size() - 1;
  std::vector<Placement> cand(tracked);
  // births: one placement per hit of this keyframe (heading from the hit, position exact)
  for (int j = 0; j < b.k; ++j) {
    double th = radians(b.e_psi(t, j) - b.o_psi[t]);
    cand.push_back({th, b.e_xy(t, j) - rotate(std::cos(th), std::sin(th), b.o[t])});
  }
  cand.insert(cand.end(), carried_.begin(), carried_.end());
  for (int it = 0; it < p_.n_refine; ++it) cand = refine(b, cand, t, it == 0 ? 2.0 * p_.radius : p_.radius);
  std::vector<double> lo = check(b, cand, t, p_.radius);
  size_t best = 0;
  for (size_t i = 1; i < lo.size(); ++i)
    if (lo[i] > lo[best]) best = i;
  std::vector<size_t> order(lo.size());
  std::iota(order.begin(), order.end(), size_t(0));
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t c) { return -lo[a] < -lo[c]; });
  carried_.clear();
  for (size_t i = 0; i < order.size() && static_cast<int>(i) < p_.carry; ++i) carried_.push_back(cand[order[i]]);
  const Placement& P = cand[best];
  xy = rotate(std::cos(P.theta), std::sin(P.theta), b.o[t]) + P.t;
  psi_deg = pymod(degrees(P.theta) + b.o_psi[t], 360.0);
}

// ---- the window graph: odometry between consecutive poses, sigmas growing with the step, and one
// Geman-McClure prior per front-end hit; a chain, solved by the chain solver (chain.hpp) ----------

GraphParams GraphParams::drifting() {
  GraphParams p;
  p.window_m = 100.0, p.odo_scale = 0.03, p.odo_yaw = 0.7, p.odo_floor_m = 0.05, p.odo_floor_deg = 0.2;
  p.hit_sigma = 3.0, p.gm_c = 1.0;
  return p;
}

Pose2d relative(const KeyframeBuffer& b, int u, int v) {
  double a = -b.o_psi_rad[u], c = std::cos(a), s = std::sin(a);
  Vec2 d = b.o[v] - b.o[u];
  return Pose2d::from(c * d[0] + -s * d[1], s * d[0] + c * d[1], b.o_psi_rad[v] - b.o_psi_rad[u]);
}

namespace {

// the chain over keyframes `keys` (ascending): odometry between neighbours, every hit as a prior
ChainGraph chain(const KeyframeBuffer& b, const GraphParams& p, const std::vector<int>& keys) {
  ChainGraph g;
  g.gm_c = p.gm_c;
  g.hit_isig = Vec3(p.hit_sigma, p.hit_sigma, radians(p.hit_psi)).cwiseInverse();
  for (size_t i = 0; i + 1 < keys.size(); ++i) {
    int u = keys[i], v = keys[i + 1];
    double step = b.dist[v] - b.dist[u];
    g.odo.push_back(relative(b, u, v));
    g.odo_isig.push_back(Vec3(p.odo_scale * step + p.odo_floor_m, p.odo_scale * step + p.odo_floor_m,
                              radians(p.odo_yaw) * std::sqrt(step) + radians(p.odo_floor_deg))
                             .cwiseInverse());
  }
  for (int u : keys) {
    std::vector<Pose2d> h;
    for (int j = 0; j < b.k; ++j) {
      Vec2 e = b.e_xy(u, j);
      h.push_back(Pose2d::from(e[0], e[1], radians(b.e_psi(u, j))));
    }
    g.hits.push_back(std::move(h));
  }
  return g;
}

}  // namespace

std::vector<int> WindowGraph::window(int t) const {
  const auto& d = buf_.dist;
  // one keyframe per `every_m` bucket of path behind t (the first in each), plus t itself
  std::vector<int> sub;
  double last_key = std::numeric_limits<double>::quiet_NaN();
  for (int u = 0; u <= t; ++u) {
    if (!(d[u] > d[t] - p_.window_m)) continue;
    double key = std::floor((d[t] - d[u]) / p_.every_m);
    if (key != last_key) sub.push_back(u), last_key = key;
  }
  if (sub.empty() || sub.back() != t) sub.push_back(t);
  return sub;
}

void WindowGraph::step(int t, const Vec2& rigid_xy, double rigid_psi, Vec2& xy, double& psi_deg) {
  const auto& o = buf_.o;
  const auto& o_psi = buf_.o_psi_rad;
  std::vector<int> sub = window(t);
  ChainGraph g = chain(buf_, p_, sub);
  LMParams lm;
  lm.max_iters = p_.iters;
  // rigid: the history check's placement applied to the window
  Eigen::Matrix2d R = rot2(rigid_psi - o_psi[t]);
  std::vector<Pose2d> rigid;
  for (int u : sub) {
    Vec2 q = R * (o[u] - o[t]) + rigid_xy;
    rigid.push_back(Pose2d::from(q[0], q[1], o_psi[u] + rigid_psi - o_psi[t]));
  }
  std::vector<Pose2d> best = optimize(g, rigid, lm);
  // carry: the previous solution, new poses chained on by odometry
  std::vector<int> have;
  for (int u : sub)
    if (prev_.count(u)) have.push_back(u);
  if (!have.empty()) {
    int first = *std::min_element(have.begin(), have.end()), last = have.back();
    std::vector<Pose2d> carry;
    for (int u : sub) {
      if (prev_.count(u))
        carry.push_back(prev_.at(u));
      else if (u > last)
        carry.push_back(prev_.at(last) * relative(buf_, last, u));
      else
        carry.push_back(prev_.at(first) * relative(buf_, first, u));
    }
    std::vector<Pose2d> sol = optimize(g, carry, lm);
    if (g.error(sol) < g.error(best) - std::max(p_.carry_margin, 1e-9)) best = sol;
  }
  sol_ = best;
  sub_ = sub;
  prev_.clear();
  for (size_t i = 0; i < sub.size(); ++i) prev_.emplace(sub[i], best[i]);
  const Pose2d& q = best.back();
  xy = Vec2(q.x, q.y);
  psi_deg = pymod(degrees(q.theta()), 360.0);
}

Pose2d WindowGraph::pose_of(int u) const {
  size_t f = 0;
  for (size_t i = 0; i < sub_.size(); ++i)
    if (std::abs(sub_[i] - u) < std::abs(sub_[f] - u)) f = i;
  return sol_[f] * relative(buf_, sub_[f], u);
}

// ---- the localizer ----------------------------------------------------------------------------

Output output_of(const std::string& name) {
  static const char* names[kNumOutputs] = {"rigid", "graph", "hybrid", "lag", "filtered"};
  for (int o = 0; o < kNumOutputs; ++o)
    if (name == names[o]) return static_cast<Output>(o);
  throw std::invalid_argument("no output '" + name + "': rigid, graph, hybrid, lag or filtered");
}

Localizer::Localizer(std::shared_ptr<const Matcher> matcher, int k, const GraphParams& graph, double agree_m,
                     double lag_m, const FilterParams& filter)
    : match_(std::move(matcher)), k_(k), agree_m_(agree_m), lag_m_(lag_m), params_(graph), buf_(k),
      graph_(buf_, graph), filter_(filter) {}

Estimate Localizer::step(const double* q_desc, const std::optional<Vec2>& d_body, double d_psi) {
  if (!match_) throw std::logic_error("no matcher: use step_hits");
  Stopwatch sw;
  Hits hits = (*match_)(q_desc, k_);
  timing.match += sw.seconds();
  return step_hits(hits, d_body, d_psi);
}

void Localizer::settle(int u, const Pose2d& q) {
  xy_[LAG][u] = q.xy(), psi_[LAG][u] = pymod(degrees(q.theta()), 360.0);
}

Estimate Localizer::step_hits(const Hits& hits, const std::optional<Vec2>& d_body, double d_psi) {
  Stopwatch sw;
  int t = buf_.append(d_body, d_psi, hits);
  rigid_.step(hits, d_body, d_psi);
  Estimate e;
  e.t = t;
  history_.step(buf_, rigid_.placements(), e.xy[RIGID], e.psi[RIGID]);
  double t1 = sw.seconds();
  graph_.step(t, e.xy[RIGID], radians(e.psi[RIGID]), e.xy[GRAPH], e.psi[GRAPH]);
  e.agree = (e.xy[GRAPH] - e.xy[RIGID]).norm() < agree_m_;
  const Output h = e.agree ? RIGID : GRAPH;
  e.xy[HYBRID] = e.xy[h], e.psi[HYBRID] = e.psi[h];
  // filtered: carried by odometry, pulled toward the graph
  e.xy[FILTERED] = e.xy[GRAPH], e.psi[FILTERED] = e.psi[GRAPH];
  if (t > 0) {
    const auto& o = buf_.o;
    Vec2 step = rot2(-buf_.o_psi_rad[t - 1]) * (o[t] - o[t - 1]);
    Vec2 f_xy = xy_[FILTERED].back() + rot2(radians(psi_[FILTERED].back())) * step;
    double f_psi = psi_[FILTERED].back() + (buf_.o_psi[t] - buf_.o_psi[t - 1]);
    if ((e.xy[GRAPH] - f_xy).norm() > filter_.snap_m) {
      f_xy = e.xy[GRAPH], f_psi = e.psi[GRAPH];
    } else {
      f_xy += filter_.alpha * (e.xy[GRAPH] - f_xy);
      f_psi += filter_.alpha * wrap180(e.psi[GRAPH] - f_psi);
    }
    e.xy[FILTERED] = f_xy, e.psi[FILTERED] = pymod(f_psi, 360.0);
  }
  e.xy[LAG] = Vec2(kNaN, kNaN), e.psi[LAG] = kNaN;
  for (int o = 0; o < kNumOutputs; ++o) xy_[o].push_back(e.xy[o]), psi_[o].push_back(e.psi[o]);
  if (lag_m_ > 0) {
    const auto& d = buf_.dist;
    for (int u = 0; u <= t; ++u)
      if (std::isnan(xy_[LAG][u][0]) && d[t] - d[u] >= lag_m_ && d[t] - d[u] < params_.window_m) {
        settle(u, graph_.pose_of(u));
        e.lagged.push_back(u);
      }
  }
  timing.rigid += t1;
  timing.graph += sw.seconds() - t1;
  return e;
}

std::vector<int> Localizer::finish_lag() {
  const int n = size();
  const auto& d = buf_.dist;
  std::vector<int> settled;
  for (int u = 0; u < n; ++u) {
    if (!std::isnan(xy_[LAG][u][0])) continue;
    if (d[n - 1] - d[u] < params_.window_m)
      settle(u, graph_.pose_of(u));
    else
      xy_[LAG][u] = xy_[GRAPH][u], psi_[LAG][u] = psi_[GRAPH][u];
    settled.push_back(u);
  }
  return settled;
}

MatX2 Localizer::trajectory(Output o) const {
  MatX2 out(xy_[o].size(), 2);
  for (size_t i = 0; i < xy_[o].size(); ++i) out.row(i) = xy_[o][i].transpose();
  return out;
}

}  // namespace olb
