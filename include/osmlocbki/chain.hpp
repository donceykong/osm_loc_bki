// A Levenberg-Marquardt solver for the one graph shape the localizer builds: a chain of Pose2
// linked by odometry (BetweenFactor), each pose with robust (Geman-McClure) priors from the
// front-end's hits. The Hessian is block tridiagonal, so a step is a block Thomas solve: O(n).
//
// It reproduces GTSAM 4.2's LevenbergMarquardtOptimizer with default parameters on that graph:
// the same factor errors and Jacobians (first-order Pose2 chart, between = inverse() * g),
// whitening and block reweighting, damping (lambda * I), step acceptance (model fidelity),
// lambda schedule and convergence tests. Results agree with GTSAM to rounding
// (tests/test_backend.cpp checks against GTSAM's own solutions).
#pragma once

#include <vector>

#include "osmlocbki/common.hpp"

namespace olb {

// A Pose2 stored as GTSAM stores it: translation and a (cos, sin) pair.
struct Pose2d {
  double x = 0, y = 0, c = 1, s = 0;
  static Pose2d from(double x, double y, double theta) { return {x, y, std::cos(theta), std::sin(theta)}; }
  double theta() const { return std::atan2(s, c); }
  Vec2 xy() const { return Vec2(x, y); }
  Pose2d inverse() const;
  Pose2d operator*(const Pose2d& p) const;
  Pose2d between(const Pose2d& p) const { return inverse() * p; }
  // chart coordinates (x, y, theta) of between(p)
  Eigen::Vector3d local(const Pose2d& p) const;
  Pose2d retract(const Eigen::Vector3d& v) const { return *this * from(v[0], v[1], v[2]); }
  Eigen::Matrix3d adjoint() const;
};

struct ChainGraph {
  std::vector<Pose2d> odo;                // n - 1: measured motion from pose i to pose i + 1
  std::vector<Eigen::Vector3d> odo_isig;  // n - 1: 1 / sigma of each
  std::vector<std::vector<Pose2d>> hits;  // n: prior measurements on each pose
  Eigen::Vector3d hit_isig;               // 1 / sigma of every hit
  double gm_c = 0.5;                      // Geman-McClure width
  double error(const std::vector<Pose2d>& x) const;
};

struct LMParams {
  int max_iters = 100;
  double lambda_initial = 1e-5, lambda_factor = 10.0, lambda_upper = 1e5, lambda_lower = 0.0;
  double min_model_fidelity = 1e-3, relative_tol = 1e-5, absolute_tol = 1e-5, error_tol = 0.0;
};

// -> the optimized poses; `iters` = successful iterations
std::vector<Pose2d> optimize(const ChainGraph& g, std::vector<Pose2d> x, const LMParams& p, int* iters = nullptr);

}  // namespace olb
