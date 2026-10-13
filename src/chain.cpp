// Not compiled with FMA, like the GTSAM library it reproduces: contracted multiply-adds round
// differently.
#include "osmlocbki/chain.hpp"

#include <limits>

namespace olb {

namespace {

using M3 = Eigen::Matrix3d;
using V3 = Eigen::Vector3d;

// Rot2::fromCosSin: normalized only when off the unit circle by more than 1e-10
inline void normalize(double& c, double& s) {
  double scale = c * c + s * s;
  if (std::abs(scale - 1.0) > 1e-10) {
    scale = 1 / std::sqrt(scale);
    c *= scale, s *= scale;
  }
}

}  // namespace

Pose2d Pose2d::inverse() const {
  // Pose2(r.inverse(), r.unrotate(-t))
  double tx = -x, ty = -y;
  return {c * tx + s * ty, -s * tx + c * ty, c, -s};
}

Pose2d Pose2d::operator*(const Pose2d& p) const {
  double rc = c * p.c - s * p.s, rs = s * p.c + c * p.s;
  normalize(rc, rs);
  return {x + (c * p.x + -s * p.y), y + (s * p.x + c * p.y), rc, rs};
}

V3 Pose2d::local(const Pose2d& p) const {
  Pose2d h = between(p);
  return V3(h.x, h.y, h.theta());
}

M3 Pose2d::adjoint() const {
  M3 A;
  A << c, -s, y, s, c, -x, 0.0, 0.0, 1.0;
  return A;
}

namespace {

double gm_loss(double c, double r2) { return 0.5 * (c * c * r2) / (c * c + r2); }
double gm_sqrt_weight(double c, double r) {
  double c2 = c * c, c2e = c2 + r * r;
  return std::sqrt(c2 * c2 / (c2e * c2e));
}

// The linearized system at one estimate: whitened (and reweighted) Jacobians and right-hand sides
// per factor, and the block tridiagonal Hessian and gradient they add up to.
struct Linear {
  // priors: A = w * diag(isig) (block on pose i), b
  std::vector<double> pw;    // per prior: w
  std::vector<V3> pb;        // per prior: b
  std::vector<int> pk;       // per prior: pose
  // between i -> i + 1: A1, A2 = diag(isig), b
  std::vector<M3> a1;
  std::vector<V3> bb;
  std::vector<M3> D, U;      // Hessian: diagonal blocks, (i, i + 1) blocks
  std::vector<V3> g;         // A^T b

  double error(const ChainGraph& G, const std::vector<V3>* d) const {
    double e = 0;
    for (size_t k = 0; k < pb.size(); ++k) {
      V3 r = -pb[k];
      if (d) r += pw[k] * G.hit_isig.cwiseProduct((*d)[pk[k]]);
      e += r.squaredNorm();
    }
    for (size_t i = 0; i < bb.size(); ++i) {
      V3 r = -bb[i];
      if (d) r += a1[i] * (*d)[i] + G.odo_isig[i].cwiseProduct((*d)[i + 1]);
      e += r.squaredNorm();
    }
    return 0.5 * e;
  }
};

Linear linearize(const ChainGraph& G, const std::vector<Pose2d>& x) {
  const size_t n = x.size();
  Linear L;
  L.D.assign(n, M3::Zero());
  L.U.assign(n > 0 ? n - 1 : 0, M3::Zero());
  L.g.assign(n, V3::Zero());
  const V3& hs = G.hit_isig;
  const M3 Ah = hs.asDiagonal();
  for (size_t i = 0; i < n; ++i)
    for (const Pose2d& z : G.hits[i]) {
      // PriorFactor: error -Local(x, z), H = I; whitened, then block-reweighted
      V3 b = x[i].local(z).cwiseProduct(hs);  // b = -error, whitened
      double w = gm_sqrt_weight(G.gm_c, b.norm());
      b *= w;
      L.pw.push_back(w), L.pb.push_back(b), L.pk.push_back(static_cast<int>(i));
      M3 A = w * Ah;
      L.D[i] += A.transpose() * A;
      L.g[i] += A.transpose() * b;
    }
  for (size_t i = 0; i + 1 < n; ++i) {
    // BetweenFactor: hx = between(x_i, x_i+1), error Local(z, hx), H1 = -hx.inverse().AdjointMap(), H2 = I
    Pose2d hx = x[i].between(x[i + 1]);
    V3 e = G.odo[i].local(hx);
    const V3& is = G.odo_isig[i];
    M3 A1 = is.asDiagonal() * (-hx.inverse().adjoint());
    M3 A2 = is.asDiagonal();
    V3 b = -e.cwiseProduct(is);
    L.a1.push_back(A1), L.bb.push_back(b);
    L.D[i] += A1.transpose() * A1;
    L.D[i + 1] += A2.transpose() * A2;
    L.U[i] += A1.transpose() * A2;
    L.g[i] += A1.transpose() * b;
    L.g[i + 1] += A2.transpose() * b;
  }
  return L;
}

// (H + lambda I) d = g on the block tridiagonal H -> false if not positive definite
bool solve(const Linear& L, double lambda, std::vector<V3>& d) {
  const size_t n = L.D.size();
  std::vector<Eigen::LLT<M3>> S(n);
  std::vector<V3> y(n);
  for (size_t i = 0; i < n; ++i) {
    M3 Si = L.D[i] + lambda * M3::Identity();
    V3 yi = L.g[i];
    if (i > 0) {
      M3 Lt = S[i - 1].solve(L.U[i - 1]);  // S_{i-1}^-1 U_{i-1}
      Si -= L.U[i - 1].transpose() * Lt;
      yi -= Lt.transpose() * y[i - 1];
    }
    S[i].compute(Si);
    if (S[i].info() != Eigen::Success) return false;
    y[i] = yi;
  }
  d.assign(n, V3::Zero());
  for (size_t k = n; k-- > 0;) {
    V3 r = y[k];
    if (k + 1 < n) r -= L.U[k] * d[k + 1];
    d[k] = S[k].solve(r);
  }
  return true;
}

}  // namespace

double ChainGraph::error(const std::vector<Pose2d>& x) const {
  double e = 0;
  for (size_t i = 0; i < x.size(); ++i)
    for (const Pose2d& z : hits[i]) e += gm_loss(gm_c, (-x[i].local(z)).cwiseProduct(hit_isig).squaredNorm());
  for (size_t i = 0; i + 1 < x.size(); ++i)
    e += 0.5 * odo[i].local(x[i].between(x[i + 1])).cwiseProduct(odo_isig[i]).squaredNorm();
  return e;
}

std::vector<Pose2d> optimize(const ChainGraph& G, std::vector<Pose2d> x, const LMParams& p, int* iters) {
  double err = G.error(x), lambda = p.lambda_initial;
  int it = 0;
  if (iters) *iters = 0;
  if (err <= p.error_tol || it >= p.max_iters) return x;
  double new_err = err, cur;
  std::vector<V3> d;
  for (;;) {
    cur = new_err;
    // one outer iteration: linearize, then raise lambda until a step is taken (or give up)
    Linear L = linearize(G, x);
    for (;;) {
      bool success = false, stop = false;
      std::vector<Pose2d> xn;
      double step_err = std::numeric_limits<double>::infinity();
      if (solve(L, lambda, d)) {
        double old_lin = L.error(G, nullptr), new_lin = L.error(G, &d);
        double lin_change = old_lin - new_lin;
        if (lin_change >= 0) {
          xn.resize(x.size());
          for (size_t i = 0; i < x.size(); ++i) xn[i] = x[i].retract(d[i]);
          step_err = G.error(xn);
          double change = err - step_err;
          if (lin_change > std::numeric_limits<double>::epsilon() * old_lin)
            success = change / lin_change > p.min_model_fidelity;
          else
            success = true;
          if (std::abs(change) < p.relative_tol * err) stop = true;
        }
      }
      if (success) {
        x.swap(xn);
        err = step_err;
        lambda = std::max(p.lambda_lower, lambda / p.lambda_factor);
        ++it;
        break;
      }
      if (stop) break;
      lambda *= p.lambda_factor;
      if (lambda >= p.lambda_upper) break;
    }
    new_err = err;
    if (it >= p.max_iters || !std::isfinite(cur)) break;
    // GTSAM's checkConvergence
    if (new_err <= p.error_tol) break;
    double abs_dec = cur - new_err, rel_dec = abs_dec / cur;
    if (rel_dec <= p.relative_tol || abs_dec <= p.absolute_tol) break;
  }
  if (iters) *iters = it;
  return x;
}

}  // namespace olb
