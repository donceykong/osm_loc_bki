// Shared types and the few NumPy semantics the port has to reproduce (floor-mod, round half to even).
#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <vector>

namespace olb {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat4 = Eigen::Matrix4d;
using MatX2 = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using MatX3 = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatX3f = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VecXd = Eigen::VectorXd;

constexpr double kPi = 3.14159265358979323846;

inline double radians(double d) { return d * (kPi / 180.0); }
inline double degrees(double r) { return r * (180.0 / kPi); }

// np.mod / torch.remainder: the result takes the divisor's sign
inline double pymod(double a, double m) {
  double r = std::fmod(a, m);
  if (r != 0.0) {
    if ((m < 0) != (r < 0)) r += m;
  } else {
    r = std::copysign(0.0, m);
  }
  return r;
}

inline long pymod(long a, long m) {
  long r = a % m;
  return (r != 0 && ((r < 0) != (m < 0))) ? r + m : r;
}

inline double wrap180(double a) { return pymod(a + 180.0, 360.0) - 180.0; }

// np.round: half to even (the default rounding mode)
inline double round_even(double x) { return std::nearbyint(x); }

inline Eigen::Matrix2d rot2(double a) {
  double c = std::cos(a), s = std::sin(a);
  Eigen::Matrix2d R;
  R << c, -s, s, c;
  return R;
}

inline Eigen::Matrix3d rz(double a) {
  double c = std::cos(a), s = std::sin(a);
  Eigen::Matrix3d R;
  R << c, -s, 0, s, c, 0, 0, 0, 1;
  return R;
}

class Stopwatch {
 public:
  Stopwatch() : t0_(std::chrono::steady_clock::now()) {}
  double seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
  }

 private:
  std::chrono::steady_clock::time_point t0_;
};

}  // namespace olb
