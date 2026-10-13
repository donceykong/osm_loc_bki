#include "osmlocbki/survey.hpp"

#include <algorithm>
#include <stdexcept>

#include "osmlocbki/npz.hpp"

namespace olb {

Survey::Survey(const std::string& path) {
  Npz z = load_npz(path);
  auto o = z.at("origin").as<double>();
  origin_ = Vec2(o[0], o[1]);
  res_ = z.at("res").scalar<double>();
  epsg_ = static_cast<int>(z.at("epsg").scalar<int64_t>());
  const auto& dem = z.at("dem");
  nx_ = dem.shape[0], ny_ = dem.shape[1];
  dem_ = dem.as<float>();
  pts_ = z.at("pts").as<float>();
  auto b = z.at("bins").as<long>();
  bins_[0] = b[0], bins_[1] = b[1];
  starts_ = z.at("starts").as<int64_t>();
  if (static_cast<long>(starts_.size()) != bins_[0] * bins_[1] + 1) throw std::runtime_error(path + ": bins and starts disagree");
}

float Survey::terrain(const Vec2& utm) const {
  long i = std::clamp(static_cast<long>(std::floor((utm[0] - origin_[0]) / res_)), 0L, nx_ - 1);
  long j = std::clamp(static_cast<long>(std::floor((utm[1] - origin_[1]) / res_)), 0L, ny_ - 1);
  return dem_[i * ny_ + j];
}

MatX3 Survey::crop(const Vec2& utm, double r) const {
  const Vec2 c = utm - origin_;
  long b0[2], b1[2];
  for (int a = 0; a < 2; ++a) {
    b0[a] = std::clamp(static_cast<long>(std::floor((c[a] - r) / kCell)), 0L, bins_[a] - 1);
    b1[a] = std::clamp(static_cast<long>(std::floor((c[a] + r) / kCell)), 0L, bins_[a] - 1);
  }
  std::vector<double> out;
  for (long i = b0[0]; i <= b1[0]; ++i)
    for (int64_t k = starts_[i * bins_[1] + b0[1]]; k < starts_[i * bins_[1] + b1[1] + 1]; ++k) {
      double x = pts_[3 * k], y = pts_[3 * k + 1];
      if ((x - c[0]) * (x - c[0]) + (y - c[1]) * (y - c[1]) < r * r)
        out.insert(out.end(), {x + origin_[0], y + origin_[1], static_cast<double>(pts_[3 * k + 2])});
    }
  return Eigen::Map<MatX3>(out.data(), static_cast<long>(out.size() / 3), 3);
}

}  // namespace olb
