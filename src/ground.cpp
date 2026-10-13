#include "osmlocbki/ground.hpp"

#include <patchwork/patchworkpp.h>

namespace olb {

GroundSegmenter::GroundSegmenter(const GroundParams& p) : p_(p) {
  patchwork::Params q;
  q.sensor_height = p.sensor_height;
  q.min_range = p.min_range, q.max_range = p.max_range;
  q.enable_RNR = p.reflected_noise_removal;
  q.verbose = false;
  pw_ = std::make_unique<patchwork::PatchWorkpp>(q);
}

GroundSegmenter::~GroundSegmenter() = default;

std::vector<int8_t> GroundSegmenter::segment(const MatXf& pts) {
  const long n = pts.rows();
  std::vector<int8_t> out(n, -1);
  if (n == 0) return out;
  const int cols = p_.reflected_noise_removal && pts.cols() >= 4 ? 4 : 3;
  Eigen::MatrixXf m = pts.leftCols(cols);
  pw_->estimateGround(m);
  // Patchwork++ files points outside its zone model under non-ground; they get no verdict here,
  // or the disc under the sensor (mostly road) would lose its ground classes
  auto in_zone = [&](long i) {
    double r = std::hypot(pts(i, 0), pts(i, 1));
    return r > p_.min_range && r <= p_.max_range;
  };
  auto mark = [&](const Eigen::VectorXi& idx, int8_t v) {
    for (int j = 0; j < idx.size(); ++j)
      if (idx[j] >= 0 && idx[j] < n && in_zone(idx[j])) out[idx[j]] = v;
  };
  mark(pw_->getGroundIndices(), 1);
  mark(pw_->getNongroundIndices(), 0);
  return out;
}

}  // namespace olb
