#include "osmlocbki/frontend.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <numeric>

#include "osmlocbki/npz.hpp"
#include "osmlocbki/site.hpp"

namespace fs = std::filesystem;

namespace olb {

// ---- descriptor -----------------------------------------------------------------------------

std::vector<double> lidar_descriptor(const float* pts, long n, int stride, int num_bin, double sensor_range) {
  std::vector<double> out(num_bin, std::numeric_limits<double>::infinity());
  for (long i = 0; i < n; ++i) {
    double x = pts[i * stride], y = pts[i * stride + 1];
    double r = std::hypot(x, y), az = std::atan2(y, x);
    if (az < 0) az += 2 * kPi;
    long b = pymod(static_cast<long>(round_even(az * num_bin / (2 * kPi))), static_cast<long>(num_bin));
    out[b] = std::min(out[b], r);
  }
  for (auto& d : out) d = d < sensor_range ? d : 0.0;
  return out;
}

std::vector<double> scan_descriptor(const float* pts, long n, int stride, const int32_t* labels,
                                    const std::vector<int>& building_ids, int* kept, int num_bin,
                                    double sensor_range) {
  std::vector<float> sel;
  sel.reserve(n / 4 * 2);
  for (long i = 0; i < n; ++i) {
    if (std::find(building_ids.begin(), building_ids.end(), labels[i]) == building_ids.end()) continue;
    double x = pts[i * stride], y = pts[i * stride + 1];
    if (std::hypot(x, y) <= kMinScanRange) continue;
    sel.push_back(pts[i * stride]);
    sel.push_back(pts[i * stride + 1]);
  }
  if (kept) *kept = static_cast<int>(sel.size() / 2);
  return lidar_descriptor(sel.data(), static_cast<long>(sel.size() / 2), 2, num_bin, sensor_range);
}

template <typename T>
void rotation_invariant(const T* d, int num_bin, float* out, double sensor_range, double bin_size) {
  int n_bins = static_cast<int>(round_even(sensor_range / bin_size));
  std::fill(out, out + n_bins, 0.0f);
  for (int b = 0; b < num_bin; ++b) {
    double v = d[b];
    if (v == 0) continue;
    int k = static_cast<int>(std::ceil(v / bin_size)) - 1;
    out[std::clamp(k, 0, n_bins - 1)] += 1.0f;
  }
}

template void rotation_invariant<float>(const float*, int, float*, double, double);
template void rotation_invariant<double>(const double*, int, float*, double, double);

// ---- ray casting ------------------------------------------------------------------------------

namespace {

int64_t cell_key(long i, long j) { return (static_cast<int64_t>(i) << 32) ^ (static_cast<int64_t>(j) & 0xffffffff); }

bool in_bbox(double lon, double lat, const std::array<double, 4>& b) {
  return lat >= b[0] && lat <= b[2] && lon >= b[1] && lon <= b[3];
}

}  // namespace

RayCaster::RayCaster(const std::vector<Ring>& rings, int num_bin, double sensor_range)
    : rings_(rings), range_(sensor_range), n_(num_bin) {
  for (int j = 1; j <= n_; ++j) {
    double a = j * 2.0 * kPi / n_;
    cos_.push_back(std::cos(a)), sin_.push_back(std::sin(a));
  }
  for (size_t r = 0; r < rings_.size(); ++r)
    for (const auto& v : rings_[r]) {
      grid_[cell_key(static_cast<long>(std::floor(v[0] / range_)), static_cast<long>(std::floor(v[1] / range_)))]
          .push_back(static_cast<int>(vxy_.size()));
      vxy_.push_back(v);
      owner_.push_back(static_cast<int>(r));
    }
}

void RayCaster::cast(const Vec2& o, float* out) const {
  // rings with a vertex within range (make_descriptor.m's prefilter)
  std::vector<int> near;
  long ci = static_cast<long>(std::floor(o[0] / range_)), cj = static_cast<long>(std::floor(o[1] / range_));
  const double r2 = range_ * range_;
  for (long i = ci - 1; i <= ci + 1; ++i)
    for (long j = cj - 1; j <= cj + 1; ++j) {
      auto it = grid_.find(cell_key(i, j));
      if (it == grid_.end()) continue;
      for (int k : it->second)
        if ((vxy_[k] - o).squaredNorm() <= r2) near.push_back(owner_[k]);
    }
  std::sort(near.begin(), near.end());
  near.erase(std::unique(near.begin(), near.end()), near.end());
  std::vector<double> best(n_, std::numeric_limits<double>::infinity());
  const double step = 2.0 * kPi / n_;
  for (int r : near) {
    const Ring& ring = rings_[r];
    for (size_t e = 0; e < ring.size(); ++e) {
      const Vec2 a = ring[e] - o, s = ring[(e + 1) % ring.size()] - ring[e];
      // ray t*d meets segment a + u*s; only rays inside the edge's angular span can (plus a bin)
      double t0 = std::atan2(a[1], a[0]), t1 = std::atan2(a[1] + s[1], a[0] + s[0]);
      double d = std::remainder(t1 - t0, 2 * kPi);
      long k0, k1;
      double cross = a[0] * s[1] - a[1] * s[0];
      if (std::abs(cross) <= 1e-12 * (a.norm() * s.norm() + 1e-300) || a.norm() < 1e-9 || (a + s).norm() < 1e-9) {
        k0 = 1, k1 = n_;  // degenerate: on or along the segment, test every ray
      } else {
        double lo = d >= 0 ? t0 : t1, w = std::abs(d);
        k0 = static_cast<long>(std::floor(lo / step)) - 1;
        k1 = static_cast<long>(std::ceil((lo + w) / step)) + 1;
      }
      for (long k = k0; k <= k1; ++k) {
        int j = static_cast<int>(pymod(k - 1, static_cast<long>(n_)));  // ray j + 1, angle (j + 1) * step
        double dx = cos_[j], dy = sin_[j];
        double den = dx * s[1] - dy * s[0];
        double t = (a[0] * s[1] - a[1] * s[0]) / den, u = (a[0] * dy - a[1] * dx) / den;
        if (std::isfinite(t) && t >= 0 && u >= 0 && u <= 1 && t < best[j]) best[j] = t;
      }
    }
  }
  for (int j = 0; j < n_; ++j) out[j] = static_cast<float>(best[j] < range_ ? best[j] : 0.0);
}

// ---- the database -----------------------------------------------------------------------------

Database build_database(const OsmMap& map, const std::optional<std::array<double, 4>>& bbox, double sp) {
  // a grid aligned to multiples of `sp` over the map box (or the buildings' extent)
  Vec2 lo = Vec2::Constant(1e300), hi = -lo;
  if (bbox) {
    const auto& b = *bbox;
    for (const auto& p : {to_utm(b[0], b[1], map.epsg), to_utm(b[0], b[3], map.epsg), to_utm(b[2], b[1], map.epsg),
                          to_utm(b[2], b[3], map.epsg)})
      lo = lo.cwiseMin(p), hi = hi.cwiseMax(p);
  } else {
    for (const auto& r : map.descriptor_rings) {
      Box b = bounds(r);
      lo = lo.cwiseMin(b.head<2>()), hi = hi.cwiseMax(b.tail<2>());
    }
  }
  lo = (lo / sp).array().floor() * sp;
  long nx = static_cast<long>(std::ceil((hi[0] - lo[0]) / sp)), ny = static_cast<long>(std::ceil((hi[1] - lo[1]) / sp));
  std::vector<Vec2> xy, ll;
  for (long j = 0; j < ny; ++j)
    for (long i = 0; i < nx; ++i) {
      Vec2 p(lo[0] + i * sp, lo[1] + j * sp);
      Vec2 q = from_utm(p[0], p[1], map.epsg);  // (lat, lon)
      if (bbox && !in_bbox(q[1], q[0], *bbox)) continue;
      xy.push_back(p), ll.emplace_back(q[1], q[0]);
    }
  // free space is outside buildings by definition
  Footprints fp(map.buildings);
  std::vector<char> inside(xy.size());
#pragma omp parallel for schedule(static)
  for (long i = 0; i < static_cast<long>(xy.size()); ++i) inside[i] = fp.contains(xy[i]);
  std::fprintf(stderr, "  ray-casting %zu candidates against %zu buildings, EPSG:%d\n", xy.size(),
               map.descriptor_rings.size(), map.epsg);
  RayCaster rc(map.descriptor_rings);
  std::vector<float> desc(xy.size() * kNumBin, 0.0f);
#pragma omp parallel for schedule(dynamic, 256)
  for (long i = 0; i < static_cast<long>(xy.size()); ++i)
    if (!inside[i]) rc.cast(xy[i], desc.data() + i * kNumBin);
  std::vector<size_t> keep;
  for (size_t i = 0; i < xy.size(); ++i)
    if (!inside[i] && std::any_of(desc.begin() + i * kNumBin, desc.begin() + (i + 1) * kNumBin, [](float v) { return v != 0; }))
      keep.push_back(i);
  std::fprintf(stderr, "  kept %zu candidates (outside footprints, a building in range)\n", keep.size());
  Database db;
  db.epsg = map.epsg;
  db.xy.resize(keep.size(), 2), db.ll.resize(keep.size(), 2);
  db.desc.resize(keep.size() * kNumBin);
  for (size_t k = 0; k < keep.size(); ++k) {
    size_t i = keep[k];
    db.xy.row(k) = xy[i].transpose(), db.ll.row(k) = ll[i].transpose();
    std::copy(desc.begin() + i * kNumBin, desc.begin() + (i + 1) * kNumBin, db.desc.begin() + k * kNumBin);
  }
  return db;
}

void save_database(const Database& db, const std::string& path) {
  const long M = db.size();
  save_npz(path, {{"db_xy", NpyArray::from(db.xy.data(), {M, 2})},
                  {"db_ll", NpyArray::from(db.ll.data(), {M, 2})},
                  {"db_full", NpyArray::from(db.desc.data(), {M, db.num_bin})},
                  {"epsg", NpyArray::scalar_of<int64_t>(db.epsg)}});
}

Database load_database(const std::string& path) {
  Npz z = load_npz(path);
  Database db;
  auto xy = z.at("db_xy").as<double>(), ll = z.at("db_ll").as<double>();
  const long M = z.at("db_xy").shape[0];
  db.xy = Eigen::Map<MatX2>(xy.data(), M, 2);
  db.ll = Eigen::Map<MatX2>(ll.data(), M, 2);
  db.num_bin = static_cast<int>(z.at("db_full").shape[1]);
  db.desc = z.at("db_full").as<float>();
  db.epsg = static_cast<int>(z.at("epsg").scalar<int64_t>());
  return db;
}

std::shared_ptr<Database> site_database(const Site& site, double spacing, bool rebuild) {
  char name[96];
  std::snprintf(name, sizeof name, "/db_%s_free%g_n%d_r%g.npz", site.map.c_str(), spacing, kNumBin, kSensorRange);
  std::string path = data_dir() + name;
  if (fs::exists(path) && !rebuild) return std::make_shared<Database>(load_database(path));
  std::fprintf(stderr, "OSM database for map %s:\n", site.map.c_str());
  auto db = std::make_shared<Database>(build_database(load_map(site.map_prefix()), site.map_bbox, spacing));
  fs::create_directories(data_dir());
  save_database(*db, path);
  std::fprintf(stderr, "  saved %s\n", path.c_str());
  return db;
}

// ---- matching -------------------------------------------------------------------------------

Matcher::Matcher(std::shared_ptr<const Database> db, int top_n)
    : db_(std::move(db)), n_ri_(static_cast<int>(round_even(kSensorRange / kBinSize))) {
  top_n_ = static_cast<int>(std::min<long>(top_n, db_->size()));
  ri_.resize(db_->size() * n_ri_);
#pragma omp parallel for schedule(static)
  for (long i = 0; i < db_->size(); ++i) rotation_invariant(db_->row(i), db_->num_bin, ri_.data() + i * n_ri_);
}

std::vector<long> Matcher::stage1(const double* q) const {
  std::vector<float> qr(n_ri_);
  rotation_invariant(q, db_->num_bin, qr.data());
  const long M = db_->size();
  std::vector<float> d(M);
#pragma omp parallel for schedule(static)
  for (long i = 0; i < M; ++i) {
    const float* r = ri_.data() + i * n_ri_;
    float s = 0;
    for (int b = 0; b < n_ri_; ++b) s += std::abs(qr[b] - r[b]);
    d[i] = s;
  }
  // the top_n smallest, ties broken by row (the counts are integers, so ties are common)
  std::vector<long> idx(M);
  std::iota(idx.begin(), idx.end(), 0L);
  auto less = [&](long a, long b) { return d[a] < d[b] || (d[a] == d[b] && a < b); };
  std::nth_element(idx.begin(), idx.begin() + top_n_ - 1, idx.end(), less);
  idx.resize(top_n_);
  std::sort(idx.begin(), idx.end(), less);
  return idx;
}

void Matcher::stage2(const double* q, const std::vector<long>& cand, std::vector<long>& idx, std::vector<long>& shift) const {
  const int n = db_->num_bin;
  const long C = static_cast<long>(cand.size());
  Eigen::ArrayXf qf(n);
  for (int b = 0; b < n; ++b) qf[b] = static_cast<float>(q[b]);
  std::vector<float> best(C);
  std::vector<long> best_k(C);
#pragma omp parallel
  {
    Eigen::ArrayXf dd(2 * n);
#pragma omp for schedule(static)
    for (long c = 0; c < C; ++c) {
      const float* r = db_->row(cand[c]);
      for (int b = 0; b < 2 * n; ++b) dd[b] = r[b % n];
      // circshift by k: rolled[b] = r[(b - k) mod n] = dd[n - k + b]
      float bd = std::numeric_limits<float>::infinity();
      long bk = 0;
      for (int k = 1; k <= n; ++k) {
        float s = (qf - dd.segment(n - k, n)).abs().sum();
        if (s < bd) bd = s, bk = k;
      }
      best[c] = bd;
      best_k[c] = bk;
    }
  }
  std::vector<long> order(C);
  std::iota(order.begin(), order.end(), 0L);
  std::stable_sort(order.begin(), order.end(), [&](long a, long b) { return best[a] < best[b]; });
  idx.resize(C), shift.resize(C);
  for (long i = 0; i < C; ++i) idx[i] = cand[order[i]], shift[i] = best_k[order[i]];
}

Hits Matcher::operator()(const double* q, int k) const {
  std::vector<long> idx, shift;
  stage2(q, stage1(q), idx, shift);
  k = std::min<int>(k, static_cast<int>(idx.size()));
  Hits h{MatX2(k, 2), VecXd(k)};
  for (int i = 0; i < k; ++i) {
    h.xy.row(i) = db_->xy.row(idx[i]);
    h.psi[i] = psi_of(shift[i], db_->num_bin);
  }
  return h;
}

}  // namespace olb
