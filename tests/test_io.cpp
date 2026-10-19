// .npz files and UTM (against pyproj, tests/data/utm.npz). Localization itself is osm_localization's
// (third_party/osm_localization), tested there.
#include <cstdio>
#include <filesystem>

#include "check.hpp"
#include "osmlocbki/geo.hpp"
#include "osmlocbki/npz.hpp"

using namespace olb;
using olb_test::data;

TEST(npz_round_trip) {
  std::vector<double> a{1.5, -2, 3e9};
  std::vector<int32_t> b{1, 2, 3, 4, 5, 6};
  std::string path = (std::filesystem::temp_directory_path() / "osmlocbki_test.npz").string();
  save_npz(path, {{"a", NpyArray::from(a, {3})}, {"b", NpyArray::from(b, {2, 3})}, {"s", NpyArray::scalar_of<int64_t>(7)}});
  Npz z = load_npz(path);
  CHECK(z.at("a").as<double>() == a);
  CHECK(z.at("b").as<int32_t>() == b);
  CHECK((z.at("b").shape == std::vector<long>{2, 3}));
  CHECK(z.at("s").scalar<int64_t>() == 7);
  std::filesystem::remove(path);
}

TEST(utm_matches_pyproj) {
  auto rows = load_npz(data("utm.npz")).at("rows");
  auto v = rows.as<double>();
  for (long i = 0; i < rows.shape[0]; ++i) {
    double lat = v[5 * i], lon = v[5 * i + 1], x = v[5 * i + 3], y = v[5 * i + 4];
    int epsg = static_cast<int>(v[5 * i + 2]);
    CHECK(utm_epsg(lat, lon) == epsg);
    Vec2 p = to_utm(lat, lon, epsg);
    CHECK_NEAR(p[0], x, 1e-6);
    CHECK_NEAR(p[1], y, 1e-6);
    Vec2 q = from_utm(x, y, epsg);
    CHECK_NEAR(q[0], lat, 1e-10);
    CHECK_NEAR(q[1], lon, 1e-10);
  }
}
