// NumPy .npz archives: read (stored or deflated, zip64) and write (stored), so every cache and
// result file stays readable with np.load.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace olb {

struct NpyArray {
  char kind = 'f';  // NumPy kind: f float, i int, u unsigned, b bool
  int word = 8;     // bytes per element
  std::vector<long> shape;
  std::vector<char> data;

  long size() const {
    long n = 1;
    for (long s : shape) n *= s;
    return n;
  }
  // the elements converted to T (any numeric dtype)
  template <typename T>
  std::vector<T> as() const;
  template <typename T>
  T scalar() const { return as<T>().at(0); }

  template <typename T>
  static NpyArray from(const T* p, std::vector<long> shape);
  template <typename T>
  static NpyArray from(const std::vector<T>& v, std::vector<long> shape) { return from(v.data(), std::move(shape)); }
  template <typename T>
  static NpyArray scalar_of(T v) { return from(&v, {}); }
};

using Npz = std::map<std::string, NpyArray>;

Npz load_npz(const std::string& path);
void save_npz(const std::string& path, const Npz& arrays);

}  // namespace olb
