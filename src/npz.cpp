#include "osmlocbki/npz.hpp"

#include <zlib.h>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <type_traits>

namespace olb {

namespace {

template <typename T>
constexpr char kind_of() {
  if constexpr (std::is_same_v<T, bool>) return 'b';
  else if constexpr (std::is_floating_point_v<T>) return 'f';
  else if constexpr (std::is_signed_v<T>) return 'i';
  else return 'u';
}

[[noreturn]] void fail(const std::string& path, const std::string& what) {
  throw std::runtime_error(path + ": " + what);
}

// ---- .npy ----------------------------------------------------------------------------------

NpyArray parse_npy(const std::vector<char>& buf, const std::string& where) {
  if (buf.size() < 10 || std::memcmp(buf.data(), "\x93NUMPY", 6) != 0) fail(where, "not a .npy array");
  int major = static_cast<unsigned char>(buf[6]);
  size_t hlen, off;
  if (major == 1) {
    hlen = static_cast<unsigned char>(buf[8]) | (static_cast<unsigned char>(buf[9]) << 8);
    off = 10;
  } else {
    uint32_t h;
    std::memcpy(&h, buf.data() + 8, 4);
    hlen = h, off = 12;
  }
  std::string header(buf.data() + off, hlen);
  NpyArray a;
  auto descr = header.find("'descr'");
  auto q = header.find('\'', header.find(':', descr) + 1);
  std::string d = header.substr(q + 1, header.find('\'', q + 1) - q - 1);  // e.g. <f8, |b1, |u1
  if (d[0] == '>') fail(where, "big-endian arrays are not supported");
  a.kind = d[1];
  a.word = std::stoi(d.substr(2));
  if (header.find("'fortran_order': True") != std::string::npos) fail(where, "Fortran-order arrays are not supported");
  auto s0 = header.find('(', header.find("'shape'")), s1 = header.find(')', s0);
  std::string shp = header.substr(s0 + 1, s1 - s0 - 1);
  for (size_t p = 0; p < shp.size();) {
    while (p < shp.size() && (shp[p] == ' ' || shp[p] == ',')) ++p;
    if (p >= shp.size()) break;
    size_t e = p;
    while (e < shp.size() && shp[e] != ',') ++e;
    a.shape.push_back(std::stol(shp.substr(p, e - p)));
    p = e;
  }
  size_t bytes = static_cast<size_t>(a.size()) * a.word, start = off + hlen;
  if (buf.size() < start + bytes) fail(where, "truncated array");
  a.data.assign(buf.begin() + start, buf.begin() + start + bytes);
  return a;
}

std::vector<char> make_npy(const NpyArray& a) {
  std::string shape = "(";
  for (size_t i = 0; i < a.shape.size(); ++i) shape += std::to_string(a.shape[i]) + (a.shape.size() == 1 ? "," : i + 1 < a.shape.size() ? ", " : "");
  shape += ")";
  char order = a.word == 1 ? '|' : '<';
  std::string h = "{'descr': '" + std::string(1, order) + a.kind + std::to_string(a.word) +
                  "', 'fortran_order': False, 'shape': " + shape + ", }";
  size_t total = 10 + h.size() + 1;
  h.append((64 - total % 64) % 64, ' ');
  h += '\n';
  std::vector<char> out = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
  out.push_back(static_cast<char>(h.size() & 0xff));
  out.push_back(static_cast<char>(h.size() >> 8));
  out.insert(out.end(), h.begin(), h.end());
  out.insert(out.end(), a.data.begin(), a.data.end());
  return out;
}

// ---- zip -----------------------------------------------------------------------------------

template <typename T>
T rd(const std::vector<char>& b, size_t o) {
  T v;
  std::memcpy(&v, b.data() + o, sizeof(T));
  return v;
}

std::vector<char> inflate_raw(const char* src, size_t n, size_t out_n, const std::string& where) {
  std::vector<char> out(out_n);
  z_stream z{};
  if (inflateInit2(&z, -MAX_WBITS) != Z_OK) fail(where, "inflate init failed");
  z.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(src));
  z.avail_in = static_cast<uInt>(n);
  size_t done = 0;
  while (done < out_n) {  // in pieces: avail_out is 32-bit
    size_t piece = std::min<size_t>(out_n - done, 1u << 30);
    z.next_out = reinterpret_cast<Bytef*>(out.data() + done);
    z.avail_out = static_cast<uInt>(piece);
    int r = inflate(&z, Z_NO_FLUSH);
    done += piece - z.avail_out;
    if (r == Z_STREAM_END) break;
    if (r != Z_OK) {
      inflateEnd(&z);
      fail(where, "inflate failed");
    }
  }
  inflateEnd(&z);
  return out;
}

void put16(std::vector<char>& b, uint16_t v) { b.insert(b.end(), reinterpret_cast<char*>(&v), reinterpret_cast<char*>(&v) + 2); }
void put32(std::vector<char>& b, uint32_t v) { b.insert(b.end(), reinterpret_cast<char*>(&v), reinterpret_cast<char*>(&v) + 4); }
void put64(std::vector<char>& b, uint64_t v) { b.insert(b.end(), reinterpret_cast<char*>(&v), reinterpret_cast<char*>(&v) + 8); }

}  // namespace

template <typename T>
std::vector<T> NpyArray::as() const {
  const long n = size();
  std::vector<T> out(n);
  auto conv = [&](auto tag) {
    using S = decltype(tag);
    const S* p = reinterpret_cast<const S*>(data.data());
    for (long i = 0; i < n; ++i) out[i] = static_cast<T>(p[i]);
  };
  if (kind == 'f' && word == 8) conv(double());
  else if (kind == 'f' && word == 4) conv(float());
  else if (kind == 'i' && word == 8) conv(int64_t());
  else if (kind == 'i' && word == 4) conv(int32_t());
  else if (kind == 'i' && word == 2) conv(int16_t());
  else if (kind == 'i' && word == 1) conv(int8_t());
  else if (kind == 'u' && word == 8) conv(uint64_t());
  else if (kind == 'u' && word == 4) conv(uint32_t());
  else if (kind == 'u' && word == 2) conv(uint16_t());
  else if ((kind == 'u' || kind == 'b') && word == 1) conv(uint8_t());
  else throw std::runtime_error(std::string("unsupported dtype ") + kind + std::to_string(word));
  return out;
}

template <typename T>
NpyArray NpyArray::from(const T* p, std::vector<long> shape) {
  NpyArray a;
  a.kind = kind_of<T>();
  a.word = sizeof(T);
  a.shape = std::move(shape);
  a.data.assign(reinterpret_cast<const char*>(p), reinterpret_cast<const char*>(p) + a.size() * sizeof(T));
  return a;
}

#define OSMLOC_NPY(T)                                                     \
  template std::vector<T> NpyArray::as<T>() const;                        \
  template NpyArray NpyArray::from<T>(const T*, std::vector<long>);
OSMLOC_NPY(double)
OSMLOC_NPY(float)
OSMLOC_NPY(int64_t)
OSMLOC_NPY(int32_t)
OSMLOC_NPY(int8_t)
OSMLOC_NPY(uint8_t)
OSMLOC_NPY(bool)
OSMLOC_NPY(long long)
#undef OSMLOC_NPY

Npz load_npz(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) fail(path, "cannot open");
  std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  // end of central directory (and its zip64 locator)
  size_t eocd = std::string::npos;
  for (size_t i = b.size() >= 22 ? b.size() - 22 : 0;; --i) {
    if (rd<uint32_t>(b, i) == 0x06054b50) {
      eocd = i;
      break;
    }
    if (i == 0 || b.size() - i > 70000) break;
  }
  if (eocd == std::string::npos) fail(path, "not a zip archive");
  uint64_t n_entries = rd<uint16_t>(b, eocd + 10), cd = rd<uint32_t>(b, eocd + 16);
  if (eocd >= 20 && rd<uint32_t>(b, eocd - 20) == 0x07064b50) {
    size_t z64 = rd<uint64_t>(b, eocd - 12);
    n_entries = rd<uint64_t>(b, z64 + 32);
    cd = rd<uint64_t>(b, z64 + 48);
  }
  Npz out;
  size_t p = cd;
  for (uint64_t e = 0; e < n_entries; ++e) {
    if (rd<uint32_t>(b, p) != 0x02014b50) fail(path, "bad central directory");
    uint16_t method = rd<uint16_t>(b, p + 10);
    uint64_t csize = rd<uint32_t>(b, p + 20), usize = rd<uint32_t>(b, p + 24), local = rd<uint32_t>(b, p + 42);
    uint16_t nlen = rd<uint16_t>(b, p + 28), xlen = rd<uint16_t>(b, p + 30), clen = rd<uint16_t>(b, p + 32);
    std::string name(b.data() + p + 46, nlen);
    for (size_t x = p + 46 + nlen; x < p + 46 + nlen + xlen;) {  // zip64 extra field
      uint16_t id = rd<uint16_t>(b, x), sz = rd<uint16_t>(b, x + 2);
      if (id == 1) {
        size_t q = x + 4;
        if (usize == 0xffffffff) usize = rd<uint64_t>(b, q), q += 8;
        if (csize == 0xffffffff) csize = rd<uint64_t>(b, q), q += 8;
        if (local == 0xffffffff) local = rd<uint64_t>(b, q);
      }
      x += 4 + sz;
    }
    size_t data = local + 30 + rd<uint16_t>(b, local + 26) + rd<uint16_t>(b, local + 28);
    std::vector<char> raw;
    if (method == 0)
      raw.assign(b.begin() + data, b.begin() + data + usize);
    else if (method == 8)
      raw = inflate_raw(b.data() + data, csize, usize, path + ":" + name);
    else
      fail(path, "unsupported compression in " + name);
    if (name.size() > 4 && name.substr(name.size() - 4) == ".npy") name.resize(name.size() - 4);
    out[name] = parse_npy(raw, path + ":" + name);
    p += 46 + nlen + xlen + clen;
  }
  return out;
}

void save_npz(const std::string& path, const Npz& arrays) {
  std::ofstream f(path, std::ios::binary);
  if (!f) fail(path, "cannot write");
  std::vector<char> cd;
  uint64_t off = 0;
  for (const auto& [key, arr] : arrays) {
    std::vector<char> npy = make_npy(arr);
    std::string name = key + ".npy";
    uint32_t crc = 0;
    for (size_t done = 0; done < npy.size();) {
      size_t piece = std::min<size_t>(npy.size() - done, 1u << 30);
      crc = crc32(crc, reinterpret_cast<const Bytef*>(npy.data() + done), static_cast<uInt>(piece));
      done += piece;
    }
    // local header, sizes in a zip64 extra field
    std::vector<char> h;
    put32(h, 0x04034b50), put16(h, 45), put16(h, 0), put16(h, 0), put16(h, 0), put16(h, 0x21);
    put32(h, crc), put32(h, 0xffffffff), put32(h, 0xffffffff);
    put16(h, static_cast<uint16_t>(name.size())), put16(h, 20);
    h.insert(h.end(), name.begin(), name.end());
    put16(h, 1), put16(h, 16), put64(h, npy.size()), put64(h, npy.size());
    f.write(h.data(), h.size());
    f.write(npy.data(), npy.size());
    // central directory entry
    put32(cd, 0x02014b50), put16(cd, 45), put16(cd, 45), put16(cd, 0), put16(cd, 0), put16(cd, 0), put16(cd, 0x21);
    put32(cd, crc), put32(cd, 0xffffffff), put32(cd, 0xffffffff);
    put16(cd, static_cast<uint16_t>(name.size())), put16(cd, 28), put16(cd, 0), put16(cd, 0), put16(cd, 0);
    put32(cd, 0), put32(cd, 0xffffffff);
    cd.insert(cd.end(), name.begin(), name.end());
    put16(cd, 1), put16(cd, 24), put64(cd, npy.size()), put64(cd, npy.size()), put64(cd, off);
    off += h.size() + npy.size();
  }
  std::vector<char> tail;
  uint64_t n = arrays.size();
  put32(tail, 0x06064b50), put64(tail, 44), put16(tail, 45), put16(tail, 45), put32(tail, 0), put32(tail, 0);
  put64(tail, n), put64(tail, n), put64(tail, cd.size()), put64(tail, off);
  put32(tail, 0x07064b50), put32(tail, 0), put64(tail, off + cd.size()), put32(tail, 1);
  put32(tail, 0x06054b50), put16(tail, 0), put16(tail, 0), put16(tail, 0xffff), put16(tail, 0xffff);
  put32(tail, 0xffffffff), put32(tail, 0xffffffff), put16(tail, 0);
  f.write(cd.data(), cd.size());
  f.write(tail.data(), tail.size());
  if (!f) fail(path, "write failed");
}

}  // namespace olb
