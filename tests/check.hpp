// A minimal test harness: TEST(name) { CHECK(...); CHECK_NEAR(a, b, tol); }, run by tests/main.cpp.
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace olb_test {

struct Case {
  const char* name;
  std::function<void()> fn;
};

inline std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}

inline int& failures() {
  static int n = 0;
  return n;
}

struct Register {
  Register(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};

inline void fail(const char* file, int line, const std::string& what) {
  std::printf("  FAIL %s:%d: %s\n", file, line, what.c_str());
  ++failures();
}

inline std::string data(const std::string& f) { return std::string(OSMLOCBKI_TEST_DATA) + "/" + f; }

}  // namespace olb_test

#define TEST(name)                                                         \
  static void name();                                                      \
  static olb_test::Register name##_reg(#name, name);                    \
  static void name()

#define CHECK(c) \
  do {           \
    if (!(c)) olb_test::fail(__FILE__, __LINE__, #c); \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                                               \
  do {                                                                                                      \
    double a_ = (a), b_ = (b);                                                                              \
    if (!(std::abs(a_ - b_) <= (tol)))                                                                      \
      olb_test::fail(__FILE__, __LINE__, std::string(#a " = ") + std::to_string(a_) + ", " #b " = " +     \
                                                std::to_string(b_) + ", tolerance " + std::to_string(tol)); \
  } while (0)
