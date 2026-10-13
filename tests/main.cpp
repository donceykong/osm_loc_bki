// Runs every TEST, or those whose name contains the first argument.
#include <chrono>
#include <cstring>

#include "check.hpp"

int main(int argc, char** argv) {
  int run = 0;
  for (const auto& c : olb_test::registry()) {
    if (argc > 1 && !std::strstr(c.name, argv[1])) continue;
    int before = olb_test::failures();
    auto t0 = std::chrono::steady_clock::now();
    c.fn();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%s %s (%.2f s)\n", olb_test::failures() == before ? "ok  " : "FAIL", c.name, s);
    ++run;
  }
  std::printf("%d tests, %d failed checks\n", run, olb_test::failures());
  return olb_test::failures() ? 1 : 0;
}
