// Built without nvblox (no CUDA toolkit, or cmake -DOSMLOCBKI_NVBLOX=OFF): only the CPU store.
#include <stdexcept>

#include "osmlocbki/voxel_store.hpp"

namespace olb {

bool have_nvblox() { return false; }

std::unique_ptr<VoxelStore> make_nvblox_store(int, const StoreParams&, const NvbloxParams&) {
  throw std::runtime_error("osmlocbki was built without nvblox: install a CUDA toolkit and rebuild (see the README)");
}

}  // namespace olb
