cmake_minimum_required(VERSION 3.25)

# Inside a conda environment (environment.yaml), build with its compiler and libraries even when
# it is not activated (e.g. while `conda env create` runs pip).
if(Python_EXECUTABLE AND NOT CMAKE_CXX_COMPILER)
  get_filename_component(_env_bin "${Python_EXECUTABLE}" DIRECTORY)
  if(EXISTS "${_env_bin}/x86_64-conda-linux-gnu-c++")
    set(CMAKE_CXX_COMPILER "${_env_bin}/x86_64-conda-linux-gnu-c++")
  endif()
endif()

project(osmlocbki LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

if(Python_EXECUTABLE)
  get_filename_component(_env "${Python_EXECUTABLE}" DIRECTORY)
  get_filename_component(_env "${_env}" DIRECTORY)
  list(APPEND CMAKE_PREFIX_PATH "${_env}")
  set(CMAKE_INSTALL_RPATH "$ORIGIN;${_env}/lib")  # the nvblox library sits next to the module
endif()

option(OSMLOCBKI_AVX2 "Compile the numeric code with AVX2/FMA" ON)
option(OSMLOCBKI_PYTHON "Build the Python module" ON)
option(OSMLOCBKI_TESTS "Build the tests" ON)
set(OSMLOCBKI_NVBLOX AUTO CACHE STRING "The nvblox (GPU) map backend: AUTO (when CUDA and nvblox are found), ON, OFF")
set(NVBLOX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../nvblox" CACHE PATH "An nvblox checkout (github.com/nvidia-isaac/nvblox)")
set(OSMLOCBKI_MAX_CLASSES 9 CACHE STRING "Classes an nvblox semantic voxel holds (config/labels.yaml has 9)")

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(ZLIB REQUIRED)

# Patchwork++ (BSD-2, third_party/patchworkpp): ground segmentation
add_library(patchworkpp STATIC third_party/patchworkpp/src/patchworkpp.cpp)
target_include_directories(patchworkpp PUBLIC third_party/patchworkpp/include)
target_link_libraries(patchworkpp PUBLIC Eigen3::Eigen)
target_compile_options(patchworkpp PRIVATE -w)

# nvblox: the map's voxels in GPU voxel blocks next to its TSDF (src/nvblox_store.cu)
set(_nvblox OFF)
if(NOT OSMLOCBKI_NVBLOX STREQUAL "OFF")
  include(CheckLanguage)
  if(NOT CMAKE_CUDA_HOST_COMPILER AND CMAKE_CXX_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  endif()
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER AND EXISTS "${NVBLOX_SOURCE_DIR}/nvblox/CMakeLists.txt")
    set(_nvblox ON)
  elseif(OSMLOCBKI_NVBLOX STREQUAL "ON")
    message(FATAL_ERROR "OSMLOCBKI_NVBLOX=ON needs a CUDA compiler and an nvblox checkout at NVBLOX_SOURCE_DIR (${NVBLOX_SOURCE_DIR})")
  endif()
endif()
if(_nvblox)
  set(_cuda_arch_given ${CMAKE_CUDA_ARCHITECTURES})
  enable_language(CUDA)
  if(NOT _cuda_arch_given AND CMAKE_CUDA_ARCHITECTURES_NATIVE)
    set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_NATIVE})  # this machine's GPU
  endif()
  # the nvblox library alone, with this build's Eigen
  set(BUILD_TESTING OFF)
  set(BUILD_BENCHMARKS OFF)
  set(BUILD_PYTORCH_WRAPPER OFF)
  set(BUILD_RENDERER OFF)
  set(USE_SYSTEM_EIGEN ON)
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)  # its fetched gflags predates CMake 4
  add_subdirectory("${NVBLOX_SOURCE_DIR}" nvblox EXCLUDE_FROM_ALL SYSTEM)
  add_library(osmlocbki_nvblox STATIC src/nvblox_store.cu)
  set_nvblox_compiler_options_nowarnings(osmlocbki_nvblox)  # its Thrust/CUB ABI settings
  target_include_directories(osmlocbki_nvblox PRIVATE include)
  target_link_libraries(osmlocbki_nvblox PUBLIC nvblox_lib PRIVATE nvblox_stdgpu Eigen3::Eigen)
  target_compile_definitions(osmlocbki_nvblox PRIVATE EIGEN_MAX_ALIGN_BYTES=16 EIGEN_MALLOC_ALREADY_ALIGNED=1
                             OSMLOCBKI_MAX_CLASSES=${OSMLOCBKI_MAX_CLASSES})
  message(STATUS "osmlocbki: nvblox backend ON (${NVBLOX_SOURCE_DIR})")
else()
  add_library(osmlocbki_nvblox STATIC src/nvblox_stub.cpp)
  target_include_directories(osmlocbki_nvblox PRIVATE include)
  target_link_libraries(osmlocbki_nvblox PRIVATE Eigen3::Eigen)
  target_compile_definitions(osmlocbki_nvblox PRIVATE EIGEN_MAX_ALIGN_BYTES=16 EIGEN_MALLOC_ALREADY_ALIGNED=1)
  message(STATUS "osmlocbki: nvblox backend OFF")
endif()

add_library(osmlocbki STATIC
  src/npz.cpp src/geo.cpp src/site.cpp src/labels.cpp
  src/frontend.cpp src/chain.cpp src/backend.cpp
  src/osm_prior.cpp src/ground.cpp src/voxel_store_cpu.cpp src/semantic_map.cpp src/survey.cpp
  src/pipeline.cpp)
target_include_directories(osmlocbki PUBLIC include)
target_link_libraries(osmlocbki PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX yaml-cpp::yaml-cpp
                      nlohmann_json::nlohmann_json ZLIB::ZLIB osmlocbki_nvblox PRIVATE patchworkpp)
# where config/ and data/ are found unless $OSMLOCBKI_ROOT says otherwise
target_compile_definitions(osmlocbki PRIVATE OSMLOCBKI_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
# Every file (Patchwork++ included: it trades Eigen objects with us) allocates and frees Eigen heap
# memory the same way, whether or not it is compiled with AVX.
target_compile_definitions(osmlocbki PUBLIC EIGEN_MAX_ALIGN_BYTES=16 EIGEN_MALLOC_ALREADY_ALIGNED=1)
target_compile_definitions(patchworkpp PUBLIC EIGEN_MAX_ALIGN_BYTES=16 EIGEN_MALLOC_ALREADY_ALIGNED=1)
if(OSMLOCBKI_AVX2)
  # the numeric kernels; chain.cpp stays without FMA, rounding like the GTSAM it reproduces
  set_source_files_properties(src/frontend.cpp src/backend.cpp src/voxel_store_cpu.cpp src/semantic_map.cpp src/osm_prior.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

if(OSMLOCBKI_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_osmlocbki bindings/osmlocbki.cpp)
  target_link_libraries(_osmlocbki PRIVATE osmlocbki)
  # importable from python/ without installing (PYTHONPATH=python)
  if(NOT SKBUILD)
    set_target_properties(_osmlocbki PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/osmlocbki")
  endif()
endif()

if(OSMLOCBKI_TESTS)
  enable_testing()
  add_executable(osmlocbki_tests tests/main.cpp tests/test_localization.cpp tests/test_mapping.cpp tests/test_end_to_end.cpp)
  target_link_libraries(osmlocbki_tests PRIVATE osmlocbki)
  target_compile_definitions(osmlocbki_tests PRIVATE OSMLOCBKI_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
  add_test(NAME osmlocbki_tests COMMAND osmlocbki_tests)
endif()

# pip install (scikit-build-core): the module into the osmlocbki package
if(SKBUILD)
  install(TARGETS _osmlocbki LIBRARY DESTINATION osmlocbki)
  if(_nvblox)
    install(TARGETS nvblox_lib LIBRARY DESTINATION osmlocbki)
  endif()
endif()
