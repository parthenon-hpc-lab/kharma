# KHARMA CMake top file.
# Only for general options, not KHARMA build specifics.
# That is, what would still be relevant if we were building
# multiple codes in this directory?

cmake_minimum_required(VERSION 3.10)
project(kharma LANGUAGES C CXX)

option(KHARMA_BUILD_TESTS "Build unit tests" OFF)
option(KHARMA_BUILD_IRIS "Build iris *instead* of KHARMA" OFF)

# We follow Parthenon in requiring C++17 going forward
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX17_STANDARD_COMPILE_OPTION "-std=c++20")

# Set the path to include cmake/ dir
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Parthenon options
# Coordinates
set(PARTHENON_COORDINATES "GRCoordinates" CACHE STRING "KHARMA Override")
set(PARTHENON_DISABLE_EXAMPLES ON CACHE BOOL "KHARMA Override")
set(PARTHENON_LINT_DEFAULT OFF CACHE BOOL "KHARMA Override")
# Attempt HDF5 compression, requires recent/standard HDF5. YMMV
set(PARTHENON_DISABLE_HDF5_COMPRESSION OFF CACHE BOOL "KHARMA Override")
# Don't build sparse (selectively-allocated) variable support
set(PARTHENON_DISABLE_SPARSE OFF CACHE BOOL "KHARMA Override")
# Set to move MPI buffers to host; slower but less crashy
# Favor setting this per-machine in machines/
set(PARTHENON_ENABLE_HOST_COMM_BUFFERS OFF CACHE BOOL "KHARMA Override")

# Parthenon internal build options
set(BUILD_TESTING OFF CACHE BOOL "KHARMA Override")
set(ENABLE_COMPILER_WARNINGS OFF CACHE BOOL "KHARMA Override")
# Don't require static HDF5, as conda and other environments don't provide it
set(HDF5_USE_STATIC_LIBRARIES OFF CACHE BOOL "KHARMA Override")


# Kokkos options
# Set ON for possible speed improvement?
set(Kokkos_ENABLE_HWLOC OFF CACHE BOOL "KHARMA Override")
set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL "KHARMA Override")

# For including the current git revision in the exe
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
git_describe_working_tree(GIT_VERSION --tags)

# Offer a KHARMA option to disable the MPI requirement
# The only difference from setting PARTHENON_DISABLE_MPI is that
# the configure step no longer searches for/fails without it
if (KHARMA_DISABLE_MPI)
  set(PARTHENON_DISABLE_MPI ON CACHE BOOL "KHARMA Override")
else()
  set(PARTHENON_DISABLE_MPI OFF CACHE BOOL "KHARMA Override")
  find_package(MPI REQUIRED)
  include_directories(SYSTEM ${MPI_INCLUDE_PATH})
endif()

set(HDF5_PREFER_PARALLEL (NOT KHARMA_DISABLE_MPI))
find_package(HDF5 COMPONENTS C HL)
if (NOT HDF5_FOUND)
  message(FATAL_ERROR "HDF5 is required but couldn't be found.")
endif()

# Build Parthenon
add_subdirectory(external/parthenon)
include_directories(external/parthenon/src)
# variant is header only, don't build anything
include_directories(external/singularity-eos/utils/ports-of-call)

# Inherit pristine kokkos-kernels
include_directories(external/kokkos-kernels)
# kokkos-kernels want to import from their own directories
include_directories(external/kokkos-kernels/batched)
include_directories(external/kokkos-kernels/batched/dense)
include_directories(external/kokkos-kernels/batched/dense/src)
include_directories(external/kokkos-kernels/batched/dense/impl)
include_directories(external/kokkos-kernels/blas/src)
include_directories(external/kokkos-kernels/blas/impl)
include_directories(external/kokkos-kernels/common/src)

# Needed for running singularity-eos tests
#set(SINGULARITY_USE_HDF5 ON CACHE BOOL "" FORCE)
#set(SINGULARITY_USE_SPINER ON CACHE BOOL "" FORCE)
#set(SINGULARITY_USE_SPINER_WITH_HDF5 ON CACHE BOOL "" FORCE)
#set(PARTHENON_DISABLE_HDF5 OFF CACHE BOOL "" FORCE)
#set(SINGULARITY_USE_STELLAR_COLLAPSE ON CACHE BOOL "" FORCE)
set(SINGULARITY_FORCE_SUBMODULE_MODE ON CACHE BOOL "" FORCE)

# Build Singularity-eos
add_subdirectory(external/singularity-eos)

# Build Singularity-opac
add_subdirectory(external/singularity-opac)

# Build KHARMA
if (KHARMA_BUILD_IRIS)
  add_subdirectory(iris)  
else()
  add_subdirectory(kharma)
endif()

# Install scripts and stuff
# run.sh as 'kharma' w/o suffix (for now?)
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/run.sh TYPE BIN RENAME "kharma")
# and our utilities, for fun
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/ TYPE BIN)

if (KHARMA_BUILD_TESTS)
   enable_testing()
   add_subdirectory(tests/unit)
endif()

