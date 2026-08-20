# This file was generated in part with generative AI

cmake_minimum_required(VERSION 3.16)

set(CATCH_INSTALL_DOCS OFF CACHE BOOL "Do not install Catch2 docs")
set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "Do not install Catch2 extras")
set(CATCH_BUILD_TESTING OFF CACHE BOOL "Do not build Catch2 self-tests")

if(NOT TARGET Catch2)
add_subdirectory(
  ${PROJECT_SOURCE_DIR}/external/Catch2
  ${CMAKE_BINARY_DIR}/external/Catch2
)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external/Catch2/extras)
include(Catch)

set(KHARMA_TEST_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/kharma
  ${PROJECT_SOURCE_DIR}/kharma/prob
  ${PROJECT_SOURCE_DIR}/kharma/prob/elec
  ${PROJECT_SOURCE_DIR}/kharma/prob/emhd
  ${PROJECT_SOURCE_DIR}/kharma/prob/utils
  ${PROJECT_SOURCE_DIR}/kharma/coordinates
  ${PROJECT_SOURCE_DIR}/kharma/flux
  ${PROJECT_SOURCE_DIR}/kharma/b_cd
  ${PROJECT_SOURCE_DIR}/kharma/b_cleanup
  ${PROJECT_SOURCE_DIR}/kharma/b_cleanup_gmg
  ${PROJECT_SOURCE_DIR}/kharma/b_ct
  ${PROJECT_SOURCE_DIR}/kharma/b_flux_ct
  ${PROJECT_SOURCE_DIR}/kharma/boundaries
  ${PROJECT_SOURCE_DIR}/kharma/coord_output
  ${PROJECT_SOURCE_DIR}/kharma/current
  ${PROJECT_SOURCE_DIR}/kharma/driver
  ${PROJECT_SOURCE_DIR}/kharma/electrons
  ${PROJECT_SOURCE_DIR}/kharma/emhd
  ${PROJECT_SOURCE_DIR}/kharma/floors
  ${PROJECT_SOURCE_DIR}/kharma/grmhd
  ${PROJECT_SOURCE_DIR}/kharma/implicit
  ${PROJECT_SOURCE_DIR}/kharma/inverter
  ${PROJECT_SOURCE_DIR}/kharma/ismr
  ${PROJECT_SOURCE_DIR}/kharma/reductions
  ${PROJECT_SOURCE_DIR}/kharma/wind
)

set(KHARMA_UNIT_TEST_SOURCES
  catch2_define.cpp
  test_kokkos.cpp
  test_gray_opacities.cpp
)

add_executable(kharma_unit_tests ${KHARMA_UNIT_TEST_SOURCES})

target_include_directories(kharma_unit_tests PRIVATE ${KHARMA_TEST_INCLUDE_DIRS})

target_link_libraries(kharma_unit_tests
  PRIVATE
    Catch2::Catch2
    Kokkos::kokkos
    parthenon
    singularity-opac::flags
    singularity-eos::singularity-eos
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(kharma_unit_tests PRIVATE DEBUG=1)
else()
  target_compile_definitions(kharma_unit_tests PRIVATE DEBUG=0)
endif()

if(KHARMA_TRACE)
  target_compile_definitions(kharma_unit_tests PRIVATE TRACE=1)
else()
  target_compile_definitions(kharma_unit_tests PRIVATE TRACE=0)
endif()

if(FAST_CARTESIAN)
  target_compile_definitions(kharma_unit_tests PRIVATE FAST_CARTESIAN=1)
else()
  target_compile_definitions(kharma_unit_tests PRIVATE FAST_CARTESIAN=0)
endif()

if(KHARMA_DISABLE_IMPLICIT)
  target_compile_definitions(kharma_unit_tests PRIVATE DISABLE_IMPLICIT=1)
else()
  target_compile_definitions(kharma_unit_tests PRIVATE DISABLE_IMPLICIT=0)
endif()

if(KHARMA_DISABLE_CLEANUP)
  target_compile_definitions(kharma_unit_tests PRIVATE DISABLE_CLEANUP=1)
else()
  target_compile_definitions(kharma_unit_tests PRIVATE DISABLE_CLEANUP=0)
endif()

if(KHARMA_DISABLE_MPI)
  target_compile_definitions(kharma_unit_tests PRIVATE ENABLE_MPI=0)
else()
  target_compile_definitions(kharma_unit_tests PRIVATE ENABLE_MPI=1)
endif()

target_compile_definitions(kharma_unit_tests PRIVATE USE_FFTW=0)

catch_discover_tests(kharma_unit_tests
  DISCOVERY_MODE PRE_TEST
  TEST_PREFIX kharma.unit.
)
