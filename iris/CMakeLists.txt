#
# kharma executable
#
if (Kokkos_ENABLE_CUDA)
    set(EXE_NAME "iris.cuda")
elseif(Kokkos_ENABLE_SYCL)
    set(EXE_NAME "iris.sycl")
elseif(Kokkos_ENABLE_HIP)
    set(EXE_NAME "iris.hip")
else()
    set(EXE_NAME "iris.host")
endif()

AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR} EXE_NAME_SRC)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/../kharma/coordinates EXE_NAME_SRC)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/../kharma/prob/utils EXE_NAME_SRC)
# PACKAGES
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/cameras EXE_NAME_SRC)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/driver EXE_NAME_SRC)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/file EXE_NAME_SRC)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/model EXE_NAME_SRC)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/rays EXE_NAME_SRC)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../kharma/coordinates)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../kharma/prob/utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../kharma) # Headers only, don't compile KHARMA
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../kharma/boundaries)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../kharma/grmhd)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/coefficients) # Header-only "package"
# PACKAGES
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cameras)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/driver)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/file)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/model)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/rays)

# generate version.cpp file from current git commit in a way that
# 1. Re-generates if hash changes
# 2. Does not re-build everything when hash changes
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/version.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/version.cpp" @ONLY)
list(APPEND EXE_NAME_SRC "${CMAKE_CURRENT_BINARY_DIR}/version.cpp" version.hpp)

add_executable(${EXE_NAME} ${EXE_NAME_SRC})

target_link_libraries(${EXE_NAME} PUBLIC kokkos)
target_link_libraries(${EXE_NAME} PUBLIC parthenon)
# Sometimes helps with OpenMP
#target_link_libraries(${EXE_NAME} PUBLIC gomp)
target_link_libraries(${EXE_NAME} PUBLIC z)

# OPTIONS
# TODO is there any way to make compile options less painful in CMake?
#option(IRIS_OPTION "Description" OFF)
#if(IRIS_OPTION)
#    message("Option description")
#    target_compile_definitions(${EXE_NAME} PUBLIC INTERNAL_OPTION=1)
#else()
#    target_compile_definitions(${EXE_NAME} PUBLIC INTERNAL_OPTION=0)
#endif()

# FLAGS
if(CMAKE_BUILD_TYPE)
    if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        message("Debug build")
        target_compile_definitions(${EXE_NAME} PUBLIC DEBUG=1)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
    else()
        message("Explicit non-Debug build")
        target_compile_definitions(${EXE_NAME} PUBLIC DEBUG=0)
    endif()
else()
    message("Implicit non-Debug build")
    target_compile_definitions(${EXE_NAME} PUBLIC DEBUG=0)
endif()

if (Kokkos_ENABLE_HWLOC)
    target_compile_definitions(${EXE_NAME} PUBLIC Kokkos_ENABLE_HWLOC)
    target_link_libraries(${EXE_NAME} PUBLIC hwloc)
endif()
