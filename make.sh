#!/bin/bash

# Make script for KHARMA
# Used to set sensible default flags and call cmake/make
# Usage:
# ./make.sh [option1] [option2]
#
# clean: BUILD by re-running cmake, restarting the make process from nothing.
#        That is, "./make.sh clean" == "make clean" + "make"
#        Always use 'clean' when switching Release<->Debug or OpenMP<->CUDA
# cuda:  Build for GPU with CUDA
# sycl:  Build for GPU with SYCL
# hip:   Build for GPU with HIP
# debug: Configure with debug flags: mostly array bounds checks
#        Note, though, many sanity checks during the run are
#        actually *runtime* parameters e.g. verbose, flag_verbose, etc
# trace: Configure with execution tracing: print at the beginning and end
#        of most host-side function calls during a step
# hdf5:  Download & compile HDF5, rather than looking for a system version
# cleanhdf5:  Reconfigure HDF5 from scratch, rather than just recompiling
# nompi:      Disable MPI and don't search/link it
# noimplicit: Disable implicit solver, avoids pulling in Kokkos-kernels
# nocleanup:  Disable magnetic field cleaning code for resizing, avoids
#             pulling in some unofficial Parthenon code.
# test:       Build unit tests and register them with CTest
# Many machine files have additional options, check machines/machinename.sh

# Make processes to use
# Set conservatively as nvcc/nvc++ uses a *lot* of memory
# Set in environment or override in machine file
NPROC=${NPROC:-8}

### Load basic stuff ###
HOST=$(hostname -f)
if [ -z $HOST ]; then
  HOST=$(hostname)
fi
ARGS="$*"
SOURCE_DIR=$(dirname "$(readlink -f "$0")")

# Parse options in a slightly less insane way than before
# At least this checks for the full space-separated word as a flag
args_array=( "$@" )
option() {
  printf '%s\0' "${args_array[@]}" | grep -Fxqz -- $1
}
export -f option

# A machine config in .config overrides our defaults
if [ -f $HOME/.config/kharma.sh ]; then
  source $HOME/.config/kharma.sh
else
  for machine in $SOURCE_DIR/machines/*.sh
  do
    source $machine
  done
fi

# Default to compiling for the host architecture
# Always better to specify, though, for cross-compile/older Kokkos support
EXTRA_FLAGS="-DKokkos_ARCH_${HOST_ARCH:-NATIVE}=ON $EXTRA_FLAGS"

# Kokkos does *not* support compiling for multiple devices!
# But if they ever do, you can separate a list of DEVICE_ARCH
# with commas.
if [[ -v DEVICE_ARCH ]]; then
  readarray -t arch_array < <(awk -F',' '{ for( i=1; i<=NF; i++ ) print $i }' <<<"$DEVICE_ARCH")
  for arch in "${arch_array[@]}"; do
    EXTRA_FLAGS="-DKokkos_ARCH_${arch}=ON $EXTRA_FLAGS"
  done
fi
if option "trace"; then
  EXTRA_FLAGS="-DKHARMA_TRACE=ON $EXTRA_FLAGS"
fi
if option "nompi"; then
  EXTRA_FLAGS="-DKHARMA_DISABLE_MPI=ON $EXTRA_FLAGS"
fi
if option "noimplicit"; then
  EXTRA_FLAGS="-DKHARMA_DISABLE_IMPLICIT=ON $EXTRA_FLAGS"
fi
# Always disable old resizing, it's broken w/new tasking
#if option "nocleanup"; then
EXTRA_FLAGS="-DKHARMA_DISABLE_CLEANUP=ON $EXTRA_FLAGS"
#fi
if option "split_implicit"; then
  EXTRA_FLAGS="-DKHARMA_SPLIT_IMPLICIT_SOLVE=ON $EXTRA_FLAGS"
fi
if option "test"; then
  EXTRA_FLAGS="-DKHARMA_BUILD_TESTS=ON $EXTRA_FLAGS"
fi
# This builds iris binary *instead* of KHARMA.  I have insufficient CMake-fu to do them together
if option "iris"; then
  EXTRA_FLAGS="-DKHARMA_BUILD_IRIS=ON $EXTRA_FLAGS"
fi

### Enivoronment Prep ###
if [[ "$(which python3 2>/dev/null)" == *"conda"* ]]; then
  echo "make.sh note:"
  echo "It looks like you have Anaconda loaded."
  echo "This is usually okay, but double-check the line 'Found MPI_CXX:' below!"
  echo
fi
# Save arguments if we've changed them
# Used in run.sh for loading the same modules/etc.
if option "clean"; then
  echo "$ARGS" > $SOURCE_DIR/make_args
fi
# Choose configuration
if option "debug"; then
  TYPE=Debug
else
  TYPE=Release
fi

### Set KHARMA Flags ###
SCRIPT_DIR=$( dirname "$0" )
cd $SCRIPT_DIR
SCRIPT_DIR=$PWD

# Generally best to set CXX_NATIVE yourself if you want to be sure,
# but we try to be smart about loading the most specific/advanced/
# capable compiler available in PATH.
if [[ -z "$CXX_NATIVE" ]]; then
  # If Cray environment is loaded, use their wrappers
  if which CC >/dev/null 2>&1; then
    CXX_NATIVE=CC
    C_NATIVE=cc
    # Don't set an OMP flag to use
    # This could call through to any compiler, & sometimes (Frontier)
    # we want no OpenMP at all
  # Prefer Intel oneAPI compiler over legacy, both over generic
  elif which icpx >/dev/null 2>&1; then
    CXX_NATIVE=icpx
    C_NATIVE=icx
  elif which icpc >/dev/null 2>&1; then
    CXX_NATIVE=icpc
    C_NATIVE=icc
  # Prefer NVHPC over generic compilers
  elif which nvc++ >/dev/null 2>&1; then
    CXX_NATIVE=nvc++
    C_NATIVE=nvc
  # Maybe we overwrote 'c++' to point to something
  # Usually this is GCC on Linux systems, which is fine
  elif which c++ >/dev/null 2>&1; then
    CXX_NATIVE=c++
    C_NATIVE=cc
  # Otherwise, trusty system GCC
  else
    CXX_NATIVE=g++
    C_NATIVE=gcc
  fi
  # TODO(CEP) finally use clang/++ automatically, just w/o OpenMP?
fi

# Set flags, incl. correct OpenMP flag for our compiler
if [[ $CXX_NATIVE == *"icpx" ]]; then
  # Avoid icpx's astonishing DEFAULT -ffast-math
  export CXXFLAGS="-fno-fast-math $CXXFLAGS"
  OMP_FLAG="-fiopenmp"
elif [[ $CXX_NATIVE == *"icpc" ]]; then
  # Avoid warning on nvcc pragmas Intel doesn't like
  export CXXFLAGS="-Wno-unknown-pragmas $CXXFLAGS"
  OMP_FLAG="-qopenmp"
elif [[ $CXX_NATIVE == *"nvc++" ]]; then
  OMP_FLAG="-mp"
elif [[ $CXX_NATIVE == *"c++" || $CXX_NATIVE == *"g++" ]]; then
  OMP_FLAG="-fopenmp"
fi

# Disable OpenMP for HIP compiles, it gets confused
# and thinks we want to use OMP 5.0 offload stuff
if ! option "hip"; then
  export CXXFLAGS="$OMP_FLAG $CXXFLAGS"
fi

# Set compilers
# Options are named different so we can override w/wrapper for CUDA
export CXX="$CXX_NATIVE"
export CC="$C_NATIVE"

# CUDA loop options: MANUAL1D_LOOP > MDRANGE_LOOP, TPTTR_LOOP & TPTTRTVR_LOOP don't compile
# Inner loop must be TVR_INNER_LOOP
# OpenMP loop options for KNL:
# Outer: SIMDFOR_LOOP;MANUAL1D_LOOP;MDRANGE_LOOP;TPTTR_LOOP;TPTVR_LOOP;TPTTRTVR_LOOP
# Inner: SIMDFOR_INNER_LOOP;TVR_INNER_LOOP
if option "sycl"; then
  # TODO CXXFLAGS add -Wno-dynamic-class-memaccess
  OUTER_LAYOUT="MANUAL1D_LOOP"
  INNER_LAYOUT="TVR_INNER_LOOP"
  ENABLE_OPENMP="ON"
  ENABLE_CUDA="OFF"
  ENABLE_SYCL="ON"
  ENABLE_HIP="OFF"
elif option "hip"; then
  OUTER_LAYOUT="MANUAL1D_LOOP"
  INNER_LAYOUT="TVR_INNER_LOOP"
  ENABLE_OPENMP="OFF"
  ENABLE_CUDA="OFF"
  ENABLE_SYCL="OFF"
  ENABLE_HIP="ON"
elif option "cuda"; then
  export CXX="$SCRIPT_DIR/external/parthenon/external/Kokkos/bin/nvcc_wrapper"
  if option "wrapper_dryrun"; then
    export CXXFLAGS="-dryrun $CXXFLAGS"
    echo "Dry-running the nvcc wrapper with $CXXFLAGS"
  fi
  export NVCC_WRAPPER_DEFAULT_COMPILER="$CXX_NATIVE"
  EXTRA_FLAGS="$EXTRA_FLAGS -DKokkos_ENABLE_CUDA_CONSTEXPR=ON"
  OUTER_LAYOUT="MANUAL1D_LOOP"
  INNER_LAYOUT="TVR_INNER_LOOP"
  ENABLE_OPENMP="OFF"
  ENABLE_CUDA="ON"
  ENABLE_SYCL="OFF"
  ENABLE_HIP="OFF"
elif option "nvc++"; then
  OUTER_LAYOUT="MANUAL1D_LOOP"
  INNER_LAYOUT="TVR_INNER_LOOP"
  ENABLE_OPENMP="OFF"
  ENABLE_CUDA="ON"
  ENABLE_SYCL="OFF"
  ENABLE_HIP="OFF"
elif option "noopenmp"; then
  OUTER_LAYOUT="SIMDFOR_LOOP"
  INNER_LAYOUT="SIMDFOR_INNER_LOOP"
  ENABLE_OPENMP="OFF"
  ENABLE_CUDA="OFF"
  ENABLE_SYCL="OFF"
  ENABLE_HIP="OFF"
else
  OUTER_LAYOUT="MANUAL1D_LOOP"
  INNER_LAYOUT="SIMDFOR_INNER_LOOP"
  ENABLE_OPENMP="ON"
  ENABLE_CUDA="OFF"
  ENABLE_SYCL="OFF"
  ENABLE_HIP="OFF"
fi

# Allow for a custom linker program, but use CXX by
# default as system linker may be older/incompatible
if [[ -v LINKER ]]; then
  EXTRA_FLAGS="$EXTRA_FLAGS -DCMAKE_LINKER=$LINKER"
fi
if option "special_link_line"; then
  EXTRA_FLAGS="$EXTRA_FLAGS -DCMAKE_CXX_LINK_EXECUTABLE='<CMAKE_LINKER> <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>'"
fi

### Build HDF5 ###
# If we're building HDF5, do it after we set *all flags*
if option "hdf5" && option "clean" && ! option "dryrun"; then
  H5VER=1.14.2
  H5VERU=1_14_2

  cd external
  # Allow complete reconfigure (for switching compilers, takes longer)
  if option "cleanhdf5"; then
    rm -rf hdf5-${H5VER}/
  fi
  # Download if needed
  if [ ! -f hdf5-${H5VER}.tar.gz ]; then
    curl https://hdf-wordpress-1.s3.amazonaws.com/wp-content/uploads/manual/HDF5/HDF5_${H5VERU}/src/hdf5-${H5VER}.tar.gz -o hdf5-${H5VER}.tar.gz
  fi
  # Unpack if needed (or deleted)
  if [ ! -d hdf5-${H5VER}/ ]; then
    tar xf hdf5-${H5VER}.tar.gz
  fi
  cd hdf5-${H5VER}/
  # TODO better ensure we're using C_NATIVE underneath.  e.g. MPI_CFLAGS with -cc
  if  option "nompi"; then
    HDF_CC=$C_NATIVE
    HDF_EXTRA=""
  else
    if [[ "$C_NATIVE" == *"icx"* ]]; then
      HDF_CC=mpiicx
    elif [[ "$C_NATIVE" == *"icc"* ]]; then
      HDF_CC=mpiicc
    elif [[ "$C_NATIVE" == "cc" ]]; then
      # Cray wrappers include MPI
      HDF_CC=cc
    else
      HDF_CC=mpicc
    fi
    HDF_EXTRA="--enable-parallel"
  fi

  echo Configuring HDF5...

  export CFLAGS="-fPIC $CFLAGS"
  CC=$HDF_CC sh configure -C $HDF_EXTRA --prefix=$SOURCE_DIR/external/hdf5 --enable-build-mode=production \
  --disable-dependency-tracking --disable-tests --disable-tools --disable-shared --disable-deprecated-symbols > build-hdf5.log
  sleep 1

  echo "Building HDF5 (probably 30s-2min)"
  # Compiling C takes less memory
  if [[ -v $NPROC ]]; then
    make -j$(( $NPROC * 2 )) >> build-hdf5.log 2>&1
  else
    make -j >> build-hdf5.log 2>&1
  fi
  make install >> build-hdf5.log 2>&1
  make clean >> build-hdf5.log 2>&1
  cd ../..

  echo Built HDF5 version $H5VER
fi

# Compile against our hdf5 if specified
if option "hdf5"; then
  PREFIX_PATH="$SOURCE_DIR/external/hdf5;$PREFIX_PATH"
  EXTRA_FLAGS="$EXTRA_FLAGS -DHDF5_USE_STATIC_LIBRARIES=ON"
fi



### Build KHARMA ###
# If we're doing a clean build, prep the source and
# delete the build directory
if option "clean"; then

  # Should do this manually when compiling on backend nodes!
  if [ ! -f external/parthenon/CMakeLists.txt -o \
       ! -f external/singularity-eos/CMakeLists.txt ]; then
    git submodule update --recursive --init
  fi

  # Patch parthenon to use KHARMA's coordinates, anything incidental
  cd external/parthenon
  if [[ $(( $(git --version | cut -d '.' -f 2) > 35 )) == "1" ]]; then
    git apply --quiet ../patches/parthenon-*.patch
  else
    echo "make.sh note: You may see errors applying patches below. These are normal."
    git apply ../patches/parthenon-*.patch
  fi
  cd -

  # Patches for ports-of-call
  #cd external/singularity-eos/utils/ports-of-call
  #if [[ $(( $(git --version | cut -d '.' -f 2) > 35 )) == "1" ]]; then
  #  git apply --quiet ../../../patches/ports-of-call-*.patch
  #else
  #  echo "make.sh note: You may see errors applying patches below. These are normal."
  #  git apply ../../../patches/ports-of-call-*.patch
  #fi
  #cd -

  rm -rf build
fi
mkdir -p build
cd build

if option "clean"; then

  # Print cmake command
  if option "dryrun"; then
    set -x
  fi

  cmake ..\
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_PREFIX_PATH="$PREFIX_PATH;$CMAKE_PREFIX_PATH" \
    -DCMAKE_BUILD_TYPE=$TYPE \
    -DPAR_LOOP_LAYOUT=$OUTER_LAYOUT \
    -DPAR_LOOP_INNER_LAYOUT=$INNER_LAYOUT \
    -DKokkos_ENABLE_OPENMP=$ENABLE_OPENMP \
    -DKokkos_ENABLE_CUDA=$ENABLE_CUDA \
    -DKokkos_ENABLE_SYCL=$ENABLE_SYCL \
    -DKokkos_ENABLE_HIP=$ENABLE_HIP \
    $EXTRA_FLAGS

  # Stop printing
  if option "dryrun"; then
    set +x
  fi

  # Describe the kokkos version, for debugging
  echo "--- Using Kokkos version: ---"
  (cd $SCRIPT_DIR/external/parthenon/external/Kokkos && git describe --tags --always && cd -)
  echo "-----------------------------"
fi

if ! option "dryrun"; then
  make -j$NPROC

  if option "iris"; then
    cp iris/iris.* ..
  else
    cp kharma/kharma.* ..
  fi

  # Needed now that we build packages
  if option "install"; then
    make install
  fi
fi
