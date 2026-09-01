
if [[ $HOST == *".astro.illinois.edu" ]]; then
  if [[ $HOST == "bh29"* ]]; then
    # BH29: Zen2 AMD EPYC 7742
    HOST_ARCH="ZEN2"
    # BH29 benefits from using just 1 thread/core
    export OMP_NUM_THREADS=64
    NPROC=64
  else
    # Other machines are Skylake
    HOST_ARCH="SKX"
    NPROC=36
  fi

  # Compile our own HDF5 by default
  PREFIX_PATH="$SOURCE_DIR/external/hdf5"

  if [[ $ARGS == *"icc"* ]]; then
    # Intel ICC ("classic")
    module purge
    source /opt/intel/oneapi/setvars.sh
    C_NATIVE="icc"
    CXX_NATIVE="icpc"
    C_FLAGS="-diag-disable=10441"
    CXX_FLAGS="-diag-disable=10441"

  elif [[ $ARGS == *"aocc"* ]]; then
    # AMD AOCC (BH29 only)
    source /opt/AMD/aocc-compiler-3.1.0.sles15/setenv_AOCC.sh
    C_NATIVE="clang"
    CXX_NATIVE="clang++"

  else
    # GCC 7.5 is too old to compile KHARMA at all. Use new Intel compiler by default
    module purge
    source /opt/intel/oneapi/setvars.sh
    C_NATIVE="icx"
    CXX_NATIVE="icpx"
  fi
fi
