
# ALCF: Crux (CPU machine, for testing)
if [[ $HOST == *".crux.alcf.anl.gov" ]]; then
  HOST_ARCH=ZEN3

  if [[ $ARGS == *"gcc"* ]]; then
    # untested, don't use this, the cray compilers are fine
    module load PrgEnv-gnu
  fi
  # Common modules
  # You'd think we'd load cmake but noooo Crux doesn't even ship it
  # Compile your own and ensure it's in PATH
  module load cray-fftw cray-hdf5-parallel
  
  EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON $EXTRA_FLAGS"
fi
