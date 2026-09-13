# Config for NCSA Delta, ACCESS GPU resource

if [[ $HOST == *".delta.internal.ncsa.edu" || $HOST == *".delta.ncsa.illinois.edu" ]]
then
  HOST_ARCH=ZEN3
  DEVICE_ARCH=AMPERE80
  MPI_EXE=srun
  NPROC=64

  # Generally we want the Cray comipiler/environment,
  # for better device-side MPI support
  if [[ $ARGS == *"cray"* ]]; then
    module swap PrgEnv-gnu PrgEnv-cray
  fi

  module load cmake
  module load cray-hdf5-parallel
  module list

  export C_NATIVE=cc
  export CXX_NATIVE=CC

  if [[ $ARGS == *"cuda"* ]]; then
    # GPU Compile
    if [[ "$ARGS" == *"hostside"* ]]; then
      # Device-side buffers are broken on some Nvidia machines
      EXTRA_FLAGS="-DPARTHENON_ENABLE_HOST_COMM_BUFFERS=ON $EXTRA_FLAGS"
    fi
  else
    # CPU Compile (TODO test)
    #module load modtree/cpu gcc
    MPI_NUM_PROCS=1
  fi
fi
