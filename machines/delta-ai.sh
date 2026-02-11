
# Config for NCSA DeltaAI (RH9), ACCESS GPU resource

if [[ $HOST =~ gh-login[0-9]+\.delta\.ncsa\.illinois\.edu ]]
then
  echo "Building on DeltaAI."
  HOST_ARCH=ARMV9_GRACE
  DEVICE_ARCH=HOPPER90
  MPI_EXE=srun
  NPROC=8

  module purge

  # `cuda` argument still means to build with GPU support.
  if [[ $ARGS == *"cuda"* ]]
  then
    # Enable GPU RDMA for cray-mpich
    export MPICH_GPU_SUPPORT_ENABLED=1
    export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1

    if [[ "$ARGS" == *"hostside"* ]]; then
      # Device-side buffers are broken on some Nvidia machines
      EXTRA_FLAGS="-DPARTHENON_ENABLE_HOST_COMM_BUFFERS=ON $EXTRA_FLAGS"
    fi

    # GNU is the only build tested as of now. 
    # ToDo: Test nvidia HPC SDK and Cray builds.
    module load PrgEnv-gnu
    module load default # Needed for cmake on DeltaAI
    module load cmake
    # Unload gcc-native/13 and load gcc-native/12 to enable gpudirect.
    module unload gcc-native/13
    module load gcc-native/12
    module load cudatoolkit
    module load craype-arm-grace
    module load craype-accel-nvidia90
    module load cray-hdf5-parallel
    module list

    export C_NATIVE=cc
    export CXX_NATIVE=CC

  # CPU-only build. To be tested.
  elif [[ $ARGS == *"cpu"* ]]
  then
    # CPU Compile
    module load PrgEnv-gnu
    module load craype/2.7.34
    module load gcc-native/13.2
    module load craype-network-ofi
    module load libfabric/1.22.0
    module load cray-mpich/8.1.32
    module load craype-x86-milan
    module load cray-hdf5-parallel
    module list

    export C_NATIVE=cc
    export CXX_NATIVE=CC
  else
    echo "Error: No valid build type specified. Use 'cuda' for GPU build or 'cpu' for CPU-only build."
    exit 1
  fi
fi
