
# Config for NCSA Delta (RH9), ACCESS GPU resource
# Uses Cray Programming Environment (CrayPE) + cray-mpich (no OpenMPI anymore!!)

# With the major update to Delta in Fall 2025 (transition to Redhat 9 and newer compilers),
# the cluster now provides MPI implementation and other packages via CrayPE, similar to how
# NCSA does it on DeltaAI.
# The default environment will be based on the GNU CrayPE PrgEnv-gnu.
# Other cray environments are available (PrgEnv-nvidia, PrgEnv-cray).
# What's new: https://docs.ncsa.illinois.edu/systems/delta/en/rh9/whats_new.html

# HDF5: prefer KHARMA's vendored build:  ./make.sh clean cuda hdf5

if [[ $HOST == *".delta.internal.ncsa.edu" || $HOST == *".delta.ncsa.illinois.edu" ]]
then
  HOST_ARCH=ZEN3
  DEVICE_ARCH=AMPERE80
  MPI_EXE=srun
  NPROC=64

  module purge
  module load cmake

  # `cuda` argument still means to build with GPU support.
  if [[ $ARGS == *"cuda"* ]]
  then
    # GPU Compile
    # 4-device MPI w/mapping, should play nice with different numbers
    # MPI_NUM_PROCS=${MPI_NUM_PROCS:-4}
    # MPI_EXTRA_ARGS="--map-by ppr:$MPI_NUM_PROCS:node:pe=16"

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
    module load craype/2.7.34
    module load gcc-native/13.2
    module load craype-network-ofi
    module load libfabric/1.22.0
    module load cray-mpich/8.1.32
    module load craype-x86-milan
    module load cudatoolkit/25.3_12.8
    module load craype-accel-nvidia80
    module load cray-hdf5-parallel
    module list

    export C_NATIVE=cc
    export CXX_NATIVE=CC

  # CPU-only build. To be tested.
  else
    # CPU Compile
    module load modtree/cpu gcc
    MPI_NUM_PROCS=1
  fi

  echo "--- Using Kokkos version: ---"
    (cd external/parthenon/external/Kokkos && git describe --tags --always)
  echo "-----------------------------"
fi
