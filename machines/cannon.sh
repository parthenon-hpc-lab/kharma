# Harvard Cannon

if [[ $HOST == *"rc.fas.harvard.edu" ]]; then
  HOST_ARCH=SKX
  # Use all of our usual compute job
  NPROC=48

  if [[ $HOST == *"login"* ]]; then
    echo "Processes on head nodes are killed. Using one thread!"
    export NPROC=1
  fi

  # System HDF5 doesn't have compression, we don't need it
  EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON" # -DPARTHENON_ENABLE_HOST_COMM_BUFFERS=ON"

  module load gcc/14.2.0-fasrc01 openmpi/5.0.5-fasrc01
  module load hdf5
  module load cmake

  MPI_EXE="srun"
  MPI_EXTRA_ARGS="--mpi=pmix"

  C_NATIVE=gcc
  CXX_NATIVE=g++
  if [[ "$ARGS" == *"cuda"* ]]; then
    if [[ "$ARGS" == *"volta"* ]]; then
      DEVICE_ARCH=VOLTA70
    else
      DEVICE_ARCH=AMPERE80
    fi
    export MPICH_GPU_SUPPORT_ENABLED=1
    module load cuda

  else
    # General Cannon CPU machines have 112 threads
    export OMP_NUM_THREADS=56
  fi
fi

