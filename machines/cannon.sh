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

  MPI_EXE="srun"
  MPI_EXTRA_ARGS="--mpi=pmix"

  if option "cuda"; then
    if option "volta"; then
      DEVICE_ARCH=VOLTA70
    else
      DEVICE_ARCH=AMPERE80
    fi

    if option "nompi"; then
      module load gcc/14.2.0-fasrc01
      module load cuda
    else
      # NVHPC includes Nvidia's MPI that has GPUDirect
      # Note this compiles hdf5 with nvc, because mpicc->nvc non-negotiably
      # This is fine as long as you allow shared libraries in HDF5 config
      export MPICH_GPU_SUPPORT_ENABLED=1
      module load gcc/13.2.0-fasrc01
      module load nvhpc
      # Didn't work for me, but you're welcome to try!
      #export CXXFLAGS="$CXXFLAGS -allow-unsupported-compiler"
    fi

    # Avoid Nvidia's compilers
    C_NATIVE=gcc
    CXX_NATIVE=g++
  else
    # TODO Intel's or AMD's compiler is probably faster
    module load gcc/14.2.0-fasrc01
    module load hdf5
    # General Cannon CPU machines have 112 threads
    export OMP_NUM_THREADS=56
    C_NATIVE=gcc
    CXX_NATIVE=g++
  fi

  # Load last, might be gated on having a compiler
  module load cmake
fi

