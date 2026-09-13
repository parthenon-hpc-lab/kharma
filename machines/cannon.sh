# Harvard Cannon

if [[ $HOST == *"rc.fas.harvard.edu" ]]; then
  HOST_ARCH=SKX

  if [[ "$ARGS" != *"login"* ]]; then
    if [[ $HOST == *"login"* ]]; then
      echo "Processes on head nodes are killed. Compile on a compute node!"
      exit
    fi

    # Use all of the compute node, we're impatient
    NPROC=48

  else
    # Stay out of the way on login nodes
    NPROC=1
  fi

  # System HDF5 doesn't have compression, we don't need it
  EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON" # -DPARTHENON_ENABLE_HOST_COMM_BUFFERS=ON"

  MPI_EXE=${MPI_EXE:-"srun"}
  MPI_EXTRA_ARGS=${MPI_EXTRA_ARGS:-"--mpi=pmix"}

  if option "cuda"; then
    if option "volta"; then
      DEVICE_ARCH=VOLTA70
    else
      DEVICE_ARCH=AMPERE80
    fi

    # Never hurts unless we enable HOST_BUFFERS
    export MPICH_GPU_SUPPORT_ENABLED=1

    if option "nvhpc"; then
      # NVHPC includes Nvidia's MPI that has GPUDirect
      # Currently segfaults...

      # Note this compiles hdf5 with nvc, because mpicc->nvc non-negotiably
      # This is fine as long as you allow shared libraries in HDF5 config
      module load gcc/13.2.0-fasrc01
      module load nvhpc
      # Didn't work for me to get GCC14.2 going, but maybe someday?
      #export CXXFLAGS="$CXXFLAGS -allow-unsupported-compiler"

      # Use Nvidia's openmpi
      MPI_EXE="mpirun"
    else
      module load gcc/14.2.0-fasrc01
      module load openmpi/5.0.5-fasrc01
      module load hdf5
      module load cuda
    fi

    # Avoid Nvidia's compilers
    C_NATIVE=gcc
    CXX_NATIVE=g++
  else
    # TODO Intel's or AMD's compiler is probably faster
    module load gcc/14.2.0-fasrc01
    module load openmpi/5.0.5-fasrc01
    module load hdf5
    # General Cannon CPU machines have 112 threads
    export OMP_NUM_THREADS=56
    C_NATIVE=gcc
    CXX_NATIVE=g++
  fi

  # Load last, might be gated on having a compiler
  module load cmake
  #module list
fi

