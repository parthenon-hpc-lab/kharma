# Harvard Cannon

if [[ $HOST == *"rc.fas.harvard.edu" ]]; then
    if [[ $HOST == *"login"* ]]; then
      echo "Processes on head nodes are killed. Compile on a compute node!"
      exit
    fi

    # Use all of our usual compute job
    NPROC=48

    HOST_ARCH=SKX
    EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON" # -DPARTHENON_ENABLE_HOST_COMM_BUFFERS=ON"

    module purge
    module load gcc/12.2.0-fasrc01 openmpi/4.1.5-fasrc02 hdf5/1.12.2-fasrc01

    if [[ "$ARGS" == *"rdma"* ]]; then
      module load cmake/3.27.5-fasrc01
    else
      module load cmake/3.25.2-fasrc01
    fi
    if [[ "$ARGS" == *"2gpu"* ]]; then
      # CUDA aware MPI
      MPI_NUM_PROCS=2
    fi
    if [[ "$ARGS" == *"cudaaware"* ]]; then
      # CUDA aware MPI
      #module load ucx/1.14.1-fasrc02
      MPI_NUM_PROCS=4
      #export KOKKOS_NUM_DEVICES=4
    fi
    
    #module load intel/23.0.0-fasrc01 openmpi/4.1.4-fasrc01 cmake/3.25.2-fasrc01
    #MPI_EXTRA_ARGS="--map-by ppr:4:node:pe=16"
    MPI_EXE="srun"
    MPI_EXTRA_ARGS="--mpi=pmix"

    #source /n/holylfs05/LABS/bhi/Users/hyerincho/grmhd/spack/share/spack/setup-env.sh
    #spack clean -m
    #spack install gcc@10.2.0 openmpi@4.1.3 cmake@3.23.2

    C_NATIVE=gcc
    CXX_NATIVE=g++
    if [[ "$ARGS" == *"cuda"* ]]; then
      DEVICE_ARCH=AMPERE80
      export MPICH_GPU_SUPPORT_ENABLED=1
      # Need CUDA >=12.2.2 in modern KHARMA,
      # this is CUDA 12.9
      module load cuda
      if [[ "$ARGS" == *"volta"* ]]; then
        DEVICE_ARCH=VOLTA70
      fi
    else
      # General Cannon CPU machines have 112 threads
      export OMP_NUM_THREADS=56
    fi
fi

