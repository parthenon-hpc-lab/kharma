# ALCF: Polaris
if [[ $HOST == *"polaris"* ]]; then
  HOST_ARCH=ZEN3
  DEVICE_ARCH=AMPERE80
  NPROC=64

  module restore

  # Re-enable the ALCF software module tree that purge wiped out
  module use /soft/modulefiles
  module use /soft/modulefiles/core
  
  # Clean up any active environments defensively without throwing errors
  module unload PrgEnv-gnu PrgEnv-nvidia PrgEnv-cray PrgEnv-intel
  
  if [[ $ARGS == *"gcc"* ]]; then
      module load PrgEnv-gnu
      module load cuda
  else
    # Defaulting to the modern system-recommended NVIDIA path
      module load PrgEnv-nvidia
      module load cuda
  fi
  # Common modules
  # UNLOCK THE SPACK UTILITY TREE
  module load spack-pe-base
  
  module load cray-hdf5-parallel
  module load craype-accel-nvidia80
  module load cmake
  module list
  
  
  # Ensure the Cray compiler wrappers explicitly target the host architecture
  export CRAY_CPU_TARGET=x86-64
  # TODO(CEP) need to set CRAYPE_LINK_TYPE=dynamic long-term?

  # Globally enforce the C++17 standard for all targets and submodules
  EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON $EXTRA_FLAGS"

  # ==============================================================================
  # Runtime Configuration for Execution via run.sh
  # ==============================================================================
  
  # Set the primary parallel launcher for Polaris
  MPI_EXE="mpiexec"

  if [[ $ARGS == *"cuda"* ]]; then
    # Enable hardware-accelerated MPI and orchestrate Kokkos device mappings
    export MPICH_GPU_SUPPORT_ENABLED=1
    export KOKKOS_MAP_DEVICE_ID_BY=mpi_rank

    # 1. Fallback default: If MPI_NUM_PROCS isn't set, default to 1
    # (run.sh naturally initializes MPI_NUM_PROCS=1, so we catch it here)
    if [[ -z "$MPI_NUM_PROCS" ]]; then
      MPI_NUM_PROCS=1
    fi

    # 2. Dynamic Hardware Math:
    # Set PPN to match MPI_NUM_PROCS up to the maximum 4 GPUs per node.
    if (( MPI_NUM_PROCS < 4 )); then
      LOCAL_PPN=$MPI_NUM_PROCS
    else
      LOCAL_PPN=4
    fi

    # 3. Assign the dynamically calculated arguments
    MPI_EXTRA_ARGS="--ppn ${LOCAL_PPN} --depth=16 --cpu-bind depth"

  else
    # Host/CPU-only fallback execution profile
    MPI_NUM_PROCS=${MPI_NUM_PROCS:-1}
    MPI_EXTRA_ARGS="--ppn 1"
  fi
fi
