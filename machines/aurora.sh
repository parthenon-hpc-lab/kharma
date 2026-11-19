
# Config for ALCF Aurora

if [[ $HOST == *".aurora.alcf.anl.gov" ]]
then

  # KHARMA is currently built & run against the previous production MPICH.
  # Running a binary built with this one under the default (87e2045) segfaults on the GPU.
  module load mpich/opt/5.0.0.aurora_test.3c70a61
  module load cmake

  if [[ ! "$*" == *"hdf5"* ]]; then
    module load hdf5
  else
    module unload hdf5
  fi

  MPI_EXE=mpiexec
  NPROC=32

  export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}
  export OMP_PLACES=${OMP_PLACES:-threads}

  # Aurora's HDF5 does not include ZLIB
  # There may be a real fix, but Flare is fast
  EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON $EXTRA_FLAGS"

  if [[ $ARGS == *"sycl"* ]]; then
    # SYCL compile for Intel GPUs
    DEVICE_ARCH=INTEL_PVC

    # I think this is faster on Intel like on AMD?
    EXTRA_FLAGS="-DKHARMA_SPLIT_IMPLICIT_SOLVE=ON $EXTRA_FLAGS"

    # Aurora jobs export OMP_NUM_THREADS=208, i.e. a thread per core in *every* rank.
    # Override with ./run.sh -nt N
    export OMP_NUM_THREADS=1

    # GPU-aware MPI (default, but to be safe)
    export MPIR_CVAR_ENABLE_GPU=1
    # Since the 2026-09 system update MPICH doesn't cache GPU IPC handles by default,
    # making intra-node halo exchanges ~30x slower. Takes unlimited|limited|disabled, not a number.
    export MPIR_CVAR_CH4_IPC_GPU_CACHE_SIZE=${MPIR_CVAR_CH4_IPC_GPU_CACHE_SIZE:-unlimited}

    # Runtime: WTF Intel, this is so complicated
    export CPU_BIND_SCHEME="--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
    MPI_NUM_PROCS=${MPI_NUM_PROCS:-12}
    MPI_EXTRA_ARGS="$MPI_EXTRA_ARGS -ppn 12 $CPU_BIND_SCHEME gpu_tile_compact.sh"

    #export CPU_BIND_SCHEME="--cpu-bind=list:1-8:17-24:33-40:53-60:69-76:85-92"
    #MPI_NUM_PROCS=${MPI_NUM_PROCS:-6}
    #MPI_EXTRA_ARGS="-ppn 6 $CPU_BIND_SCHEME gpu_dev_compact.sh"

  else
    # CPU Compile
    # TODO -c etc etc
    MPI_NUM_PROCS=${MPI_NUM_PROCS:-1}
  fi
fi
