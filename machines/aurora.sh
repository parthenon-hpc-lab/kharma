
# Config for ALCF Aurora

if [[ $HOST == *".aurora.alcf.anl.gov" ]]
then

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
