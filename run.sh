#!/bin/bash

### System-specific parameters
# Override these with your compile file in machines/!
# For running different configs on the fly, you can use the options
# -n (number of MPI procs)
# -nt (number of OpenMP threads)
# -b (KHARMA binary to use)
# Note these options must be BEFORE any KHARMA options!

### Kokkos tools and profiling
# KHARMA can automatically add some of the Kokkos-tools,
# if the 'kokkos-tools' library
# Specify these FIRST, before any other options including the above
# 'trace': activates the Kokkos "kernel-logger" printing all
#          kernel and profiling region (function) names
# 'prof': activates the Kokkos "kernel-timer," with JSON kernel timing output
# 'nvprof': activates the Kokkos nvprof connector, for demangling names (sometimes)
# And for Nvidia's tools:
# 'ncu_basic name_of_output': runs under the Nsight Compute 'ncu' profiler/analyzer, basic profile
# 'ncu_full name_of_output': runs under the Nsight Compute 'ncu' profiler/analyzer, full profile
# The last two run just one step, but repeat 10-40 times for accurate measurements

# Default MPI parameters: don't use MPI or run with 1 process
MPI_EXE=${MPI_EXE:-}
MPI_NUM_PROCS=${MPI_NUM_PROCS:-1}
MPI_EXTRA_ARGS=${MPI_EXTRA_ARGS:-}

### General run script

# Map each MPI rank to one device with Kokkos
export KOKKOS_MAP_DEVICE_ID_BY=mpi_rank
# If you see weird GPU race conditions, setting this
# to 1 *might* fix them. Maybe.
export CUDA_LAUNCH_BLOCKING=0
# Kokkos can be forced to use only a particular device:
#export KOKKOS_DEVICE_ID=0

# Choose the kharma binary from compiled options in order of preference
KHARMA_DIR="$(dirname "${BASH_SOURCE[0]}")"

# Load environment from the same files as the compile process
HOST=$(hostname -f)
ARGS=${ARGS:-$(cat $KHARMA_DIR/make_args)}
SOURCE_DIR=$(dirname "$(readlink -f "$0")")

# A machine config in .config overrides our defaults
if [ -f $HOME/.config/kharma.sh ]; then
  source $HOME/.config/kharma.sh
else
  for machine in $SOURCE_DIR/machines/*.sh
  do
    source $machine
  done
fi

if [[ "$1" == "trace" ]]; then
  export KOKKOS_TOOLS_LIBS=$KHARMA_DIR/../kokkos-tools/kp_kernel_logger.so
  shift
fi
if [[ "$1" == "prof" ]]; then
  export KOKKOS_TOOLS_LIBS=$KHARMA_DIR/../kokkos-tools/kp_kernel_timer.so
  shift
fi
if [[ "$1" == "nvprof" ]]; then
  export KOKKOS_TOOLS_LIBS=$KHARMA_DIR/../kokkos-tools/kp_nvprof_connector.so
  shift
fi
PROF_EXE=""
PROF_OPTS=${PROF_OPTS:-""}
KHARMA_PROF_OPTS=""
if [[ "$1" == "ncu_basic" ]]; then
  PROF_EXE="ncu"
  PROF_OPTS=${PROF_OPTS:-"--set basic --replay-mode application -k regex:cuda_parallel_launch_constant_memory"}
  PROF_OPTS="$PROF_OPTS -o $2"
  # We want short runs, no MPI ever under ncu
  KHARMA_PROF_OPTS="parthenon/time/nlim=1"
  MPI_EXE=""
  shift
  shift
fi
if [[ "$1" == "ncu_full" ]]; then
  PROF_EXE="ncu"
  PROF_OPTS=${PROF_OPTS:-"--set full --replay-mode application -k regex:cuda_parallel_launch_constant_memory"}
  PROF_OPTS="$PROF_OPTS -o $2"
  # We want short runs, no MPI ever under ncu
  KHARMA_PROF_OPTS="parthenon/time/nlim=1"
  MPI_EXE=""
  shift
  shift
fi

# Override MPI_NUM_PROCS at user option "-n"
# and OMP_NUM_THREADS at option "-nt"
if [[ "$1" == "-n" ]]; then
  MPI_NUM_PROCS="$2"
  if [[ -z $MPI_EXE && $(( $MPI_NUM_PROCS > 1 )) ]]; then
    MPI_EXE="mpirun"
  fi
  shift
  shift
fi
if [[ "$1" == "-nt" ]]; then
  export OMP_NUM_THREADS="$2"
  shift
  shift
fi
if [[ "$1" == "-b" ]]; then
  EXE_NAME="$2"
  shift
  shift
fi

# Set default exe only if we didn't specify it
PROJ_NAME="iris"
if [ -z "$EXE_NAME" ]; then
  if [ -f $KHARMA_DIR/${PROJ_NAME}.cuda ]; then
    EXE_NAME=${PROJ_NAME}.cuda
  elif [ -f $KHARMA_DIR/${PROJ_NAME}.sycl ]; then
    EXE_NAME=${PROJ_NAME}.sycl
  elif [ -f $KHARMA_DIR/${PROJ_NAME}.hip ]; then
    EXE_NAME=${PROJ_NAME}.hip
  elif [ -f $KHARMA_DIR/${PROJ_NAME}.host ]; then
    EXE_NAME=${PROJ_NAME}.host
    # Enable OpenMP to use all threads only where not counterproductive
    #export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}
    #export OMP_PLACES=${OMP_PLACES:-threads}
    # Force a number of OpenMP threads if it doesn't autodetect
    #export OMP_NUM_THREADS=${OMP_NUM_THREADS:-28}
  else
    echo "KHARMA executable not found!"
    exit
  fi
fi

# Run based on preferences
# TODO can we just set +x to print commands, like does that play nice with exec?
if [ -z "$MPI_EXE" ]; then
  echo "Running $PROF_EXE $PROF_OPTS $KHARMA_DIR/$EXE_NAME $@ $KHARMA_PROF_OPTS"
  exec $PROF_EXE $PROF_OPTS $KHARMA_DIR/$EXE_NAME "$@" $KHARMA_PROF_OPTS
else
  echo "Running $MPI_EXE -n $MPI_NUM_PROCS $MPI_EXTRA_ARGS $KHARMA_DIR/$EXE_NAME $@"
  exec $MPI_EXE -n $MPI_NUM_PROCS $MPI_EXTRA_ARGS $KHARMA_DIR/$EXE_NAME "$@"
fi
