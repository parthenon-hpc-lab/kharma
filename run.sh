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
OUTDIR=${OUTDIR:-dumps_kharma}

### General run script

# Map each MPI rank to one device with Kokkos
export KOKKOS_MAP_DEVICE_ID_BY=mpi_rank
# If you see weird GPU race conditions, setting this
# to 1 *might* fix them. Maybe.
export CUDA_LAUNCH_BLOCKING=0
# Kokkos can be forced to use only a particular device:
#export KOKKOS_DEVICE_ID=0

### Load basic stuff ###
KHARMA_DIR=$(dirname "$(readlink -f "$0")")
# Old run.sh version. Why?
#KHARMA_DIR="$(dirname "${BASH_SOURCE[0]}")"

HOST=$(hostname -f)
if [ -z $HOST ]; then
  HOST=$(hostname)
fi
ARGS=${ARGS:-$(cat $KHARMA_DIR/make_args)}

# Parse options in a slightly less insane way than before
# At least this checks for the full space-separated word as a flag
args_array=( $ARGS )
option() {
  printf '%s\0' "${args_array[@]}" | grep -Fxqz -- $1
}
export -f option

# Parse options in a slightly less insane way than before
# At least this checks for the full space-separated word as a flag
args_array=($ARGS)
option() {
  printf '%s\0' "${args_array[@]}" | grep -Fxqz -- $1
}

# A machine config in .config overrides our defaults
if [ -f $HOME/.config/kharma.sh ]; then
  source $HOME/.config/kharma.sh
else
  for machine in $KHARMA_DIR/machines/*.sh
  do
    source $machine
  done
fi

# Run-script-specific stuff
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

DRYRUN=0
if [[ "$1" == "--dryrun" ]]; then
  DRYRUN=1
  shift
fi

# Override MPI_NUM_PROCS at user option "-n"
# and OMP_NUM_THREADS at option "-nt"
if [[ "$1" == "-n" ]]; then
  MPI_NUM_PROCS="$2"
  if [[ -z $MPI_EXE && $(( $MPI_NUM_PROCS > 1 )) ]]; then
    MPI_EXE="srun"
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
if [[ "$1" == "-d" ]]; then
  OUTDIR="$2"
  shift
  shift
fi

# Set default exe only if we didn't specify it
if [ -z "$EXE_NAME" ]; then
  if [ -f $KHARMA_DIR/kharma.cuda ]; then
    EXE_NAME=kharma.cuda
  elif [ -f $KHARMA_DIR/kharma.sycl ]; then
    EXE_NAME=kharma.sycl
  elif [ -f $KHARMA_DIR/kharma.hip ]; then
    EXE_NAME=kharma.hip
  elif [ -f $KHARMA_DIR/kharma.host ]; then
    EXE_NAME=kharma.host
    # Enable OpenMP to use all threads only where not counterproductive
    #export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}
    #export OMP_PLACES=${OMP_PLACES:-threads}
    # Force a number of OpenMP threads if it doesn't autodetect
    #export OMP_NUM_THREADS=${OMP_NUM_THREADS:-28}
  else
    if [ -f $KHARMA_DIR/build-artifacts.zip ]; then
      cd $KHARMA_DIR
      unzip build-artifacts.zip
      cd -
    fi
    echo "KHARMA executable not found!"
    exit
  fi
fi

chmod +x $KHARMA_DIR/$EXE_NAME

# Run based on preferences
if [ -z "$MPI_EXE" ]; then
  echo "Running $PROF_EXE $PROF_OPTS $KHARMA_DIR/$EXE_NAME $@ $KHARMA_PROF_OPTS"
  if [[ $DRYRUN != 1 ]]; then
    exec $PROF_EXE $PROF_OPTS $KHARMA_DIR/$EXE_NAME -d "$OUTDIR" "$@" $KHARMA_PROF_OPTS
  fi
else
  echo "Running $MPI_EXE -n $MPI_NUM_PROCS $MPI_EXTRA_ARGS $KHARMA_DIR/$EXE_NAME $@"
  if [[ $DRYRUN != 1 ]]; then
    exec $MPI_EXE -n $MPI_NUM_PROCS $MPI_EXTRA_ARGS $KHARMA_DIR/$EXE_NAME -d "$OUTDIR" "$@"
  fi
fi
