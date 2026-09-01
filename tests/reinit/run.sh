#!/bin/bash
set -euo pipefail

# Bash script testing determinism of problem initialization and first steps

# TODO figure out why I need the following.  Sure smells like a Parthenon bug
#export MPI_NUM_PROCS=1

# Set paths
KHARMADIR=../..

echo "First run"
$KHARMADIR/run.sh -d . -i $KHARMADIR/pars/tori_3d/sane.par perturbation/u_jitter=0 driver/two_sync=true \
                     parthenon/time/nlim=5 parthenon/job/archive_parameters=false \
                     parthenon/mesh/nx1=128 parthenon/mesh/nx2=64 parthenon/mesh/nx3=64 \
                     parthenon/meshblock/nx1=64 parthenon/meshblock/nx2=32 parthenon/meshblock/nx3=64 \
                     >log_reinit_1.txt 2>&1

mv torus.out1.final.rhdf torus.out1.final.first.rhdf

echo "Second run"
$KHARMADIR/run.sh -d . -i $KHARMADIR/pars/tori_3d/sane.par perturbation/u_jitter=0 driver/two_sync=true \
                     parthenon/time/nlim=5 parthenon/job/archive_parameters=false \
                     parthenon/mesh/nx1=128 parthenon/mesh/nx2=64 parthenon/mesh/nx3=64 \
                     parthenon/meshblock/nx1=64 parthenon/meshblock/nx2=32 parthenon/meshblock/nx3=64 \
                     >log_reinit_2.txt 2>&1

mv torus.out1.final.rhdf torus.out1.final.second.rhdf

echo "Checking..."
# This one's a clear case.  Binary or bust, even the input params
# /Info includes walltime, which obvs can change
h5diff --exclude-path=/Info torus.out1.final.first.rhdf torus.out1.final.second.rhdf
# And that's the exit code.  One and done.
