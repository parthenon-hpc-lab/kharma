#!/bin/bash
set -euo pipefail

# Bash script testing a fresh Orszag-Tang vortex vs a version
# re-gridded to 64^2 tiles in the middle of the run,
# and then a version resized to twice the resolution

# TODO the first comparison should really be binary-identical

exit_code=0

# Set paths
KHARMADIR=../..

$KHARMADIR/run.sh -d . -i ./orszag_tang_with_restarts.par >log_orig.txt 2>&1

mv orszag_tang.out0.final.phdf orszag_tang.out0.final.orig.phdf

sleep 1

pyharm-convert --to-restart orszag_tang.out1.00005.rhdf orszag_tang.out1.00009.rhdf

sleep 1

$KHARMADIR/run.sh -d . -i ./regrid_orszag_tang.par >log_regrid.txt 2>&1

mv resize_restart.out0.final.phdf resize_restart.out0.final.regrid.phdf

check_code=0
pyharm-diff orszag_tang.out0.final.orig.phdf resize_restart.out0.final.regrid.phdf --no_plot --rel_tol=0.002 || check_code=$?
if [[ $check_code != 0 ]]; then
    echo Regrid test FAIL: $check_code
    exit_code=1
else
    echo Regrid test success
fi

# TODO(CEP) restore when solvers are back
# Finally, test that we can sanely resize the dump, too
# This won't output .phdf files, only restarts (.rhdf)
#$KHARMADIR/run.sh -d . -i ./resize_orszag_tang.par >log_resize.txt 2>&1

# Check the final .rhdf file for sanity (i.e., divB small)
#check_code=0
#pyharm-check-basics resize_restart.out1.final.rhdf || check_code=$?
#if [[ $check_code != 0 ]]; then
#    echo Resize test FAIL: $check_code
#    exit_code=1
#else
#    echo Resize test success
#fi

exit $exit_code
