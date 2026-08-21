#!/bin/bash
set -euo pipefail

# Bash script to run Hubble flow e- heating test

# Set paths
KHARMADIR=../..

exit_code=0

noh_test() {
    ALL_RES="512"
    for res in 512
    do
        eighth=$(($res / 8))
        $KHARMADIR/run.sh -d . -i $KHARMADIR/pars/electrons/hubble.par debug/verbose=1 parthenon/output0/dt=1000 \
                            parthenon/mesh/nx1=$res parthenon/meshblock/nx1=$eighth \
                            >log_hubble_${res}.txt 2>&1

        #cp hubble.out0.final.phdf hubble.out0.final.res$res.phdf
    done
    check_code=0
    python make_plots.py
    #python check.py . . $ALL_RES 1.666667 || check_code=$?
    #if [[ $check_code != 0 ]]; then
    #    echo Noh shock test FAIL: $check_code
    #    exit_code=1
    #else
    #    echo Noh shock test success
    #fi
}

noh_test

exit $exit_code
