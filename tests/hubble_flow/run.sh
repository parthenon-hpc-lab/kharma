#!/bin/bash
set -euo pipefail

# Bash script to run Hubble flow e- heating test
# Ressler+ 2015 sec. 4.1.1: the explicit heating term must be time-centred, so
# everything here should converge at 2nd order.

# Set paths
KHARMADIR=../..

exit_code=0

hubble_test() {
    ALL_RES="64,128,256,512"
    for res in 64 128 256 512
    do
        $KHARMADIR/run.sh -d . -i $KHARMADIR/pars/electrons/hubble.par debug/verbose=1 \
                            parthenon/mesh/nx1=$res parthenon/meshblock/nx1=$res \
                            parthenon/output0/dt=1e5 \
                            >log_hubble_${res}.txt 2>&1

        mv hubble.out0.final.phdf hubble.out0.final.res$res.phdf
    done

    check_code=0
    python3 check.py ${ALL_RES//,/ } || check_code=$?
    if [[ $check_code != 0 ]]; then
        echo Hubble flow test FAIL: $check_code
        exit_code=1
    else
        echo Hubble flow test success
    fi
}

hubble_test

exit $exit_code
