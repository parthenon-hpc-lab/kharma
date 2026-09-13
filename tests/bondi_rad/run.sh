#!/bin/bash
set -euo pipefail

KHARMADIR=../..

exit_code=0

rad_shocktube_test() {
    local test_num=$1
    local parfile=$KHARMADIR/pars/radM1/bondi_rad/bondi_rad${test_num}.par
    local all_res="32,64,128,256"

    for res in 32 64 128 256
    do
        $KHARMADIR/run.sh -i $parfile debug/verbose=1 parthenon/output0/dt=25000 \
                            parthenon/mesh/nx1=$res parthenon/meshblock/nx1=$((res / 2)) \
                            >shock_test${test_num}_${res}.log 2>&1

        cp dumps_kharma/bondi_rad.out0.final.phdf bondi_rad${test_num}.out0.final.res${res}.phdf
        rm ./dumps_kharma/*
    done

    check_code=0
    python3 check.py . . $test_num $all_res || check_code=$?
    if [[ $check_code != 0 ]]; then
        echo "Radiative shock tube test $test_num FAIL: $check_code"
        exit_code=1
    else
        echo "Radiative shock tube test $test_num success"
    fi
}

for test_num in 1 2 3 4 5
do
    rad_shocktube_test $test_num
done

exit $exit_code
