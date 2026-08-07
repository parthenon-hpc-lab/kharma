#!/bin/bash
set -euo pipefail

# Bash script to run all four PLUTO-paper radiative M1 shock tube tests
# (Melon Fuksman & Mignone 2019, 
# check convergence of the L1 norm against a frozen 3200-zone
# reference solution for each test.

KHARMADIR=../..

exit_code=0

rad_shocktube_test() {
    local test_num=$1
    local parfile=$KHARMADIR/pars/radM1/shocktube_pluto/shocktube${test_num}.par
    local all_res="100,200,400,800"

    for res in 100 200 400 800
    do
        $KHARMADIR/run.sh -i $parfile debug/verbose=1 parthenon/output0/dt=1000 \
                            parthenon/mesh/nx1=$res parthenon/meshblock/nx1=$res \
                            >log_radshock_test${test_num}_${res}.txt 2>&1

        cp shock.out0.final.phdf shock_test${test_num}.out0.final.res${res}.phdf
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

for test_num in 1 2 3 4
do
    rad_shocktube_test $test_num
done

exit $exit_code
