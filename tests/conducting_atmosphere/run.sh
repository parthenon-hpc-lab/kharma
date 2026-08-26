#!/bin/bash
set -euo pipefail

BASE=../..

exit_code=0

# Extended MHD atmosphere test convergence to exercise geometrical terms
# We'll use just 1 MPI rank to circumvent the somewhat annoying ODE initialization

conv_2d() {
    IFS=',' read -ra RES_LIST <<< "$ALL_RES"
    for res in "${RES_LIST[@]}"
    do
        cp conducting_atmosphere_${res}_default/atmosphere_soln_*.txt .
        $BASE/run.sh -d . -i ./conducting_atmosphere.par debug/verbose=1 \
            parthenon/time/tlim=200 parthenon/output0/dt=1000000 \
            parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=1 \
            parthenon/meshblock/nx1=$res parthenon/meshblock/nx2=$res parthenon/meshblock/nx3=1 \
            $2 >log_${1}_${res}.txt 2>&1

        mv conducting_atmosphere.out0.00000.phdf emhd_2d_${res}_start_${1}.phdf
        mv conducting_atmosphere.out0.final.phdf emhd_2d_${res}_end_${1}.phdf
        rm atmosphere_soln_*.txt
    done
    check_code=0
    pyharm-convert --double *.phdf
    python3 check.py $ALL_RES $1 2d || check_code=$?
    if [[ $check_code != 0 ]]; then
        echo Conducting atmosphere test $3 FAIL: $check_code
        exit_code=1
    else
        echo Conducting atmosphere test $3 success
    fi
}

ALL_RES="64,128,256,512"
conv_2d emhd2d_weno driver/reconstruction=weno5 "in 2D, WENO5"
# Test if it works with ideal solution as guess
conv_2d emhd2d_weno_ideal_guess emhd/ideal_guess=true "in 2D, WENO5, Ideal guess"

exit $exit_code
