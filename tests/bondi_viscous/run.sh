#!/bin/bash
set -euo pipefail

BASE=../..

exit_code=0

# Viscous bondi inflow convergence to exercise all terms in the evolution equation of dP

conv_2d() {
    IFS=',' read -ra RES_LIST <<< "$ALL_RES"
    for res in "${RES_LIST[@]}"
    do
        # Four blocks
        half=$(( $res / 2 ))
        $BASE/run.sh -d . -i $BASE/pars/emhd/bondi_viscous.par debug/verbose=1 \
            parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=1 \
            parthenon/meshblock/nx1=$half parthenon/meshblock/nx2=$half parthenon/meshblock/nx3=1 \
            b_field/implicit=false $2 >log_${1}_${res}.txt 2>&1

        mv bondi.out0.00000.phdf emhd_2d_${res}_start_${1}.phdf
        mv bondi.out0.final.phdf emhd_2d_${res}_end_${1}.phdf
    done
    check_code=0
    python3 check.py $ALL_RES $1 2d || check_code=$?
    rm -r *.out0*
    if [[ $check_code != 0 ]]; then
            echo Viscous Bondi test $3 FAIL: $check_code
            exit_code=1
    else
            echo Viscous Bondi test $3 success
    fi
}

ALL_RES="8,16,32,64"
conv_2d emhd2d_weno driver/reconstruction=weno5 "Viscous Bondi in 2D, WENO5"
# Test if it works with ideal solution as guess
conv_2d emhd2d_weno_ideal_guess emhd/ideal_guess=true "Viscous bondi in 2D, WENO5, Ideal guess"

exit $exit_code
