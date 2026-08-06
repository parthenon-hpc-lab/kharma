#!/bin/bash
set -euo pipefail

# Bash script to run the radiative M1 shock tube test 4 of the PLUTO M1
# module paper (Melon Fuksman & Mignone 2019. Currently checks
# convergence of the L1 norm against a Chebyshev-polynomial fit of a former dump that was matching the paper.
# This is done because the analytic solution in Farris et al (2008) uses the Eddington approximation, instead of the M1 closure, therefore not valid.

KHARMADIR=../..
PARFILE=$KHARMADIR/pars/radM1/shocktube_pluto/shocktube4.par

exit_code=0

rad_shocktube_test() {
    ALL_RES="100,200,400,800"
    for res in 100 200 400 800
    do
        $KHARMADIR/run.sh -i $PARFILE debug/verbose=1 parthenon/output0/dt=500 \
                            parthenon/mesh/nx1=$res parthenon/meshblock/nx1=$res \
                            >log_radshock_${res}.txt 2>&1

        cp shock.out0.final.phdf shock.out0.final.res$res.phdf
    done

    check_code=0
    python3 check.py . . $ALL_RES || check_code=$?
    if [[ $check_code != 0 ]]; then
        echo Radiative shock tube test FAIL: $check_code
        exit_code=1
    else
        echo Radiative shock tube test success
    fi
}

rad_shocktube_test

exit $exit_code
