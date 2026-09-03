#!/bin/bash
set -euo pipefail

# Bash script to run Hubble flow e- heating test
# Ressler+ 2015 sec. 4.1.1.
#
# Two cases, both of which should converge at 2nd order:
#   cooling:  the actual test: an explicit heating term is added to the energy
#                equation, and the analytic solution is only maintained if that source
#                is time-centred consistently with the integrator
#   nocooling: no source term at all, so the flow is adiabatic (u ~ rho^gam, electron
#                entropy constant).  This isolates transport and the boundaries, so if
#                both cases fail the problem is not in the heating.

# Set paths
KHARMADIR=../..

ALL_RES="64,128,256,512"
exit_code=0

# $1: tag used to name dumps & logs, remaining args: extra parameters
run_convergence() {
    tag=$1; shift
    for res in ${ALL_RES//,/ }
    do
        $KHARMADIR/run.sh -d . -i $KHARMADIR/pars/electrons/hubble.par debug/verbose=1 \
                            parthenon/mesh/nx1=$res parthenon/meshblock/nx1=$res \
                            parthenon/output0/dt=1e5 "$@" \
                            >log_hubble_${tag}_${res}.txt 2>&1

        mv hubble.out0.final.phdf hubble_${tag}.res${res}.phdf
        rm -f hubble.out0.0*.phdf
    done
}

hubble_test() {
    run_convergence cooling   hubble/cooling=true
    run_convergence nocooling hubble/cooling=false

    check_code=0
    python3 check.py $ALL_RES cooling nocooling || check_code=$?
    if [[ $check_code != 0 ]]; then
        echo Hubble flow test FAIL: $check_code
        exit_code=1
    else
        echo Hubble flow test success
    fi
}

hubble_test

exit $exit_code
