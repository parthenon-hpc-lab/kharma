#!/bin/bash
set -euo pipefail

BASE=../..

exit_code=0

# Extended MHD atmosphere test convergence to exercise geometrical terms
# We use many ranks to make this test faster on CPU,
# But testing GPU w/MPI on our poor CI machine is out of scope
if [[ $MPI_NUM_PROCS > 1 ]]; then
  export MPI_NUM_PROCS=8
fi

conv_2d() {
    IFS=',' read -ra RES_LIST <<< "$ALL_RES"
    for res in "${RES_LIST[@]}"
    do
        echo Running conducting atmosphere test $3
        # Run this with more cores, we can usually spare them and it's looooooong
        # Must still be one block in r!
        eighth=$(( $res / 8 ))
        cp conducting_atmosphere_${res}_default/atmosphere_soln_*.txt .
        $BASE/run.sh -d . -i ./conducting_atmosphere.par debug/verbose=1 \
            parthenon/time/tlim=200 parthenon/output0/dt=1000000 \
            parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=1 \
            parthenon/meshblock/nx1=$res parthenon/meshblock/nx2=$eighth parthenon/meshblock/nx3=1 \
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

# TODO(CEP) remove 512? Remember can't go below 64 w/8 ranks
ALL_RES="64,128,256,512"
conv_2d emhd2d_weno driver/reconstruction=weno5 "in 2D, WENO5"
# Test if it works with ideal solution as guess
conv_2d emhd2d_weno_ideal_guess emhd/ideal_guess=true "in 2D, WENO5, Ideal guess"

exit $exit_code
