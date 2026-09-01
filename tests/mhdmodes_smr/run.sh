#!/bin/bash
set -euo pipefail

BASE=../..

# This test confirms that all of the many transport options in KHARMA
# can converge when modeling each of the basic linearized modes:
# slow, fast, and alfven waves

# It tests:
# 1. different reconstructions WENO vs linear
# 2. different drivers, simple, KHARMA, & ImEx
# 3. different B field transports, Flux-CT and Face-CT

exit_code=0

conv_2d() {
    IFS=',' read -ra RES_LIST <<< "$ALL_RES"
    for res in "${RES_LIST[@]}"
    do
      # 3x3 & refine center
      block=$(($res / 3))
      $BASE/run.sh -d . -i $BASE/pars/smr/mhdmodes_refined.par debug/verbose=2 mhdmodes/dir=3 \
                      parthenon/output0/single_precision_output=false parthenon/output0/dt=100. \
                      parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=1 \
                      parthenon/meshblock/nx1=$block parthenon/meshblock/nx2=$block parthenon/meshblock/nx3=1 \
                      $2 >log_2d_${1}_${res}.txt 2>&1
        mv mhdmodes.out0.00000.phdf mhd_2d_${1}_${res}_start.phdf
        mv mhdmodes.out0.final.phdf mhd_2d_${1}_${res}_end.phdf
    done
    check_code=0
    python check.py $ALL_RES "$3" $1  2d || check_code=$?
    if [[ $check_code != 0 ]]; then
        echo MHD modes test \"$3\" FAIL: $check_code
        exit_code=1
    else
        echo MHD modes test \"$3\" success
    fi
}

# Normal MHD modes, 2D, defaults
ALL_RES="24,48,96,192"
conv_2d slow mhdmodes/nmode=1 "slow mode in 2D"
conv_2d alfven mhdmodes/nmode=2 "Alfven mode in 2D"
conv_2d fast mhdmodes/nmode=3 "fast mode in 2D"

exit $exit_code
