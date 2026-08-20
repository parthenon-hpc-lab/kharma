#!/bin/bash
set -euo pipefail

exit_code=0

tilt_init() {
    # Run default tilted problem to 5 steps
    ../../run.sh -d . -i ../../pars/tori_3d/mad.par parthenon/time/nlim=5 debug/verbose=1 \
                    parthenon/mesh/nx1=128 parthenon/mesh/nx2=64 parthenon/mesh/nx3=64 \
                    parthenon/meshblock/nx1=128 parthenon/meshblock/nx2=32 parthenon/meshblock/nx3=64 \
                    parthenon/job/archive_parameters=false \
                    parthenon/output0/single_precision_output=false \
                    parthenon/output0/variables=prims,jcon,fflag,pflag,divB \
                    boundaries/inner_x2=reflecting boundaries/outer_x2=reflecting \
                    $2 >log_tilt_init_${1}.txt 2>&1

    # Image the first dump
    python ./check.py

    check_code=0
    # Check some basics (divB) of the first dump
    pyharm check-basics --allowed-divb=${4} torus.out0.final.phdf || check_code=$?

    if [[ $check_code != 0 ]]; then
        echo Tilt init test \"$3\" FAIL: $check_code
        exit_code=1
    else
        echo Tilt init test \"$3\" success
    fi
}

tilt_init cell "b_field/solver=flux_ct torus/tilt=30" "Cell-centered B" "1e-9"
tilt_init face "b_field/solver=face_ct torus/tilt=30" "Face-centered B" "1e-9"

exit $exit_code
