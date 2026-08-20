#!/bin/bash
set -euo pipefail

BASE=../..
exit_code=0

check_sanity() {
    # mad_test.par is basically only used for this, so common options are there.
    $BASE/run.sh -d . -i ./mad_test.par $2 >log_divb_${1}.txt 2>&1 || exit_code=$?

    pyharm check-basics -d --allowed-divb=1e-10 torus.out0.final.phdf || exit_code=$?
}

check_sanity imex driver/type=imex
check_sanity harm driver/type=harm

exit $exit_code
