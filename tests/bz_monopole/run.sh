#!/bin/bash
set -euo pipefail

BASE=../..

exit_code=0

# Full run to test stability to completion
$BASE/run.sh -d . -i ./bz_monopole.par debug/verbose=1 >log_bz_monopole_full.txt 2>&1 || exit_code=$?

# At *least* check divB
pyharm-check-basics bz_monopole.out0.final.phdf || exit_code=$?

# Take 1 step to look for early signs of non-fatal instabilities
$BASE/run.sh -d . -i ./bz_monopole.par parthenon/time/nlim=1 parthenon/output0/dt=0.0 >log_bz_monopole_step.txt 2>&1 || exit_code=$?

# This just makes plots, it doesn't check anything
python ./check.py

exit $exit_code
