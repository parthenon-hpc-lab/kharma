#!/bin/bash
set -euo pipefail

# Advected entropy wave (2D): convergence test for the Entropy package's Ktot_adv.
#
# The point of this test is that the entropy is *non-uniform*.  Bondi, Noh and Hubble all
# have spatially constant K, and a uniform passive scalar is advected exactly by any
# scheme whose scalar flux is proportional to the mass flux -- so none of them exercise
# the Ktot_adv flux term at all.  Here K = p/rho^gam varies by a factor ~6 across the
# grid and is carried diagonally across it for a full period.
#
# Ktot is checked here by an exact identity rather than a convergence order (it is reset
# from rho,u every sub-step, so its accuracy is just rho and u's).  See check.py.
#
# Each run is NxN with NxN/4 meshblocks, so the block-to-block ghost exchange of the
# entropy variables is exercised too.
#
# check.py writes entropy_wave_convergence.png.  For maps of rho and Ktot_adv at the
# first and last dump, run ./make_plots.py afterwards.

KHARMADIR=../..

ALL_RES="64,128,256,512"
exit_code=0

entropy_wave_test() {
    for res in ${ALL_RES//,/ }
    do
        half=$(($res / 2))
        $KHARMADIR/run.sh -d . -i $KHARMADIR/pars/tests/entropy_wave.par debug/verbose=1 \
                            parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res \
                            parthenon/meshblock/nx1=$half parthenon/meshblock/nx2=$half \
                            >log_entropy_wave_${res}.txt 2>&1

        mv entropy_wave.out0.00000.phdf entropy_wave.init.res${res}.phdf
        mv entropy_wave.out0.final.phdf entropy_wave.res${res}.phdf
        rm -f entropy_wave.out0.0*.phdf
    done

    check_code=0
    python3 check.py $ALL_RES || check_code=$?
    if [[ $check_code != 0 ]]; then
        echo Entropy wave test FAIL: $check_code
        exit_code=1
    else
        echo Entropy wave test success
    fi
}

entropy_wave_test

exit $exit_code
