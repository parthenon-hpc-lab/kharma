#!/bin/bash
set -euo pipefail

# Bash script testing initialization vs restart of a torus problem
# Require similarity to round-off after 5 steps

# TODO figure out why I need the following.  Sure smells like a Parthenon bug
export MPI_NUM_PROCS=1

# Set paths
KHARMADIR=../..

exit_code=0

test_restart() {
    $KHARMADIR/run.sh -i $KHARMADIR/pars/tori_3d/sane.par parthenon/time/nlim=5 driver/two_sync=true \
                         parthenon/job/archive_parameters=false \
                         parthenon/mesh/nx1=128 parthenon/mesh/nx2=64 parthenon/mesh/nx3=64 \
                         parthenon/meshblock/nx1=128 parthenon/meshblock/nx2=32 parthenon/meshblock/nx3=64 \
                         parthenon/output0/single_precision_output=false \
                         $2 >log_restart_${1}_first.txt 2>&1

    mv torus.out0.final.phdf restart_${1}_first.phdf

    sleep 1

    $KHARMADIR/run.sh -r torus.out1.00000.rhdf >log_restart_${1}_second.txt 2>&1

    mv torus.out0.final.phdf restart_${1}_second.phdf

    check_code=0
    # Compare to some high degree of accuracy
    pyharm diff --rel_tol 1e-9 restart_${1}_first.phdf restart_${1}_second.phdf --no_plot || check_code=$?
    # Compare binary. For someday
    #h5diff --exclude-path=/Info --exclude-path=/Input --exclude-path=/divB \
    #       --relative=1e-5 \
    #       restart_${1}_first.phdf restart_${1}_second.phdf || check_code=$?
    if [[ $check_code != 0 ]]; then
        echo Restart test \"$3\" FAIL: $check_code
        exit_code=1
    else
        echo Restart test \"$3\" success
    fi
}
test_restart_phdf() {
    $KHARMADIR/run.sh -i $KHARMADIR/pars/tori_3d/sane.par parthenon/time/nlim=5 driver/two_sync=true \
                         parthenon/job/archive_parameters=false \
                         parthenon/mesh/nx1=128 parthenon/mesh/nx2=64 parthenon/mesh/nx3=64 \
                         parthenon/meshblock/nx1=128 parthenon/meshblock/nx2=32 parthenon/meshblock/nx3=64 \
                         parthenon/output0/single_precision_output=false \
                         $2 >log_restart_${1}_first.txt 2>&1

    mv torus.out0.final.phdf restart_${1}_first.phdf

    sleep 1

    $KHARMADIR/run.sh -r torus.out0.00000.phdf b_field/restart_from_prims=true b_field/initial_cleanup=true >log_restart_${1}_second.txt 2>&1

    mv torus.out0.final.phdf restart_${1}_second.phdf

    check_code=0
    # Only check basics for now
    #pyharm diff --rel_tol 1e-3 restart_${1}_first.phdf restart_${1}_second.phdf --no_plot || check_code=$?
    pyharm check-basics restart_${1}_second.phdf || check_code=$?

    if [[ $check_code != 0 ]]; then
        echo Restart from dump test \"$3\" FAIL: $check_code
        exit_code=1
    else
        echo Restart from dump test \"$3\" success
    fi
}
test_restart_smr() {
    $KHARMADIR/run.sh -i $KHARMADIR/pars/smr/sane2d_refined.par parthenon/time/nlim=5 \
                         parthenon/job/archive_parameters=false \
                         driver/two_sync=true parthenon/output0/single_precision_output=false \
                         $2 >log_restart_${1}_first.txt 2>&1

    mv torus.out0.final.phdf restart_${1}_first.phdf

    sleep 1

    $KHARMADIR/run.sh -r torus.out1.00000.rhdf >log_restart_${1}_second.txt 2>&1

    mv torus.out0.final.phdf restart_${1}_second.phdf

    check_code=0
    # Compare to some high degree of accuracy
    pyharm diff --rel_tol 1e-9 restart_${1}_first.phdf restart_${1}_second.phdf --no_plot || check_code=$?
    # Compare binary. For someday
    #h5diff --exclude-path=/Info --exclude-path=/Input --exclude-path=/divB \
    #       --relative=1e-5 \
    #       restart_${1}_first.phdf restart_${1}_second.phdf || check_code=$?
    if [[ $check_code != 0 ]]; then
        echo Restart test \"$3\" FAIL: $check_code
        exit_code=1
    else
        echo Restart test \"$3\" success
    fi
}

test_restart kharma "driver/type=kharma b_field/solver=flux_ct" "KHARMA driver"
test_restart imex "driver/type=imex b_field/solver=flux_ct" "ImEx driver"
#test_restart imex_emhd "driver/type=imex emhd/on=true" "ImEx driver, EMHD"
test_restart kharma_face "driver/type=kharma b_field/solver=face_ct" "KHARMA driver, face CT"
test_restart imex_face "driver/type=imex b_field/solver=face_ct" "ImEx driver, face CT"
TWO_D="parthenon/mesh/nx3=1 parthenon/meshblock/nx3=1"
REFLECTING="boundaries/inner_x2=reflecting boundaries/outer_x2=reflecting boundaries/excise_polar_flux=false"
test_restart kharma_face_2d "driver/type=kharma b_field/solver=face_ct $TWO_D $REFLECTING" "KHARMA driver, face CT, 2D"
test_restart imex_face_2d   "driver/type=imex b_field/solver=face_ct $TWO_D $REFLECTING" "ImEx driver, face CT, 2D"
# SMR
test_restart_smr kharma_face_smr "driver/type=kharma b_field/solver=face_ct" "KHARMA driver, face CT, SMR"
test_restart_smr imex_face_smr "driver/type=imex b_field/solver=face_ct" "ImEx driver, face CT, SMR"
# phdf
test_restart_phdf kharma_face_phdf "driver/type=kharma b_field/solver=face_ct" "KHARMA driver from normal dump, face CT"

exit $exit_code
