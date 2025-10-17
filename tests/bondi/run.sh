#!/bin/bash
set -euo pipefail

BASE=../..

exit_code=0

conv_2d() {
    IFS=',' read -ra RES_LIST <<< "$ALL_RES"
    for res in "${RES_LIST[@]}"
    do
      # Four blocks
      half=$(( $res / 2 ))
      $BASE/run.sh -i $BASE/pars/bondi/bondi.par debug/verbose=1 debug/flag_verbose=2 parthenon/time/tlim=50 \
                                           parthenon/output0/dt=1000 parthenon/output0/single_precision_output=false \
                                           parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=1 \
                                           parthenon/meshblock/nx1=$half parthenon/meshblock/nx2=$half parthenon/meshblock/nx3=1 \
                                           $2 >log_${1}_${res}.txt 2>&1 || check_code=$?
        mv bondi.out0.00000.phdf bondi_2d_${res}_start_${1}.phdf
        mv bondi.out0.final.phdf bondi_2d_${res}_end_${1}.phdf
    done
    check_code=0
    python check.py $ALL_RES "$3" $1 || check_code=$?
    if [[ $check_code != 0 ]]; then
        echo Bondi test $3 FAIL: $check_code
        exit_code=1
    else
        echo Bondi test $3 success
    fi
}

# Test boundaries
ALL_RES="16,24,32,48,64"
conv_2d base " " "in 2D, baseline"
conv_2d onedw "inverter/type=onedw" "in 2D, 1Dw inverter"

conv_2d dirichlet      "boundaries/inner_x1=dirichlet boundaries/outer_x1=dirichlet" "in 2D, Dirichlet boundaries"
conv_2d dirichlet_imex "boundaries/inner_x1=dirichlet boundaries/outer_x1=dirichlet driver/type=imex" "in 2D, Dirichlet boundaries+ImEx Driver"

# Test coordinates
conv_2d eks "coordinates/transform=eks" "in 2D, EKS coordinates"
# Some coordinate systems do better/worse than 2o at low res
ALL_RES="48,64,96,128"
conv_2d fmks "coordinates/transform=fmks" "in 2D, FMKS coordinates"
conv_2d ks   "coordinates/transform=null" "in 2D, KS coordinates"

# Recon
ALL_RES="16,24,32,48,64"
conv_2d linear_mc GRMHD/reconstruction=linear_mc "in 2D, linear recon with MC limiter"
conv_2d linear_vl GRMHD/reconstruction=linear_vl "in 2D, linear recon with VL limiter"

# And the GRIM/classic driver
conv_2d imex    driver/type=imex "in 2D, with Imex driver"
conv_2d imex_im "driver/type=imex GRMHD/implicit=true" "in 2D, semi-implicit stepping"

ALL_RES="16,24,32,48,64"
conv_2d b_flux_ct "b_field/type=monopole_cube b_field/B10=1 b_field/solver=flux_ct" "in 2D, monopole B, Flux-CT"
conv_2d b_face_ct "b_field/type=monopole_cube b_field/B10=1 b_field/solver=face_ct" "in 2D, monopole B, Face-CT"
#conv_2d b_face_ct "b_field/type=monopole_cube b_field/B10=1 b_field/solver=face_ct b_field/consistent_face_b=false" "in 2D, monopole B, Face-CT reconstructed"

ALL_RES="24,32,48,64" # TODO idk why this doesn't work at 16^2
conv_2d b_face_ct_dirichlet "boundaries/inner_x1=dirichlet boundaries/outer_x1=dirichlet b_field/type=monopole_cube b_field/B10=1 b_field/solver=face_ct" "in 2D, monopole B, face-centered+Dirichlet"

# TODO 3D?

exit $exit_code
