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

conv_3d() {
    IFS=',' read -ra RES_LIST <<< "$ALL_RES"
    for res in "${RES_LIST[@]}"
    do
      # Eight blocks
      half=$(( $res / 2 ))
      $BASE/run.sh -d . -i $BASE/pars/tests/mhdmodes.par debug/verbose=2 mhdmodes/dir=0 \
                      parthenon/output0/single_precision_output=false parthenon/output0/dt=100. \
                      parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=$res \
                      parthenon/meshblock/nx1=$half parthenon/meshblock/nx2=$half parthenon/meshblock/nx3=$half \
                      $2 >log_3d_${1}_${res}.txt 2>&1
        mv mhdmodes.out0.00000.phdf mhd_3d_${1}_${res}_start.phdf
        mv mhdmodes.out0.final.phdf mhd_3d_${1}_${res}_end.phdf
    done
    check_code=0
    python3 check.py $ALL_RES "$3" $1 3d 3 || check_code=$?
    if [[ $check_code != 0 ]]; then
        echo MHD modes test \"$3\" FAIL: $check_code
        exit_code=1
    else
        echo MHD modes test \"$3\" success
    fi
}
conv_2d() {
    IFS=',' read -ra RES_LIST <<< "$ALL_RES"
    for res in "${RES_LIST[@]}"
    do
      # Four blocks
      half=$(( $res / 2 ))
      $BASE/run.sh -d . -i $BASE/pars/tests/mhdmodes.par debug/verbose=2 mhdmodes/dir=3 \
                      parthenon/output0/single_precision_output=false parthenon/output0/dt=100. \
                      parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=1 \
                      parthenon/meshblock/nx1=$half parthenon/meshblock/nx2=$half parthenon/meshblock/nx3=1 \
                      $2 >log_2d_${1}_${res}.txt 2>&1
        mv mhdmodes.out0.00000.phdf mhd_2d_${1}_${res}_start.phdf
        mv mhdmodes.out0.final.phdf mhd_2d_${1}_${res}_end.phdf
    done
    check_code=0
    python3 check.py $ALL_RES "$3" $1  2d || check_code=$?
    if [[ $check_code != 0 ]]; then
        echo MHD modes test \"$3\" FAIL: $check_code
        exit_code=1
    else
        echo MHD modes test \"$3\" success
    fi
}

ALL_RES="16,24,32,48,64"

# Normal MHD modes, 2D, defaults
conv_2d slow   mhdmodes/nmode=1 "slow mode in 2D"
conv_2d alfven mhdmodes/nmode=2 "Alfven mode in 2D"
conv_2d fast   mhdmodes/nmode=3 "fast mode in 2D"

# Entropy mode as reconstruction demo
# Most everything is run with the default weno5 reconstruction since that sees a lot of use;
# replace it in mhdmodes.par and re-run this script to evaluate how the other recons would converge!
conv_2d entropy_nob "mhdmodes/nmode=0 b_field/solver=none" "entropy mode in 2D, no B field"
# The resolutions we test are too low for these two
#conv_2d entropy_donor "mhdmodes/nmode=0 flux/reconstruction=donor_cell" "entropy mode in 2D, Donor Cell reconstruction"
#conv_2d entropy_vl "mhdmodes/nmode=0 flux/reconstruction=linear_vl" "entropy mode in 2D, linear/VL reconstruction"
conv_2d entropy_mc   "mhdmodes/nmode=0 flux/reconstruction=linear_mc" "entropy mode in 2D, linear/MC reconstruction"
conv_2d entropy_weno "mhdmodes/nmode=0 flux/reconstruction=weno5" "entropy mode in 2D, WENO reconstruction"
conv_2d entropy_weno_lin "mhdmodes/nmode=0 flux/reconstruction=weno5_linear" "entropy mode in 2D, WENO linearized reconstruction"
conv_2d entropy_ppm  "mhdmodes/nmode=0 flux/reconstruction=ppm" "entropy mode in 2D, PPM reconstruction"
# TODO this gives some absurd results of like 1e-16
#conv_2d entropy_ppmx "mhdmodes/nmode=0 flux/reconstruction=ppmx" "entropy mode in 2D, PPMX reconstruction"
conv_2d entropy_mp5  "mhdmodes/nmode=0 flux/reconstruction=mp5" "entropy mode in 2D, MP5 reconstruction"

# KHARMA driver
conv_2d slow_kharma   "mhdmodes/nmode=1 driver/type=kharma" "slow mode in 2D, KHARMA driver"
conv_2d alfven_kharma "mhdmodes/nmode=2 driver/type=kharma" "Alfven mode in 2D, KHARMA driver"
conv_2d fast_kharma   "mhdmodes/nmode=3 driver/type=kharma" "fast mode in 2D, KHARMA driver"
# HLLE flux.  This will probably only be used with kharma driver
# TODO weno5_linear doesn't seem to like HLLE very well for some reason?  Mildly unexpected convergence rates
conv_2d slow_hlle   "mhdmodes/nmode=1 driver/type=kharma b_field/solver=face_ct flux/type=hlle" "slow mode in 2D, HLLE flux"
conv_2d alfven_hlle "mhdmodes/nmode=2 driver/type=kharma b_field/solver=face_ct flux/type=hlle" "Alfven mode in 2D, HLLE flux"
conv_2d fast_hlle   "mhdmodes/nmode=3 driver/type=kharma b_field/solver=face_ct flux/type=hlle" "fast mode in 2D, HLLE flux"
# ImEx driver
conv_2d slow_imex   "mhdmodes/nmode=1 driver/type=imex" "slow mode in 2D, ImEx explicit"
conv_2d alfven_imex "mhdmodes/nmode=2 driver/type=imex" "Alfven mode in 2D, ImEx explicit"
conv_2d fast_imex   "mhdmodes/nmode=3 driver/type=imex" "fast mode in 2D, ImEx explicit"
# B field totally explicit
conv_2d slow_imex_semi   "mhdmodes/nmode=1 driver/type=imex GRMHD/implicit=true b_field/implicit=false" "slow mode 3D, ImEx semi-implicit"
conv_2d alfven_imex_semi "mhdmodes/nmode=2 driver/type=imex GRMHD/implicit=true b_field/implicit=false" "Alfven mode 3D, ImEx semi-implicit"
conv_2d fast_imex_semi   "mhdmodes/nmode=3 driver/type=imex GRMHD/implicit=true b_field/implicit=false" "fast mode 3D, ImEx semi-implicit"

# Face-CT using KHARMA driver
conv_2d slow_kharma_ct   "mhdmodes/nmode=1 driver/type=kharma b_field/solver=face_ct b_field/ct_scheme=bs99" "slow mode in 2D, face CT"
conv_2d alfven_kharma_ct "mhdmodes/nmode=2 driver/type=kharma b_field/solver=face_ct b_field/ct_scheme=bs99" "Alfven mode in 2D, face CT"
conv_2d fast_kharma_ct   "mhdmodes/nmode=3 driver/type=kharma b_field/solver=face_ct b_field/ct_scheme=bs99" "fast mode in 2D, face CT"
# Face-CT using ImEx driver
conv_2d slow_imex_ct   "mhdmodes/nmode=1 driver/type=imex b_field/solver=face_ct b_field/ct_scheme=bs99" "slow mode in 2D, ImEx explicit w/face CT"
conv_2d alfven_imex_ct "mhdmodes/nmode=2 driver/type=imex b_field/solver=face_ct b_field/ct_scheme=bs99" "Alfven mode in 2D, ImEx explicit w/face CT"
conv_2d fast_imex_ct   "mhdmodes/nmode=3 driver/type=imex b_field/solver=face_ct b_field/ct_scheme=bs99" "fast mode in 2D, ImEx explicit w/face CT"
# Fake upwinded fluxes
conv_2d slow_kharma_ct_gs05_0   "mhdmodes/nmode=1 driver/type=kharma b_field/solver=face_ct b_field/ct_scheme=gs05_0" "slow mode in 2D, KHARMA driver w/epsilon_0 flux"
conv_2d alfven_kharma_ct_gs05_0 "mhdmodes/nmode=2 driver/type=kharma b_field/solver=face_ct b_field/ct_scheme=gs05_0" "Alfven mode in 2D, KHARMA driver w/epsilon_0 flux"
conv_2d fast_kharma_ct_gs05_0   "mhdmodes/nmode=3 driver/type=kharma b_field/solver=face_ct b_field/ct_scheme=gs05_0" "fast mode in 2D, KHARMA driver w/epsilon_0 flux"
# Real upwinded fluxes (note this is now equivalent to sg07/gs07)
conv_2d slow_kharma_ct_gs05_c   "mhdmodes/nmode=1 driver/type=kharma b_field/solver=face_ct b_field/ct_scheme=gs05_c" "slow mode in 2D, KHARMA driver w/epsilon_c flux"
conv_2d alfven_kharma_ct_gs05_c "mhdmodes/nmode=2 driver/type=kharma b_field/solver=face_ct b_field/ct_scheme=gs05_c" "Alfven mode in 2D, KHARMA driver w/epsilon_c flux"
conv_2d fast_kharma_ct_gs05_c   "mhdmodes/nmode=3 driver/type=kharma b_field/solver=face_ct b_field/ct_scheme=gs05_c" "fast mode in 2D, KHARMA driver w/epsilon_c flux"

# Test old 1Dw primitive recovery
conv_2d slow_onedw   "mhdmodes/nmode=1 inverter/type=onedw driver/type=imex" "slow mode in 2D, 1Dw inversion"
conv_2d alfven_onedw "mhdmodes/nmode=2 inverter/type=onedw driver/type=imex" "Alfven mode in 2D, 1Dw inversion"
conv_2d fast_onedw   "mhdmodes/nmode=3 inverter/type=onedw driver/type=imex" "fast mode in 2D, 1Dw inversion"


# simple driver, high res
ALL_RES="16,24,32,48,64,96,128,192,256"
conv_2d slow_highres   "mhdmodes/nmode=1 driver/type=simple" "slow mode in 2D, high res"
conv_2d alfven_highres "mhdmodes/nmode=2 driver/type=simple" "Alfven mode in 2D, high res"
conv_2d fast_highres   "mhdmodes/nmode=3 driver/type=simple" "fast mode in 2D, high res"

# Trying for convergence down to 8 zone. This configuration should be able to go the lowest?
# ALL_RES="8,16,32,64"
# conv_2d slow_ln "mhdmodes/nmode=1 driver/type=kharma driver/flux=hlle driver/reconstruction=weno5 b_field/solver=face_ct" "slow mode in 2D"
# conv_2d alfven_ln "mhdmodes/nmode=2 driver/type=kharma driver/flux=hlle driver/reconstruction=weno5 b_field/solver=face_ct" "Alfven mode in 2D"
# conv_2d fast_ln "mhdmodes/nmode=3 driver/type=kharma driver/flux=hlle driver/reconstruction=weno5 b_field/solver=face_ct" "fast mode in 2D"

# Mimicry
#ALL_RES="16,24,32,48,64"
# TODO PPM and PPMX behave badly here, but would be more "authentic"
#OPTS="driver/type=kharma inverter/type=kastaun flux/type=hlle flux/reconstruction=weno5 b_field/solver=face_ct b_field/ct_scheme=gs05_c"
#conv_2d entropy_ak "mhdmodes/nmode=0 $OPTS" "entropy mode in 2D, AthenaK mode"
#conv_2d slow_ak   "mhdmodes/nmode=1 $OPTS" "slow mode in 2D, AthenaK mode"
#conv_2d alfven_ak "mhdmodes/nmode=2 $OPTS" "Alfven mode in 2D, AthenaK mode"
#conv_2d fast_ak   "mhdmodes/nmode=3 $OPTS" "fast mode in 2D, AthenaK mode"

#OPTS="$OPTS b_field/consistent_face_b=false"
#conv_2d slow_ak_rec   "mhdmodes/nmode=1 $OPTS" "slow mode in 2D, reconstructed B"
#conv_2d alfven_ak_rec "mhdmodes/nmode=2 $OPTS" "Alfven mode in 2D, reconstructed B"
#conv_2d fast_ak_rec   "mhdmodes/nmode=3 $OPTS" "fast mode in 2D, reconstructed B"

# TODO fully iharm3d mode?

# In 3D, test the default config only
ALL_RES="16,24,32"
conv_3d slow "mhdmodes/nmode=1 mhdmodes/dir=3" "slow mode in 3D"
conv_3d alfven "mhdmodes/nmode=2 mhdmodes/dir=3" "Alfven mode in 3D"
conv_3d fast "mhdmodes/nmode=3 mhdmodes/dir=3" "fast mode in 3D"

exit $exit_code
