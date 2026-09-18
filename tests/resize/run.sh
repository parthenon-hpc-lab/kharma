#!/bin/bash
set -euo pipefail

# Bash script testing starting a simulation, then resizing it up
# TODO store return codes?  Don't think it helps much having both results if cell failed...

# Set paths
KHARMADIR=../..

test_resize () {
    # This at least stirs up the field slightly vs initialization
    $KHARMADIR/run.sh -d . -i $KHARMADIR/pars/tori_3d/sane.par \
                    b_field/solver=face_ct parthenon/time/nlim=5 $2 >log_resize_${1}_1.txt 2>&1

    # We can only resize/restart from iharm3d-format files
    pyharm convert --to-restart torus.out0.final.phdf

    sleep 1

    # Tolerance is generous to make test short, while testing solver is at least approaching correct vals
    # Real simulation resizes should use tighter tolerance than this
    $KHARMADIR/run.sh -d . -i $KHARMADIR/pars/restarts/resize_restart.par $2 resize_restart/fname=torus.out0.final.h5 \
                    b_cleanup/tolerance=1e-10 b_cleanup/no_clean_below=0 parthenon/time/nlim=1 \
                    parthenon/output0/single_precision_output=false >log_resize_${1}_2.txt 2>&1

    # Check divB on the re-meshed output.  Tolerate some divB as we set the tolerance loosely above for speed
    pyharm check-basics --allowed-divb=1e-9 resize_restart.out0.final.phdf
}
test_resize_smr () {
    # This at least stirs up the field slightly vs initialization
    $KHARMADIR/run.sh -d . -i $KHARMADIR/pars/smr/sane2d_refined.par parthenon/time/nlim=5 $2 >log_resize_${1}_1.txt 2>&1

    # We can only resize/restart from iharm3d-format files
    pyharm convert --to-restart torus.out0.final.phdf

    sleep 1

    # Tolerance is generous to make test short, while testing solver is at least approaching correct vals
    # Real simulation resizes should use tighter abs_tolerance than this
    $KHARMADIR/run.sh -d . -i $KHARMADIR/pars/restarts/resize_restart_smr.par $2 resize_restart/fname=torus.out0.final.h5 \
                    b_cleanup/tolerance=1e-10 b_cleanup/no_clean_below=0 parthenon/time/nlim=1 \
                    parthenon/output0/single_precision_output=false >log_resize_${1}_2.txt 2>&1

    # Check divB on the re-meshed output.  Tolerate some divB as we set the tolerance loosel>
    pyharm check-basics --allowed-divb=1e-9 resize_restart.out0.final.phdf
}

# These use Parthenon's new GMG solver
test_resize face ""
# SMR not working due to NaN face B out of DangerousPtoU
#test_resize_smr smr ""
