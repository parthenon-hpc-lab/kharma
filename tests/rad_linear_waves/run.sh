#!/bin/bash
set -euo pipefail

KHARMADIR=../..

exit_code=0

WAVETYPES="sonic slow fast"
REGIMES="puremhd thin thick"
RESOLUTIONS="64 128 256 512"

declare -A REGIME_LABEL
REGIME_LABEL[puremhd]=pure
REGIME_LABEL[thin]=thin
REGIME_LABEL[thick]=thick

for regime in $REGIMES
do
    rlabel=${REGIME_LABEL[$regime]}
    for wavetype in $WAVETYPES
    do
        outdir=${rlabel}_${wavetype}
        mkdir -p $outdir
        for res in $RESOLUTIONS
        do
            $KHARMADIR/run.sh -i $KHARMADIR/pars/radM1/radmhdmodes/radmhdmodes.par \
                                mhdmodes/wavetype=$wavetype mhdmodes/regime=$regime \
                                parthenon/mesh/nx1=$res parthenon/meshblock/nx1=$((res / 2)) \
                                >${outdir}/log_${res}.txt 2>&1

            cp dumps_kharma/radmhdmodes.out0.final.phdf ${outdir}/mhd_modes_${res}_${rlabel}_${wavetype}.phdf
            rm ./dumps_kharma/*
        done
    done
done

check_code=0
python3 check.py . . || check_code=$?
if [[ $check_code != 0 ]]; then
    echo "Radiation-modified MHD linear waves FAIL: $check_code"
    exit_code=1
else
    echo "Radiation-modified MHD linear waves success"
fi

exit $exit_code
