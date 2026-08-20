#!/bin/bash
set -euo pipefail

return_code=0

if [ -f ../../kharma.host ]; then
  FOLDERS="bondi electrons emhd shocks smr tests tori_2d tori_3d"
else
  # driven_turbulence problem needs FFTW, which is not on GPU
  # Also Noh shock requests too much shmem for some reason
  # 3D Tori take up too much memory for one little test GPU
  FOLDERS="bondi emhd shocks smr tests tori_2d"
fi

# Skip testing the restarting & benchmark scripts
for folder in $FOLDERS
do
  for fil in ../../pars/$folder/*.par
  do
    exit_code=0
    par=$(basename $fil)
    prob=${par%.*}
    ../../run.sh -d . -i $fil parthenon/time/nlim=2 parthenon/job/archive_parameters=false &>log_${prob}.txt || exit_code=$?
    rm -f *.{hst,phdf,rhdf,xdmf}
    if [ $exit_code -ne 0 ]; then
      printf "%-40s %s\n" $par FAIL
      return_code=1
    else
      printf "%-40s %s\n" $par PASS
    fi
  done
done

exit $return_code

