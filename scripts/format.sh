#!/bin/bash

#------------------------------------------------------------------------------
# © 2021-2025. Triad National Security, LLC. All rights reserved.  This
# program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S.  Department of Energy/National
# Nuclear Security Administration. All rights in the program are
# reserved by Triad National Security, LLC, and the U.S. Department of
# Energy/National Nuclear Security Administration. The Government is
# granted for itself and others acting on its behalf a nonexclusive,
# paid-up, irrevocable worldwide license in this material to reproduce,
# prepare derivative works, distribute copies to the public, perform
# publicly and display publicly, and to permit others to do so.
#------------------------------------------------------------------------------


: ${CFM:=clang-format-19}
: ${VERBOSE:=0}

if ! command -v ${CFM} &> /dev/null; then
    >&2 echo "Error: No clang format found! Looked for ${CFM}"
    >&2 echo "Download? [Y/n]: "
    read do_download
    echo $do_download
    if [[ "$do_download" == "Y" || "$do_download" == "y" || "$do_download" == "" ]]; then
      wget https://github.com/muttleyxd/clang-tools-static-binaries/releases/download/master-796e77c/clang-format-19_linux-amd64
      mv clang-format-19_linux-amd64 bin/clang-format-19
      chmod +x bin/clang-format-19
      CFM=$PWD/bin/clang-format-19
    else
      exit 1
    fi
else
    CFM=$(command -v ${CFM})
    echo "Clang format found: ${CFM}"
fi

# clang format major version
TARGET_CF_VRSN=19
CF_VRSN=$(${CFM} --version)
echo "Note we assume clang format version ${TARGET_CF_VRSN}."
echo "You are using ${CF_VRSN}."
echo "If these differ, results may not be stable."


xargs_command="-I {} -0 -t ${CFM} -style=file -i {}"
# (The xargs insertion brackets '{}' are helpful if there are no cpp/hpp
#  files being added, they will be left as-is and avoid error messages)

if [ -n "$KDEV_FORMAT_STAGED" ]; then
    echo "Formatting staged files..."
    git diff --staged --name-only --diff-filter=ACMR -z \
        'kharma/**.[ch]pp' 'iris/**.[ch]pp' 'tests/**.[ch]pp' |
        xargs $xargs_command

elif [ -n "$KDEV_FORMAT_TRACKED" ]; then
    echo "Formatting tracked files..."
    git ls-files --cached -z 'kharma/**.[ch]pp' 'iris/**.[ch]pp' 'tests/**.[ch]pp' |
        xargs $xargs_command

else
    # format both tracked and untracked cpp/hpp files in our directories.
    # git ls-files seemed more reliable here than ls or find.
    echo "Formatting all files..."
    git ls-files --cached --other -z 'kharma/**.[ch]pp' 'iris/**.[ch]pp' 'tests/**.[ch]pp' |
        xargs $xargs_command
fi
echo "...Done"
