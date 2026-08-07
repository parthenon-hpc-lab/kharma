#!/bin/bash

# This is for users of newer hardware who wish to use newer Kokkos versions.
# Specifically, Kokkos 4.7, which is the last compatible version with this
# parthenon/KHARMA codebase.
# Version 4.7 is not the default due to issues with some MPI implementations
# hanging at the end of simulations. Most large-scale clusters do not
# have this problem, but you should test for it if upgrading.

# The next version of KHARMA will use a much newer Parthenon version and fix
# this properly, I swear

KHARMA_DIR="$(dirname "${BASH_SOURCE[0]}")/.."

# Ensure we're up to date
git submodule update --recursive --init

# Bump version of kokkos itself, ahead of what our
# Parthenon wants
cd ${KHARMA_DIR}/external/parthenon/external/Kokkos
git fetch
git checkout release-candidate-4.7.04
cd ../../../..

# Apply patch to add an old definition back to the current
# kokkos-kernels.  Kernels will be updated with new Parthenon
git apply external/patches/kokkos-kernels-bumpkokkos.patch
