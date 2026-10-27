if [[ "$OSTYPE" == "darwin"* ]]; then

  echo "MacOS is only partially supported!"
  echo "Make sure homebrew is installed, and you've installed the packages"
  echo "     hdf5-mpi, llvm, and cmake! (and optionally fftw)"
  echo "Remember to invoke this script with 'zsh ./make.sh <arguments>'!"

  # Initialize LDFLAGS so later appends ("$LDFLAGS -L...") are safe under
  # `set -u`, regardless of which branch below runs first.
  export LDFLAGS="${LDFLAGS:-}"
  # Neither branch below goes through make.sh's own compiler auto-detect
  # (it's skipped whenever CXX_NATIVE is already set), so OMP_FLAG would
  # otherwise never be set at all. -fopenmp is correct for both Homebrew
  # gcc and Homebrew llvm/clang (Apple's own clang has no OpenMP support,
  # which is the whole reason this file reaches for Homebrew's instead).
  OMP_FLAG="-fopenmp"

  if option "gcc"; then
    # afaict there's no path to "most recent homebrew gcc" which is
    # independent of both Homebrew location and gcc version
    C_NATIVE=$(brew --prefix gcc)/bin/gcc-16
    CXX_NATIVE=$(brew --prefix gcc)/bin/g++-16
  else
    BREW_LLVM="$(brew --prefix llvm)"
    export LDFLAGS="$LDFLAGS -L$BREW_LLVM/lib/c++"
    C_NATIVE="$BREW_LLVM/bin/clang"
    CXX_NATIVE="$BREW_LLVM/bin/clang++"
  fi

  # Also have add fftw manually if it exists
  if [ -d "$(brew --prefix fftw)" ]; then
    export LDFLAGS="$LDFLAGS -L$(brew --prefix fftw)/lib"
  fi

  # Point CMake at Homebrew's parallel HDF5/MPI explicitly, if installed --
  # Anaconda (if loaded) often ships its own copies earlier on PATH, which
  # find_package() will otherwise pick up in preference to Homebrew's.
  # No-ops if these weren't installed via Homebrew.
  if [ -d "$(brew --prefix hdf5-mpi 2>/dev/null)" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS -DHDF5_ROOT=$(brew --prefix hdf5-mpi)"
  fi
  if [ -d "$(brew --prefix open-mpi 2>/dev/null)" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS -DMPI_HOME=$(brew --prefix open-mpi)"
  fi
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    EXTRA_FLAGS="$EXTRA_FLAGS -DCMAKE_IGNORE_PATH=${CONDA_BASE}/bin;${CONDA_BASE}/lib;${CONDA_BASE}/include"
  fi

  # Parthenon doesn't notice brew HDF5 2.0 is parallel. "Convince" it
  BASE=$PWD
  cd $SOURCE_DIR/external/parthenon
  git apply --quiet ../patches/mac-parthenon-no-complain-hdf5.patch || true
  cd $BASE
fi
