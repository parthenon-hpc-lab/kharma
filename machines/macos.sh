if [[ "$OSTYPE" == "darwin"* ]]; then

  echo "MacOS is only partially supported!"
  echo "Make sure homebrew is installed, and you've installed the packages"
  echo "     hdf5-mpi, llvm, and cmake! (and optionally fftw)"
  echo "Remember to invoke this script with 'zsh ./make.sh <arguments>'!"

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

  # Parthenon doesn't notice brew HDF5 2.0 is parallel. "Convince" it
  BASE=$PWD
  cd $SOURCE_DIR/external/parthenon
  git apply --quiet ../patches/mac-parthenon-no-complain-hdf5.patch
  cd $BASE
fi
