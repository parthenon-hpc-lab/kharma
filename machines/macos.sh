if [[ "$OSTYPE" == "darwin"* ]]; then

  echo "MacOS is only partially supported!"
  echo "Make sure homebrew is installed, and you've installed the packages"
  echo "     hdf5-mpi, llvm, and cmake! (and optionally fftw)"
  echo "Remember to invoke this script with 'zsh ./make.sh <arguments>'!"

  BREW_LLVM="$(brew --prefix llvm)"
  export LDFLAGS="$LDFLAGS -L$BREW_LLVM/lib/c++"
  C_NATIVE="$BREW_LLVM/bin/clang"
  CXX_NATIVE="$BREW_LLVM/bin/clang++"

  # Also have to add fftw if it exists
  if [ -d "$(brew --prefix fftw)" ]; then
    export LDFLAGS="$LDFLAGS -L$(brew --prefix fftw)/lib"
  fi

  # Parthenon doesn't notice brew HDF5 2.0 is parallel    
  cd external/parthenon
  git apply --quiet ../patches/mac-parthenon-no-complain-hdf5.patch
  cd -
fi
