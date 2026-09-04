# KHARMA
KHARMA is an implementation of the HARM scheme for general relativistic magnetohydrodynamics (GRMHD) in C++.  It is based on the Parthenon AMR framework, using Kokkos for parallelism and GPU support.  It is composed of modular "packages," which in theory make it easy to add or swap components representing different algorithmic components or physics processes.

KHARMA is capable of closely matching other HARM implementations, e.g. [iharm3d](https://github.com/AFD-Illinois/iharm3d). However, it also updates the scheme to support static and adaptive mesh refinement, new methods for primitive variable recovery, new boundary conditions, and new stability features for running difficult simulations at high resolutions reliably.

There is a bunch of documentation on the [wiki](https://github.com/parthenon-hpc-lab/kharma/wiki).  If you have a basic question, it might be answered there!  There is also a Slack workspace for users of KHARMA and the associated imaging and analysis codes -- message or email @c-prather on GitHub for the link.

## Getting KHARMA
KHARMA requires that the system have:
* a C++17-compliant compiler
* HDF5 (optional)
* MPI (optional)

KHARMA can be cloned from this repository by running
```bash
git clone --recursive https://github.com/parthenon-hpc-lab/kharma.git
```

This will automatically pull all of KHARMA's internal dependenices, or "submodules." Submodules can be updated by running
```bash
git submodule update --recursive
```
Old submodules are a common cause of compile errors!

## Compiling
On directly supported systems, or systems with standard install locations, you may be able to run:
```bash
./make.sh clean
```
or if you're compiling for Nvidia GPUs,
```bash
./make.sh clean cuda
```
(KHARMA can also be compiled for AMD and Intel GPUs, see the [wiki page](https://github.com/parthenon-hpc-lab/kharma/wiki/Building-KHARMA))

If you're on Mac, you will have to use `zsh` because the MacOS version of `bash` is too old:
```bash
zsh ./make.sh <arguments>
```

### Compiling on macOS in more detail

First, install the required dependencies via Homebrew:
```bash
brew install hdf5-mpi llvm cmake fftw
```

By default, `machines/macos.sh` builds with Homebrew's LLVM `clang` rather than Apple's own Xcode `clang`, because Apple's `clang` has no OpenMP support. However, that clang/libc++ combination is stricter about implicit standard-library includes than what this codebase is usually built against, and can fail to compile some vendored submodule code as a result. For example, `external/singularity-eos/utils/ports-of-call/ports-of-call/portability.hpp` uses `std::copy` without including `<algorithm>`, and a similar missing `<iterator>` include shows up further into Parthenon's `src/utils/unique_id.cpp`. These are real missing-include bugs in the vendored code, not local misconfiguration -- they just happen not to matter under GCC's libstdc++, which tends to pull those headers in transitively anyway.

Rather than patch every missing include as it's discovered, we recommend building with GCC instead:
```bash
brew install gcc
```

Then create `~/.config/kharma.sh` (if this file exists, `make.sh` and `run.sh` source it *instead of* anything in `machines/`, entirely bypassing the clang-forcing logic above):
```bash
C_NATIVE=/opt/homebrew/bin/gcc-16
CXX_NATIVE=/opt/homebrew/bin/g++-16

if [ -d "$(brew --prefix fftw)" ]; then
  export LDFLAGS="$LDFLAGS -L$(brew --prefix fftw)/lib"
fi
```
(adjust `gcc-16`/`g++-16` to whatever version Homebrew installs; Intel Macs should use `/usr/local` instead of `/opt/homebrew`)

Because this bypasses `machines/macos.sh` entirely, you'll also need to manually apply its Parthenon compatibility patch once -- Homebrew's `hdf5-mpi` doesn't self-report as parallel-capable in a way Parthenon's CMake reliably detects, which otherwise aborts the build with a `FATAL_ERROR`:
```bash
cd external/parthenon
git apply --quiet ../patches/mac-parthenon-no-complain-hdf5.patch
cd -
```

With those in place, `zsh ./make.sh clean` should compile `kharma.host` cleanly.

If your system does not have HDF5, KHARMA can attempt to compile it for you -- just add `hdf5` when you run `make.sh`.  If you want to omit MPI, you can add `nompi` (the resulting binary can still be run on multiple CPU cores!  Just not multiple GPUs, or multiple nodes of a cluster).

After a successful configuration (after you see `-- Generating done (X.Ys)`), subsequent invocations can omit `clean`.  If `./make.sh` is not working on a supported machine (those with a file in `machines/`), please open an issue.  Broken builds aren't uncommon, as HPC machines change software all the time.

If you're having trouble or need more options, check out the [wiki page](https://github.com/parthenon-hpc-lab/kharma/wiki/Building-KHARMA) describing the build system.

## Running
Run a particular problem with e.g.
```bash
./run.sh -i pars/tests/orszag_tang.par
```
note that *all* options are runtime.  The single KHARMA binary can run any of the parameter files in `pars/`, and indeed this is checked as a part of the regression tests.  Note you can still disable some sub-systems manually at compile time, and of course in that case the accompanying problems will crash.

As a broad and capable code, KHARMA has quite a lot of configuration parameters.  Most are documented [here](https://github.com/parthenon-hpc-lab/kharma/wiki/Parameters), with specific problem setups described [here](https://github.com/parthenon-hpc-lab/kharma/wiki/Problems).

If you need more control of where and how KHARMA runs, you can use the binary, e.g. `kharma.host` or `kharma.cuda`, directly.  `run.sh` is provided mostly to load any modules or environment variables a machine needs (again, soruced from the file in `machines/`), regardless of whether you're running interactively or as part of a batch script.

Further information can be found on the [wiki page](https://github.com/parthenon-hpc-lab/kharma/wiki/Running-KHARMA).

## Hacking
KHARMA has some documentation for developers on the [wiki](https://github.com/parthenon-hpc-lab/kharma/wiki).  The docs cover some quirks of coding in C++, in particular with Kokkos/GPU programming, and in particular with Parthenon.

## Licenses
KHARMA is made available under the BSD 3-clause license included in each file and in the file LICENSE at the root of this repository.

This repository also carries a substantial portion of the [Kokkos Kernels](https://github.com/kokkos/kokkos-kernels), in the directory `external/kokkos-kernels`, which is provided under the license included in that directory.

Submodules of this repository are subject to their own licenses -- universally BSD-style permissive terms.
