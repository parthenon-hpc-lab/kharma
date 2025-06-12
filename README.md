# KHARMA
KHARMA is an implementation of the HARM scheme for gerneral relativistic magnetohydrodynamics (GRMHD) in C++.  It is based on the Parthenon AMR framework, using Kokkos for parallelism and GPU support.  It is composed of modular "packages," which in theory make it easy to add or swap components representing different algorithmic components or physics processes.

KHARMA is capable of closely matching other HARM implementations, e.g. [iharm3d](https://github.com/AFD-Illinois/iharm3d). However, it also updates the scheme to support static and adaptive mesh refinement, new methods for primitive variable recovery, new boundary conditions, and new stability features for running difficult simulations at high resolutions reliably.

There is a bunch of documentation on the [wiki](https://github.com/AFD-Illinois/kharma/wiki).  If you have a basic question, it might be answered there!  There is also a Slack workspace for users of KHARMA and the associated imaging and analysis codes -- message or email @bprather on GitHub for the link.

## Prerequisites
KHARMA requires that the system have a C++17-compliant compiler, MPI, and parallel HDF5.  All other dependencies are included as submodules, and can be checked out with `git` by running
```bash
$ git submodule update --init --recursive
```

When updating the KHARMA source code, you may also have to update the submodules with
```bash
$ git submodule update --recursive
```
Old submodules are a common cause of compile errors!

## Compiling
On directly supported systems, or systems with standard install locations, you may be able to run:
```bash
./make.sh clean [cuda hip sycl]
```
after a successful configuration (after you see `-- Generating done (X.Ys)`), subsequent invocations can omit `clean`.  If this command fails on supported machines (those with a file in `machines/`), please open an issue.  Broken builds aren't uncommon, as HPC machines change software all the time.

If (when) you run into any trouble, take a look at the [wiki page](https://github.com/AFD-Illinois/kharma/wiki/Building-KHARMA) describing the build system.

## Running
Run a particular problem with e.g.
```bash
$ ./run.sh -i pars/tests/orszag_tang.par
```
note that *all* options are runtime.  The single KHARMA binary can run any of the parameter files in `pars/`, and indeed this is checked as a part of the regression tests.  Note you can still disable some sub-systems manually at compile time, and of course in that case the accompanying problems will crash.

As a broad and capable code, KHARMA has quite a lot of options.  Most are documented [here](https://github.com/AFD-Illinois/kharma/wiki/Parameters), with specific problem setups described [here](https://github.com/AFD-Illinois/kharma/wiki/Problems).

Using `run.sh` is not necessary, feel free to use `kharma.host` or `kharma.cuda` directly.  The script is provided mostly to load any modules or environment variables a machine needs (again, soruced from the file in `machines/`), regardless of whether you're running interactively or as part of a batch script.

Further information can be found on the [wiki page](https://github.com/AFD-Illinois/kharma/wiki/Running-KHARMA).

## Hacking
KHARMA has some documentation for developers on the [wiki](https://github.com/AFD-Illinois/kharma/wiki).  The docs cover some quirks of coding in C++, in particular with Kokkos/GPU programming, and in particular with Parthenon.

## Licenses
KHARMA is made available under the BSD 3-clause license included in each file and in the file LICENSE at the root of this repository.

This repository also carries a substantial portion of the [Kokkos Kernels](https://github.com/kokkos/kokkos-kernels), in the directory `external/kokkos-kernels`, which is provided under the license included in that directory.

Submodules of this repository are subject to their own licenses.
