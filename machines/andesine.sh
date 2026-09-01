
echo $HOST
# cprather/CI machine andesine
if [[ "$HOST" == "andesine"* ]]; then
  echo "Running on CI machine"
  HOST_ARCH="AMDAVX"

  if ! option "nompi"; then
    module load mpi/openmpi-x86_64
  fi

  if option "hip"; then
    module load rocm

    # We patch Kokkos to make this into gfx1101==rx7800xt
    DEVICE_ARCH="AMD_GFX1100"

    C_NATIVE=hipcc
    CXX_NATIVE=hipcc
  elif option "cuda"; then
    module load nvhpc
    DEVICE_ARCH="TURING75"
  else
    # GCC by default
    CXX_NATIVE=g++
    C_NATIVE=gcc
  fi
fi
