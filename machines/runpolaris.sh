
HOST_ARCH=ZEN3
DEVICE_ARCH=AMPERE80
NPROC=64

module restore

# Re-enable the ALCF software module tree that purge wiped out
module use /soft/modulefiles
module use /soft/modulefiles/core
  
# Clean up any active environments defensively without throwing errors
module unload PrgEnv-gnu PrgEnv-nvidia PrgEnv-cray PrgEnv-intel
  
if [[ $ARGS == *"gcc"* ]]; then
    module load PrgEnv-gnu
    module load cuda
else
    # Defaulting to the modern system-recommended NVIDIA path
    module load PrgEnv-nvidia
    module load cuda
fi
# Common modules
# UNLOCK THE SPACK UTILITY TREE
module load spack-pe-base
  
module load cray-hdf5-parallel
module load craype-accel-nvidia80
module load cmake
module list
  
  
# Ensure the Cray compiler wrappers explicitly target the host architecture
export CRAY_CPU_TARGET=x86-64
# TODO(BSP) need to set CRAYPE_LINK_TYPE=dynamic long-term?

# Globally enforce the C++17 standard for all targets and submodules
EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON $EXTRA_FLAGS"
