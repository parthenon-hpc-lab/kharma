// © 2021-2023. Triad National Security, LLC. All rights reserved.  This
// program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.

#ifndef MICROPHYSICS_EOS_EOS_HPP_
#define MICROPHYSICS_EOS_EOS_HPP_

#include <memory>
#include <parthenon/package.hpp>

#include <singularity-eos/eos/eos.hpp>

using namespace parthenon::package::prelude;

namespace Microphysics {

//  using MyEOS=singularity::impl::Variant<IdealGas>;

namespace EOS {

using EOS = singularity::Variant<
    singularity::UnitSystem<singularity::IdealGas>, singularity::IdealGas
#ifdef SPINER_USE_HDF
    ,
    singularity::UnitSystem<singularity::StellarCollapse>, singularity::StellarCollapse
#endif // SPINER_USE_HDF
    >;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
} // namespace EOS

} // namespace Microphysics

#endif // MICROPHYSICS_EOS_EOS_HPP_
