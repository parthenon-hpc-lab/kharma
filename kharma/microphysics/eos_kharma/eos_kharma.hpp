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

#include "kharma_package.hpp"

#include <singularity-eos/eos/eos_ideal.hpp>
#include <singularity-eos/eos/eos_stellar_collapse.hpp>
#include <singularity-eos/eos/eos_variant.hpp>
#include <singularity-eos/eos/modifiers/eos_unitsystem.hpp>

#ifdef SPINER_USE_HDF
#include <singularity-eos/eos/eos_stellar_collapse.hpp>
#endif

using namespace parthenon::package::prelude;
namespace Microphysics
{

//  using MyEOS=singularity::impl::Variant<IdealGas>;

namespace EOS
{

using EOS = singularity::Variant<singularity::UnitSystem<singularity::IdealGas>,
    singularity::IdealGas
#ifdef SPINER_USE_HDF
    ,
    singularity::UnitSystem<singularity::StellarCollapse>, singularity::StellarCollapse
#endif // SPINER_USE_HDF
    >;

std::shared_ptr<KHARMAPackage> Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages);
<<<<<<< HEAD

/**
 * Specific internal energy from (rho, P, lambda), for any EOS in the variant.
 *
 * singularity-eos's generic InternalEnergyFromDensityPressure can't be used for
 * tabulated, composition-dependent EOSs: it computes its root-finding bracket without
 * passing lambda (eos_base.hpp), which aborts for StellarCollapse. So for non-ideal EOSs
 * we bisect in log(T) over the table's temperature range instead, since at fixed rho and
 * Ye, P is non-decreasing in T. IdealGas uses its exact closed form.
 *
 * T_min/T_max: the "eos" package's "T_min"/"T_max" params (code units). Unused for
 * IdealGas. A target P outside [P(T_min), P(T_max)] silently returns the table-edge
 * value.
 */
KOKKOS_INLINE_FUNCTION Real SieFromDensityPressure(const EOS& eos, const bool is_ideal,
    const Real rho, const Real P, Real lambda[2], const Real T_min, const Real T_max)
{
    if (is_ideal) {
        Real sie = 0.0;
        eos.InternalEnergyFromDensityPressure(rho, P, sie, lambda);
        return sie;
    }
    Real lT_lo = m::log(T_min), lT_hi = m::log(T_max);
    for (int iter = 0; iter < 64; ++iter) {
        const Real lT_mid = 0.5 * (lT_lo + lT_hi);
        if (eos.PressureFromDensityTemperature(rho, m::exp(lT_mid), lambda) < P)
            lT_lo = lT_mid;
        else
            lT_hi = lT_mid;
    }
    return eos.InternalEnergyFromDensityTemperature(
        rho, m::exp(0.5 * (lT_lo + lT_hi)), lambda);
}

=======
>>>>>>> origin/feature/RadM1
} // namespace EOS

} // namespace Microphysics

#endif // MICROPHYSICS_EOS_EOS_HPP_
