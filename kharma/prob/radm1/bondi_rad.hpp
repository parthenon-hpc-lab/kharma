#pragma once

#include "decs.hpp"

#include "coordinate_utils.hpp"
#include "flux_functions.hpp"
#include "gr_coordinates.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

/**
 * Mckinney Radiative Bondi Problem as done in
 * (MNRAS, 441, 4, 11 July 2024, 3177-3208), section 5.10:
 * Radiative spherical (Bondi) accretion in 1D, for a non-rotating BH.
 */
TaskStatus InitializeRadiativeBondi(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin);

/**
 * Record parameters the RadiativeBondi problem (or boundaries) will need throughout the
 * run. Uses "GRMHD" package as convenient proxy, same as bondi.cpp.
 */
void AddBondiRadParameters(ParameterInput* pin, Packages_t& packages);

/**
 * Set all values on a given domain to the radiative Bondi analytic initial state
 * (eqs. 93-95 of McKinney et al. 2014). Use the template version when possible, which
 * just calls through.
 */
TaskStatus SetBondiRadImpl(
    std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse);

template<IndexDomain domain>
TaskStatus SetBondiRad(std::shared_ptr<MeshBlockData<Real>>& rc, bool coarse = false)
{
    return SetBondiRadImpl(rc, domain, coarse);
}
