#pragma once

#include "decs.hpp"

#include "coordinate_utils.hpp"
#include "flux_functions.hpp"
#include "gr_coordinates.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>


TaskStatus InitializeBeamOfLight(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin);

void AddBeamOfLightParameters(ParameterInput* pin, Packages_t& packages);


TaskStatus SetBeamOfLightImpl(
    std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse);

template<IndexDomain domain>
TaskStatus SetBeamOfLight(std::shared_ptr<MeshBlockData<Real>>& rc, bool coarse = false)
{
    return SetBeamOfLightImpl(rc, domain, coarse);
}
