/*
 *  File: temperature.cpp
 *
 *  BSD 3-Clause License
 *
 *  Copyright (c) 2020, AFD Group at UIUC
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "temperature.hpp"

#include "decs.hpp"
#include "domain.hpp"
#include "kharma_driver.hpp"
#include "microphysics/eos_kharma/eos_kharma.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

using namespace parthenon;

namespace Temperature
{

std::shared_ptr<KHARMAPackage> Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Temperature");

    auto& driver = packages->Get("Driver")->AllParams();
    auto flags_prim = driver.Get<std::vector<MetadataFlag>>("prim_flags");
    flags_prim.push_back(
        Metadata::Cell); // prim_flags has no location; these are per-cell

    pkg->AddField("prims.Temperature", flags_prim);
    pkg->AddField("prims.lT_guess", flags_prim);
    return pkg;
}

TaskStatus BlockUpdateTemperature(MeshBlockData<Real>* rc)
{
    Flag("BlockUpdateTemperature");
    auto pmb = rc->GetBlockPointer();

    PackIndexMap prims_map;
    auto P = rc->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    const VarMap m_p(prims_map, false);

    const auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS");

    // Entire domain, ghosts included: this runs after boundary conditions, and every
    // zone's EOS calls next substep read their own cached guess.
    const IndexRange ib = rc->GetBoundsI(IndexDomain::entire);
    const IndexRange jb = rc->GetBoundsJ(IndexDomain::entire);
    const IndexRange kb = rc->GetBoundsK(IndexDomain::entire);
    pmb->par_for("update_temperature", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            const Real rho = P(m_p.RHO, k, j, i);
            const Real sie = P(m_p.UU, k, j, i) / rho;
            Real lambda[2];
            fill_eos_lambda(P, m_p, k, j, i, lambda);
            P(m_p.TEMP, k, j, i) =
                eos.TemperatureFromDensityInternalEnergy(rho, sie, lambda);
            // StellarCollapse writes the solved log(T) back into lambda[1]
            P(m_p.LT_GUESS, k, j, i) = lambda[1];
        });

    EndFlag();
    return TaskStatus::complete;
}

} // namespace Temperature
