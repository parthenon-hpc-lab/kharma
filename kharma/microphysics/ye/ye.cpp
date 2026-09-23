/*
 *  File: ye.cpp
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
#include "ye.hpp"

#include "decs.hpp"
#include "domain.hpp"
#include "floors.hpp"
#include "flux.hpp"
#include "kharma_driver.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

using namespace parthenon;

namespace Ye
{

std::shared_ptr<KHARMAPackage> Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Ye");
    Params& params = pkg->AllParams();

    const bool use_ye = pin->GetOrAddBoolean("fluid", "Ye", false);
    params.Add("use_ye", use_ye);

    std::vector<MetadataFlag> flags_ye = {Metadata::Cell, Metadata::GetUserFlag("Explicit")};

    auto& driver = packages->Get("Driver")->AllParams();
    auto flags_prim = driver.Get<std::vector<MetadataFlag>>("prim_flags");
    flags_prim.insert(flags_prim.end(), flags_ye.begin(), flags_ye.end());
    auto flags_cons = driver.Get<std::vector<MetadataFlag>>("cons_flags");
    flags_cons.insert(flags_cons.end(), flags_ye.begin(), flags_ye.end());

    pkg->AddField("cons.Ye", flags_cons);
    pkg->AddField("prims.Ye", flags_prim);

    pkg->BlockUtoP = Ye::BlockUtoP;
    pkg->BoundaryUtoP = Ye::BlockUtoP;

    return pkg;
}

void BlockUtoP(MeshBlockData<Real>* rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();

    GridScalar Ye_P = rc->Get("prims.Ye").data;
    GridScalar Ye_U = rc->Get("cons.Ye").data;
    GridScalar rho_U = rc->Get("cons.rho").data;

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    int is = bounds.is(domain), ie = bounds.ie(domain);
    int js = bounds.js(domain), je = bounds.je(domain);
    int ks = bounds.ks(domain), ke = bounds.ke(domain);

    // Ye is specific (per baryon).
    pmb->par_for("UtoP_Ye", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            Ye_P(k, j, i) = Ye_U(k, j, i) / rho_U(k, j, i);
        });
}

void ApplyFloors(MeshBlockData<Real>* mbd, IndexDomain domain)
{
    auto pmb = mbd->GetBlockPointer();
    auto packages = pmb->packages;

    PackIndexMap prims_map;
    auto P = mbd->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    const VarMap m_p(prims_map, false);

    auto fflag = mbd->PackVariables(std::vector<std::string>{"fflag"}, prims_map);

    const Real ye_min = packages.Get("eos")->Param<Real>("ye_min");
    const Real ye_max = packages.Get("eos")->Param<Real>("ye_max");

    const IndexRange3 b = KDomain::GetRange(mbd, domain);
    pmb->par_for("apply_ye_bounds", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            if (P(m_p.YE, k, j, i) < ye_min) {
                fflag(0, k, j, i) = Floors::FFlag::YE | (int)fflag(0, k, j, i);
                P(m_p.YE, k, j, i) = ye_min;
            }
            if (P(m_p.YE, k, j, i) > ye_max) {
                fflag(0, k, j, i) = Floors::FFlag::YE | (int)fflag(0, k, j, i);
                P(m_p.YE, k, j, i) = ye_max;
            }

        });
    Flux::BlockPtoU(mbd, domain);
}

} // namespace Ye
