/* 
 *  File: dirichlet.cpp
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

#include "dirichlet.hpp"

#include "boundary_types.hpp"
#include "domain.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

using namespace parthenon;

void KBoundaries::DirichletImpl(MeshBlockData<Real> *rc, BoundaryFace bface, bool coarse, bool set)
{
    // Get all cell-centered ghosts, minus anything just used at startup
    using FC = Metadata::FlagCollection;
    FC ghost_vars = FC({Metadata::FillGhost, Metadata::Conserved})
                  + FC({Metadata::FillGhost, Metadata::GetUserFlag("Primitive")})
                  - FC({Metadata::GetUserFlag("StartupOnly")});
    auto q = rc->PackVariables(ghost_vars, coarse);
    auto bound = rc->PackVariables(std::vector<std::string>{"Boundaries." + BoundaryName(bface)});
    DirichletSetFromField(rc, q, bound, bface, coarse, set, false);

    FC ghost_vars_f = FC({Metadata::FillGhost, Metadata::Face})
                  - FC({Metadata::GetUserFlag("StartupOnly")});
    auto q_f = rc->PackVariables(ghost_vars_f, coarse);
    auto bound_f = rc->PackVariables(std::vector<std::string>{"Boundaries.f." + BoundaryName(bface)});
    DirichletSetFromField(rc, q_f, bound_f, bface, coarse, set, true);
}

void KBoundaries::DirichletSetFromField(MeshBlockData<Real> *rc, VariablePack<Real> &q, VariablePack<Real> &bound,
                                        BoundaryFace bface, bool coarse, bool set, bool do_face)
{
    // We're sometimes called without any variables to sync (e.g. syncing flags, EMFs), just return
    if (q.GetDim(4) == 0) return;

    // Determine elements to run over
    std::vector<TopologicalElement> el_list;
    if (do_face) {
        el_list = {F1, F2, F3};
    } else {
        el_list = {CC};
    }
    int el_tot = el_list.size();

    // Warn when the cache doesn't match the variables we've been given
    if ((el_tot * q.GetDim(4)) != bound.GetDim(4)) {
        std::cerr << "Dirichlet boundary mismatch! Boundary cache: " << bound.GetDim(4) << " for pack: " << el_tot * q.GetDim(4) << std::endl;
    }

    // Indices
    auto pmb = rc->GetBlockPointer();
    const bool binner = BoundaryIsInner(bface);
    const int dir = BoundaryDirection(bface);
    const auto domain = BoundaryDomain(bface);
    const auto bname = BoundaryName(bface);
    for (auto el : el_list) {
        // This is the domain of the boundary/ghost zones
        IndexRange3 b = KDomain::GetBoundaryRange(rc, domain, el, coarse);

        // Flatten TopologicalElements when reading/writing to boundaries cache
        pmb->par_for(
            "dirichlet_boundary_" + bname, 0, q.GetDim(4)-1, b.ks, b.ke, b.js, b.je, b.is, b.ie,
            KOKKOS_LAMBDA (const int &v, const int &k, const int &j, const int &i) {
                if (set) {
                    bound(el_tot*v + (static_cast<int>(el) % el_tot), k - b.ks, j - b.js, i - b.is) = q(el, v, k, j, i);
                } else {
                    q(el, v, k, j, i) = bound(el_tot*v + (static_cast<int>(el) % el_tot), k - b.ks, j - b.js, i - b.is);
                }
            }
        );
    }
}

void KBoundaries::FreezeDirichlet(std::shared_ptr<MeshData<Real>> &md)
{
    // For each face...
    for (int i=0; i < BOUNDARY_NFACES; i++) {
        BoundaryFace bface = (BoundaryFace) i;
        auto bname = BoundaryName(bface);
        auto pmesh = md->GetMeshPointer();
        // ...if this boundary is dirichlet...
        if (pmesh->packages.Get("Boundaries")->Param<std::string>(bname) == "dirichlet") {
            //std::cout << "Freezing dirichlet " << bname << " on mesh." << std::endl;
            // ...on all blocks...
            for (int i=0; i < md->NumBlocks(); i++) {
                auto rc = md->GetBlockData(i).get();
                auto pmb = rc->GetBlockPointer();
                // ...if the face is not internal
                if (KBoundaries::IsPhysicalBoundary(pmb, bface)) {
                    auto domain = BoundaryDomain(bface);
                    // Set whatever is in that domain as the Dirichlet bound
                    SetDomainDirichlet(rc, domain, false);
                }
            }
        }
    }
}
void KBoundaries::FreezeDirichletBlock(MeshBlockData<Real> *rc)
{
    // For each face...
    for (int i=0; i < BOUNDARY_NFACES; i++) {
        BoundaryFace bface = (BoundaryFace) i;
        auto bname = BoundaryName(bface);
        auto pmb = rc->GetBlockPointer();
        // ...if this boundary is dirichlet and physical...
        if (pmb->packages.Get("Boundaries")->Param<std::string>(bname) == "dirichlet" &&
            KBoundaries::IsPhysicalBoundary(pmb, bface)) {
            //std::cout << "Freezing dirichlet " << bname << " on block." << std::endl;
            auto domain = BoundaryDomain(bface);
            // Set whatever is in that domain as the Dirichlet bound
            SetDomainDirichlet(rc, domain, false);
        }
    }
}
