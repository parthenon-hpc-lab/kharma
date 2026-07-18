/*
 *  File: ismr.cpp
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
#include "ismr.hpp"

#include "domain.hpp"
#include "inverter.hpp"
#include "kharma.hpp"
#include <stdexcept>

std::shared_ptr<KHARMAPackage> ISMR::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("ISMR");
    Params &params = pkg->AllParams();

    // Parameters
    // TODO add "poles" specifically if we ever support other areas
    uint nlevels = (uint) pin->GetOrAddInteger("ismr", "nlevels", 1);
    params.Add("nlevels", nlevels);
    if (nlevels < 1)
        throw std::runtime_error("Internal SMR requires a positive number of levels!");

    // ISMR cache: not evolved, immediately copied to fluid state after averaging
    // Must be total size of variable list
    using FC = Metadata::FlagCollection;
    FC fluid_vars = FC({Metadata::Conserved, Metadata::Cell, Metadata::Independent});
    int nvar = KHARMA::PackDimension(packages.get(), fluid_vars);
    std::vector<int> s_avg({nvar});
    auto m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_avg);
    pkg->AddField("ismr.vars_avg", m);

    // Incompatible with B_FluxCT due to non-local divB, yell
    if (packages->AllPackages().count("B_FluxCT"))
        throw std::runtime_error("Internal SMR is not compatible with Flux-CT magnetic field transport!");
    // Otherwise declare a face temporary
    if (packages->AllPackages().count("B_CT")) {
        m = Metadata({Metadata::Real, Metadata::Face, Metadata::Derived, Metadata::OneCopy});
        pkg->AddField("ismr.fB_avg", m);
    }

    // Incompatible with 2D simulations
    if (pin->GetInteger("parthenon/meshblock", "nx3") == 1)
        throw std::runtime_error("Internal SMR is not compatible with 2D blocks or meshes!");

    // TODO register a split-operator callback?

    return pkg;
}

TaskStatus ISMR::DerefinePoles(MeshData<Real> *md)
{
    Flag("ISMR_DerefinePoles");
    // TODO this routine only applies to polar boundaries for now.
    auto pmesh = md->GetMeshPointer();
    const uint nlevels = pmesh->packages.Get("ISMR")->Param<uint>("nlevels");

    // Figure out indices
    int ng = Globals::nghost;
    for (int iblock=0; iblock < md->NumBlocks(); iblock++) {
        auto& rc = md->GetBlockData(iblock);
        auto pmb = rc->GetBlockPointer();
        PackIndexMap cons_map, cons_map_utop;
        auto vars = rc->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved, Metadata::Cell, Metadata::Independent}, cons_map);
        auto vars_avg = rc->PackVariables(std::vector<std::string>{"ismr.vars_avg"});
        auto vars_utop = rc->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved, Metadata::Cell}, cons_map_utop);
        const int nvar = vars.GetDim(4);
        for (int i = 0; i < BOUNDARY_NFACES; i++) {
            BoundaryFace bface = (BoundaryFace) i;
            auto bdir = KBoundaries::BoundaryDirection(bface);
            auto domain = KBoundaries::BoundaryDomain(bface);
            auto binner = KBoundaries::BoundaryIsInner(bface);
            if (bdir == X2DIR && KBoundaries::IsPhysicalBoundary(pmb, bface)) {
                // indices
                IndexRange3 bCC = KDomain::GetRange(rc, IndexDomain::interior, CC);
                // last physical face
                const int j_f = (binner) ? bCC.js : bCC.je + 1;
                // start of the lowest level of derefinement
                const int jps = (binner) ? j_f + (nlevels - 1) : j_f - (nlevels - 1);
                // Range of x2 to be de-refined
                const IndexRange j_p = IndexRange{(binner) ? j_f : jps, (binner) ? jps : j_f};

                // TODO the following could be done in 1 kernel w/nested parallelism
                // fluid variables average
                pmb->par_for("DerefinePoles_avg_fluid", 0, nvar-1, bCC.ks, bCC.ke, j_p.s, j_p.e, bCC.is, bCC.ie,
                    KOKKOS_LAMBDA (const int &v, const int &k, const int &j, const int &i) {
                        const int coarse_cell_len = m::pow(2, ((binner) ? jps - j : j - jps) + 1);
                        // cell center
                        const int j_c = j + ((binner) ? 0 : -1);
                        // this fine cell's k-index within the coarse cell
                        const int k_fine = (k - ng) % coarse_cell_len;
                        // starting k-index of the coarse cell
                        const int k_start = k - k_fine;

                        // average each var over next `coarse_cell_len` cells
                        // Lots of repeated ops but we don't care, this is applied over a small region
                        Real avg = 0.;
                        for (int ktemp = 0; ktemp < coarse_cell_len; ++ktemp)
                            avg += vars(v, k_start + ktemp, j_c, i);
                        avg /= coarse_cell_len;
                        vars_avg(v, k, j_c, i) = avg;
                    }
                );
                // fluid variables write
                pmb->par_for("DerefinePoles_write_fluid", 0, nvar-1, bCC.ks, bCC.ke, j_p.s, j_p.e, bCC.is, bCC.ie,
                    KOKKOS_LAMBDA (const int &v, const int &k, const int &j, const int &i) {
                        const int j_c = j + ((binner) ? 0 : -1); // cell center
                        vars(v, k, j_c, i) = vars_avg(v, k, j_c, i);
                    }
                );

                // UtoP for the GRMHD variables
                PackIndexMap prims_map;
                auto P = rc->PackVariables(std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive"), Metadata::Cell}, prims_map);
                VarMap m_u(cons_map_utop, true), m_p(prims_map, false);
                const auto& G = pmb->coords;
                const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
                const Floors::Prescription floors = pmb->packages.Get("Floors")->Param<Floors::Prescription>("prescription");
                pmb->par_for("DerefinePoles_UtoP", bCC.ks, bCC.ke, j_p.s, j_p.e, bCC.is, bCC.ie,
                    KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                        const int j_c = j + ((binner) ? 0 : -1); // cell center
                        // The usual inverter is not EMHD-aware, so it's going to dump all of T into the
                        // ideal GRMHD fluid variables
                        Inverter::u_to_p<Inverter::Type::kastaun>(G, vars_utop, m_u, gam, k, j_c, i, P, m_p,
                                                                  Loci::center, 25, 1e-12);
                        // Consistent with that, we zero out the EMHD extra variables.  This switches theories to
                        // evolving ideal GRMHD in ISMR region, but conserves the components of T themselves
                        if (m_u.Q >= 0) vars_utop(m_u.Q, k, j_c, i) = 0.;
                        if (m_p.Q >= 0) P(m_p.Q, k, j_c, i) = 0.;
                        if (m_u.DP >= 0) vars_utop(m_u.DP, k, j_c, i) = 0.;
                        if (m_p.DP >= 0) P(m_p.DP, k, j_c, i) = 0.;
                    }
                );
                // TODO there SHOULD be no need for floors here. Should test or prove this is always true
            }
        }
    }
    EndFlag();
    return TaskStatus::complete;
}
