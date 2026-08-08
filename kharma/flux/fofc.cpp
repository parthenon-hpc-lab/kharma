/*
 *  File: fofc.cpp
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

#include "flux.hpp"

#include "domain.hpp"
#include "floors_functions.hpp"
#include "inverter.hpp"

// phoebus includes
#include "microphysics/eos_kharma/eos_kharma.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "phoebus_utils/variables.hpp"

using namespace parthenon;

// Very bad definition. TODO get rid of this eventually
#define PLOOP for (int ip = 0; ip < nvar; ++ip)

TaskStatus Flux::MarkFOFC(MeshData<Real>* guess)
{
    auto pmb0 = guess->GetBlockData(0)->GetBlockPointer();

    // flags of the guess indicate where we lower
    // (not that it matters, the flags are OneCopy)
    auto fflag = guess->PackVariables(std::vector<std::string>{"fflag"});
    auto pflag = guess->PackVariables(std::vector<std::string>{"pflag"});
    auto fofcflag = guess->PackVariables(std::vector<std::string>{"fofcflag"});

    PackIndexMap cons_map, prims_map;
    std::vector<MetadataFlag> prims_flags = {
        Metadata::GetUserFlag("Primitive"), Metadata::Cell};
    std::vector<MetadataFlag> cons_flags = {Metadata::WithFluxes, Metadata::Cell};
    const auto& P = guess->PackVariables(prims_flags, prims_map);
    const auto& U = guess->PackVariablesAndFluxes(cons_flags, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    const Real gam = pmb0->packages.Get("GRMHD")->Param<Real>("gamma");

    // Use values from floors package if it's enabled, otherwise any we've been asked to
    // apply
    const Floors::Prescription floors =
        pmb0->packages.AllPackages().count("Floors")
            ? pmb0->packages.Get("Floors")->Param<Floors::Prescription>("prescription")
            : pmb0->packages.Get("Inverter")
                  ->Param<Floors::Prescription>("inverter_prescription");
    const Floors::Prescription floors_inner =
        pmb0->packages.AllPackages().count("Floors")
            ? pmb0->packages.Get("Floors")->Param<Floors::Prescription>(
                  "prescription_inner")
            : pmb0->packages.Get("Inverter")
                  ->Param<Floors::Prescription>("inverter_prescription");

    // Parameters
    const auto& pars = pmb0->packages.Get("Fluxes")->AllParams();
    const bool spherical = pmb0->coords.coords.is_spherical();
    const int polar_cells = pars.Get<int>("fofc_polar_cells");
    const GReal r_eh = pmb0->coords.coords.get_horizon();
    const GReal fofc_radius = pars.Get<bool>("fofc_use_eh_buffer")
                                  ? r_eh + pars.Get<GReal>("fofc_eh_buffer")
                                  : 0.;

    // Pre-mark cells which will need fluxes reduced.
    // This avoids a race condition marking them multiple times when iterating faces,
    // and isolates the potentially slow/weird integer conversion stuff so we can measure
    // the kernel time
    const IndexRange3 b = KDomain::GetRange(guess, IndexDomain::entire);
    const IndexRange block = IndexRange{0, fofcflag.GetDim(5) - 1};
    pmb0->par_for("fofc_mark", block.s, block.e, b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &bl, const int &k, const int &j, const int &i)
        {
            const auto& G = fofcflag.GetCoords(bl);
            // if cell failed to invert or would call floors...
            // TODO preserve cause in the fofcflag
            // If the solve failed, because we reconstructed a
            // negative or zero internal energy (even after floors!)
            Real rhomin_geom, umin_geom;
            determine_geo_floors(G, P(bl), m_p, gam, k, j, i, floors, floors_inner,
                rhomin_geom, umin_geom);
            const Real umin = umin_geom;
            if (Inverter::failed(pflag(bl, 0, k, j, i)) &&
                (P(bl, m_p.UU, k, j, i) < umin)) {
                fofcflag(bl, 0, k, j, i) = 1;
            } else {
                fofcflag(bl, 0, k, j, i) = 0;
            }
        });

    if (spherical && polar_cells > 0) {
        for (int i_block = 0; i_block < guess->NumBlocks(); i_block++) {
            auto& rc = guess->GetBlockData(i_block);
            auto pmb = rc->GetBlockPointer();
            const bool is_inner_x2 =
                KBoundaries::IsPhysicalBoundary(pmb, BoundaryFace::inner_x2);
            const bool is_outer_x2 =
                KBoundaries::IsPhysicalBoundary(pmb, BoundaryFace::outer_x2);
            if (is_inner_x2 || is_outer_x2) {
                auto lfofcflag = rc->PackVariables(std::vector<std::string>{"fofcflag"});
                if (is_inner_x2) {
                    const IndexRange3 b = KDomain::GetRange(guess, IndexDomain::inner_x2);
                    int jstart = b.je + 1;
                    int jend = jstart + polar_cells - 1;
                    pmb0->par_for("fofc_mark_inner_x2", b.ks, b.ke, jstart, jend, b.is,
                        b.ie,
                                  KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
                        {
                            lfofcflag(0, k, j, i) = 1;
                        });
                }
                if (is_outer_x2) {
                    const IndexRange3 b = KDomain::GetRange(guess, IndexDomain::outer_x2);
                    int jend = b.js - 1;
                    int jstart = jend - polar_cells + 1;
                    pmb0->par_for("fofc_mark_outer_x2", b.ks, b.ke, jstart, jend, b.is,
                        b.ie,
                                  KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
                        {
                            lfofcflag(0, k, j, i) = 1;
                        });
                }
            }
        }
    }

    return TaskStatus::complete;
}

TaskStatus Flux::FOFC(MeshData<Real>* md, MeshData<Real>* guess)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    auto& packages = pmb0->packages;
    const auto& eos_params = packages.Get("eos")->AllParams();
    auto eos = eos_params.Get<Microphysics::EOS::EOS>("d.EOS");
    auto pmesh = md->GetMeshPointer();
    const int ndim = pmesh->ndim;

    // Pick up flag. Optionally synced
    auto fofcflag = guess->PackVariables(std::vector<std::string>{"fofcflag"});

    // But we're modifying the live temporaries, and eventually fluxes, here
    const auto& Pl_all = md->PackVariables(std::vector<std::string>{"Flux.Pl"});
    const auto& Pr_all = md->PackVariables(std::vector<std::string>{"Flux.Pr"});
    const auto& Ul_all = md->PackVariables(std::vector<std::string>{"Flux.Ul"});
    const auto& Ur_all = md->PackVariables(std::vector<std::string>{"Flux.Ur"});
    const auto& Fl_all = md->PackVariables(std::vector<std::string>{"Flux.Fl"});
    const auto& Fr_all = md->PackVariables(std::vector<std::string>{"Flux.Fr"});
    // I assume we should update cmax/cmin. Else we should use the old ones, so
    const auto& cmax = md->PackVariables(std::vector<std::string>{"Flux.cmax"});
    const auto& cmin = md->PackVariables(std::vector<std::string>{"Flux.cmin"});

    // TODO this does NOT necessarily leave Flux.xyz vars in a matching state with GetFlux
    // It will be filled according to m_u/cons_map, which does not contain B
    PackIndexMap cons_map, prims_map;
    std::vector<MetadataFlag> prims_flags = {
        Metadata::GetUserFlag("Primitive"), Metadata::Cell};
    std::vector<MetadataFlag> cons_flags = {Metadata::WithFluxes, Metadata::Cell};
    const auto& P_all = md->PackVariables(prims_flags, prims_map);
    const auto& U_all = md->PackVariablesAndFluxes(cons_flags, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    const int nvar = U_all.GetDim(4);
    // Okay if this is empty since we won't access it then
    const auto& Bf = md->PackVariables(std::vector<std::string>{"cons.fB"});

    // Parameters
    const auto& pars = packages.Get("Fluxes")->AllParams();
    const Real gam = packages.Get("GRMHD")->Param<Real>("gamma");
    const bool use_global = pars.Get<bool>("fofc_use_glf");
    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(packages);
    // Only fix faces if they exist
    const bool face_b = (Bf.GetDim(4) > 0 && pars.Get<bool>("fofc_consistent_face_b"));

    for (int dir = 1; dir <= ndim; dir++) { // TODO if(trivial_direction) etc
        const TE el = FaceOf(dir);
        const Loci loc = loc_of(dir);
        const IndexRange3 b = KDomain::GetRange(md, IndexDomain::interior, el, -1, 1);
        const IndexRange block = IndexRange{0, P_all.GetDim(5) - 1};
        pmb0->par_for("fofc_replacement", block.s, block.e, b.ks, b.ke, b.js, b.je, b.is,
            b.ie,
            KOKKOS_LAMBDA(const int& b, const int& k, const int& j, const int& i)
            {
                const auto& G = P_all.GetCoords(b);

                // Face i,j,k borders cell with same index and 1 left with
                // index:
                int kk = (dir == 3) ? k - 1 : k;
                int jj = (dir == 2) ? j - 1 : j;
                int ii = (dir == 1) ? i - 1 : i;
                // If either bordering cell is marked
                if (static_cast<int>(fofcflag(b, 0, k, j, i)) ||
                    static_cast<int>(
                        fofcflag(b, 0, kk, jj, ii))) { // TODO allow customizing

                    // "Reconstruct" left & right of this face: left is left
                    // cell, right is shared-index
                    PLOOP
                        Pl_all(b, ip, k, j, i) = P_all(b, ip, kk, jj, ii);
                    PLOOP
                        Pr_all(b, ip, k, j, i) = P_all(b, ip, k, j, i);
                    // Preserve the existing field at the face
                    if (face_b) {
                        Pl_all(b, m_p.B1 + dir - 1, k, j, i) =
                            Bf(b, el, 0, k, j, i) / G.gdet(loc, j, i);
                        Pr_all(b, m_p.B1 + dir - 1, k, j, i) =
                            Bf(b, el, 0, k, j, i) / G.gdet(loc, j, i);
                    }

                    FourVectors Dtmp;
                    // Left
                    GRMHD::calc_4vecs(G, Pl_all(b), m_p, k, j, i, loc, Dtmp);
                    Flux::prim_to_flux(G, Pl_all(b), m_p, Dtmp, emhd_params, eos, k, j, i,
                        0, Ul_all(b), m_u, loc);
                    Flux::prim_to_flux(G, Pl_all(b), m_p, Dtmp, emhd_params, eos, k, j, i,
                        dir, Fl_all(b), m_u, loc);
                    // Magnetosonic speeds
                    Real cmaxL, cminL;
                    Flux::vchar_global(G, Pl_all(b), m_p, Dtmp, eos, emhd_params, k, j, i,
                        loc, dir, cmaxL, cminL);
                    // Record speeds
                    cmax(b, dir - 1, k, j, i) = m::max(0., cmaxL);
                    cmin(b, dir - 1, k, j, i) = m::min(0., cminL);

                    // Right
                    GRMHD::calc_4vecs(G, Pr_all(b), m_p, k, j, i, loc, Dtmp);
                    Flux::prim_to_flux(G, Pr_all(b), m_p, Dtmp, emhd_params, eos, k, j, i,
                        0, Ur_all(b), m_u, loc);
                    Flux::prim_to_flux(G, Pr_all(b), m_p, Dtmp, emhd_params, eos, k, j, i,
                        dir, Fr_all(b), m_u, loc);
                    // Magnetosonic speeds
                    Real cmaxR, cminR;
                    Flux::vchar_global(G, Pr_all(b), m_p, Dtmp, eos, emhd_params, k, j, i,
                        loc, dir, cmaxR, cminR);
                    // Calculate cmax/min based on comparison with cached values
                    if (!use_global) {
                        cmax(b, dir - 1, k, j, i) =
                            m::max(cmax(b, dir - 1, k, j, i), cmaxR);
                        cmin(b, dir - 1, k, j, i) =
                            -m::min(cmin(b, dir - 1, k, j, i), cminR);
                    } else {
                        // This conveniently also reduces the timestep if
                        // necessary -- though, you should almost certainly set
                        // use_dt_light w/this
                        cmax(b, dir - 1, k, j, i) = 1.;
                        cmin(b, dir - 1, k, j, i) = 1.;
                    }

                    // Use LLF flux. Note we replace fluxes of all variables
                    // (including B!) This is for a consistent scheme, i.e. all
                    // cells FOFC == using DC+LLF
                    PLOOP
                        U_all(b).flux(dir, ip, k, j, i) =
                            llf(Fl_all(b, ip, k, j, i), Fr_all(b, ip, k, j, i),
                                cmax(b, dir - 1, k, j, i), cmin(b, dir - 1, k, j, i),
                                Ul_all(b, ip, k, j, i), Ur_all(b, ip, k, j, i));
                }
            });
    }

    return TaskStatus::complete;
}
