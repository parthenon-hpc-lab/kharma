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

    // Use values from floors package if it's enabled, otherwise any we've been asked to
    // apply
    const Floors::Prescription floors =
        pmb0->packages.Get("Floors")->Param<Floors::Prescription>("prescription");

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
            if (Inverter::failed(pflag(bl, 0, k, j, i)) || fflag(bl, 0, k, j, i) != 0. ||
                G.r(k, j, i) < fofc_radius) {
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
                    Flux::vchar(G, Pl_all(b), m_p, Dtmp, eos, emhd_params, k, j, i, loc,
                        dir, cmaxL, cminL);
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
                    Flux::vchar(G, Pr_all(b), m_p, Dtmp, eos, emhd_params, k, j, i, loc,
                        dir, cmaxR, cminR);
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

// We want a stupid user-settable power
// TODO obviously replace with something better
template<int power>
KOKKOS_FORCEINLINE_FUNCTION Real ipow(Real x)
{}
template<>
KOKKOS_FORCEINLINE_FUNCTION Real ipow<1>(Real x)
{
    return x;
}
template<>
KOKKOS_FORCEINLINE_FUNCTION Real ipow<2>(Real x)
{
    return x * x;
}

TaskStatus Flux::FOFC_PCP(MeshData<Real>* md, MeshData<Real>* guess, const Real dt)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    auto& packages = pmb0->packages;
    auto pmesh = md->GetMeshPointer();
    const int ndim = pmesh->ndim;

    auto& flpars = pmb0->packages.Get("Floors")->AllParams();
    const Floors::Prescription floors = flpars.Get<Floors::Prescription>("prescription");

    // Pick up flag. Optionally synced
    auto fofcflag = guess->PackVariables(std::vector<std::string>{"fofcflag"});

    // Temporary for amount to adjust
    auto alpha_norm =
        guess->PackVariables(std::vector<std::string>{"Flux.fofc_pcp_alpha"});
    // auto wsum = guess->PackVariables(std::vector<std::string>{"Flux.fofc_pcp_wsum"});

    // We want to update fluxes in md, based on prims from guess
    PackIndexMap cons_map, prims_map;
    std::vector<MetadataFlag> prims_flags = {
        Metadata::GetUserFlag("Primitive"), Metadata::Cell};
    std::vector<MetadataFlag> cons_flags = {Metadata::WithFluxes};
    const auto& P_all = guess->PackVariables(prims_flags, prims_map);
    const auto& U_all = md->PackVariablesAndFluxes(cons_flags, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    const int nvar = U_all.GetDim(4);

    const auto& B_Uf = guess->PackVariables(std::vector<std::string>{"cons.fB"});

    // Parameters
    const auto& pars = packages.Get("Fluxes")->AllParams();
    // const int chi = pars.Get<int>("fofc_pcp_chi");
    // const Real umin = pars.Get<Real>("fofc_pcp_umin");

    const IndexRange3 b = KDomain::GetRange(md, IndexDomain::interior);
    const IndexRange block = IndexRange{0, P_all.GetDim(5) - 1};

    // utilde -> w -> normalize -> alpha
    pmb0->par_for("fix_FOFC_PCP", block.s, block.e, b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &bl, const int &k, const int &j, const int &i)
        {
            const auto& G = U_all.GetCoords(bl);

            Real rhomin_geom, umin_geom;
            determine_geo_floors(
                G, P_all(bl), m_p, k, j, i, floors, rhomin_geom, umin_geom);
            const Real& umin = umin_geom; // Keep flexibility

            if (static_cast<int>(fofcflag(bl, 0, k, j, i)) &&
                (P_all(bl, m_p.UU, k, j, i) < umin)) { // ||
                // P_all(bl, m_p.RHO, k, j, i) < rhomin_geom)) {

                // Weights w from Balsara+
                Real wts[6]; // TODO(CEP) assign on creation?
                // TODO template to respect user chi? Also use istrivial()
                wts[0] = ipow<2>(m::max(P_all(bl, m_p.UU, k, j, i + 1) - umin, 0.));
                wts[1] = ipow<2>(m::max(P_all(bl, m_p.UU, k, j, i - 1) - umin, 0.));
                wts[2] = ndim > 1
                             ? ipow<2>(m::max(P_all(bl, m_p.UU, k, j + 1, i) - umin, 0.))
                             : 0.;
                wts[3] = ndim > 1
                             ? ipow<2>(m::max(P_all(bl, m_p.UU, k, j - 1, i) - umin, 0.))
                             : 0.;
                wts[4] = ndim > 2
                             ? ipow<2>(m::max(P_all(bl, m_p.UU, k + 1, j, i) - umin, 0.))
                             : 0.;
                wts[5] = ndim > 2
                             ? ipow<2>(m::max(P_all(bl, m_p.UU, k - 1, j, i) - umin, 0.))
                             : 0.;

                // Normalize
                Real wsum_local = 0.;
                for (int ii = 0; ii < 6; ii++) wsum_local += wts[ii];
                for (int ii = 0; ii < 6; ii++) wts[ii] /= wsum_local;

                // If we have available neighbors...
                if (wsum_local > 0.) {
                    // Preserve the sum for reweighting fluxes later
                    // wsum(bl, 0, k, j, i) = wsum_local;

                    // Compute bsq in proper frame
                    const Real uvec[NVEC] = {P_all(bl, m_p.U1, k, j, i),
                        P_all(bl, m_p.U2, k, j, i), P_all(bl, m_p.U3, k, j, i)};

                    // Central B components when calculated from face
                    Real B_fP[NVEC];
                    B_fP[V1] = (B_Uf(bl, F1, 0, k, j, i) / G.gdet(Loci::face1, j, i) +
                                   B_Uf(bl, F1, 0, k, j, i + 1) /
                                       G.gdet(Loci::face1, j, i + 1)) /
                               2;
                    B_fP[V2] =
                        (ndim > 1)
                            ? (B_Uf(bl, F2, 0, k, j, i) / G.gdet(Loci::face2, j, i) +
                                  B_Uf(bl, F2, 0, k, j + 1, i) /
                                      G.gdet(Loci::face2, j + 1, i)) /
                                  2
                            : B_Uf(bl, F2, 0, k, j, i) / G.gdet(Loci::face2, j, i);
                    B_fP[V3] =
                        (ndim > 2)
                            ? (B_Uf(bl, F3, 0, k, j, i) / G.gdet(Loci::face3, j, i) +
                                  B_Uf(bl, F3, 0, k + 1, j, i) /
                                      G.gdet(Loci::face3, j, i)) /
                                  2
                            : B_Uf(bl, F3, 0, k, j, i) / G.gdet(Loci::face3, j, i);

                    // Central B components updated from cells alone
                    const Real B_cP[NVEC] = {P_all(bl, m_p.B1, k, j, i),
                        P_all(bl, m_p.B2, k, j, i), P_all(bl, m_p.B3, k, j, i)};

                    // TODO Surely we can save on this with algebra
                    FourVectors Dtmp;
                    Real T[GR_DIM];
                    GRMHD::calc_4vecs(G, uvec, B_fP, k, j, i, Loci::center, Dtmp);
                    GRMHD::calc_tensor(0, 0, 0, Dtmp, 0, T);
                    const Real T0_face = T[0];
                    GRMHD::calc_4vecs(G, uvec, B_cP, k, j, i, Loci::center, Dtmp);
                    GRMHD::calc_tensor(0, 0, 0, Dtmp, 0, T);
                    const Real T0_cell = T[0];

                    // Mark separately to track
                    fofcflag(bl, 0, k, j, i) = (int)Flux::Correction::pcp;

                    // This is alpha/dt as is customary for fluxes
                    // If the magnetic field stress-energy component (T0_face) will be
                    // different than the PCP value (T0_cell), we need to adjust our
                    // energy to reality
                    const Real alpha = (T0_face - T0_cell);
                    // We additionally normalize by wsum so we don't have to hold onto it
                    alpha_norm(bl, 0, k, j, i) = alpha / wsum_local *
                                                 G.gdet(Loci::center, j, i) *
                                                 G.CellVolume(k, j, i) / dt;

                    // TODO bring back print on debug build?
                    // if (m::abs(alpha / T0_face) > 1e-3) {
                    //     printf("Total alpha %g (proportion %g)\n"
                    //         "First flux %g changed by %g\n", alpha, alpha / T0_face,
                    //         U_all(bl).flux(1, m_u.UU, k, j, i) * G.FaceArea<X1DIR>(k,
                    //         j, i + 1), wts[0] * alpha_norm);
                    // }
                }
            } else {
                alpha_norm(bl, 0, k, j, i) = 0.;
            }
        });

    // Now revise F
    for (int dir = 1; dir <= ndim; dir++) { // TODO if(trivial_direction) etc
        const TE el = FaceOf(dir);
        const IndexRange3 bf = KDomain::GetRange(md, IndexDomain::interior, el);
        pmb0->par_for("fix_FOFC_PCP_flux", block.s, block.e, bf.ks, bf.ke, bf.js, bf.je,
            bf.is, bf.ie,
            KOKKOS_LAMBDA (const int &bl, const int &k, const int &j, const int &i)
            {
                const auto& G = U_all.GetCoords(bl);

                // Face i,j,k borders cell with same index and 1 left with
                // index:
                int kk = (dir == 3) ? k - 1 : k;
                int jj = (dir == 2) ? j - 1 : j;
                int ii = (dir == 1) ? i - 1 : i;
                // If either bordering cell is marked (TODO test flag instead?)
                if (alpha_norm(bl, 0, k, j, i) != 0. ||
                    alpha_norm(bl, 0, kk, jj, ii) != 0.) {

                    Real rhomin_geom, umin_geom;
                    determine_geo_floors(
                        G, P_all(bl), m_p, k, j, i, floors, rhomin_geom, umin_geom);
                    const Real& umin = umin_geom; // Keep flexibility

                    // Flux corrections to satisfy neighbors of this face.
                    // This recalculates & must match some of the weights wts[N] for each
                    // cell Rightward flux from left/offset cell to our same-index cell,
                    // minus leftward flux from us to the offset/left cell
                    U_all(bl).flux(dir, m_u.UU, k, j, i) +=
                        (ipow<2>(m::max(P_all(bl, m_p.UU, kk, jj, ii) - umin, 0.)) *
                                alpha_norm(bl, 0, k, j, i) -
                            ipow<2>(m::max(P_all(bl, m_p.UU, k, j, i) - umin, 0.)) *
                                alpha_norm(bl, 0, kk, jj, ii)) /
                        G.FaceArea(dir, k, j, i);
                }
            });
    }

    return TaskStatus::complete;
}
