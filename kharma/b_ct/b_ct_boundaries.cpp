/*
 *  File: b_ct_boundaries.cpp
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
#include "b_ct.hpp"

#include "decs.hpp"
#include "domain.hpp"
#include "floors.hpp"
#include "grmhd.hpp"
#include "grmhd_functions.hpp"
#include "inverter.hpp"
#include "kharma.hpp"

void B_CT::ZeroBoundaryEMF(MeshBlockData<Real> *rc, IndexDomain domain, const VariablePack<Real> &emfpack, bool coarse)
{
    auto pmb = rc->GetBlockPointer();
    const BoundaryFace bface = KBoundaries::BoundaryFaceOf(domain);
    const std::string bname = KBoundaries::BoundaryName(bface);
    const int bdir = KBoundaries::BoundaryDirection(bface);
    const bool binner = KBoundaries::BoundaryIsInner(bface);
    // Select edges which lie on the domain face, zero only those
    for (auto &el : OrthogonalEdges(bdir)) {
        auto b = KDomain::GetBoundaryRange(rc, domain, el, coarse);
        int i_face = (binner) ? b.ie : b.is;
        int j_face = (binner) ? b.je : b.js;
        int k_face = (binner) ? b.ke : b.ks;
        IndexRange ib = (bdir == 1) ? IndexRange{i_face, i_face} : IndexRange{b.is, b.ie};
        IndexRange jb = (bdir == 2) ? IndexRange{j_face, j_face} : IndexRange{b.js, b.je};
        IndexRange kb = (bdir == 3) ? IndexRange{k_face, k_face} : IndexRange{b.ks, b.ke};
        pmb->par_for(
            "zero_EMF_" + bname, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                emfpack(el, 0, k, j, i) = 0;
            }
        );
    }
}

void B_CT::AverageBoundaryEMF(MeshBlockData<Real> *rc, IndexDomain domain, const VariablePack<Real> &emfpack, bool coarse)
{
    auto pmb = rc->GetBlockPointer();
    const BoundaryFace bface = KBoundaries::BoundaryFaceOf(domain);
    const std::string bname = KBoundaries::BoundaryName(bface);
    const int bdir = KBoundaries::BoundaryDirection(bface);
    const bool binner = KBoundaries::BoundaryIsInner(bface);
    const int ndim = KDomain::GetNDim(rc);

    for (auto &el : OrthogonalEdges(bdir)) {
        if (bdir == X2DIR && el == E3 && pmb->coords.coords.is_spherical()) {
            // X3 EMF must be zero *on* polar face, since edge size is 0
            IndexRange3 b = KDomain::GetBoundaryRange(rc, domain, el, coarse);
            pmb->par_for(
                "zero_polar_EMF3_" + bname, b.ks, b.ke, b.js, b.je, b.is, b.ie,
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    emfpack(el, 0, k, j, i) = 0;
                }
            );
        } else if (ndim < 3 && ((bdir == X2DIR && el == E1) || (bdir == X1DIR && el == E2))) {
            // In 2D, "averaging" should just mean not zeroing E1 on X2 or E2 on X1
            continue;
        } else {
            // Otherwise the EMF at `el` is *averaged* along its perpendicular direction
            IndexRange3 b = KDomain::GetRange(rc, domain, el, coarse);
            IndexRange3 bi = KDomain::GetRange(rc, IndexDomain::interior, el, coarse);
            // Calculate face index and outer sum index
            int cface, inner_dir;
            IndexRange outer;
            if (bdir == X1DIR) {
                cface = (binner) ? bi.is : bi.ie;
                if (el == E2) {
                    outer = {b.js, b.je};
                    inner_dir = X3DIR;
                } else {
                    outer = {b.ks, b.ke};
                    inner_dir = X2DIR;
                }
            } else if (bdir == X2DIR) {
                cface = (binner) ? bi.js : bi.je;
                if (el == E1) { // COMMON CASE
                    outer = {b.is, b.ie};
                    inner_dir = X3DIR;
                } else {
                    outer = {b.ks, b.ke};
                    inner_dir = X1DIR;
                }
            } else {
                cface = (binner) ? bi.ks : bi.ke;
                if (el == E1) {
                    outer = {b.is, b.ie};
                    inner_dir = X2DIR;
                } else {
                    outer = {b.js, b.je};
                    inner_dir = X1DIR;
                }
            }
            parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "reduce_EMF_" + bname, pmb->exec_space,
                0, 1, outer.s, outer.e,
                KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& o) {
                    double emf_sum = 0.;
                    Kokkos::Sum<double> sum_reducer(emf_sum);

                    // One of these won't be used
                    // Outer loop is along face corresponding to our element
                    const int ii = (el == E1) ? o : cface;
                    const int jj = (el == E2) ? o : cface;
                    const int kk = (el == E3) ? o : cface;

                    // Sum the non-ghost fluxes in our desired averaging direction
                    int len;
                    if (inner_dir == X1DIR) {
                        len = bi.ie - bi.is;
                        parthenon::par_reduce_inner(member, bi.is, bi.ie - 1,
                            [&](const int& i, double& local_result) {
                                local_result += emfpack(el, 0, kk, jj, i);
                            }
                        , sum_reducer);
                    } else if (inner_dir == X2DIR) {
                        len = bi.je - bi.js;
                        parthenon::par_reduce_inner(member, bi.js, bi.je - 1,
                            [&](const int& j, double& local_result) {
                                local_result += emfpack(el, 0, kk, j, ii);
                            }
                        , sum_reducer);
                    } else {
                        len = bi.ke - bi.ks;
                        parthenon::par_reduce_inner(member, bi.ks, bi.ke - 1,
                            [&](const int& k, double& local_result) {
                                local_result += emfpack(el, 0, k, jj, ii);
                            }
                        , sum_reducer);
                    }
                    member.team_barrier();

                    // Calculate the average
                    const double emf_av = emf_sum / len;

                    // Set all EMFs identically (even ghosts, to keep divB)
                    if (inner_dir == X1DIR) {
                        parthenon::par_for_inner(member, b.is, b.ie,
                            [&](const int& i) {
                                emfpack(el, 0, kk, jj, i) = emf_av;
                            }
                        );
                    } else if (inner_dir == X2DIR) {
                        parthenon::par_for_inner(member, b.js, b.je,
                            [&](const int& j) {
                                emfpack(el, 0, kk, j, ii) = emf_av;
                            }
                        );
                    } else {
                        parthenon::par_for_inner(member, b.ks, b.ke,
                            [&](const int& k) {
                                emfpack(el, 0, k, jj, ii) = emf_av;
                            }
                        );
                    }
                }
            );
        }
    }
}

void B_CT::DestructiveBoundaryClean(MeshBlockData<Real> *rc, IndexDomain domain, const VariablePack<Real> &fpack, bool coarse)
{
    // Set XN faces to keep clean divergence at outflow XN boundary
    // Feels wrong to work backward from no divergence, but they are just outflow...
    auto pmb = rc->GetBlockPointer();
    const BoundaryFace bface = KBoundaries::BoundaryFaceOf(domain);
    const std::string bname = KBoundaries::BoundaryName(bface);
    const int bdir = KBoundaries::BoundaryDirection(bface);
    const bool binner = KBoundaries::BoundaryIsInner(bface);
    const TopologicalElement face = FaceOf(bdir);
    // Correct last domain face, too
    auto b = KDomain::GetRange(rc, domain, face, (binner) ? 0 : -1, (binner) ? 1 : 0, coarse);
    // Need the coordinates for this boundary, uniquely
    auto G = pmb->coords;
    const int ndim = pmb->pmy_mesh->ndim;
    if (bdir == X1DIR) {
        const int i_face = (binner) ? b.ie : b.is;
        for (int iadd = 0; iadd <= (b.ie - b.is); iadd++) {
            const int i = (binner) ? i_face - iadd : i_face + iadd;
            const int last_rank_f  = (binner) ? i + 1 : i - 1;
            const int last_rank_c  = (binner) ? i     : i - 1;
            const int outward_sign = (binner) ? -1.   : 1.;
            pmb->par_for(
                "correct_face_vector_" + bname, b.ks, b.ke, b.js, b.je, i, i,
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    // Other faces have been updated, just need to clean divergence
                    // Subtract off their contributions to find ours. Note our partner face contributes differently,
                    // depending on whether we're the i+1 "outward" face, or the i "innward" face
                    Real new_face = - (-outward_sign) * fpack(F1, 0, k, j, last_rank_f) * G.Volume<F1>(k, j, last_rank_f)
                                    - (fpack(F2, 0, k, j + 1, last_rank_c) * G.Volume<F2>(k, j + 1, last_rank_c)
                                        - fpack(F2, 0, k, j, last_rank_c) * G.Volume<F2>(k, j, last_rank_c));
                    if (ndim > 2)
                        new_face -= fpack(F3, 0, k + 1, j, last_rank_c) * G.Volume<F3>(k + 1, j, last_rank_c)
                                    - fpack(F3, 0, k, j, last_rank_c) * G.Volume<F3>(k, j, last_rank_c);

                    fpack(F1, 0, k, j, i) = outward_sign * new_face / G.Volume<F1>(k, j, i);
                }
            );
        }
    } else if (bdir == X2DIR) {
        const int j_face = (binner) ? b.je : b.js;
        for (int jadd = 0; jadd <= (b.je - b.js); jadd++) {
            const int j = (binner) ? j_face - jadd : j_face + jadd;
            const int last_rank_f  = (binner) ? j + 1 : j - 1;
            const int last_rank_c  = (binner) ? j     : j - 1;
            const int outward_sign = (binner) ? -1.   : 1.;
            pmb->par_for(
                "correct_face_vector_" + bname, b.ks, b.ke, j, j, b.is, b.ie,
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    Real new_face = - (-outward_sign) * fpack(F2, 0, k, last_rank_f, i) * G.Volume<F2>(k, last_rank_f, i)
                                    - (fpack(F1, 0, k, last_rank_c, i + 1) * G.Volume<F1>(k, last_rank_c, i + 1)
                                        - fpack(F1, 0, k, last_rank_c, i) * G.Volume<F1>(k, last_rank_c, i));
                    if (ndim > 2)
                        new_face -= fpack(F3, 0, k + 1, last_rank_c, i) * G.Volume<F3>(k + 1, last_rank_c, i)
                                    - fpack(F3, 0, k, last_rank_c, i) * G.Volume<F3>(k, j, last_rank_c, i);

                    fpack(F2, 0, k, j, i) = outward_sign * new_face / G.Volume<F2>(k, j, i);
                }
            );
        }
    } else {
        const int k_face = (binner) ? b.ie : b.is;
        for (int kadd = 0; kadd <= (b.ie - b.is); kadd++) {
            const int k = (binner) ? k_face - kadd : k_face + kadd;
            const int last_rank_f  = (binner) ? k + 1 : k - 1;
            const int last_rank_c  = (binner) ? k     : k - 1;
            const int outward_sign = (binner) ? -1.   : 1.;
            pmb->par_for(
                "correct_face_vector_" + bname, k, k, b.js, b.je, b.is, b.ie,
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    Real new_face = - (-outward_sign) * fpack(F3, 0, last_rank_f, j, i) * G.Volume<F3>(last_rank_f, j, i)
                                    - (fpack(F1, 0, last_rank_c, j, i + 1) * G.Volume<F1>(last_rank_c, j, i + 1)
                                        - fpack(F1, 0, last_rank_c, j, i) * G.Volume<F1>(last_rank_c, j, i))
                                    - (fpack(F2, 0, last_rank_c, j + 1, i) * G.Volume<F2>(last_rank_c, j + 1, i)
                                        - fpack(F2, 0, last_rank_c, j, i) * G.Volume<F2>(last_rank_c, j, i));

                    fpack(F3, 0, k, j, i) = outward_sign * new_face / G.Volume<F3>(k, j, i);
                }
            );
        }
    }
}

void B_CT::ReconnectBoundaryB3(MeshBlockData<Real> *rc, IndexDomain domain, const VariablePack<Real> &fpack, bool coarse)
{
    // We're also sometimes called on coarse buffers with or without AMR.
    // Use of transmitting polar conditions when coarse buffers matter (e.g., refinement
    // boundary touching the pole) is UNSUPPORTED
    if (coarse) return;

    // Pull boundary properties
    auto pmb = rc->GetBlockPointer();
    const BoundaryFace bface = KBoundaries::BoundaryFaceOf(domain);
    const bool binner = KBoundaries::BoundaryIsInner(bface);
    const int bdir = KBoundaries::BoundaryDirection(bface);
    const auto bname = KBoundaries::BoundaryName(bface);

    // Pull cell-centered values, as we need to update fluid primitives
    // TODO standardize on passing Packs or Datas...
    PackIndexMap prims_map, cons_map;
    auto P = rc->PackVariables({Metadata::GetUserFlag("Primitive"), Metadata::Cell}, prims_map);
    auto U = rc->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved, Metadata::Cell}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    const auto& G = pmb->coords;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    const Floors::Prescription floors = pmb->packages.Get("Floors")->Param<Floors::Prescription>("prescription");
    // Don't be fooled, this function does *not* support/preserve EMHD values
    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb->packages);

    // Subtract the average B3 as "reconnection"
    IndexRange3 b = KDomain::GetRange(rc, domain, F3, coarse);
    IndexRange3 bi = KDomain::GetRange(rc, IndexDomain::interior, F3, coarse);
    const int jf = (binner) ? bi.js : bi.je; // j index of last zone next to pole
    parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "reduce_B3_" + bname, pmb->exec_space,
        0, 1, 0, fpack.GetDim(4)-1, b.is, b.ie,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int &v, const int& i) {
            // Sum the first rank of B3
            double B3_sum = 0.;
            Kokkos::Sum<double> sum_reducer(B3_sum);
            parthenon::par_reduce_inner(member, bi.ks, bi.ke - 1,
                [&](const int& k, double& local_result) {
                    local_result += fpack(F3, v, k, jf, i);
                }
            , sum_reducer);
            member.team_barrier();

            // Calculate the average and modify all B3 identically
            // This will preserve their differences->divergence
            const double B3_av = B3_sum / (bi.ke - bi.ks);
            parthenon::par_for_inner(member, b.ks, b.ke,
                [&](const int& k) {
                    fpack(F3, v, k, jf, i) -= B3_av;
                }
            );
            member.team_barrier();

            // Update cell-centered conserved & primitive B, and cell primitive fluid variables, in the zones we touched
            parthenon::par_for_inner(member, b.ks, b.ke-1, // iterate over all *cells* k
                [&](const int& k) {
                    P(m_p.B3, k, jf, i) =  (fpack(F3, 0, k, jf, i) / G.gdet(Loci::face3, jf, i)
                                        + fpack(F3, 0, k + 1, jf, i) / G.gdet(Loci::face3, jf, i)) / 2;
                    U(m_u.B3, k, jf, i) = P(m_p.B3, k, jf, i) * G.gdet(Loci::center, jf, i);

                    // Recover primitive GRMHD variables from our modified U
                    Inverter::u_to_p<Inverter::Type::kastaun>(G, U, m_u, gam, k, jf, i, P, m_p, Loci::center,
                                                              floors, 8, 1e-8);
                    // Floor them
                    int fflag = Floors::apply_geo_floors(G, P, m_p, gam, k, jf, i, floors, floors, Loci::center);
                    // Recalculate U on anything we floored
                    if (fflag)
                        GRMHD::p_to_u(G, P, m_p, gam, k, jf, i, U, m_u, Loci::center);

                }
            );
        }
    );
}
