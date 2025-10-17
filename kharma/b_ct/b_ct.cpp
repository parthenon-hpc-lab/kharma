/*
 *  File: b_ct.cpp
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
#include "grmhd.hpp"
#include "grmhd_functions.hpp"
#include "kharma.hpp"

#include <parthenon/parthenon.hpp>
#include <prolong_restrict/pr_ops.hpp>
#include <stdexcept>

using namespace parthenon;
using parthenon::refinement_ops::ProlongateSharedMinMod;
using parthenon::refinement_ops::RestrictAverage;
using parthenon::refinement_ops::ProlongateInternalAverage;

std::shared_ptr<KHARMAPackage> B_CT::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("B_CT");
    Params &params = pkg->AllParams();

    // Diagnostic flags

    // Default to stopping execution when divB is large, which generally indicates something
    // has gone wrong.  As always, can be disabled by the brave.
    bool kill_on_large_divb = pin->GetOrAddBoolean("b_field", "kill_on_large_divb", true);
    params.Add("kill_on_large_divb", kill_on_large_divb);
    Real kill_on_divb_over = pin->GetOrAddReal("b_field", "kill_on_divb_over", 1.e-3);
    params.Add("kill_on_divb_over", kill_on_divb_over);

    // TODO gs05_alpha, LDZ04 UCT1, LDZ07 UCT2?
    std::vector<std::string> ct_scheme_options = {"bs99", "gs05_0", "gs05_c", "sg07"};
    std::string ct_scheme = pin->GetOrAddString("b_field", "ct_scheme", "gs05_c", ct_scheme_options);
    params.Add("ct_scheme", ct_scheme);

    // Use the default Parthenon prolongation operator, rather than the divergence-preserving one
    // This relies entirely on the EMF communication for preserving the divergence
    bool lazy_prolongation = pin->GetOrAddBoolean("b_field", "lazy_prolongation", false);
    // Need to preserve divergence if you refine/derefine during sim i.e. AMR
    if (lazy_prolongation && pin->GetString("parthenon/mesh", "refinement") == "adaptive")
        throw std::runtime_error("Cannot use non-divergence-preserving prolongation in AMR!");

    // TODO don't set this unless we're reconnecting at boundaries (can't just check, we load Boundaries pkg later)
    int reconnection_outer_buffer = pin->GetOrAddBoolean("b_field", "reconnection_outer_buffer", 5);
    params.Add("reconnection_outer_buffer", reconnection_outer_buffer);

    // FIELDS

    // Flags for B fields on faces.
    // We don't mark these as "Conserved" else they'd be bundled
    // with all the cell vars in a bunch of places we don't want
    // Also note we *always* sync B field conserved var
    std::vector<MetadataFlag> flags_cons_f = {Metadata::Real, Metadata::Face, Metadata::Independent, Metadata::Restart, Metadata::FillGhost,
                                              Metadata::GetUserFlag("Explicit"), Metadata::GetUserFlag("SplitVector")};
    auto m = Metadata(flags_cons_f);
    if (!lazy_prolongation)
        m.RegisterRefinementOps<ProlongateSharedMinMod, RestrictAverage, ProlongateInternalOlivares>();
    else
        m.RegisterRefinementOps<ProlongateSharedMinMod, RestrictAverage, ProlongateInternalAverage>();
    pkg->AddField("cons.fB", m);

    // Cell-centered versions.  Needed for BS, not for other schemes.
    // Probably will want to keep primitives for e.g. correct PtoU of MHD vars, but cons maybe can go
    std::vector<MetadataFlag> flags_prim = {Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::GetUserFlag("Primitive"),
                                            Metadata::GetUserFlag("MHD"), Metadata::GetUserFlag("Explicit"), Metadata::Vector};
    std::vector<MetadataFlag> flags_cons = {Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::Conserved, Metadata::WithFluxes,
                                            Metadata::GetUserFlag("MHD"), Metadata::GetUserFlag("Explicit"), Metadata::Vector};

    // KHARMA now (vaguely) supports restarting from dump (.phdf) files.
    // To do so we need to read cell-centered primitive B and interpolate to faces, as we do with iharm3d restart files
    if (pin->GetOrAddBoolean("b_field", "restart_from_prims", false)) {
        flags_prim.push_back(Metadata::Restart);
    }

    std::vector<int> s_vector({NVEC});
    m = Metadata(flags_prim, s_vector);
    pkg->AddField("prims.B", m);
    m = Metadata(flags_cons, s_vector);
    pkg->AddField("cons.B", m);

    // EMF on edges.
    std::vector<MetadataFlag> flags_emf = {Metadata::Real, Metadata::Edge, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost};
    m = Metadata(flags_emf);
    pkg->AddField("B_CT.emf", m);

    if (ct_scheme != "bs99") {
        std::vector<MetadataFlag> flags_emf_c = {Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy};
        m = Metadata(flags_emf_c, s_vector);
        pkg->AddField("B_CT.cemf", m);
    }

    // CALLBACKS

    // We implement a source term replacement, rather than addition,
    // but same difference really
    pkg->AddSource = B_CT::AddSource;

    // Also ensure that prims get filled, both during step and on boundaries
    //pkg->MeshUtoP = B_CT::MeshUtoP;
    pkg->BlockUtoP = B_CT::BlockUtoP;
    pkg->BoundaryUtoP = B_CT::BlockUtoP;

    // Register the other callbacks
    pkg->PostStepDiagnosticsMesh = B_CT::PostStepDiagnostics;

    // The definition of MaxDivB we care about actually changes per-transport,
    // so calculating it is handled by the transport package
    // We'd only ever need to declare or calculate divB for output (getting the max is independent)
    if (KHARMA::FieldIsOutput(pin, "divB")) {
        pkg->BlockUserWorkBeforeOutput = B_CT::FillOutput;
        m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
        pkg->AddField("divB", m);
    }

    // List (vector) of HistoryOutputVars that will all be enrolled as output variables
    parthenon::HstVar_list hst_vars = {};
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, B_CT::MaxDivB, "MaxDivB"));
    // Event horizon magnetization.  Might be the same or different for different representations?
    if (pin->GetBoolean("coordinates", "domain_intersects_eh")) {
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAt0<Reductions::Var::phi>, "Phi_0"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAtEH<Reductions::Var::phi>, "Phi_EH"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAt5M<Reductions::Var::phi>, "Phi_5M"));
    }
    // add callbacks for HST output to the Params struct, identified by the `hist_param_key`
    pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

    return pkg;
}

TaskStatus B_CT::MeshUtoP(MeshData<Real> *md, IndexDomain domain, bool coarse)
{
    // TODO later
    for (int i=0; i < md->NumBlocks(); i++)
        B_CT::BlockUtoP(md->GetBlockData(i).get(), domain, coarse);
    return TaskStatus::complete;
}

TaskStatus B_CT::BlockUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();
    const int ndim = pmb->pmy_mesh->ndim;
    auto B_Uf = rc->PackVariables(std::vector<std::string>{"cons.fB"});
    auto B_U = rc->PackVariables(std::vector<std::string>{"cons.B"});
    auto B_P = rc->PackVariables(std::vector<std::string>{"prims.B"});
    const auto& G = pmb->coords;
    // Return if we're not syncing U & P at all (e.g. edges)
    if (B_Uf.GetDim(4) == 0) return TaskStatus::complete;

    // We need one cell inside of the domain, since it will be updated by the domain-face field
    // this oversteps "interior" by a zone
    IndexRange3 bct;
    if (domain == IndexDomain::interior || domain == IndexDomain::entire) {
        bct = KDomain::GetRange(rc, domain, coarse);
    } else {
        const BoundaryFace bface = KBoundaries::BoundaryFaceOf(domain);
        const bool binner = KBoundaries::BoundaryIsInner(bface);
        bct = KDomain::GetRange(rc, domain, (binner) ? 0 : -1, (binner) ? 1 : 0, coarse);
    }
    const IndexRange3 bc = bct;

    // Average the primitive vals to zone centers
    pmb->par_for("UtoP_B_center", bc.ks, bc.ke, bc.js, bc.je, bc.is, bc.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            B_P(V1, k, j, i) = (B_Uf(F1, 0, k, j, i) / G.gdet(Loci::face1, j, i)
                              + B_Uf(F1, 0, k, j, i + 1) / G.gdet(Loci::face1, j, i + 1)) / 2;
            B_P(V2, k, j, i) = (ndim > 1) ? (B_Uf(F2, 0, k, j, i) / G.gdet(Loci::face2, j, i)
                                           + B_Uf(F2, 0, k, j + 1, i) / G.gdet(Loci::face2, j + 1, i)) / 2
                                           : B_Uf(F2, 0, k, j, i) / G.gdet(Loci::face2, j, i);
            B_P(V3, k, j, i) = (ndim > 2) ? (B_Uf(F3, 0, k, j, i) / G.gdet(Loci::face3, j, i)
                                           + B_Uf(F3, 0, k + 1, j, i) / G.gdet(Loci::face3, j, i)) / 2
                                           : B_Uf(F3, 0, k, j, i) / G.gdet(Loci::face3, j, i);
        }
    );
    // Recover conserved B at centers
    pmb->par_for("UtoP_B_centerPtoU", 0, NVEC-1, bc.ks, bc.ke, bc.js, bc.je, bc.is, bc.ie,
        KOKKOS_LAMBDA (const int &v, const int &k, const int &j, const int &i) {
            B_U(v, k, j, i) = B_P(v, k, j, i) * G.gdet(Loci::center, j, i);
        }
    );

    return TaskStatus::complete;
}

TaskStatus B_CT::DangerousPtoU(MeshData<Real> *md, IndexDomain domain, bool coarse)
{
    auto B_Uf = md->PackVariables(std::vector<std::string>{"cons.fB"});
    auto B_U = md->PackVariables(std::vector<std::string>{"cons.B"});
    auto B_P = md->PackVariables(std::vector<std::string>{"prims.B"});

    // Figure out indices
    const IndexRange block = IndexRange{0, B_Uf.GetDim(5)-1};

    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Average the primitive vals to faces and multiply by gdet
    const IndexRange3 bf1 = KDomain::GetRange(md, domain, F1, coarse);
    pmb0->par_for("PtoU_B_F1", block.s, block.e, bf1.ks, bf1.ke, bf1.js, bf1.je, bf1.is, bf1.ie,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i) {
            const auto& G = B_Uf.GetCoords(b);
            B_Uf(b, F1, 0, k, j, i) = G.gdet(Loci::face1, j, i) * (B_P(b, V1, k, j, i-1) + B_P(b, V1, k, j, i)) / 2;
        }
    );
    const IndexRange3 bf2 = KDomain::GetRange(md, domain, F2, coarse);
    pmb0->par_for("PtoU_B_F2", block.s, block.e, bf2.ks, bf2.ke, bf2.js, bf2.je, bf2.is, bf2.ie,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i) {
            const auto& G = B_Uf.GetCoords(b);
            B_Uf(b, F2, 0, k, j, i) = G.gdet(Loci::face2, j, i) * (B_P(b, V2, k, j-1, i) + B_P(b, V2, k, j, i)) / 2;
        }
    );
    const IndexRange3 bf3 = KDomain::GetRange(md, domain, F3, coarse);
    pmb0->par_for("PtoU_B_F3", block.s, block.e, bf3.ks, bf3.ke, bf3.js, bf3.je, bf3.is, bf3.ie,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i) {
            const auto& G = B_Uf.GetCoords(b);
            B_Uf(b, F3, 0, k, j, i) = G.gdet(Loci::face3, j, i) * (B_P(b, V3, k-1, j, i) + B_P(b, V3, k, j, i)) / 2;
        }
    );

    // Make sure B on poles is still zero, even though we've interpolated
    if (pmb0->coords.coords.is_spherical()) {
        for (int i=0; i < md->GetMeshPointer()->GetNumMeshBlocksThisRank(); i++) {
            auto rc = md->GetBlockData(i);
            auto pmb = rc->GetBlockPointer();
            auto dB_block = rc->PackVariables(std::vector<std::string>{"dB"});
            if (KBoundaries::IsPhysicalBoundary(pmb, BoundaryFace::inner_x2)) {
                pmb->par_for("dB_boundary", bf2.ks, bf2.ke, bf2.is, bf2.ie,
                    KOKKOS_LAMBDA (const int &k, const int &i) {
                        dB_block(F2, 0, k, bf2.js, i) = 0.;
                    }
                );
            }
            if (KBoundaries::IsPhysicalBoundary(pmb, BoundaryFace::outer_x2)) {
                pmb->par_for("dB_boundary", bf2.ks, bf2.ke, bf2.is, bf2.ie,
                    KOKKOS_LAMBDA (const int &k, const int &i) {
                        dB_block(F2, 0, k, bf2.je, i) = 0.;
                    }
                );
            }
        }
    }

    // Also recover conserved B at centers, just in case
    const IndexRange3 bc = KDomain::GetRange(md, domain, CC, coarse);
    pmb0->par_for("UtoP_B_centerPtoU", block.s, block.e, 0, NVEC-1, bc.ks, bc.ke, bc.js, bc.je, bc.is, bc.ie,
        KOKKOS_LAMBDA (const int &b, const int &v, const int &k, const int &j, const int &i) {
            const auto& G = B_U.GetCoords(b);
            B_U(b, v, k, j, i) = B_P(b, v, k, j, i) * G.gdet(Loci::center, j, i);
        }
    );

    return TaskStatus::complete;
}

TaskStatus B_CT::CalculateEMF(MeshData<Real> *md)
{
    auto pmesh = md->GetMeshPointer();
    const int ndim = pmesh->ndim;
    if (ndim < 2) throw std::runtime_error("Face-centered constrained transport does not support 1D! Use `flux_ct` instead.");

    // EMF temporary
    auto& emf_pack = md->PackVariables(std::vector<std::string>{"B_CT.emf"});

    // Figure out indices
    const IndexRange3 b = KDomain::GetRange(md, IndexDomain::interior, 0, 0);
    const IndexRange3 b1 = KDomain::GetRange(md, IndexDomain::interior, 0, 1);
    const IndexRange block = IndexRange{0, emf_pack.GetDim(5)-1};

    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // Calculate circulation by averaging fluxes
    // This is the base of most other schemes, which make corrections
    // It is the entirety of B&S '99
    auto& B_U = md->PackVariablesAndFluxes(std::vector<std::string>{"cons.B"});
    pmb0->par_for("B_CT_emf_BS", block.s, block.e, b1.ks, b1.ke, b1.js, b1.je, b1.is, b1.ie,
        KOKKOS_LAMBDA (const int &bl, const int &k, const int &j, const int &i) {
            // The basic EMF per length along edges is the B field flux
            // We use this form rather than multiply by edge length here,
            // since the default restriction op averages values
            const auto& G = B_U.GetCoords(bl);
            if (ndim > 2) {
                emf_pack(bl, E1, 0, k, j, i) =
                    0.25*(-B_U(bl).flux(X2DIR, V3, k - 1, j, i) - B_U(bl).flux(X2DIR, V3, k, j, i)
                         + B_U(bl).flux(X3DIR, V2, k, j - 1, i) + B_U(bl).flux(X3DIR, V2, k, j, i));
                emf_pack(bl, E2, 0, k, j, i) =
                    0.25*(-B_U(bl).flux(X3DIR, V1, k, j, i - 1) - B_U(bl).flux(X3DIR, V1, k, j, i)
                         + B_U(bl).flux(X1DIR, V3, k - 1, j, i) + B_U(bl).flux(X1DIR, V3, k, j, i));
                emf_pack(bl, E3, 0, k, j, i) =
                    0.25*(-B_U(bl).flux(X1DIR, V2, k, j - 1, i) - B_U(bl).flux(X1DIR, V2, k, j, i)
                        + B_U(bl).flux(X2DIR, V1, k, j, i - 1)  + B_U(bl).flux(X2DIR, V1, k, j, i));
            } else if (ndim > 1) {
                emf_pack(bl, E1, 0, k, j, i) = -B_U(bl).flux(X2DIR, V3, k, j, i);
                emf_pack(bl, E2, 0, k, j, i) =  B_U(bl).flux(X1DIR, V3, k, j, i);
                emf_pack(bl, E3, 0, k, j, i) =
                    0.25*(-B_U(bl).flux(X1DIR, V2, k, j - 1, i) - B_U(bl).flux(X1DIR, V2, k, j, i)
                         + B_U(bl).flux(X2DIR, V1, k, j, i - 1) + B_U(bl).flux(X2DIR, V1, k, j, i));
            } else {
                emf_pack(bl, E1, 0, k, j, i) = 0;
                emf_pack(bl, E2, 0, k, j, i) =  B_U(bl).flux(X1DIR, V3, k, j, i);
                emf_pack(bl, E3, 0, k, j, i) = -B_U(bl).flux(X1DIR, V2, k, j, i);
            }
        }
    );
    // All corrections require/are only necessary for 2D+
    if (ndim < 2) return TaskStatus::complete;

    std::string scheme = pmesh->packages.Get("B_CT")->Param<std::string>("ct_scheme");
    if (scheme != "bs99") {
        // Additional terms for Stone & Gardiner '09
        // Caclulate the EMF at zone centers with primitive B, U1-3
        PackIndexMap prims_map;
        auto& P = md->PackVariables(std::vector<std::string>{"prims.uvec", "prims.B"}, prims_map);
        const VarMap m_p(prims_map, false);
        auto& emfc = md->PackVariables(std::vector<std::string>{"B_CT.cemf"});
        // Need this over whole domain to have halo around EMF caclulation
        const IndexRange3 be = KDomain::GetRange(md, IndexDomain::entire);
        pmb0->par_for("B_CT_emfc", block.s, block.e, be.ks, be.ke, be.js, be.je, be.is, be.ie,
            KOKKOS_LAMBDA (const int &bl, const int &k, const int &j, const int &i) {
                const auto& G = P.GetCoords(bl);
                Real gdet = G.gdet(Loci::center, j, i);

                // Get the 4vecs
                FourVectors D;
                GRMHD::calc_4vecs(G, P(bl), m_p, k, j, i, Loci::center, D);

                // Calculate cell-center EMF w/v x B
                emfc(bl, V1, k, j, i) = (D.bcon[2]*D.ucon[3] - D.bcon[3]*D.ucon[2]) * gdet;
                emfc(bl, V2, k, j, i) = (D.bcon[3]*D.ucon[1] - D.bcon[1]*D.ucon[3]) * gdet;
                emfc(bl, V3, k, j, i) = (D.bcon[1]*D.ucon[2] - D.bcon[2]*D.ucon[1]) * gdet;
            }
        );

        if (scheme == "gs05_0") {
            pmb0->par_for("B_CT_emf_GS05_0", block.s, block.e, b1.ks, b1.ke, b1.js, b1.je, b1.is, b1.ie,
                KOKKOS_LAMBDA (const int &bl, const int &k, const int &j, const int &i) {
                    const auto& G = emfc.GetCoords(bl);
                    // Just subtract centered emf from twice the face version
                    // More stable for planar flows even without anything fancy
                    if (ndim > 2) {
                        emf_pack(bl, E1, 0, k, j, i) = 2 * emf_pack(bl, E1, 0, k, j, i)
                            - 0.25*(emfc(bl, V1, k, j, i)      + emfc(bl, V1, k, j - 1, i)
                                  + emfc(bl, V1, k, j - 1, i)  + emfc(bl, V1, k - 1, j - 1, i));
                        emf_pack(bl, E2, 0, k, j, i) = 2 * emf_pack(bl, E2, 0, k, j, i)
                            - 0.25*(emfc(bl, V2, k, j, i)      + emfc(bl, V2, k, j, i - 1)
                                  + emfc(bl, V2, k - 1, j, i)  + emfc(bl, V2, k - 1, j, i - 1));
                    }
                    emf_pack(bl, E3, 0, k, j, i) = 2 * emf_pack(bl, E3, 0, k, j, i)
                        - 0.25*(emfc(bl, V3, k, j, i)     + emfc(bl, V3, k, j, i - 1)
                              + emfc(bl, V3, k, j - 1, i) + emfc(bl, V3, k, j - 1, i - 1));
                }
            );
        } else if (scheme == "gs05_c" || scheme == "sg07") {
            auto& rho = md->PackVariablesAndFluxes(std::vector<std::string>{"cons.rho"});
            pmb0->par_for("B_CT_emf_SG07", block.s, block.e, b1.ks, b1.ke, b1.js, b1.je, b1.is, b1.ie,
                KOKKOS_LAMBDA (const int &bl, const int &k, const int &j, const int &i) {
                    // Following adapted closely from AthenaK, including clever use of the mass flux for the
                    // sign of the contact mode.
                    if (ndim > 2) {
                        // Integrate EMF to the corner using GS07 i.e. GS05 E^c upwinding
                        Real e1_l3 = (rho(bl).flux(X2DIR, 0, k-1, j, i) >= 0.0) ?
                                    B_U(bl).flux(X3DIR, V2, k, j-1, i) - emfc(bl, V1, k-1, j-1, i) :
                                    B_U(bl).flux(X3DIR, V2, k, j  , i) - emfc(bl, V1, k-1, j  , i);
                        Real e1_r3 = (rho(bl).flux(X2DIR, 0, k  , j, i  ) >= 0.0) ?
                                    B_U(bl).flux(X3DIR, V2, k, j-1, i) - emfc(bl, V1, k  , j-1, i) :
                                    B_U(bl).flux(X3DIR, V2, k, j  , i) - emfc(bl, V1, k  , j  , i);
                        Real e1_l2 = (rho(bl).flux(X3DIR, 0, k, j-1, i) >= 0.0) ?
                                    -B_U(bl).flux(X2DIR, V3, k-1, j, i) - emfc(bl, V1, k-1, j-1, i) :
                                    -B_U(bl).flux(X2DIR, V3, k  , j, i) - emfc(bl, V1, k  , j-1, i);
                        Real e1_r2 = (rho(bl).flux(X3DIR, 0, k, j  , i) >= 0.0) ?
                                    -B_U(bl).flux(X2DIR, V3, k-1, j, i) - emfc(bl, V1, k-1, j  , i) :
                                    -B_U(bl).flux(X2DIR, V3, k  , j, i) - emfc(bl, V1, k  , j  , i);
                        emf_pack(bl, E1, 0, k, j, i) += 0.25*(e1_l3 + e1_r3 + e1_l2 + e1_r2);

                        Real e2_l3 = (rho(bl).flux(X1DIR, 0, k-1, j, i) >= 0.0) ?
                                    -B_U(bl).flux(X3DIR, V1, k, j, i-1) - emfc(bl, V2, k-1, j, i-1) :
                                    -B_U(bl).flux(X3DIR, V1, k, j, i  ) - emfc(bl, V2, k-1, j, i  );
                        Real e2_r3 = (rho(bl).flux(X1DIR, 0, k  , j, i) >= 0.0) ?
                                    -B_U(bl).flux(X3DIR, V1, k, j, i-1) - emfc(bl, V2, k  , j, i-1) :
                                    -B_U(bl).flux(X3DIR, V1, k, j, i  ) - emfc(bl, V2, k  , j, i  );
                        Real e2_l1 = (rho(bl).flux(X3DIR, 0, k, j, i-1) >= 0.0) ?
                                    B_U(bl).flux(X1DIR, V3, k-1, j, i) - emfc(bl, V2, k-1, j, i-1) :
                                    B_U(bl).flux(X1DIR, V3, k  , j, i) - emfc(bl, V2, k  , j, i-1);
                        Real e2_r1 = (rho(bl).flux(X3DIR, 0, k, j, i  ) >= 0.0) ?
                                    B_U(bl).flux(X1DIR, V3, k-1, j, i) - emfc(bl, V2, k-1, j, i  ) :
                                    B_U(bl).flux(X1DIR, V3, k  , j, i) - emfc(bl, V2, k  , j, i  );
                        emf_pack(bl, E2, 0, k, j, i) += 0.25*(e2_l3 + e2_r3 + e2_l1 + e2_r1);
                    }

                    Real e3_l2 = (rho(bl).flux(X1DIR, 0, k, j-1, i) >= 0.0) ?
                                B_U(bl).flux(X2DIR, V1, k, j, i-1) - emfc(bl, V3, k, j-1, i-1) :
                                B_U(bl).flux(X2DIR, V1, k, j, i  ) - emfc(bl, V3, k, j-1, i  );
                    Real e3_r2 = (rho(bl).flux(X1DIR, 0, k, j  , i) >= 0.0) ?
                                B_U(bl).flux(X2DIR, V1, k, j, i-1) - emfc(bl, V3, k, j  , i-1) :
                                B_U(bl).flux(X2DIR, V1, k, j, i  ) - emfc(bl, V3, k, j  , i  );
                    Real e3_l1 = (rho(bl).flux(X2DIR, 0, k, j, i-1) >= 0.0) ?
                                -B_U(bl).flux(X1DIR, V2, k, j-1, i) - emfc(bl, V3, k, j-1, i-1) :
                                -B_U(bl).flux(X1DIR, V2, k, j  , i) - emfc(bl, V3, k, j  , i-1);
                    Real e3_r1 = (rho(bl).flux(X2DIR, 0, k, j, i  ) >= 0.0) ?
                                -B_U(bl).flux(X1DIR, V2, k, j-1, i) - emfc(bl, V3, k, j-1, i  ) :
                                -B_U(bl).flux(X1DIR, V2, k, j  , i) - emfc(bl, V3, k, j  , i  );
                    emf_pack(bl, E3, 0, k, j, i) += 0.25*(e3_l2 + e3_r2 + e3_l1 + e3_r1);
                }
            );
        }
    }

    return TaskStatus::complete;
}

TaskStatus B_CT::AddSource(MeshData<Real> *md, MeshData<Real> *mdudt, IndexDomain domain)
{
    auto pmesh = md->GetMeshPointer();
    const int ndim = pmesh->ndim;

    // EMF temporary
    auto& emf_pack = md->PackVariables(std::vector<std::string>{"B_CT.emf"});

    // Figure out indices
    const IndexRange block = IndexRange{0, emf_pack.GetDim(5)-1};

    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // This is what we're replacing
    auto& dB_Uf_dt = mdudt->PackVariables(std::vector<std::string>{"cons.fB"});
    // Circulation -> change in flux at face
    const IndexRange3 bf1 = KDomain::GetRange(md, domain, F1);
    pmb0->par_for("B_CT_Circ_1", block.s, block.e, bf1.ks, bf1.ke, bf1.js, bf1.je, bf1.is, bf1.ie,
        KOKKOS_LAMBDA (const int &bl, const int &k, const int &j, const int &i) {
            const auto& G = dB_Uf_dt.GetCoords(bl);
            dB_Uf_dt(bl, F1, 0, k, j, i) = (-G.Volume<E3>(k, j + 1, i) * emf_pack(bl, E3, 0, k, j + 1, i)
                                           + G.Volume<E3>(k, j, i)     * emf_pack(bl, E3, 0, k, j, i));
            if (ndim > 2)
                dB_Uf_dt(bl, F1, 0, k, j, i) += (G.Volume<E2>(k + 1, j, i) * emf_pack(bl, E2, 0, k + 1, j, i)
                                                - G.Volume<E2>(k, j, i)    * emf_pack(bl, E2, 0, k, j, i));
            dB_Uf_dt(bl, F1, 0, k, j, i) /= G.Volume<F1>(k, j, i);
        }
    );
    const IndexRange3 bf2 = KDomain::GetRange(md, domain, F2);
    pmb0->par_for("B_CT_Circ_2", block.s, block.e, bf2.ks, bf2.ke, bf2.js, bf2.je, bf2.is, bf2.ie,
        KOKKOS_LAMBDA (const int &bl, const int &k, const int &j, const int &i) {
            const auto& G = dB_Uf_dt.GetCoords(bl);
            dB_Uf_dt(bl, F2, 0, k, j, i) = (G.Volume<E3>(k, j, i + 1) * emf_pack(bl, E3, 0, k, j, i + 1)
                                           - G.Volume<E3>(k, j, i)    * emf_pack(bl, E3, 0, k, j, i));
            if (ndim > 2)
                dB_Uf_dt(bl, F2, 0, k, j, i) += (-G.Volume<E1>(k + 1, j, i) * emf_pack(bl, E1, 0, k + 1, j, i)
                                                + G.Volume<E1>(k, j, i)     * emf_pack(bl, E1, 0, k, j, i));
            dB_Uf_dt(bl, F2, 0, k, j, i) /= G.Volume<F2>(k, j, i);
        }
    );
    const IndexRange3 bf3 = KDomain::GetRange(md, domain, F3);
    pmb0->par_for("B_CT_Circ_3", block.s, block.e, bf3.ks, bf3.ke, bf3.js, bf3.je, bf3.is, bf3.ie,
        KOKKOS_LAMBDA (const int &bl, const int &k, const int &j, const int &i) {
            const auto& G = dB_Uf_dt.GetCoords(bl);
            dB_Uf_dt(bl, F3, 0, k, j, i) = (- G.Volume<E2>(k, j, i + 1) * emf_pack(bl, E2, 0, k, j, i + 1)
                                            + G.Volume<E2>(k, j, i)     * emf_pack(bl, E2, 0, k, j, i)
                                            + G.Volume<E1>(k, j + 1, i) * emf_pack(bl, E1, 0, k, j + 1, i)
                                            - G.Volume<E1>(k, j, i)     * emf_pack(bl, E1, 0, k, j, i)) / G.Volume<F3>(k, j, i);
        }
    );

    return TaskStatus::complete;
}

TaskStatus B_CT::DerefinePoles(MeshData<Real> *md)
{
    // HYERIN (01/17/24) this routine is not general yet and only applies to polar boundaries for now.
    auto pmesh = md->GetMeshPointer();
    const uint nlevels = pmesh->packages.Get("ISMR")->Param<uint>("nlevels");

    // Figure out indices
    int ng = Globals::nghost;
    for (int iblock=0; iblock < md->NumBlocks(); iblock++) {
        auto& rc = md->GetBlockData(iblock);
        auto pmb = rc->GetBlockPointer();
        const auto& G = pmb->coords;
        auto B_Uf = rc->PackVariables(std::vector<std::string>{"cons.fB"});
        auto B_avg = rc->PackVariables(std::vector<std::string>{"ismr.fB_avg"});
        for (int i = 0; i < BOUNDARY_NFACES; i++) {
            BoundaryFace bface = (BoundaryFace) i;
            auto bname = KBoundaries::BoundaryName(bface);
            auto bdir = KBoundaries::BoundaryDirection(bface);
            auto domain = KBoundaries::BoundaryDomain(bface);
            auto binner = KBoundaries::BoundaryIsInner(bface);
            if (bdir == X2DIR && KBoundaries::IsPhysicalBoundary(pmb, bface)) {
                // indices
                // TODO also get ranges in cells from the beginning rather than using j_p & calculating j_c
                IndexRange3 bCC = KDomain::GetRange(rc, IndexDomain::interior, CC);
                // Note these are invalid in X2! We use them only for X1/X3 directions
                IndexRange3 bF1 = KDomain::GetRange(rc, domain, F1, ng, -ng);
                IndexRange3 bF3 = KDomain::GetRange(rc, domain, F3, ng, -ng);
                const int j_f = (binner) ? bCC.js : bCC.je + 1; // last physical face
                const int jps = (binner) ? j_f + (nlevels - 1) : j_f - (nlevels - 1); // start of the lowest level of derefinement
                const IndexRange j_p = IndexRange{(binner) ? j_f : jps, (binner) ? jps : j_f};  // Range of x2 to be de-refined
                const int offset = (binner) ? 1 : -1; // offset to read the physical face values
                const int point_out = offset; // if F2 B field at j_f + offset face is positive when pointing out of the cell, +1.

                // Should we allow flux through the pole?
                auto &bpars = pmesh->packages.Get("Boundaries")->AllParams();
                const bool allow_flux = binner ? bpars.Get<bool>("excise_flux_inner_x2"):
                                                 bpars.Get<bool>("excise_flux_outer_x2");

                // F1 average
                pmb->par_for("B_CT_derefine_poles_avg_F1", bCC.ks, bCC.ke, j_p.s, j_p.e, bF1.is, bF1.ie,
                    KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                        const int coarse_cell_len = m::pow(2, ((binner) ? jps - j : j - jps) + 1);
                        const int j_c = j + ((binner) ? 0 : -1); // cell center
                        const int k_fine = (k - ng) % coarse_cell_len; // this fine cell's k-index within the coarse cell
                        const int k_start = k - k_fine; // starting k-index of the coarse cell

                        // average over fine cells within the coarse cell we're in
                        Real avg = 0.;
                        for (int ktemp = 0; ktemp < coarse_cell_len; ++ktemp)
                            avg += B_Uf(F1, 0, k_start + ktemp, j_c, i) * G.Volume<F1>(k_start + ktemp, j_c, i);
                        avg /= coarse_cell_len;

                        B_avg(F1, 0, k, j_c, i) = avg;
                    }
                );
                // F2 average
                pmb->par_for("B_CT_derefine_poles_avg_F2", bCC.ks, bCC.ke, j_p.s, j_p.e, bCC.is, bCC.ie,
                    KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                        const int coarse_cell_len = m::pow(2, ((binner) ? jps - j : j - jps) + 1);
                        // fine cell's k index within the coarse cell
                        const int k_fine = (k - ng) % coarse_cell_len;
                        // starting k-index of the coarse cell
                        const int k_start = k - k_fine;

                        if (!allow_flux && j == j_f) {
                            // The fine cells have 0 fluxes through the physical-ghost boundaries.
                            B_avg(F2, 0, k, j, i) = 0.;
                        } else { // average the fine cells
                            Real avg = 0.;
                            for (int ktemp = 0; ktemp < coarse_cell_len; ++ktemp)
                                avg += B_Uf(F2, 0, k_start + ktemp, j, i) * G.Volume<F2>(k_start + ktemp, j, i);
                            avg /= coarse_cell_len;

                            B_avg(F2, 0, k, j, i) = avg;
                        }
                    }
                );
                // F3 average
                pmb->par_for("B_CT_derefine_poles_avg_F3", bF3.ks, bF3.ke, j_p.s, j_p.e, bCC.is, bCC.ie,
                    KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                        // the current level of derefinement at given j
                        const int current_lv = ((binner) ? jps - j : j - jps);
                        // half of the coarse cell's length
                        const int c_half = m::pow(2, current_lv);
                        const int coarse_cell_len = 2 * c_half;
                        // cell center
                        const int j_c = j + ((binner) ? 0 : -1);
                        // this fine cell's k-index within the coarse cell
                        const int k_fine = (k - ng) % coarse_cell_len;
                        // starting k-index of the coarse cell
                        const int k_start = k - k_fine;
                        const int k_half = k_start + c_half;
                        // end k-index of the coarse cell
                        const int k_end  = k_start + coarse_cell_len;

                        if ((k - ng) % coarse_cell_len == 0) {
                            // Don't modify faces of the coarse cells
                            B_avg(F3, 0, k, j_c, i) = B_Uf(F3, 0, k, j_c, i) * G.Volume<F3>(k, j_c, i);
                        } else {
                            // F3: The internal faces will take care of the divB=0. The two faces of the coarse cell will remain unchanged.
                            // First calculate the very central internal face. In other words, deal with the highest level internal face first.
                            // Sum of F2 fluxes in the left and right half of the coarse cell each.
                            Real c_left_v = 0., c_right_v = 0.;
                            for (int ktemp = 0; ktemp < c_half; ++ktemp) {
                                c_left_v  += B_Uf(F2, 0, k_half - 1 - ktemp, j + offset, i) * G.Volume<F2>(k_half - 1 - ktemp, j + offset, i);
                                c_right_v += B_Uf(F2, 0, k_half   + ktemp, j + offset, i) * G.Volume<F2>(k_half     + ktemp, j + offset, i);
                            }
                            const Real B_start = B_Uf(F3, 0, k_start, j_c, i) * G.Volume<F3>(k_start, j_c, i);
                            const Real B_end   = B_Uf(F3, 0, k_end,   j_c, i) * G.Volume<F3>(k_end,   j_c, i);
                            const Real B_center = (B_start + B_end + point_out * (c_right_v - c_left_v)) / 2.;

                            if (k == k_half) { // if at the center, then store the calculated value.
                                B_avg(F3, 0, k, j_c, i) = B_center;
                            } else if (k < k_half) { // interpolate between B_start and B_center
                                B_avg(F3, 0, k, j_c, i) = ((c_half - k_fine) * B_start + k_fine * B_center) / (c_half);
                            } else if (k > k_half) { // interpolate between B_end and B_center
                                B_avg(F3, 0, k, j_c, i) = ((k_fine - c_half) * B_end + (coarse_cell_len - k_fine) * B_center) / (c_half);
                            }
                        }
                    }
                );

                // F1 write
                pmb->par_for("B_CT_derefine_poles_F1", bCC.ks, bCC.ke, j_p.s, j_p.e, bF1.is, bF1.ie,
                    KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                        int j_c = j + ((binner) ? 0 : -1); // cell center
                        B_Uf(F1, 0, k, j_c, i) = B_avg(F1, 0, k, j_c, i) / G.Volume<F1>(k, j_c, i);
                    }
                );
                // F2 write
                pmb->par_for("B_CT_derefine_poles_F2", bCC.ks, bCC.ke, j_p.s, j_p.e, bCC.is, bCC.ie,
                    KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                        B_Uf(F2, 0, k, j, i) = B_avg(F2, 0, k, j, i) / G.Volume<F2>(k, j, i);
                    }
                );
                // F3 write
                pmb->par_for("B_CT_derefine_poles_F3", bF3.ks, bF3.ke, j_p.s, j_p.e, bCC.is, bCC.ie,
                    KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                        int j_c = j + ((binner) ? 0 : -1); // cell center
                        B_Uf(F3, 0, k, j_c, i) = B_avg(F3, 0, k, j_c, i) / G.Volume<F3>(k, j_c, i);
                    }
                );

                // Average the primitive vals to zone centers
                const int ndim = rc->GetMeshPointer()->ndim;
                auto B_U = rc->PackVariables(std::vector<std::string>{"cons.B"});
                auto B_P = rc->PackVariables(std::vector<std::string>{"prims.B"});
                pmb->par_for("UtoP_B_center", bCC.ks, bCC.ke, j_p.s, j_p.e, bCC.is, bCC.ie,
                    KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                        int j_c = j + ((binner) ? 0 : -1); // cell center
                        B_P(V1, k, j_c, i) = (B_Uf(F1, 0, k, j_c, i) / G.gdet(Loci::face1, j_c, i)
                                        + B_Uf(F1, 0, k, j_c, i + 1) / G.gdet(Loci::face1, j_c, i + 1)) / 2;
                        B_P(V2, k, j_c, i) = (ndim > 1) ? (B_Uf(F2, 0, k, j_c, i) / G.gdet(Loci::face2, j_c, i)
                                                    + B_Uf(F2, 0, k, j_c + 1, i) / G.gdet(Loci::face2, j_c + 1, i)) / 2
                                                    : B_Uf(F2, 0, k, j_c, i) / G.gdet(Loci::face2, j_c, i);
                        B_P(V3, k, j_c, i) = (ndim > 2) ? (B_Uf(F3, 0, k, j_c, i) / G.gdet(Loci::face3, j_c, i)
                                                    + B_Uf(F3, 0, k + 1, j_c, i) / G.gdet(Loci::face3, j_c, i)) / 2
                                                    : B_Uf(F3, 0, k, j_c, i) / G.gdet(Loci::face3, j_c, i);
                    }
                );
                // Recover conserved B at centers
                pmb->par_for("UtoP_B_centerPtoU", 0, NVEC-1, bCC.ks, bCC.ke, j_p.s, j_p.e, bCC.is, bCC.ie,
                    KOKKOS_LAMBDA (const int &v, const int &k, const int &j, const int &i) {
                        int j_c = j + ((binner) ? 0 : -1); // cell center
                        B_U(v, k, j_c, i) = B_P(v, k, j_c, i) * G.gdet(Loci::center, j_c, i);
                    }
                );
            }
        }
    }
    return TaskStatus::complete;
}

double B_CT::MaxDivB(MeshData<Real> *md)
{
    auto pmesh = md->GetMeshPointer();
    const int ndim = pmesh->ndim;
    if (ndim < 2) return 0.;

    auto B_U = md->PackVariables(std::vector<std::string>{"cons.fB"});

    // Figure out indices
    const IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    const IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    const IndexRange block = IndexRange{0, B_U.GetDim(5)-1};

    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    double max_divb;
    Kokkos::Max<double> max_reducer(max_divb);
    pmb0->par_reduce("divB_max", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, double &local_result) {
            const auto& G = B_U.GetCoords(b);
            double local_divb = m::abs(face_div(G, B_U(b), ndim, k, j, i));
            if (local_divb > local_result) local_result = local_divb;
        }
    , max_reducer);

    return max_divb;
}
double B_CT::BlockMaxDivB(MeshBlockData<Real> *rc)
{
    const int ndim = KDomain::GetNDim(rc);

    auto B_U = rc->PackVariables(std::vector<std::string>{"cons.fB"});

    // Figure out indices
    const IndexRange3 b = KDomain::GetRange(rc, IndexDomain::interior);

    auto pmb = rc->GetBlockPointer();

    double max_divb;
    Kokkos::Max<double> max_reducer(max_divb);
    pmb->par_reduce("divB_max", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i, double &local_result) {
            const auto& G = B_U.GetCoords();
            double local_divb = m::abs(face_div(G, B_U, ndim, k, j, i));
            if (local_divb > local_result) local_result = local_divb;
        }
    , max_reducer);

    return max_divb;
}
double B_CT::GlobalMaxDivB(MeshData<Real> *md, bool all_reduce)
{
    if (all_reduce) {
        Reductions::StartToAll<Real>(md, 2, MaxDivB(md), MPI_MAX);
        return Reductions::CheckOnAll<Real>(md, 2);
    } else {
        Reductions::Start<Real>(md, 2, MaxDivB(md), MPI_MAX);
        return Reductions::Check<Real>(md, 2);
    }
}

TaskStatus B_CT::PrintGlobalMaxDivB(MeshData<Real> *md, bool kill_on_large_divb)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // Since this is in the history file now, I don't bother printing it
    // unless we're being verbose. It's not costly to calculate though
    const bool print = pmb0->packages.Get("Globals")->Param<int>("verbose") >= 1;
    if (print || kill_on_large_divb) {
        // Calculate the maximum from/on all nodes
        const double divb_max = B_CT::GlobalMaxDivB(md);
        // Print on rank zero
        if (MPIRank0() && print) {
            printf("Max DivB: %g\n", divb_max); // someday I'll learn stream options
        }
        if (kill_on_large_divb) {
            if (divb_max > pmb0->packages.Get("B_CT")->Param<Real>("kill_on_divb_over"))
                throw std::runtime_error("DivB exceeds maximum! Quitting...");
        }
    }

    return TaskStatus::complete;
}

// TODO unify these by adding FillOutputMesh option

void B_CT::CalcDivB(MeshData<Real> *md, std::string divb_field_name)
{
    auto pmesh = md->GetMeshPointer();
    const int ndim = pmesh->ndim;

    // Packing out here avoids frequent per-mesh packs.  Do we need to?
    auto B_U = md->PackVariables(std::vector<std::string>{"cons.fB"});
    auto divB = md->PackVariables(std::vector<std::string>{divb_field_name});

    const IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    const IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    const IndexRange block = IndexRange{0, B_U.GetDim(5)-1};

    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // See MaxDivB for details
    pmb0->par_for("calc_divB", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i) {
            const auto& G = B_U.GetCoords(b);
            divB(b, 0, k, j, i) = face_div(G, B_U(b), ndim, k, j, i);
        }
    );
}

void B_CT::FillOutput(MeshBlock *pmb, ParameterInput *pin)
{
    // This is called after the step, use "base" container
    auto rc = pmb->meshblock_data.Get("base");
    const int ndim = pmb->pmy_mesh->ndim;
    if (ndim < 2) return;

    auto B_U = rc->PackVariables(std::vector<std::string>{"cons.fB"});
    auto divB = rc->PackVariables(std::vector<std::string>{"divB"});

    const IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
    const IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = rc->GetBoundsK(IndexDomain::interior);
    const IndexRange block = IndexRange{0, B_U.GetDim(5)-1};

    pmb->par_for("divB_output", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            const auto& G = B_U.GetCoords();
            divB(0, k, j, i) = face_div(G, B_U, ndim, k, j, i);
        }
    );
}
