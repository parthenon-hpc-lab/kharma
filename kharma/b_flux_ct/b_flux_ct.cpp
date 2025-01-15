/*
 *  File: b_flux_ct.cpp
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

#include <parthenon/parthenon.hpp>

#include "b_flux_ct.hpp"

#include "decs.hpp"
#include "domain.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"

using namespace parthenon;

namespace B_FluxCT
{

std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("B_FluxCT");
    Params &params = pkg->AllParams();

    // Diagnostic & inadvisable flags
    // This enables flux corrections to ensure divB preservation even with zero flux of B2 on the polar "face."
    // It effectively makes the pole a superconducting rod
    // TODO unify all these options into zero_flux_B and bflux0_B
    bool spherical = pin->GetBoolean("coordinates", "spherical");
    bool fix_polar_flux = pin->GetOrAddBoolean("b_field", "fix_polar_flux", spherical);
    params.Add("fix_polar_flux", fix_polar_flux);
    // These options do a similar fix to the inner and outer radial edges, which is less commonly necessary.
    // They require constant (Dirichlet) boundary conditions
    // These are the "Bflux0" prescription designed by Hyerin Cho
    bool fix_flux_x1 = pin->GetOrAddBoolean("b_field", "fix_flux_x1", false);
    // Split out options. Turn off inner edge by default if inner bound is within EH (TODO turn on/off based on BOUNDARIES)
    bool r_in_eh = spherical && pin->GetBoolean("coordinates", "domain_intersects_eh");
    bool fix_flux_inner_x1 = pin->GetOrAddBoolean("b_field", "fix_flux_inner_x1", fix_flux_x1 && !r_in_eh);
    params.Add("fix_flux_inner_x1", fix_flux_inner_x1);
    bool fix_flux_outer_x1 = pin->GetOrAddBoolean("b_field", "fix_flux_outer_x1", fix_flux_x1);
    params.Add("fix_flux_outer_x1", fix_flux_outer_x1);
    // Bflux0 on x2.  NOT good for polar boundaries
    bool fix_flux_x2 = pin->GetOrAddBoolean("b_field", "fix_flux_x2", false);
    bool fix_flux_inner_x2 = pin->GetOrAddBoolean("b_field", "fix_flux_inner_x2", fix_flux_x2);
    params.Add("fix_flux_inner_x2", fix_flux_inner_x2);
    bool fix_flux_outer_x2 = pin->GetOrAddBoolean("b_field", "fix_flux_outer_x2", fix_flux_x2);
    params.Add("fix_flux_outer_x2", fix_flux_outer_x2);
    // This reverts to a more ham-fisted fix which explicitly disallows flux crossing the X1 face.
    // This version requires *inverted* B1 across the face, potentially just using reflecting conditions for B
    // Using this version is tremendously inadvisable: consult your simulator before applying.
    bool use_old_flux_fix = pin->GetOrAddBoolean("b_field", "use_old_flux_fix", false);
    params.Add("use_old_flux_fix", use_old_flux_fix);

    // KHARMA requires some kind of field transport if there is a magnetic field allocated
    // Use this if you actually want to disable all magnetic field flux corrections,
    // and allow a field divergence to grow unchecked, usually for debugging or comparison reasons
    bool disable_flux_ct = pin->GetOrAddBoolean("b_field", "disable_flux_ct", false);
    params.Add("disable_flux_ct", disable_flux_ct);

    // Default to stopping execution when divB is large, which generally indicates something
    // has gone wrong.  As always, can be disabled by the brave.
    bool kill_on_large_divb = pin->GetOrAddBoolean("b_field", "kill_on_large_divb", true);
    params.Add("kill_on_large_divb", kill_on_large_divb);
    Real kill_on_divb_over = pin->GetOrAddReal("b_field", "kill_on_divb_over", 1.e-3);
    params.Add("kill_on_divb_over", kill_on_divb_over);

    // By default, solve B explicitly
    bool implicit_b = pin->GetOrAddBoolean("b_field", "implicit", false);
    params.Add("implicit", implicit_b);

    // FIELDS
    // Vector size: 3x[grid shape]
    std::vector<int> s_vector({NVEC});

    // Mark if we're evolving implicitly
    MetadataFlag areWeImplicit = (implicit_b) ? Metadata::GetUserFlag("Implicit")
                                              : Metadata::GetUserFlag("Explicit");

    // Flags for B fields
    // We always mark conserved B to be sync'd for consistency, since it's strictly required for B_CT/AMR
    std::vector<MetadataFlag> flags_prim = {Metadata::Real, Metadata::Derived, Metadata::GetUserFlag("Primitive"),
                                            Metadata::Cell, Metadata::GetUserFlag("MHD"), areWeImplicit, Metadata::Vector};
    std::vector<MetadataFlag> flags_cons = {Metadata::Real, Metadata::Independent, Metadata::Restart, Metadata::FillGhost, Metadata::WithFluxes, Metadata::Conserved,
                                            Metadata::Cell, Metadata::GetUserFlag("MHD"), areWeImplicit, Metadata::Vector};

    auto m = Metadata(flags_prim, s_vector);
    pkg->AddField("prims.B", m);
    m = Metadata(flags_cons, s_vector);
    pkg->AddField("cons.B", m);

    // Declare EMF temporary variables, to avoid malloc/free during each step
    // Technically these are edge-centered but we only need the interior + 1-zone halo anyway, so we store as a vector
    std::vector<MetadataFlag> flags_emf = {Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy};
    m = Metadata(flags_emf, s_vector);
    pkg->AddField("emf", m);
    if (packages->Get("Globals")->Param<std::string>("problem") == "resize_restart_kharma") {
        m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::FillGhost, Metadata::Vector});
        pkg->AddField("B_Save", m);
    }

    // CALLBACKS

    // We exist basically to do this
    pkg->FixFlux = B_FluxCT::FixFlux;

    // Also ensure that prims get filled, *if* we're evolved explicitly
    if (!implicit_b) {
        pkg->MeshUtoP = B_FluxCT::MeshUtoP;
        pkg->BlockUtoP = B_FluxCT::BlockUtoP;
    }
    // Still need UtoP on boundaries
    pkg->BoundaryUtoP = B_FluxCT::BlockUtoP;

    // Register the other callbacks
    pkg->PostStepDiagnosticsMesh = B_FluxCT::PostStepDiagnostics;

    // The definition of MaxDivB we care about actually changes per-transport,
    // so calculating it is handled by the transport package
    // We'd only ever need to declare or calculate divB for output (getting the max is independent)

    if (KHARMA::FieldIsOutput(pin, "divB")) {
        pkg->BlockUserWorkBeforeOutput = B_FluxCT::FillOutput;
        m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
        pkg->AddField("divB", m);
    }

    // List (vector) of HistoryOutputVars that will all be enrolled as output variables
    parthenon::HstVar_list hst_vars = {};
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, B_FluxCT::MaxDivB, "MaxDivB"));
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

// TODO template and use as a model for future
TaskStatus MeshUtoP(MeshData<Real> *md, IndexDomain domain, bool coarse)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    const auto& B_U = md->PackVariables(std::vector<std::string>{"cons.B"});
    const auto& B_P = md->PackVariables(std::vector<std::string>{"prims.B"});

    auto bounds = coarse ? pmb0->c_cellbounds : pmb0->cellbounds;
    IndexRange ib = bounds.GetBoundsI(domain);
    IndexRange jb = bounds.GetBoundsJ(domain);
    IndexRange kb = bounds.GetBoundsK(domain);
    IndexRange vec = IndexRange{0, B_U.GetDim(4)-1};
    IndexRange block = IndexRange{0, B_U.GetDim(5)-1};

    pmb0->par_for("UtoP_B_FluxCT_Mesh", block.s, block.e, vec.s, vec.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int& b, const int &mu, const int &k, const int &j, const int &i) {
            const auto& G = B_U.GetCoords(b);
            // Update the primitive B-fields
            B_P(b, mu, k, j, i) = B_U(b, mu, k, j, i) / G.gdet(Loci::center, j, i);
        }
    );
    return TaskStatus::complete;
}
TaskStatus BlockUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();

    auto B_U = rc->PackVariables(std::vector<std::string>{"cons.B"});
    auto B_P = rc->PackVariables(std::vector<std::string>{"prims.B"});

    const auto& G = pmb->coords;

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const IndexRange ib = bounds.GetBoundsI(domain);
    const IndexRange jb = bounds.GetBoundsJ(domain);
    const IndexRange kb = bounds.GetBoundsK(domain);
    const IndexRange vec = IndexRange({0, B_U.GetDim(4)-1});

    pmb->par_for("UtoP_B_FluxCT_Block", vec.s, vec.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &mu, const int &k, const int &j, const int &i) {
            // Update the primitive B-fields
            B_P(mu, k, j, i) = B_U(mu, k, j, i) / G.gdet(Loci::center, j, i);
        }
    );
    return TaskStatus::complete;
}

void MeshPtoU(MeshData<Real> *md, IndexDomain domain, bool coarse)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    const auto& B_U = md->PackVariables(std::vector<std::string>{"cons.B"});
    const auto& B_P = md->PackVariables(std::vector<std::string>{"prims.B"});

    auto bounds = coarse ? pmb0->c_cellbounds : pmb0->cellbounds;
    IndexRange ib = bounds.GetBoundsI(domain);
    IndexRange jb = bounds.GetBoundsJ(domain);
    IndexRange kb = bounds.GetBoundsK(domain);
    IndexRange vec = IndexRange{0, B_U.GetDim(4)-1};
    IndexRange block = IndexRange{0, B_U.GetDim(5)-1};

    pmb0->par_for("PtoU_B_FluxCT_Mesh", block.s, block.e, vec.s, vec.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int& b, const int &mu, const int &k, const int &j, const int &i) {
            const auto& G = B_U.GetCoords(b);
            // Update the primitive B-fields
            B_U(b, mu, k, j, i) = B_P(b, mu, k, j, i) * G.gdet(Loci::center, j, i);
        }
    );
}
void BlockPtoU(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();

    auto B_U = rc->PackVariables(std::vector<std::string>{"cons.B"});
    auto B_P = rc->PackVariables(std::vector<std::string>{"prims.B"});

    const auto& G = pmb->coords;

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const IndexRange ib = bounds.GetBoundsI(domain);
    const IndexRange jb = bounds.GetBoundsJ(domain);
    const IndexRange kb = bounds.GetBoundsK(domain);
    const IndexRange vec = IndexRange({0, B_U.GetDim(4)-1});

    pmb->par_for("PtoU_B_FluxCT_Block", vec.s, vec.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &mu, const int &k, const int &j, const int &i) {
            // Update the conserved B-fields
            B_U(mu, k, j, i) = B_P(mu, k, j, i) * G.gdet(Loci::center, j, i);
        }
    );
}

void FixFlux(MeshData<Real> *md)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    auto& params = pmb0->packages.Get("B_FluxCT")->AllParams();
    // Poles specifically can't use Bflux0
    if (params.Get<bool>("fix_polar_flux")) {
        ZeroBoundaryFlux(md, IndexDomain::inner_x2, false);
        ZeroBoundaryFlux(md, IndexDomain::outer_x2, false);
    }
    // Everything else should default to Bflux0
    if (params.Get<bool>("fix_flux_inner_x1")) {
        if (params.Get<bool>("use_old_flux_fix")) {
            ZeroBoundaryFlux(md, IndexDomain::inner_x1, false);
        } else {
            Bflux0(md, IndexDomain::inner_x1, false);
        }
    }
    if (params.Get<bool>("fix_flux_outer_x1")) {
        if (params.Get<bool>("use_old_flux_fix")) {
            ZeroBoundaryFlux(md, IndexDomain::outer_x1, false);
        } else {
            Bflux0(md, IndexDomain::outer_x1, false);
        }
    }
    if (params.Get<bool>("fix_flux_inner_x2")) {
        if (params.Get<bool>("use_old_flux_fix")) {
            ZeroBoundaryFlux(md, IndexDomain::inner_x2, false);
        } else {
            Bflux0(md, IndexDomain::inner_x2, false);
        }
    }
    if (params.Get<bool>("fix_flux_outer_x2")) {
        if (params.Get<bool>("use_old_flux_fix")) {
            ZeroBoundaryFlux(md, IndexDomain::inner_x2, false);
        } else {
            Bflux0(md, IndexDomain::outer_x2, false);
        }
    }
    FluxCT(md);
}

// INTERNAL

void FluxCT(MeshData<Real> *md)
{
    // Pointers
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Exit on trivial operations
    const int ndim = pmesh->ndim;
    if (ndim < 2) return;

    // Pack variables
    const auto& B_F = md->PackVariablesAndFluxes(std::vector<std::string>{"cons.B"});
    const auto& emf_pack = md->PackVariables(std::vector<std::string>{"emf"});

    // Get sizes
    const IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    const IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    const IndexRange block = IndexRange{0, B_F.GetDim(5)-1};
    // One zone halo on the *right only*, except for k in 2D
    const IndexRange il = IndexRange{ib.s, ib.e + 1};
    const IndexRange jl = IndexRange{jb.s, jb.e + 1};
    const IndexRange kl = (ndim > 2) ? IndexRange{kb.s, kb.e + 1} : kb;

    // Calculate emf around each face
    pmb0->par_for("flux_ct_emf", block.s, block.e, kl.s, kl.e, jl.s, jl.e, il.s, il.e,
        KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
            if (ndim > 2) {
                emf_pack(b, V1, k, j, i) =  0.25 * (B_F(b).flux(X2DIR, V3, k, j, i) + B_F(b).flux(X2DIR, V3, k-1, j, i) -
                                            B_F(b).flux(X3DIR, V2, k, j, i) - B_F(b).flux(X3DIR, V2, k, j-1, i));
                emf_pack(b, V2, k, j, i) = 0.25 * (B_F(b).flux(X3DIR, V1, k, j, i) + B_F(b).flux(X3DIR, V1, k, j, i-1) -
                                            B_F(b).flux(X1DIR, V3, k, j, i) - B_F(b).flux(X1DIR, V3, k-1, j, i));
            }
            emf_pack(b, V3, k, j, i) =  0.25 * (B_F(b).flux(X1DIR, V2, k, j, i) + B_F(b).flux(X1DIR, V2, k, j-1, i) -
                                        B_F(b).flux(X2DIR, V1, k, j, i) - B_F(b).flux(X2DIR, V1, k, j, i-1));
        }
    );

    // Rewrite EMFs as fluxes, after Toth (2000)
    // Note that zeroing FX(BX) is *necessary* -- this flux gets filled by GetFlux
    // Note these each have different domains, eg il vs ib.  The former extends one index farther if appropriate
    pmb0->par_for("flux_ct_1", block.s, block.e, kb.s, kb.e, jb.s, jb.e, il.s, il.e,
        KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
            B_F(b).flux(X1DIR, V1, k, j, i) =  0.0;
            B_F(b).flux(X1DIR, V2, k, j, i) =  0.5 * (emf_pack(b, V3, k, j, i) + emf_pack(b, V3, k, j+1, i));
            if (ndim > 2) B_F(b).flux(X1DIR, V3, k, j, i) = -0.5 * (emf_pack(b, V2, k, j, i) + emf_pack(b, V2, k+1, j, i));
        }
    );
    pmb0->par_for("flux_ct_2", block.s, block.e, kb.s, kb.e, jl.s, jl.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
            B_F(b).flux(X2DIR, V1, k, j, i) = -0.5 * (emf_pack(b, V3, k, j, i) + emf_pack(b, V3, k, j, i+1));
            B_F(b).flux(X2DIR, V2, k, j, i) =  0.0;
            if (ndim > 2) B_F(b).flux(X2DIR, V3, k, j, i) =  0.5 * (emf_pack(b, V1, k, j, i) + emf_pack(b, V1, k+1, j, i));
        }
    );
    if (ndim > 2) {
        pmb0->par_for("flux_ct_3", block.s, block.e, kl.s, kl.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
                B_F(b).flux(X3DIR, V1, k, j, i) =  0.5 * (emf_pack(b, V2, k, j, i) + emf_pack(b, V2, k, j, i+1));
                B_F(b).flux(X3DIR, V2, k, j, i) = -0.5 * (emf_pack(b, V1, k, j, i) + emf_pack(b, V1, k, j+1, i));
                B_F(b).flux(X3DIR, V3, k, j, i) =  0.0;
            }
        );
    }
}

void ZeroBoundaryFlux(MeshData<Real> *md, IndexDomain domain, bool coarse)
{
    // TODO write ONE implementation for any boundary
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = pmesh->block_list[0];
    const int ndim = pmesh->ndim;
    if (ndim < 2) return;

    // Imagine a corner of the domain, with ghost and physical zones
    // as below, denoted w/'g' and 'p' respectively.
    //    ...
    // g | p | p
    //----------- 1
    // g | p | p ...
    //xxx-------- 0
    // g | g | g
    //-1   0   1
    // The flux through 'x' is not important for updating a physical zone,
    // as it does not border any.  However, FluxCT considers it when updating
    // nearby fluxes, two of which affect physical zones.
    // Therefore in e.g. X1 faces, we need to update fluxes on the domain:
    // [0,N1+1],[-1,N2+1],[-1,N3+1]
    // These indices arrange for that.
    auto bounds = coarse ? pmb0->c_cellbounds : pmb0->cellbounds;
    const IndexRange ib = bounds.GetBoundsI(IndexDomain::interior);
    const IndexRange jb = bounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = bounds.GetBoundsK(IndexDomain::interior);
    // For faces
    const IndexRange ibf = IndexRange{ib.s, ib.e + 1};
    const IndexRange jbf = IndexRange{jb.s, jb.e + 1};
    // Won't need X3 faces
    //const IndexRange kbf = IndexRange{kb.s, kb.e + (ndim > 2)};
    // For sides
    const IndexRange ibs = IndexRange{ib.s - 1, ib.e + 1};
    const IndexRange jbs = IndexRange{jb.s - (ndim > 1), jb.e + (ndim > 1)};
    const IndexRange kbs = IndexRange{kb.s - (ndim > 2), kb.e + (ndim > 2)};

    // Make sure the polar EMFs are 0 when performing fluxCT
    // Compare this section with calculation of emf3 in FluxCT:
    // these changes ensure that boundary emfs emf3(i,js,k)=0, etc.
    for (auto &pmb : pmesh->block_list) {
        auto& rc = pmb->meshblock_data.Get(md->StageName());
        auto& B_F = rc->PackVariablesAndFluxes(std::vector<std::string>{"cons.B"});

        if (domain == IndexDomain::inner_x2 &&
            pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::user) {
            pmb->par_for("fix_flux_b_l", kbs.s, kbs.e, jbf.s, jbf.s, ibs.s, ibs.e,
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    B_F.flux(X2DIR, V1, k, j, i) = 0.;
                    B_F.flux(X2DIR, V3, k, j, i) = 0.;
                    B_F.flux(X1DIR, V2, k, j-1, i) = -B_F.flux(X1DIR, V2, k, j, i);
                    if (ndim > 2) B_F.flux(X3DIR, V2, k, j-1, i) = -B_F.flux(X3DIR, V2, k, j, i);
                }
            );
        }

        if (domain == IndexDomain::outer_x2 &&
            pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::user) {
            pmb->par_for("fix_flux_b_r", kbs.s, kbs.e, jbf.e, jbf.e, ibs.s, ibs.e,
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    B_F.flux(X2DIR, V1, k, j, i) = 0.;
                    B_F.flux(X2DIR, V3, k, j, i) = 0.;
                    B_F.flux(X1DIR, V2, k, j, i) = -B_F.flux(X1DIR, V2, k, j - 1, i);
                    if (ndim > 2) B_F.flux(X3DIR, V2, k, j, i) = -B_F.flux(X3DIR, V2, k, j - 1, i);
                }
            );
        }
        // These boundary conditions need to arrange for B1 to be inverted in ghost cells.
        // This is no longer pure outflow, but might be thought of as a "nicer" version of
        // reflecting conditions:
        // 1. Since B1 is inverted, B1 on the domain face will tend to 0 (it's not quite reflected, but basically)
        //    (obviously don't enable this for monopole test problems!)
        // 2. However, B2 and B3 are normal outflow conditions -- despite the fluxes here, the outflow
        //    conditions will set them equal to the last zone.
        if (domain == IndexDomain::inner_x1 &&
            pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user) {
            pmb->par_for("fix_flux_b_in_old", kbs.s, kbs.e, jbs.s, jbs.e, ibf.s, ibf.s,
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    B_F.flux(X1DIR, V2, k, j, i) = 0.;
                    B_F.flux(X1DIR, V3, k, j, i) = 0.;
                    B_F.flux(X2DIR, V1, k, j, i - 1) = -B_F.flux(X2DIR, V1, k, j, i);
                    if (ndim > 2) B_F.flux(X3DIR, V1, k, j, i - 1) = -B_F.flux(X3DIR, V1, k, j, i);
                }
            );
        }

        if (domain == IndexDomain::outer_x1 &&
            pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::user) {
            pmb->par_for("fix_flux_b_out_old", kbs.s, kbs.e, jbs.s, jbs.e, ibf.e, ibf.e,
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    B_F.flux(X1DIR, V2, k, j, i) = 0.;
                    B_F.flux(X1DIR, V3, k, j, i) = 0.;
                    B_F.flux(X2DIR, V1, k, j, i) = -B_F.flux(X2DIR, V1, k, j, i - 1);
                    if (ndim > 2) B_F.flux(X3DIR, V1, k, j, i) = -B_F.flux(X3DIR, V1, k, j, i - 1);
                }
            );
        }
    }
}

void Bflux0(MeshData<Real> *md, IndexDomain domain, bool coarse)
{
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = pmesh->block_list[0];
    const int ndim = pmesh->ndim;
    if (ndim < 2) return;

    // Indices, see ZeroBoundaryFlux
    auto bounds = coarse ? pmb0->c_cellbounds : pmb0->cellbounds;
    const IndexRange ib = bounds.GetBoundsI(IndexDomain::interior);
    const IndexRange jb = bounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = bounds.GetBoundsK(IndexDomain::interior);
    // For faces
    const IndexRange ibf = IndexRange{ib.s, ib.e + 1};
    const IndexRange jbf = IndexRange{jb.s, jb.e + 1};
    // Won't need X3 faces
    //const IndexRange kbf = IndexRange{kb.s, kb.e + (ndim > 2)};
    // For sides
    const IndexRange ibs = IndexRange{ib.s - 1, ib.e + 1};
    const IndexRange jbs = IndexRange{jb.s - (ndim > 1), jb.e + (ndim > 1)};
    const IndexRange kbs = IndexRange{kb.s - (ndim > 2), kb.e + (ndim > 2)};

    // Make sure the polar EMFs are 0 when performing fluxCT
    // Compare this section with calculation of emf3 in FluxCT:
    // these changes ensure that boundary emfs emf3(i,js,k)=0, etc.
    for (auto &pmb : pmesh->block_list) {
        auto& rc = pmb->meshblock_data.Get(md->StageName());
        auto& B_F = rc->PackVariablesAndFluxes(std::vector<std::string>{"cons.B"});

        // "Bflux0" prescription for keeping divB~=0 on zone corners of the interior & exterior X1 faces
        // Courtesy of & implemented by Hyerin Cho
        // Allows nonzero flux across X1 boundary but still keeps divB=0 (turns out effectively to have 0 flux)
        // Usable only for Dirichlet conditions
        if (domain == IndexDomain::inner_x1 &&
            pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user)
        {
            pmb->par_for("fix_flux_b_in", kbs.s, kbs.e, jbs.s, jbs.e, ibf.s, ibf.s, // Hyerin (12/28/22) for 1st & 2nd prescription
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    // Allows nonzero flux across X1 boundary but still keeps divB=0 (turns out effectively to have 0 flux)
                    if (ndim > 1) B_F.flux(X2DIR, V1, k, j, i-1) = -B_F.flux(X2DIR, V1, k, j, i) + B_F.flux(X1DIR, V2, k, j, i) + B_F.flux(X1DIR, V2, k, j-1, i);
                    if (ndim > 2) B_F.flux(X3DIR, V1, k, j, i-1) = -B_F.flux(X3DIR, V1, k, j, i) + B_F.flux(X1DIR, V3, k, j, i) + B_F.flux(X1DIR, V3, k-1, j, i);
                }
            );

        }
        if (domain == IndexDomain::inner_x2 &&
            pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::user)
        {
            pmb->par_for("fix_flux_b_in", kbs.s, kbs.e, jbf.s, jbf.s, ibs.s, ibs.e,
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    if (ndim > 2) B_F.flux(X3DIR, V2, k, j-1, i) = -B_F.flux(X3DIR, V2, k, j, i) + B_F.flux(X2DIR, V3, k, j, i) + B_F.flux(X2DIR, V3, k-1, j, i);
                    if (ndim > 1) B_F.flux(X1DIR, V2, k, j-1, i) = -B_F.flux(X1DIR, V2, k, j, i) + B_F.flux(X2DIR, V1, k, j, i) + B_F.flux(X2DIR, V1, k, j, i-1);
                }
            );

        }

        // OUTER
        if (domain == IndexDomain::outer_x1 &&
            pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::user)
        {
            pmb->par_for("fix_flux_b_out", kbs.s, kbs.e, jbs.s, jbs.e, ibf.e, ibf.e, // Hyerin (12/28/22) for 1st & 2nd prescription
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    // (02/06/23) 2nd prescription that allows nonzero flux across X1 boundary but still keeps divB=0
                    if (ndim > 1) B_F.flux(X2DIR, V1, k, j, i) = -B_F.flux(X2DIR, V1, k, j, i-1) + B_F.flux(X1DIR, V2, k, j, i) + B_F.flux(X1DIR, V2, k, j-1, i);
                    if (ndim > 2) B_F.flux(X3DIR, V1, k, j, i) = -B_F.flux(X3DIR, V1, k, j, i-1) + B_F.flux(X1DIR, V3, k, j, i) + B_F.flux(X1DIR, V3, k-1, j, i);
                }
            );
        }
        if (domain == IndexDomain::outer_x2 &&
            pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::user)
        {
            pmb->par_for("fix_flux_b_out", kbs.s, kbs.e, jbf.e, jbf.e, ibs.s, ibs.e,
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    if (ndim > 2) B_F.flux(X3DIR, V2, k, j, i) = -B_F.flux(X3DIR, V2, k, j-1, i) + B_F.flux(X2DIR, V3, k, j, i) + B_F.flux(X2DIR, V3, k-1, j, i);
                    if (ndim > 1) B_F.flux(X1DIR, V2, k, j, i) = -B_F.flux(X1DIR, V2, k, j-1, i) + B_F.flux(X2DIR, V1, k, j, i) + B_F.flux(X2DIR, V1, k, j, i-1);
                }
            );
        }
    }
}

IndexRange ValidDivBX1(MeshBlock *pmb)
{
    // All user, physical (not MPI/periodic) boundary conditions in X1 will generate divB on corners
    // intersecting the interior & exterior faces. Don't report these zones, as we expect it.
    const IndexRange ibl = pmb->meshblock_data.Get("base")->GetBoundsI(IndexDomain::interior);
    bool avoid_inner = (!pmb->packages.Get("B_FluxCT")->Param<bool>("fix_flux_inner_x1") &&
        pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user);
    bool avoid_outer = (!pmb->packages.Get("B_FluxCT")->Param<bool>("fix_flux_outer_x1") &&
        pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::user);
    return IndexRange{ibl.s + (avoid_inner), ibl.e + (!avoid_outer)};
}

double MaxDivB(MeshData<Real> *md)
{
    auto pmesh = md->GetMeshPointer();
    const int ndim = pmesh->ndim;
    if (ndim < 2) return 0.;

    // Packing out here avoids frequent per-mesh packs.  Do we need to?
    auto B_U = md->PackVariables(std::vector<std::string>{"cons.B"});

    const IndexRange jbl = md->GetBoundsJ(IndexDomain::interior);
    const IndexRange kbl = md->GetBoundsK(IndexDomain::interior);

    const IndexRange jb = IndexRange{jbl.s, jbl.e + (ndim > 1)};
    const IndexRange kb = IndexRange{kbl.s, kbl.e + (ndim > 2)};
    const IndexRange block = IndexRange{0, B_U.GetDim(5)-1};

    // TODO Keep zone of max! See timestep calc
    // Will need to translate them back to KS to make them useful though

    // This is one kernel call per block, because each block will have different bounds.
    // Could consolidate at the cost of lots of bounds checking.
    double max_divb = 0.0;
    for (int b = block.s; b <= block.e; ++b) {
        auto pmb = md->GetBlockData(b)->GetBlockPointer();

        const IndexRange ib = ValidDivBX1(pmb);

        double max_divb_block;
        Kokkos::Max<double> max_reducer(max_divb_block);
        pmb->par_reduce("divB_max", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i, double &local_result) {
                const auto& G = B_U.GetCoords(b);
                const double local_divb = m::abs(corner_div(G, B_U(b), k, j, i, ndim > 2));
                if (local_divb > local_result) local_result = local_divb;
            }
        , max_reducer);

        if (max_divb_block > max_divb) max_divb = max_divb_block;
    }

    return max_divb;
}

double GlobalMaxDivB(MeshData<Real> *md, bool all_reduce)
{
    if (all_reduce) {
        Reductions::StartToAll<Real>(md, 2, MaxDivB(md), MPI_MAX);
        return Reductions::CheckOnAll<Real>(md, 2);
    } else {
        Reductions::Start<Real>(md, 2, MaxDivB(md), MPI_MAX);
        return Reductions::Check<Real>(md, 2);
    }
}

TaskStatus PrintGlobalMaxDivB(MeshData<Real> *md, bool kill_on_large_divb)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // Since this is in the history file now, I don't bother printing it
    // unless we're being verbose. It's not costly to calculate though
    const bool print = pmb0->packages.Get("Globals")->Param<int>("verbose") >= 1;
    if (print || kill_on_large_divb) {
        // Calculate the maximum from/on all nodes
        const double divb_max = B_FluxCT::GlobalMaxDivB(md);
        // Print on rank zero
        if (MPIRank0() && print) {
            // someday I'll learn stream options
            // for now this is more consistent in #digits/scientific
            printf("Max DivB: %g\n", divb_max);
        }
        if (kill_on_large_divb) {
            if (divb_max > pmb0->packages.Get("B_FluxCT")->Param<Real>("kill_on_divb_over"))
                throw std::runtime_error("DivB exceeds maximum! Quitting...");
        }
    }

    return TaskStatus::complete;
}

// TODO unify these by adding FillOutputMesh option

void CalcDivB(MeshData<Real> *md, std::string divb_field_name)
{
    auto pmesh = md->GetMeshPointer();
    const int ndim = pmesh->ndim;
    if (ndim < 2) return;

    // Packing out here avoids frequent per-mesh packs.  Do we need to?
    auto B_U = md->PackVariables(std::vector<std::string>{"cons.B"});
    auto divB = md->PackVariables(std::vector<std::string>{divb_field_name});

    const IndexRange jbl = md->GetBoundsJ(IndexDomain::interior);
    const IndexRange kbl = md->GetBoundsK(IndexDomain::interior);

    const IndexRange jb = IndexRange{jbl.s, jbl.e + (ndim > 1)};
    const IndexRange kb = IndexRange{kbl.s, kbl.e + (ndim > 2)};
    const IndexRange block = IndexRange{0, B_U.GetDim(5)-1};

    // See MaxDivB for details
    for (int b = block.s; b <= block.e; ++b) {
        auto pmb = md->GetBlockData(b)->GetBlockPointer();

        const IndexRange ib = ValidDivBX1(pmb);

        pmb->par_for("calc_divB", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                const auto& G = B_U.GetCoords(b);
                divB(b, 0, k, j, i) = corner_div(G, B_U(b), k, j, i, ndim > 2);
            }
        );
    }
}
void FillOutput(MeshBlock *pmb, ParameterInput *pin)
{
    auto rc = pmb->meshblock_data.Get("base").get();
    const int ndim = pmb->pmy_mesh->ndim;
    if (ndim < 2) return;

    auto B_U = rc->PackVariables(std::vector<std::string>{"cons.B"});
    auto divB = rc->PackVariables(std::vector<std::string>{"divB"});

    const IndexRange jbl = rc->GetBoundsJ(IndexDomain::interior);
    const IndexRange kbl = rc->GetBoundsK(IndexDomain::interior);

    const IndexRange jb = IndexRange{jbl.s, jbl.e + (ndim > 1)};
    const IndexRange kb = IndexRange{kbl.s, kbl.e + (ndim > 2)};
    const IndexRange block = IndexRange{0, B_U.GetDim(5)-1};

    const IndexRange ib = ValidDivBX1(pmb);

    pmb->par_for("divB_output", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            const auto& G = B_U.GetCoords();
            divB(0, k, j, i) = corner_div(G, B_U, k, j, i, ndim > 2);
        }
    );
}

} // namespace B_FluxCT
