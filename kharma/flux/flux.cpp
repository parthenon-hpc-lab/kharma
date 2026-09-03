/*
 *  File: flux.cpp
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
// Most includes are in the header TODO fix?

#include "b_ct.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"
#include <stdexcept>

using namespace parthenon;

// GetFlux is in the header file get_flux.hpp, as it is templated on reconstruction scheme
// and flux direction

int Flux::CountFOFCFlags(MeshData<Real>* md)
{
    return Reductions::CountFlags(md, "flags.fofc",
        std::map<int, std::string>{{1, "Flux-corrected"}}, IndexDomain::interior,
        true)[0];
}

std::shared_ptr<KHARMAPackage> Flux::Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages)
{
    Flag("Initializing Flux");
    auto pkg = std::make_shared<KHARMAPackage>("Fluxes");
    Params& params = pkg->AllParams();

    // Don't even error on this. Use LLF unless the user is very clear otherwise.
    std::string default_flux_s = "llf";
    if (pin->DoesParameterExist("driver", "flux")) {
        default_flux_s = pin->GetString("driver", "flux");
    }
    std::vector<std::string> flux_allowed_vals = {"llf", "hlle"};
    std::string flux =
        pin->GetOrAddString("flux", "type", default_flux_s, flux_allowed_vals);
    params.Add("use_hlle", (flux == "hlle"));

    // Reconstruction scheme
    // Allow from all the places it's ever been
    std::string default_recon_s = "weno5";
    if (pin->DoesParameterExist("driver", "reconstruction")) {
        default_recon_s = pin->GetString("driver", "reconstruction");
    } else if (pin->DoesParameterExist("GRMHD", "reconstruction")) {
        default_recon_s = pin->GetString("GRMHD", "reconstruction");
    }
    // Probably nobody has specified donor_cell_c in years, should remove
    // Indicated cell-wise vs row-wise donor cell recon
    std::vector<std::string> recon_allowed_vals = {"donor_cell", "donor_cell_c",
        "linear_vl", "linear_mc", "weno5", "weno5_linear", "ppm", "ppmx", "mp5"};
    std::string recon = pin->GetOrAddString(
        "flux", "reconstruction", default_recon_s, recon_allowed_vals);
    bool lower_edges = pin->GetOrAddBoolean("flux", "low_order_edges", false);
    bool lower_poles = pin->GetOrAddBoolean("flux", "low_order_poles", false);
    if (lower_edges && lower_poles)
        throw std::runtime_error(
            "Cannot enable lowered reconstruction on edges and poles!");
    if ((lower_edges || lower_poles) && recon != "weno5")
        throw std::runtime_error(
            "Lowered reconstructions can only be enabled with weno5!");

    int stencil = 0;
    if (recon == "donor_cell" || recon == "donor_cell_c") {
        params.Add("recon", KReconstruction::Type::donor_cell);
        stencil = 1;
    } else if (recon == "linear_vl") {
        params.Add("recon", KReconstruction::Type::linear_vl);
        stencil = 3;
    } else if (recon == "linear_mc") {
        params.Add("recon", KReconstruction::Type::linear_mc);
        stencil = 3;
    } else if (recon == "weno5" && lower_edges) {
        params.Add("recon", KReconstruction::Type::weno5_lower_edges);
        stencil = 5;
    } else if (recon == "weno5" && lower_poles) {
        params.Add("recon", KReconstruction::Type::weno5_lower_poles);
        stencil = 5;
    } else if (recon == "weno5") {
        params.Add("recon", KReconstruction::Type::weno5);
        stencil = 5;
    } else if (recon == "weno5_linear") {
        params.Add("recon", KReconstruction::Type::weno5_linear);
        stencil = 5;
    } else if (recon == "ppm") {
        params.Add("recon", KReconstruction::Type::ppm);
        stencil = 5;
    } else if (recon == "ppmx") {
        params.Add("recon", KReconstruction::Type::ppmx);
        stencil = 5;
        std::cout << "KHARMA WARNING: PPMX reconstruction implemention has known bugs."
                  << std::endl
                  << "Use at your own risk!" << std::endl;
    } else if (recon == "mp5") {
        params.Add("recon", KReconstruction::Type::mp5);
        stencil = 5;
    } // we only allow these options
    // Warn if using less than 3 ghost zones w/WENO etc, 2 w/Linear, etc.
    // SMR/AMR independently requires an even number of zones, so we usually use 4
    if (Globals::nghost < (stencil / 2 + 1)) {
        throw std::runtime_error("Not enough ghost zones for specified reconstruction!");
    }

    // Fallback to TVD reconstruction when these algorithms reconstruct something outside
    // the floors
    bool default_recon_fallback = (recon == "weno5" || recon == "weno5_linear" ||
                                   recon == "mp5" || recon == "ppmx");
    bool reconstruction_fallback =
        pin->GetOrAddBoolean("flux", "reconstruction_fallback", default_recon_fallback);
    params.Add("reconstruction_fallback", reconstruction_fallback);
    // Alternatively just apply the geometric floors in fluid frame like a heathen
    bool reconstruction_floors =
        pin->GetOrAddBoolean("flux", "reconstruction_floors", false);
    params.Add("reconstruction_floors", reconstruction_floors);

    // When calculating the fluxes, replace perpendicular fields (e.g. B2 at F2) with
    // the value already present at the face
    // Schemes universally do this, and it is very inadvisable to disable this
    bool consistent_face_b = false;
    if (packages->AllPackages().count("B_CT")) {
        bool default_consistent_b = true;
        if (pin->DoesParameterExist("b_field", "consistent_face_b")) {
            default_consistent_b = pin->GetBoolean("b_field", "consistent_face_b");
        }
        consistent_face_b =
            pin->GetOrAddBoolean("flux", "consistent_face_b", default_consistent_b);
        params.Add("consistent_face_b", consistent_face_b);
    }

    // We can't just use GetVariables or something since there's no mesh yet.
    // That's what this function is for.
    int nvar =
        StateDescriptor::CreateResolvedStateDescriptor(*packages)->GetPackDimension(
            Metadata::WithFluxes);
    std::vector<int> s_flux({nvar});
    if (packages->Get("Globals")->Param<int>("verbose") > 2)
        std::cout << "Allocating fluxes for " << nvar << " variables" << std::endl;
    // TODO optionally move all these to faces? Not important yet, & faces have no output,
    // more memory
    std::vector<MetadataFlag> flags_flux = {
        Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy};
    Metadata m = Metadata(flags_flux, s_flux);
    pkg->AddField("Flux.Pr", m);
    pkg->AddField("Flux.Pl", m);
    pkg->AddField("Flux.Ur", m);
    pkg->AddField("Flux.Ul", m);
    pkg->AddField("Flux.Fr", m);
    pkg->AddField("Flux.Fl", m);

    std::vector<int> s_vector({NVEC});
    std::vector<MetadataFlag> flags_speed = {
        Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy};
    m = Metadata(flags_speed, s_vector);
    pkg->AddField("Flux.cmax", m);
    pkg->AddField("Flux.cmin", m);

    // PROCESS FOFC
    // Accept this a bunch of places, maybe we'll trim this...
    bool default_fofc = false;
    if (pin->DoesParameterExist("driver", "fofc")) {
        default_fofc = pin->GetBoolean("driver", "fofc");
    } else if (pin->DoesParameterExist("flux", "fofc")) {
        default_fofc = pin->GetBoolean("flux", "fofc");
    }
    bool use_fofc = pin->GetOrAddBoolean("fofc", "on", default_fofc);
    params.Add("use_fofc", use_fofc);

    if (use_fofc) {
        if (!packages->AllPackages().count("Floors"))
            throw std::runtime_error(
                "First-order Flux Corrections cannot be used without floors!");

        // FOFC-specific options
        bool use_glf = pin->GetOrAddBoolean("fofc", "use_glf", false);
        params.Add("fofc_use_glf", use_glf);

        bool use_source_term = pin->GetOrAddBoolean("fofc", "use_source_term", false);
        params.Add("fofc_use_source_term", use_source_term);

        int fofc_polar_cells = pin->GetOrAddInteger("fofc", "polar_cells", 0);
        params.Add("fofc_polar_cells", fofc_polar_cells);
        // Usually we use LLF everywhere and this fallback is optional.
        // If we use HLLE outside EH, we need to fall back to LLF/donor-cell inside.
        const bool use_eh_buffer =
            pin->GetOrAddBoolean("fofc", "use_eh_buffer", (flux != "llf"));
        params.Add("fofc_use_eh_buffer", use_eh_buffer);
        if (use_eh_buffer) {
            const GReal eh_buffer = pin->GetOrAddReal("fofc", "eh_buffer", 0.1);
            params.Add("fofc_eh_buffer", eh_buffer);
        }

        if (packages->AllPackages().count("B_CT")) {
            // Use consistent B for FOFC (see above)
            // It is mildly inadvisable to disable this
            bool fofc_consistent_face_b =
                pin->GetOrAddBoolean("fofc", "consistent_face_b", consistent_face_b);
            params.Add("fofc_consistent_face_b", fofc_consistent_face_b);
        }

        // Use a custom block for fofc floors.  We now do the same for Kastaun, where we
        // can *also* have floors
        // TODO even post-reconstruction/reconstruction fallback?
        if (!pin->DoesBlockExist("fofc_floors")) {
            params.Add("fofc_prescription", Floors::MakePrescription(pin, "floors"));
            if (pin->DoesBlockExist("floors_inner"))
                params.Add("fofc_prescription_inner",
                    Floors::MakePrescriptionInner(
                        pin, Floors::MakePrescription(pin, "floors"), "floors_inner"));
            else
                params.Add("fofc_prescription_inner",
                    Floors::MakePrescriptionInner(
                        pin, Floors::MakePrescription(pin, "floors"), "floors"));
        } else {
            // Override inner and outer floors with `fofc_floors` block
            params.Add("fofc_prescription", Floors::MakePrescription(pin, "fofc_floors"));
            params.Add("fofc_prescription_inner",
                Floors::MakePrescriptionInner(
                    pin, Floors::MakePrescription(pin, "fofc_floors"), "fofc_floors"));
        }

        // Flag for whether FOFC was applied, for diagnostics
        // This could be another bitflag in fflag, but that would be really confusing...
        Metadata m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived,
            Metadata::OneCopy, Metadata::FillGhost});
        pkg->AddField("flags.fofc", m);

        // List (vector) of HistoryOutputVars that will all be enrolled as output
        // variables
        parthenon::HstVar_list hst_vars = {};
        // Count total floors as a history item
        hst_vars.emplace_back(parthenon::HistoryOutputVar(
            UserHistoryOperation::max, CountFOFCFlags, "FOFCFlags"));
        // TODO Domain::entire version?
        // TODO entries for each individual flag?
        // add callbacks for HST output to the Params struct, identified by the
        // `hist_param_key`
        pkg->AddParam<>(parthenon::hist_param_key, hst_vars);
    }

    // We register the geometric (\Gamma*T) source here
    pkg->AddSource = Flux::AddGeoSource;

    // And the post-step diagnostics
    pkg->PostStepDiagnosticsMesh = Flux::PostStepDiagnostics;

    EndFlag();
    return pkg;
}

TaskStatus Flux::BlockPtoUMHD(MeshBlockData<Real>* rc, IndexDomain domain, bool coarse)
{
    // Pointers
    auto pmb = rc->GetBlockPointer();
    // Options
    const auto& pars = pmb->packages.Get("GRMHD")->AllParams();
    const Real gam = pars.Get<Real>("gamma");

    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb->packages);

    // Pack variables
    PackIndexMap prims_map, cons_map;
    const auto& P = rc->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    const auto& U = rc->PackVariables({Metadata::Conserved}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const IndexRange ib = bounds.GetBoundsI(domain);
    const IndexRange jb = bounds.GetBoundsJ(domain);
    const IndexRange kb = bounds.GetBoundsK(domain);

    const auto& G = pmb->coords;

    pmb->par_for("p_to_u_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                 KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            Flux::p_to_u_mhd(G, P, m_p, emhd_params, gam, k, j, i, U, m_u);
        });

    return TaskStatus::complete;
}

TaskStatus Flux::BlockPtoU(MeshBlockData<Real>* rc, IndexDomain domain, bool coarse)
{
    // Pointers
    auto pmb = rc->GetBlockPointer();
    // Options
    const auto& pars = pmb->packages.Get("GRMHD")->AllParams();
    const Real gam = pars.Get<Real>("gamma");

    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb->packages);

    // Make sure we don't step on face CT: unnecessary so far, might fix ordering mistakes
    if (pmb->packages.AllPackages().count("B_CT")) B_CT::BlockUtoP(rc, domain, coarse);

    // Pack variables
    PackIndexMap prims_map, cons_map;
    const auto& P = rc->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    const auto& U = rc->PackVariables({Metadata::Conserved}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    const int nvar = U.GetDim(4);

    // Return if we're not syncing U & P at all (e.g. edges)
    if (P.GetDim(4) == 0) return TaskStatus::complete;

    // Indices
    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const IndexRange ib = bounds.GetBoundsI(domain);
    const IndexRange jb = bounds.GetBoundsJ(domain);
    const IndexRange kb = bounds.GetBoundsK(domain);

    const auto& G = pmb->coords;

    pmb->par_for("p_to_u", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                 KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            Flux::p_to_u(G, P, m_p, emhd_params, gam, k, j, i, U, m_u);
        });

    return TaskStatus::complete;
}

TaskStatus Flux::MeshPtoU(MeshData<Real>* md, IndexDomain domain, bool coarse)
{
    for (int i = 0; i < md->NumBlocks(); ++i)
        Flux::BlockPtoU(md->GetBlockData(i).get(), domain, coarse);
    return TaskStatus::complete;
}

TaskStatus Flux::BlockPtoU_Send(MeshBlockData<Real>* rc, IndexDomain domain, bool coarse)
{
    // Pointers
    auto pmb = rc->GetBlockPointer();
    const int ndim = pmb->pmy_mesh->ndim;
    // Options
    const auto& pars = pmb->packages.Get("GRMHD")->AllParams();
    const Real gam = pars.Get<Real>("gamma");

    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb->packages);

    // Pack variables. We never want to run this on the B field
    using FC = Metadata::FlagCollection;
    auto cons_flags =
        FC(Metadata::Conserved, Metadata::Cell, Metadata::GetUserFlag("HD"));
    if (pmb->packages.AllPackages().count("EMHD"))
        cons_flags = cons_flags + FC(Metadata::Conserved, Metadata::Cell,
                                      Metadata::GetUserFlag("EMHDVar"));
    PackIndexMap prims_map, cons_map;
    const auto& P = rc->PackVariables(
        {Metadata::GetUserFlag("Primitive"), Metadata::Cell}, prims_map);
    const auto& U = rc->PackVariables(cons_flags, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    // Return if we're not syncing U & P at all (e.g. edges)
    if (P.GetDim(4) == 0) return TaskStatus::complete;

    // Make sure we always update center conserved B from the faces, not the prims
    if (pmb->packages.AllPackages().count("B_CT"))
        B_CT::BlockUtoP(rc, IndexDomain::interior, coarse);

    // Indices
    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    IndexRange ib = bounds.GetBoundsI(domain);
    IndexRange jb = bounds.GetBoundsJ(domain);
    IndexRange kb = bounds.GetBoundsK(domain);

    // Modify the bounds to reflect zones we're sending, rather than actual ghosts
    int ng = Globals::nghost;
    if (domain == IndexDomain::inner_x1) {
        ib.s += ng;
        ib.e += ng;
    } else if (domain == IndexDomain::outer_x1) {
        ib.s -= ng;
        ib.e -= ng;
    } else if (domain == IndexDomain::inner_x2) {
        if (ndim < 2) return TaskStatus::complete;
        jb.s += ng;
        jb.e += ng;
    } else if (domain == IndexDomain::outer_x2) {
        if (ndim < 2) return TaskStatus::complete;
        jb.s -= ng;
        jb.e -= ng;
    } else if (domain == IndexDomain::inner_x3) {
        if (ndim < 3) return TaskStatus::complete;
        kb.s += ng;
        kb.e += ng;
    } else if (domain == IndexDomain::outer_x3) {
        if (ndim < 3) return TaskStatus::complete;
        kb.s -= ng;
        kb.e -= ng;
    } // TODO(CEP) error?

    const auto& G = pmb->coords;

    pmb->par_for("p_to_u_send", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                 KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            Flux::p_to_u(G, P, m_p, emhd_params, gam, k, j, i, U, m_u);
        });

    return TaskStatus::complete;
}

void Flux::AddGeoSource(MeshData<Real>* md, MeshData<Real>* mdudt, IndexDomain domain)
{
    // Pointers
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    auto pkgs = pmb0->packages;
    // Options
    const auto& pars = pkgs.Get("GRMHD")->AllParams();
    const Real gam = pars.Get<Real>("gamma");

    // All connection coefficients are zero in Cartesian Minkowski space
    // TODO do we know this fully in init?
    if (pmb0->coords.coords.is_cart_minkowski()) return;

    // Pack variables
    PackIndexMap prims_map, cons_map;
    auto P = md->PackVariables(
        std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prims_map);
    auto dUdt =
        mdudt->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map);
    const VarMap m_p(prims_map, false), m_u(cons_map, true);

    // EMHD params
    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb0->packages);

    // Get sizes
    IndexRange3 bd = KDomain::GetRange(md, domain);
    auto block = IndexRange{0, P.GetDim(5) - 1};

    pmb0->par_for("tmunu_source", block.s, block.e, bd.ks, bd.ke, bd.js, bd.je, bd.is,
        bd.ie,
        KOKKOS_LAMBDA(const int& b, const int& k, const int& j, const int& i)
        {
            const auto& G = dUdt.GetCoords(b);
            FourVectors D;
            GRMHD::calc_4vecs(G, P(b), m_p, k, j, i, Loci::center, D);
            // Call Flux::calc_tensor which will in turn call the right
            // calc_tensor based on the number of primitives
            Real Tmu[GR_DIM] = {0};
            Real new_du[GR_DIM] = {0};
            for (int mu = 0; mu < GR_DIM; ++mu) {
                Flux::calc_tensor(P(b), m_p, D, emhd_params, gam, k, j, i, mu, Tmu);
                for (int nu = 0; nu < GR_DIM; ++nu) {
                    // Contract mhd stress tensor with connection, and multiply
                    // by metric determinant
                    for (int lam = 0; lam < GR_DIM; ++lam) {
                        new_du[lam] += Tmu[nu] * G.gdet_conn(j, i, nu, lam, mu);
                    }
                }
            }

            dUdt(b, m_u.UU, k, j, i) += new_du[0];
            VLOOP
                dUdt(b, m_u.U1 + v, k, j, i) += new_du[1 + v];
        });
}

TaskStatus Flux::CheckCtop(MeshData<Real>* md)
{
    Reductions::DomainReduction<Reductions::Var::nan_ctop, UserHistoryOperation::sum,
        int>(md, 0);
    Reductions::DomainReduction<Reductions::Var::zero_ctop, UserHistoryOperation::sum,
        int>(md, 1);
    return TaskStatus::complete;
}

TaskStatus Flux::PostStepDiagnostics(const SimTime& tm, MeshData<Real>* md)
{
    auto pmesh = md->GetMeshPointer();
    // Options
    const auto& globals = pmesh->packages.Get("Globals")->AllParams();
    const int extra_checks = globals.Get<int>("extra_checks");
    const int flag_verbose = globals.Get<int>("flag_verbose");
    const auto& flux_pars = pmesh->packages.Get("Fluxes")->AllParams();
    const bool use_fofc = flux_pars.Get<bool>("use_fofc");

    // Debugging/diagnostic info about FOFC hits
    if (use_fofc && flag_verbose > 0) {
        std::map<int, std::string> fofc_label = {{1, "Flux-corrected"}};
        Reductions::StartFlagReduce(
            md, "flags.fofc", fofc_label, IndexDomain::interior, false, 10);
        // Debugging/diagnostic info about floor and inversion flags
        Reductions::CheckFlagReduceAndPrintHits(
            md, "flags.fofc", fofc_label, IndexDomain::interior, false, 10);
    }

    // Check for a soundspeed (ctop) of 0 or NaN
    // This functions as a "last resort" check to stop a
    // simulation on obviously bad data
    if (extra_checks > 0) {
        int nnan = Reductions::Check<int>(md, 0);
        int nzero = Reductions::Check<int>(md, 1);

        if (MPIRank0() && (nzero > 0 || nnan > 0)) {
            // TODO string formatting in C++ that doesn't suck
            fprintf(stderr, "Max signal speed ctop of 0 or NaN (%d zero, %d NaN)", nzero,
                nnan);
            throw std::runtime_error("Bad ctop!");
        }
    }

    return TaskStatus::complete;
}
