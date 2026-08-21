/*
 *  File: floors.cpp
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
#include "floors.hpp"
#include "floors_functions.hpp"
#include "floors_impl.hpp"

#include "domain.hpp"
#include "grmhd.hpp"
#include "grmhd_functions.hpp"
#include "inverter.hpp"
#include "kharma_driver.hpp"
#include "pack.hpp"

// Floors.  Apply limits to fluid values to maintain integrable state

std::shared_ptr<KHARMAPackage> Floors::Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Floors");
    Params& params = pkg->AllParams();

    // Parse all the particular floor values into a nice struct we can pass device-side
    params.Add("prescription", MakePrescription(pin));

    // Frame to apply floors: usually we use normal observer frame, but
    // the option exists to use the fluid frame exclusively 'fluid' or outside a
    // certain radius 'mixed'. This option adds fluid at speed, making results
    // less reliable but velocity reconstructions potentially more robust.
    // Drift frame floors are now available and preferred when using
    // the implicit solver to avoid UtoP calls.
    // TODO(CEP) automate/standardize parsing enums like this: classes w/tables like the
    // flags?
    std::vector<std::string> allowed_floor_frames = {
        "normal", "fluid", "mixed", "mixed_fluid_normal", "mixed_normal_drift", "drift"};
    std::string frame_s =
        pin->GetOrAddString("floors", "frame", "normal", allowed_floor_frames);
    InjectionFrame frame;
    if (frame_s == "normal") {
        if (pin->DoesBlockExist("inverter") &&
            pin->GetOrAddString("inverter", "type", "kastaun") == "onedw") {
            frame = InjectionFrame::normal_onedw;
        } else {
            // Use Kastaun unless we specified onedw inverter
            frame = InjectionFrame::normal_kastaun;
        }
    } else if (frame_s == "fluid") {
        frame = InjectionFrame::fluid;
    } else if (frame_s == "mixed" || frame_s == "mixed_fluid_normal") {
        frame = InjectionFrame::mixed_fluid_normal;
    } else if (frame_s == "mixed_normal_drift") {
        frame = InjectionFrame::mixed_normal_drift;
    } else if (frame_s == "drift") {
        frame = InjectionFrame::drift;
    }
    params.Add("frame", frame);

    // New inverter w/recovery only supports normal frame floors. Yell about it, but allow
    // overriding if (pin->DoesBlockExist("inverter") &&
    //     pin->GetString("inverter", "type") == "kastaun" &&
    //     frame != InjectionFrame::normal_kastaun) {
    //     if (!pin->GetOrAddBoolean("floors", "allow_unsafe", false)) {
    //         std::cerr << "With version 2025.10, KHARMA dramatically changed how floors
    //         are applied.\n"
    //                   << "The resulting algorithm is much more stable, but requires
    //                   that floors be applied in the normal observer frame.\n"
    //                   << "Consider using the <floors> parameters from
    //                   pars/tori_3d/mad.par.\n"
    //                   << "If you know what you're doing, set floors/allow_unsafe=true";
    //         throw std::runtime_error("Unsafe floors requested without override");
    //     }
    // }

    // Switch points for "mixed" frames
    // TODO no-ops under new floors
    if (frame == InjectionFrame::mixed_fluid_normal) {
        GReal frame_switch_r = pin->GetOrAddReal("floors", "frame_switch_r", 50.);
        params.Add("frame_switch_r", frame_switch_r);
    } else if (frame == InjectionFrame::mixed_normal_drift) {
        GReal frame_switch_beta = pin->GetOrAddReal("floors", "frame_switch_beta", 10.);
        params.Add("frame_switch_beta", frame_switch_beta);
    }

    // Radius-dependent floors and ceiling in a given frame
    // Default presciption struct refers to outer domain, i.e., beyond 'floor_switch_r'
    // Make an additional prescription struct for inner domain.
    // There are two Floors::Prescription objects even if there is no radius-dependence,
    // the values will simply be the same if radius-dependent floors are not enabled.
    // Avoids a bunch of if (radius_dependent_floors) else while determining floors.
    if (pin->DoesBlockExist("floors_inner"))
        params.Add(
            "prescription_inner", MakePrescriptionInner(pin, MakePrescription(pin)));
    else
        params.Add("prescription_inner",
            MakePrescriptionInner(pin, MakePrescription(pin)), "floors");

    // All of these are now the same option: disable the *call* only.
    // This lets us assume that the floors package is loaded, which is convenient many
    // places
    bool floors_on_default = !pin->GetOrAddBoolean("floors", "disable_floors", false);
    bool floors_on = pin->GetOrAddBoolean("floors", "on", floors_on_default);
    // Sometimes we want the floors package, but don't want to apply anything, for tests
    bool disable_call = pin->GetOrAddBoolean("floors", "disable_call", !floors_on);
    params.Add("disable_call", disable_call);

    // Track all conserved variable changes not produced by fluxes
    bool track_additions = pin->GetOrAddBoolean("floors", "track_additions", false);
    params.Add("track_additions", track_additions);

    // These preserve floor values between the "mark" pass and the actual floor
    // application We need them even if floors are disabled, to apply initial values based
    // on some prescription as a part of problem setup
    Metadata m =
        Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("Floors.rho_floor", m);
    pkg->AddField("Floors.u_floor", m);

    // Flag for which floor conditions were violated.  Used for diagnostics
    // TODO(CEP) Should switch these to "Integer" fields when Parthenon supports it
    pkg->AddField("fflag", m);
    // When not using UtoP, we still need a "dummy" copy of pflag to write the
    // post-flooring flag to
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy,
        Metadata::Overridable});
    pkg->AddField("pflag", m);

    if (track_additions) {
        // Track total additions to conserved variables
        // TODO add primitive versions, advect as passives
        m = Metadata(
            {Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::Conserved});
        pkg->AddField("Floors.rhou0add", m);
        std::vector<int> s_4v({4});
        m = Metadata(
            {Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::Conserved},
            s_4v);
        pkg->AddField("Floors.Tadd", m);
        // TODO(CEP) Maybe also want Floors.floorUadd, etc etc, for understanding the
        // breakdown...
    }

    // Don't actually call the usual floor function if we're using normal frame w/Kastaun,
    // floors will be applied during the inversion call.
    // Also allow manually disabling the call, for testing
    if (!disable_call) {
        // TODO(CEP) THIS IS THE ONLY MeshApplyFloors.  Any others will NOT BE CALLED.
        // Use BlockApplyFloors in your packages or fix Packages::MeshApplyFloors
        pkg->MeshApplyFloors = Floors::ApplyGRMHDFloors;
    }
    pkg->PostStepDiagnosticsMesh = Floors::PostStepDiagnostics;

    // We still need to look after the fflag, even if we're not adding to it
    pkg->PreStepWork = Floors::PreStepWork;
    pkg->PostStepDiagnosticsMesh = Floors::PostStepDiagnostics;

    // List (vector) of HistoryOutputVars that will all be enrolled as output variables
    parthenon::HstVar_list hst_vars = {};
    // Count total floors as a history item
    hst_vars.emplace_back(
        parthenon::HistoryOutputVar(UserHistoryOperation::sum, CountFFlags, "FFlags"));
    // TODO Domain::entire version?
    // TODO entries for each individual flag?
    if (track_additions) {
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum,
            Reductions::Total<Reductions::Var::rhou0add>, "rhou0add"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum,
            Reductions::Total<Reductions::Var::T00add>, "T00add"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum,
            Reductions::Total<Reductions::Var::T01add>, "T01add"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum,
            Reductions::Total<Reductions::Var::T02add>, "T02add"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum,
            Reductions::Total<Reductions::Var::T03add>, "T03add"));

        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum,
            Reductions::Total<Reductions::Var::rhou0sub>, "rhou0sub"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum,
            Reductions::Total<Reductions::Var::T00sub>, "T00sub"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum,
            Reductions::Total<Reductions::Var::T01sub>, "T01sub"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum,
            Reductions::Total<Reductions::Var::T02sub>, "T02sub"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum,
            Reductions::Total<Reductions::Var::T03sub>, "T03sub"));
    }
    // add callbacks for HST output to the Params struct, identified by the
    // `hist_param_key`
    pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

    return pkg;
}

TaskStatus Floors::ApplyInitialFloors(
    ParameterInput* pin, MeshBlockData<Real>* mbd, IndexDomain domain)
{
    Flag("ApplyInitialFloors");

    auto pmb = mbd->GetBlockPointer();

    PackIndexMap prims_map, cons_map;
    auto P = mbd->PackVariables(
        {Metadata::GetUserFlag("Primitive"), Metadata::Cell}, prims_map);
    auto U = mbd->PackVariables(
        std::vector<MetadataFlag>{Metadata::Conserved, Metadata::Cell}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    const auto& G = pmb->coords;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    // If we're going to apply floors through the run, apply the same ones at init
    // Otherwise stick to specified/default geometric floors
    Floors::Prescription floors_tmp;
    if (pmb->packages.AllPackages().count("Floors")) {
        floors_tmp =
            pmb->packages.Get("Floors")->Param<Floors::Prescription>("prescription");
    } else {
        // JUST rho & u geometric
        floors_tmp.rho_min_geom = pin->GetOrAddReal("floors", "rho_min_geom", 1e-6);
        floors_tmp.u_min_geom = pin->GetOrAddReal("floors", "u_min_geom", 1e-8);
        floors_tmp.rho_min_const = pin->GetOrAddReal("floors", "rho_min_const", 1e-20);
        floors_tmp.u_min_const = pin->GetOrAddReal("floors", "u_min_const", 1e-20);

        // Disable everything else, even if it's specified
        floors_tmp.bsq_over_rho_max = 1e20;
        floors_tmp.bsq_over_u_max = 1e20;
        floors_tmp.u_over_rho_max = 1e20;
        floors_tmp.ktot_max = 1e20;
        floors_tmp.gamma_max = 1e20;

        floors_tmp.use_r_char = false;
        floors_tmp.r_char = 0.; // unused
        floors_tmp.temp_adjust_u = false;
        floors_tmp.adjust_k = false;
    }
    const Floors::Prescription floors = floors_tmp;

    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb->packages);

    const IndexRange3 b = KDomain::GetRange(mbd, domain);
    pmb->par_for("apply_initial_floors", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
        {
            Real rhoflr_max, uflr_max;
            // Initial floors, so the radius-dependence of floors don't matter that much.
            int fflag = determine_floors(
                G, P, m_p, gam, k, j, i, floors, floors, rhoflr_max, uflr_max);
            if (fflag) {
                apply_floors<InjectionFrame::fluid>(
                    G, P, m_p, gam, k, j, i, rhoflr_max, uflr_max, U, m_u);
                apply_ceilings(G, P, m_p, gam, k, j, i, floors, floors, U, m_u);
                // P->U for any modified zones
                Flux::p_to_u_mhd(
                    G, P, m_p, emhd_params, gam, k, j, i, U, m_u, Loci::center);
            }
        });

    EndFlag();
    return TaskStatus::complete;
}

TaskStatus Floors::DetermineGRMHDFloors(MeshData<Real>* md, IndexDomain domain,
    const Floors::Prescription& floors, const Floors::Prescription& floors_inner)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // Packs of prims and cons
    PackIndexMap prims_map;
    auto& P = md->PackVariables(
        std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prims_map);
    const VarMap m_p(prims_map, false);

    auto fflag = md->PackVariables(std::vector<std::string>{"fflag"});
    auto pflag = md->PackVariables(std::vector<std::string>{"pflag"});
    PackIndexMap floors_map;
    auto floor_vals = md->PackVariables(
        std::vector<std::string>{"Floors.rho_floor", "Floors.u_floor"}, floors_map);
    const int rhofi = floors_map["Floors.rho_floor"].first;
    const int ufi = floors_map["Floors.u_floor"].first;

    const Real gam = pmb0->packages.Get("GRMHD")->Param<Real>("gamma");

    const IndexRange3 b = KDomain::GetRange(md, domain);
    const IndexRange block = IndexRange{0, P.GetDim(5) - 1};
    pmb0->par_for("determine_floors", block.s, block.e, b.ks, b.ke, b.js, b.je, b.is,
        b.ie,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i)
        {
            const auto& G = P.GetCoords(b);
            // The inverter might have set some floor flags, so we add to that
            // non-destructively
            fflag(b, 0, k, j, i) =
                static_cast<int>(fflag(b, 0, k, j, i)) |
                determine_floors(G, P(b), m_p, gam, k, j, i, floors, floors_inner,
                    floor_vals(b, rhofi, k, j, i), floor_vals(b, ufi, k, j, i));
        });

    // TODO(CEP) if we can somehow guarantee one call/rank we can start the reduction here
    // Reductions::StartFlagReduce(md, "fflag", FFlag::flag_names, IndexDomain::interior,
    // true, 0);

    return TaskStatus::complete;
}

TaskStatus Floors::ApplyGRMHDFloors(MeshData<Real>* md, IndexDomain domain)
{
    auto pmesh = md->GetMeshPointer();
    const auto& pars = pmesh->packages.Get("Floors")->AllParams();

    // Determine floors
    const Floors::Prescription floors = pars.Get<Floors::Prescription>("prescription");
    const Floors::Prescription floors_inner =
        pars.Get<Floors::Prescription>("prescription_inner");
    DetermineGRMHDFloors(md, domain, floors, floors_inner);

    if (pars.Get<InjectionFrame>("frame") == InjectionFrame::normal_kastaun) {
        return ApplyFloorsInFrame<InjectionFrame::normal_kastaun>(md, domain);
    } else if (pars.Get<InjectionFrame>("frame") == InjectionFrame::normal_onedw) {
        return ApplyFloorsInFrame<InjectionFrame::normal_onedw>(md, domain);
    } else if (pars.Get<InjectionFrame>("frame") == InjectionFrame::fluid) {
        return ApplyFloorsInFrame<InjectionFrame::fluid>(md, domain);
    } else if (pars.Get<InjectionFrame>("frame") == InjectionFrame::mixed_fluid_normal) {
        return ApplyFloorsInFrame<InjectionFrame::mixed_fluid_normal>(md, domain);
    } else if (pars.Get<InjectionFrame>("frame") == InjectionFrame::mixed_normal_drift) {
        return ApplyFloorsInFrame<InjectionFrame::mixed_normal_drift>(md, domain);
    } else if (pars.Get<InjectionFrame>("frame") == InjectionFrame::drift) {
        return ApplyFloorsInFrame<InjectionFrame::drift>(md, domain);
    } else {
        throw std::invalid_argument("Floors for requested frame not implemented!");
    }
}

TaskStatus Floors::TrackAdditions(MeshData<Real>* md, MeshData<Real>* md_save)
{
    Kokkos::Profiling::pushRegion("Task_TrackAdditions");
    PackIndexMap cons_map, tracks_map;
    const auto& U =
        md->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map);
    const auto& U_save =
        md_save->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved});
    const VarMap m_u(cons_map, true);

    const auto& tU = md->PackVariables(
        std::vector<std::string>{"Floors.rhou0add", "Floors.Tadd"}, tracks_map);
    const int rhou0i = tracks_map["Floors.rhou0add"].first;
    const int Ti = tracks_map["Floors.Tadd"].first;

    parthenon::par_for(DEFAULT_LOOP_PATTERN, "TrackAdditions", DevExecSpace(), 0,
        U.GetDim(5) - 1, 0, U.GetDim(3) - 1, 0, U.GetDim(2) - 1, 0, U.GetDim(1) - 1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i)
        {
            tU(b, rhou0i, k, j, i) = U(b, m_u.RHO, k, j, i) - U_save(b, m_u.RHO, k, j, i);
            tU(b, Ti + 0, k, j, i) = U(b, m_u.UU, k, j, i) - U_save(b, m_u.UU, k, j, i);
            tU(b, Ti + 1, k, j, i) = U(b, m_u.U1, k, j, i) - U_save(b, m_u.U1, k, j, i);
            tU(b, Ti + 2, k, j, i) = U(b, m_u.U2, k, j, i) - U_save(b, m_u.U2, k, j, i);
            tU(b, Ti + 3, k, j, i) = U(b, m_u.U3, k, j, i) - U_save(b, m_u.U3, k, j, i);
        });
    Kokkos::Profiling::popRegion(); // Task_TrackAdditions
    return TaskStatus::complete;
}

int Floors::CountFFlags(MeshData<Real>* md)
{
    return Reductions::CountFlags(
        md, "fflag", FFlag::flag_names, IndexDomain::interior, true)[0];
}

void Floors::PreStepWork(Mesh* pmesh, ParameterInput* pin, const SimTime& tm)
{
    // Clear all floor flags before each step
    auto md = pmesh->mesh_data.Get().get();
    KHARMADriver::Scale(std::vector<std::string>{"fflag"}, md, 0.);
}

TaskStatus Floors::PostStepDiagnostics(const SimTime& tm, MeshData<Real>* md)
{
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Options
    const auto& pars = pmesh->packages.Get("Globals")->AllParams();
    const int flag_verbose = pars.Get<int>("flag_verbose");

    // Debugging/diagnostic info about floor flags
    if (flag_verbose > 0) {
        Reductions::StartFlagReduce(
            md, "fflag", FFlag::flag_names, IndexDomain::interior, true, 0);
        // Debugging/diagnostic info about floor and inversion flags
        Reductions::CheckFlagReduceAndPrintHits(
            md, "fflag", FFlag::flag_names, IndexDomain::interior, true, 0);
    }

    // Anything else (energy conservation? Added material stats?)

    return TaskStatus::complete;
}
