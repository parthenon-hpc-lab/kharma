
/*
 *  File: kharma_driver.cpp
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
#include "kharma_driver.hpp"

#include "b_ct.hpp"
#include "boundaries.hpp"
#include "flux.hpp"
#include "get_flux.hpp"
#include "inverter.hpp"

std::shared_ptr<KHARMAPackage> KHARMADriver::Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages)
{
    // This function builds and returns a "KHARMAPackage" object, which is a light
    // superset of Parthenon's "StateDescriptor" class for packages.
    // The most important part of this object is a member of type "Params",
    // which acts more or less like a Python dictionary:
    // it puts values into a map of names->objects, where "objects" are usually
    // floats, strings, and ints, but can be arbitrary classes.
    // This "dictionary" is mostly immutable, and should always be treated as immutable,
    // except in the "Globals" package.
    auto pkg = std::make_shared<KHARMAPackage>("Driver");
    Params& params = pkg->AllParams();

    // Driver options
    // The two current drivers are "kharma" or "imex", with the former being the usual
    // KHARMA driver (formerly HARM driver), and the latter supporting implicit stepping
    // of some or all variables Mostly, packages should react to e.g. the "sync_prims"
    // option rather than the driver name
    std::vector<std::string> valid_drivers = {"harm", "kharma", "imex", "simple"};
    bool do_emhd = pin->GetOrAddBoolean("emhd", "on", false);
    std::string driver_type_s = pin->GetOrAddString(
        "driver", "type", (do_emhd) ? "imex" : "kharma", valid_drivers);
    DriverType driver_type;
    if (driver_type_s == "harm" || driver_type_s == "kharma") {
        driver_type = DriverType::kharma;
    } else if (driver_type_s == "imex") {
        driver_type = DriverType::imex;
    } else if (driver_type_s == "simple") {
        driver_type = DriverType::simple;
    } // We prevent this
    params.Add("type", driver_type);
    params.Add("name", driver_type_s);

    // Synchronize boundary variables twice. Ensures KHARMA is agnostic to the breakdown
    // of meshblocks, at the cost of twice the MPI overhead, for potentially worse strong
    // scaling. On by default, disable only after testing that, e.g., divB meets your
    // requirements
    bool two_sync = pin->GetOrAddBoolean("driver", "two_sync", true);
    params.Add("two_sync", two_sync);

    // When using the Implicit package we need to globally distinguish implicit & explicit
    // vars All independent variables should be marked one or the other, so we define the
    // flags here to avoid loading order issues
    Metadata::AddUserFlag("Implicit");
    Metadata::AddUserFlag("Explicit");
    // Add a flag if we wish to use ideal variables explicitly evolved as guess for
    // implicit update. GRIM uses the fluid state of the previous (sub-)step. The logic
    // here is that the non-ideal variables do not significantly contribute to the
    // stress-energy tensor. We can explicitly update the ideal MHD variables and hope
    // that this takes us to a region in the parameter space of state variables that is
    // close to the true solution. The corrections obtained from the implicit update would
    // then be small corrections.
    Metadata::AddUserFlag("IdealGuess");

    // 1. One flag to mark the primitive variables specifically
    // (Parthenon has Metadata::Conserved already)
    Metadata::AddUserFlag("Primitive");

    // Finally, a flag for anything used (and possibly sync'd) during startup,
    // but which should not be evolved (or more importantly, sync'd) during main stepping
    Metadata::AddUserFlag("StartupOnly");

    // This is a flag Parthenon should have eventually, but we'll prototype in KHARMA
    // Indicate a 1-element face-centered field is split components of a vector
    Metadata::AddUserFlag("SplitVector");

    // Synchronize primitive variables unless we're using the KHARMA driver that
    // specifically doesn't This includes for AMR w/ImEx driver Note the "conserved" B
    // field is always sync'd.  The "primitive" version only differs by sqrt(-g)
    bool sync_prims = driver_type != DriverType::kharma;
    params.Add("sync_prims", sync_prims);
    if (sync_prims) {
        // For ImEx/simple drivers, sync/prolongate/restrict primitive variables directly
        params.Add(
            "prim_flags", std::vector<MetadataFlag>{Metadata::Real, Metadata::Derived,
                              Metadata::FillGhost, Metadata::GetUserFlag("Primitive")});
        params.Add("cons_flags",
            std::vector<MetadataFlag>{Metadata::Real, Metadata::Independent,
                Metadata::Restart, Metadata::WithFluxes, Metadata::Conserved});
    } else {
        // If we're in AMR or using the KHARMA driver anyway, sync conserved vars
        params.Add(
            "prim_flags", std::vector<MetadataFlag>{Metadata::Real, Metadata::Derived,
                              Metadata::GetUserFlag("Primitive")});
        params.Add(
            "cons_flags", std::vector<MetadataFlag>{Metadata::Real, Metadata::Independent,
                              Metadata::Restart, Metadata::FillGhost,
                              Metadata::WithFluxes, Metadata::Conserved});
    }

    return pkg;
}

void KHARMADriver::AddFullSyncRegion(
    TaskCollection& tc, std::shared_ptr<MeshData<Real>>& md_sync)
{
    const TaskID t_none(0);

    // MPI boundary exchange, done over MeshData objects/partitions at once
    // Parthenon includes physical bounds
    const int num_partitions = pmesh->DefaultNumPartitions(); // Usually 1
    TaskRegion& bound_sync = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto& tl = bound_sync[i];
        AddBoundarySync(t_none, tl, md_sync);
    }
}

TaskID KHARMADriver::AddBoundarySync(
    const TaskID t_start, TaskList& tl, std::shared_ptr<MeshData<Real>>& mc1)
{
    Flag("AddBoundarySync");
    auto t_start_sync = t_start;

    // Pull the mesh pointer from mc1 so we can be a static method
    auto pmesh = mc1->GetMeshPointer();
    auto& params = pmesh->packages.Get("Driver")->AllParams();
    bool multilevel = pmesh->multilevel;

    // The Parthenon exchange tasks include applying physical boundary conditions now.
    // We generally do not take advantage of this yet, but good to know when reasoning
    // about initialization.
    Flag("ParthenonAddSync");
    auto t_sync_done =
        parthenon::AddBoundaryExchangeTasks(t_start_sync, tl, mc1, multilevel);
    auto t_bounds = t_sync_done;
    EndFlag();

    // We always just sync'd the "conserved" magnetic field
    // Translate back to "primitive" (& cell-centered) field if that's what we'll be using
    if (params.Get<bool>("sync_prims")) {
        auto pkgs = pmesh->packages.AllPackages();
        TaskID t_all_utop[mc1->NumBlocks() * BOUNDARY_NFACES];
        TaskID t_utop_final(0);
        int i_task = 0;
        for (int i_block = 0; i_block < mc1->NumBlocks(); i_block++) {
            auto& rc = mc1->GetBlockData(i_block);
            for (int i_bnd = 0; i_bnd < BOUNDARY_NFACES; i_bnd++) {
                if (rc->GetBlockPointer()->boundary_flag[i_bnd] == BoundaryFlag::block ||
                    rc->GetBlockPointer()->boundary_flag[i_bnd] ==
                        BoundaryFlag::periodic) {
                    const auto bdomain = KBoundaries::BoundaryDomain((BoundaryFace)i_bnd);
                    if (pkgs.count("B_FluxCT")) {
                        t_all_utop[i_task] = tl.AddTask(
                            t_sync_done, B_FluxCT::BlockUtoP, rc.get(), bdomain, false);
                    } else if (pkgs.count("B_CT")) {
                        t_all_utop[i_task] = tl.AddTask(
                            t_sync_done, B_CT::BlockUtoP, rc.get(), bdomain, false);
                    }
                    t_utop_final = t_utop_final | t_all_utop[i_task];
                    i_task++;
                }
            }
        }
        t_bounds = t_utop_final;
    }

    EndFlag();
    return t_bounds;
}

TaskStatus KHARMADriver::SyncAllBounds(std::shared_ptr<MeshData<Real>>& md)
{
    Flag("SyncAllBounds");
    TaskID t_none(0);

    MPIBarrier();

    TaskCollection tc;
    auto tr = tc.AddRegion(1);
    AddBoundarySync(t_none, tr[0], md);
    while (!tr.Execute());

    MPIBarrier();

    EndFlag();
    return TaskStatus::complete;
}

TaskID KHARMADriver::AddFluxCalculations(
    TaskID& t_start, TaskList& tl, MeshData<Real>* md)
{
    // Pull reconstruction option to simplify use. TODO shorten?
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    auto& pkgs = pmb0->packages.AllPackages();
    const KReconstruction::Type& recon =
        pkgs.at("Flux")->Param<KReconstruction::Type>("recon");

    // Pre-calculate B field cell-center values
    auto t_start_fluxes = t_start;
    if (md->GetMeshPointer()->packages.AllPackages().count("B_CT"))
        t_start_fluxes =
            tl.AddTask(t_start, B_CT::MeshUtoP, md, IndexDomain::entire, false);

    // Calculate fluxes in each direction using given reconstruction
    // Must be spelled out so as to generate each templated version of GetFlux<> to be
    // available at runtime Details in flux/get_flux.hpp
    // TODO(BSP) This could be a macro, maybe... But there's no easy foreach(enum_val):
    // instantiate pattern
    using RType = KReconstruction::Type;
    TaskID t_calculate_flux1, t_calculate_flux2, t_calculate_flux3;
    switch (recon) {
        case RType::donor_cell:
            t_calculate_flux1 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::donor_cell, X1DIR>, md);
            t_calculate_flux2 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::donor_cell, X2DIR>, md);
            t_calculate_flux3 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::donor_cell, X3DIR>, md);
            break;
        case RType::donor_cell_c:
            t_calculate_flux1 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::donor_cell_c, X1DIR>, md);
            t_calculate_flux2 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::donor_cell_c, X2DIR>, md);
            t_calculate_flux3 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::donor_cell_c, X3DIR>, md);
            break;
        case RType::linear_vl:
            t_calculate_flux1 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::linear_vl, X1DIR>, md);
            t_calculate_flux2 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::linear_vl, X2DIR>, md);
            t_calculate_flux3 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::linear_vl, X3DIR>, md);
            break;
        case RType::linear_mc:
            t_calculate_flux1 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::linear_mc, X1DIR>, md);
            t_calculate_flux2 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::linear_mc, X2DIR>, md);
            t_calculate_flux3 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::linear_mc, X3DIR>, md);
            break;
        case RType::weno5_lower_edges:
            t_calculate_flux1 = tl.AddTask(
                t_start_fluxes, Flux::GetFlux<RType::weno5_lower_edges, X1DIR>, md);
            t_calculate_flux2 = tl.AddTask(
                t_start_fluxes, Flux::GetFlux<RType::weno5_lower_edges, X2DIR>, md);
            t_calculate_flux3 = tl.AddTask(
                t_start_fluxes, Flux::GetFlux<RType::weno5_lower_edges, X3DIR>, md);
            break;
        case RType::weno5_lower_poles:
            t_calculate_flux1 = tl.AddTask(
                t_start_fluxes, Flux::GetFlux<RType::weno5_lower_poles, X1DIR>, md);
            t_calculate_flux2 = tl.AddTask(
                t_start_fluxes, Flux::GetFlux<RType::weno5_lower_poles, X2DIR>, md);
            t_calculate_flux3 = tl.AddTask(
                t_start_fluxes, Flux::GetFlux<RType::weno5_lower_poles, X3DIR>, md);
            break;
        case RType::weno5:
            t_calculate_flux1 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::weno5, X1DIR>, md);
            t_calculate_flux2 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::weno5, X2DIR>, md);
            t_calculate_flux3 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::weno5, X3DIR>, md);
            break;
        case RType::weno5_linear:
            t_calculate_flux1 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::weno5_linear, X1DIR>, md);
            t_calculate_flux2 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::weno5_linear, X2DIR>, md);
            t_calculate_flux3 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::weno5_linear, X3DIR>, md);
            break;
        case RType::ppm:
            t_calculate_flux1 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::ppm, X1DIR>, md);
            t_calculate_flux2 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::ppm, X2DIR>, md);
            t_calculate_flux3 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::ppm, X3DIR>, md);
            break;
        case RType::ppmx:
            t_calculate_flux1 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::ppmx, X1DIR>, md);
            t_calculate_flux2 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::ppmx, X2DIR>, md);
            t_calculate_flux3 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::ppmx, X3DIR>, md);
            break;
        case RType::mp5:
            t_calculate_flux1 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::mp5, X1DIR>, md);
            t_calculate_flux2 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::mp5, X2DIR>, md);
            t_calculate_flux3 =
                tl.AddTask(t_start_fluxes, Flux::GetFlux<RType::mp5, X3DIR>, md);
            break;
        default:
            std::cout << std::endl; // flush messages in output buffer before we error
            throw std::invalid_argument(
                "Unsupported reconstruction algorithm! Main supported algorithms: "
                "linear_mc, weno5, weno5_linear");
    }
    auto t_calc_fluxes = t_calculate_flux1 | t_calculate_flux2 | t_calculate_flux3;

    auto t_ctop = t_calc_fluxes;
    if (md->GetMeshPointer()->packages.Get("Globals")->Param<int>("extra_checks") > 0) {
        auto t_ctop = tl.AddTask(t_calc_fluxes, Flux::CheckCtop, md);
    }

    return t_ctop;
}

TaskID KHARMADriver::AddFOFC(TaskID& t_start, TaskList& tl, MeshData<Real>* md,
    MeshData<Real>* md_full_step_init, MeshData<Real>* md_sub_step_init,
    MeshData<Real>* guess_src, MeshData<Real>* guess, int stage)
{
    Flag("FOFC");
    // Pull reconstruction option to simplify use. TODO shorten?
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    auto& pkgs = pmb0->packages.AllPackages();

    const Floors::Prescription fofc_floors =
        pmb0->packages.Get("Flux")->Param<Floors::Prescription>("fofc_prescription");
    const Floors::Prescription fofc_floors_inner =
        pmb0->packages.Get("Flux")->Param<Floors::Prescription>(
            "fofc_prescription_inner");

    // Populate guess source term with divergence of the existing fluxes
    // NOTE this does not include source terms!  Though, could call them here tbh
    auto t_guess_divergence = tl.AddTask(t_start, FluxDivergence, md, guess_src,
        std::vector<MetadataFlag>{Metadata::Cell, Metadata::WithFluxes}, 3);
    // Add geometric source term to more accurately predict floor hits.
    // Could add everything here with Packages::AddSource but would be slower
    // also would need to deal with B_CT::AddSource == flux update, which we don't
    // want/need
    auto t_guess_sources = t_guess_divergence;
    if (pmb0->packages.Get("Flux")->Param<bool>("fofc_use_source_term")) {
        auto t_guess_sources = tl.AddTask(t_guess_divergence, Flux::AddGeoSourceTask, md,
            guess_src, IndexDomain::entire);
    }
    // Update the guess state with the guess source term, and our existing state
    // Note this includes updating cell-centered B with the fluxes -- we don't care if
    // this version has div
    auto t_guess_update = KHARMADriver::AddStateUpdate(t_guess_sources, tl,
        md_full_step_init, md_sub_step_init, guess_src, guess,
        {Metadata::WithFluxes, Metadata::Cell}, false, stage);
    // Recover primitive variables of the guess (carefully, since code-wide functions
    // respect B at faces!)
    auto t_guess_Bp =
        tl.AddTask(t_guess_update, B_FluxCT::MeshUtoP, guess, IndexDomain::entire, false);
    auto t_guess_prims =
        tl.AddTask(t_guess_Bp, Inverter::MeshUtoP, guess, IndexDomain::entire, false);
    // Check and mark floors
    auto t_mark_floors = tl.AddTask(t_guess_prims, Floors::DetermineGRMHDFloors, guess,
        IndexDomain::entire, fofc_floors, fofc_floors_inner);
    // Determine which cells are FOFC in our block
    auto t_mark_fofc = tl.AddTask(t_mark_floors, Flux::MarkFOFC, guess);
    // Sync with neighbor blocks.  This seems to ameliorate an increasing divB on X1
    // boundaries in GR, but it should be able to be eliminated
    auto& md_fofc = pmesh->mesh_data.AddShallow("FOFC",
        std::vector<std::string>{"fofcflag"}); // TODO this gets weird if we partition
    auto t_sync_fofc = KHARMADriver::AddBoundarySync(t_mark_fofc, tl, md_fofc);
    // Finally, replace any fluxes bordering marked zones with donor-cell/LLF versions
    auto t_fofc = tl.AddTask(t_sync_fofc, Flux::FOFC, md, guess);

    EndFlag();
    return t_fofc;
}

TaskID KHARMADriver::AddStateUpdate(TaskID& t_start, TaskList& tl,
    MeshData<Real>* md_full_step_init, MeshData<Real>* md_sub_step_init,
    MeshData<Real>* md_flux_src, MeshData<Real>* md_update,
    std::vector<MetadataFlag> flags, bool update_face, int stage)
{
    // Update all explicitly-evolved variables using the source term
    // Add any proportion of the step start required by the integrator (e.g., RK2)
    std::vector<MetadataFlag> flags_cell = flags;
    flags_cell.push_back(Metadata::Cell);
    std::vector<MetadataFlag> flags_face = flags;
    flags_face.push_back(Metadata::Face);
    // TODO splitting this is stupid, but maybe the parallelization actually helps? Eh.
    auto t_avg_data_c = tl.AddTask(t_start,
        Update::WeightedSumData<std::vector<MetadataFlag>, MeshData<Real>>,
        std::vector<MetadataFlag>(flags_cell), md_sub_step_init, md_full_step_init,
        integrator->gam0[stage - 1], integrator->gam1[stage - 1], md_update);
    auto t_avg_data = t_avg_data_c;
    if (update_face) {
        t_avg_data = tl.AddTask(t_start, WeightedSumDataFace<MetadataFlag>,
            std::vector<MetadataFlag>(flags_face), md_sub_step_init, md_full_step_init,
            integrator->gam0[stage - 1], integrator->gam1[stage - 1], md_update);
    }
    // apply du/dt to the result
    auto t_update_c = tl.AddTask(t_avg_data,
        Update::WeightedSumData<std::vector<MetadataFlag>, MeshData<Real>>,
        std::vector<MetadataFlag>(flags_cell), md_update, md_flux_src, 1.0,
        integrator->beta[stage - 1] * integrator->dt, md_update);
    auto t_update = t_update_c;
    if (update_face) {
        t_update = tl.AddTask(t_avg_data, WeightedSumDataFace<MetadataFlag>,
            std::vector<MetadataFlag>(flags_face), md_update, md_flux_src, 1.0,
            integrator->beta[stage - 1] * integrator->dt, md_update);
    }

    // We'll be running UtoP after this, which needs a guess in order to converge, so we
    // copy in md_sub_step_init
    auto t_copy_prims = t_update;
    auto pmb0 = md_full_step_init->GetBlockData(0)->GetBlockPointer();
    auto& pkgs = pmb0->packages.AllPackages();

    // If we're explicitly evolving, UtoP needs a guess
    // TODO why is this necessary still?  Is it necessary on every AddStateUpdate?
    if (!pkgs.at("GRMHD")->Param<bool>("implicit")) {
        t_copy_prims = tl.AddTask(t_start, Copy<MeshData<Real>>,
            std::vector<MetadataFlag>(
                {Metadata::GetUserFlag("MHD"), Metadata::GetUserFlag("Primitive")}),
            md_sub_step_init, md_update);
    }

    return t_copy_prims | t_update;
}

TaskID KHARMADriver::AddStateUpdateIdealGuess(TaskID& t_start, TaskList& tl,
    MeshData<Real>* md_full_step_init, MeshData<Real>* md_sub_step_init,
    MeshData<Real>* md_flux_src, MeshData<Real>* md_update,
    std::vector<MetadataFlag> flags, bool update_face, int stage)
{
    // Update variables marked as IdealGuess
    // Add any proportion of the step start required by the integrator (e.g., RK2)
    std::vector<MetadataFlag> flags_cell = flags;
    flags_cell.push_back(Metadata::Cell);
    std::vector<MetadataFlag> flags_face = flags;
    flags_face.push_back(Metadata::Face);
    // TODO splitting this is stupid, but maybe the parallelization actually helps? Eh.
    auto t_avg_data_c = tl.AddTask(t_start,
        Update::WeightedSumData<std::vector<MetadataFlag>, MeshData<Real>>,
        std::vector<MetadataFlag>(flags_cell), md_sub_step_init, md_full_step_init,
        integrator->gam0[stage - 1], integrator->gam1[stage - 1], md_update);
    auto t_avg_data = t_avg_data_c;
    if (update_face) {
        t_avg_data = tl.AddTask(t_start, WeightedSumDataFace<MetadataFlag>,
            std::vector<MetadataFlag>(flags_face), md_sub_step_init, md_full_step_init,
            integrator->gam0[stage - 1], integrator->gam1[stage - 1], md_update);
    }
    // apply du/dt to the result
    auto t_update_c = tl.AddTask(t_avg_data,
        Update::WeightedSumData<std::vector<MetadataFlag>, MeshData<Real>>,
        std::vector<MetadataFlag>(flags_cell), md_update, md_flux_src, 1.0,
        integrator->beta[stage - 1] * integrator->dt, md_update);
    auto t_update = t_update_c;
    if (update_face) {
        t_update = tl.AddTask(t_avg_data, WeightedSumDataFace<MetadataFlag>,
            std::vector<MetadataFlag>(flags_face), md_update, md_flux_src, 1.0,
            integrator->beta[stage - 1] * integrator->dt, md_update);
    }

    // We'll be running UtoP after this, which needs a guess in order to converge, so we
    // copy in md_sub_step_init
    auto t_copy_prims = t_update;
    auto pmb0 = md_full_step_init->GetBlockData(0)->GetBlockPointer();
    auto& pkgs = pmb0->packages.AllPackages();
    if (pkgs.at("GRMHD")->Param<bool>("ideal_guess")) {
        t_copy_prims = tl.AddTask(t_start, Copy<MeshData<Real>>,
            std::vector<MetadataFlag>(
                {Metadata::GetUserFlag("HD"), Metadata::GetUserFlag("Primitive")}),
            md_sub_step_init, md_update);
    }

    return t_copy_prims | t_update;
}

void KHARMADriver::SetGlobalTimeStep()
{
    // TODO(BSP) apply the limits from GRMHD package here
    if (tm.dt < 0.1 * std::numeric_limits<Real>::max()) {
        tm.dt *= 2.0;
    }
    Real big = std::numeric_limits<Real>::max();
    for (auto const& pmb : pmesh->block_list) {
        tm.dt = std::min(tm.dt, pmb->NewDt());
        pmb->SetAllowedDt(big);
    }

    // TODO(BSP) start reduce at the end of the per-meshblock stuff, then check it here
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Allreduce(
        MPI_IN_PLACE, &tm.dt, 1, MPI_PARTHENON_REAL, MPI_MIN, MPI_COMM_WORLD));
#endif

    if (tm.time < tm.tlim &&
        (tm.tlim - tm.time) < tm.dt) // timestep would take us past desired endpoint
        tm.dt = tm.tlim - tm.time;
}

void KHARMADriver::PostExecute(DriverStatus status)
{
    Packages::PostExecute(pmesh, pinput, tm);
    EvolutionDriver::PostExecute(status);
}
