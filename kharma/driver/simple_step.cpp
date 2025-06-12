/*
 *  File: simple_step.cpp
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

#include "inverter.hpp"
#include "flux.hpp"

TaskCollection KHARMADriver::MakeSimpleTaskCollection(BlockList_t &blocks, int stage)
{
    // This is probably incompatible with just about everything,
    // but we at least check the big ones
    auto& pkgs = pmesh->packages.AllPackages();
    auto& flux_pkg = pkgs.at("Fluxes")->AllParams();
    auto& inverter_pkg = pkgs.at("Inverter")->AllParams();
    PARTHENON_REQUIRE(!pkgs.count("B_CT"), "Face-centered B not compatible with simple driver!");
    PARTHENON_REQUIRE(!pkgs.count("Electrons"), "Electrons not compatible with simple driver!");
    PARTHENON_REQUIRE(!flux_pkg.Get<bool>("use_fofc"), "Flux corrections not compatible with simple driver!");
    PARTHENON_REQUIRE(inverter_pkg.Get<Inverter::Type>("inverter_type") == Inverter::Type::kastaun,
                      "Only the Kastaun primitive variable recovery is compatible with simple driver!");
    PARTHENON_REQUIRE(!pkgs.count("Current"), "4-current output not compatible with simple driver!");

    // The `TaskCollection` holds everything we do in a step.  It consists of multiple `TaskRegion`s
    // run serially.  Each `TaskRegion` can have multiple `TaskLists` which may someday be run
    // in parallel with threading.  Individual tasks in a `TaskList` may also be run concurrently
    // at some point, if their dependncies allow.
    TaskCollection tc;
    TaskID t_none(0);

    // Which packages we've loaded affects which tasks we'll add to the list
    auto& pkgs         = blocks[0]->packages.AllPackages();
    auto& flux_pkg   = pkgs.at("Flux")->AllParams();

    // Allocate the fluid states ("containers") we need for each stage
    if (stage == 1) {
        auto &base = pmesh->mesh_data.Get();
        // Fluxes
        pmesh->mesh_data.Add("dUdt", base);
        for (int i = 1; i < integrator->nstages; i++)
            pmesh->mesh_data.Add(integrator->stage_name[i], base);
    }

    // We use a single region `flux_region` with a single task list we name `tl`.
    TaskRegion &simple_region = tc.AddRegion(1);
    auto &tl = simple_region[0];
    // Container names:
    // '_full_step_init' refers to the fluid state at the start of the full time step (Si in iharm3d)
    // '_sub_step_init' refers to the fluid state at the start of the sub step (Ss in iharm3d)
    // '_sub_step_final' refers to the fluid state at the end of the sub step (Sf in iharm3d)
    // '_flux_src' refers to the mesh object corresponding to -divF + S
    auto &md_full_step_init = pmesh->mesh_data.GetOrAdd("base", 0);
    auto &md_sub_step_init  = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage - 1], 0);
    auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], 0);
    auto &md_flux_src       = pmesh->mesh_data.GetOrAdd("dUdt", 0);

    // Start by calculating the flux of each variable through each face
    // This reconstructs the primitives (P) to cell faces and uses them to calculate fluxes
    // of the conserved variables (U) through each face.
    auto t_fluxes = KHARMADriver::AddFluxCalculations(t_none, tl, md_sub_step_init.get());

<<<<<<< HEAD
    // Big synchronous region: get & apply fluxes to advance the fluid state
    // num_partitions is nearly always 1
    const int num_partitions = pmesh->DefaultNumPartitions();
    TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &tl = single_tasklist_per_pack_region[i];
        // Container names:
        // '_full_step_init' refers to the fluid state at the start of the full time step (Si in iharm3d)
        // '_sub_step_init' refers to the fluid state at the start of the sub step (Ss in iharm3d)
        // '_sub_step_final' refers to the fluid state at the end of the sub step (Sf in iharm3d)
        // '_flux_src' refers to the mesh object corresponding to -divF + S
        auto &md_full_step_init = pmesh->mesh_data.GetOrAdd("base", i);
        auto &md_sub_step_init  = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage - 1], i);
        auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], i);
        auto &md_flux_src       = pmesh->mesh_data.GetOrAdd("dUdt", i);
=======
    // Any package modifications to the fluxes, e.g. from Flux-CT
    auto t_fix_flux = tl.AddTask(t_fluxes, Packages::FixFlux, md_sub_step_init.get());
>>>>>>> parthenon-bump

    // Apply the fluxes to calculate a change in cell-centered values "md_flux_src"
    auto t_flux_div = tl.AddTask(t_fix_flux, FluxDivergence, md_sub_step_init.get(), md_flux_src.get(),
                                std::vector<MetadataFlag>{Metadata::Independent, Metadata::Cell, Metadata::WithFluxes}, 0);

<<<<<<< HEAD
        // Any package modifications to the fluxes.  e.g.:
        // 1. CT calculations for B field transport
        // 2. Zero fluxes through poles
        // etc
        auto t_fix_flux = tl.AddTask(t_fluxes, Packages::FixFlux, md_sub_step_init.get());
=======
    // Add any package source terms: geometric \Gamma * T, wind, damping, etc etc
    auto t_sources = tl.AddTask(t_flux_div, Packages::AddSource, md_sub_step_init.get(), md_flux_src.get(), IndexDomain::interior);
>>>>>>> parthenon-bump

    // Update the state with the explicit fluxes/sources
    // This includes copying in the primitive variable "guess" to md_sub_step_final
    auto t_update = KHARMADriver::AddStateUpdate(t_sources, tl, md_full_step_init.get(), md_sub_step_init.get(),
                                                    md_flux_src.get(), md_sub_step_final.get(),
                                                    std::vector<MetadataFlag>{Metadata::Independent},
                                                    false, stage);

    // Make sure the primitive values are updated.
    auto t_UtoP = tl.AddTask(t_update, Packages::MeshUtoP, md_sub_step_final.get(), IndexDomain::interior, false);

    // Apply any floors
    auto t_floors = tl.AddTask(t_UtoP, Packages::MeshApplyFloors, md_sub_step_final.get(), IndexDomain::interior);

    // Boundary sync: fill primitive variables in ghost zones
    auto t_bounds = KHARMADriver::AddBoundarySync(t_floors, tl, md_sub_step_final);

    // Re-apply boundary conditions to reflect fixes
    auto t_set_bc = tl.AddTask(t_bounds, parthenon::ApplyBoundaryConditionsOnCoarseOrFineMD, md_sub_step_final, false);

    // Make sure *all* conserved vars are synchronized at step end
    auto t_ptou = tl.AddTask(t_set_bc, Flux::MeshPtoU, md_sub_step_final.get(), IndexDomain::entire, false);
    // Estimate next time step based on ctop
    if (stage == integrator->nstages) {
        auto t_new_dt =
            tl.AddTask(t_ptou, Update::EstimateTimestep<MeshData<Real>>, md_sub_step_final.get());
    }

    return tc;
}
