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
    // This is probably incompatible with everything

    // TODO check for incompatibilities at some point:
    // At least implicit, jcon output, various electrons tests, etc.

    TaskCollection tc;
    TaskID t_none(0);

    // Which packages we've loaded affects which tasks we'll add to the list
    auto& pkgs         = blocks[0]->packages.AllPackages();
    auto& flux_pkg   = pkgs.at("Flux")->AllParams();

    // Allocate the fluid states ("containers") we need for each block
    for (auto& pmb : blocks) {
        auto &base = pmb->meshblock_data.Get("base");
        if (stage == 1) {
            pmb->meshblock_data.Add("dUdt", base);
            for (int i = 1; i < integrator->nstages; i++)
                pmb->meshblock_data.Add(integrator->stage_name[i], base);
        }
    }



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

        // Calculate the flux of each variable through each face
        // This reconstructs the primitives (P) at faces and uses them to calculate fluxes
        // of the conserved variables (U) through each face.
        auto t_fluxes = KHARMADriver::AddFluxCalculations(t_none, tl, md_sub_step_init.get());

        // Any package modifications to the fluxes.  e.g.:
        // 1. CT calculations for B field transport
        // 2. Zero fluxes through poles
        // etc
        auto t_fix_flux = tl.AddTask(t_fluxes, Packages::FixFlux, md_sub_step_init.get());

        // Apply the fluxes to calculate a change in cell-centered values "md_flux_src"
        auto t_flux_div = tl.AddTask(t_fix_flux, FluxDivergence, md_sub_step_init.get(), md_flux_src.get(),
                                    std::vector<MetadataFlag>{Metadata::Independent, Metadata::Cell, Metadata::WithFluxes}, 0);

        // Add any source terms: geometric \Gamma * T, wind, damping, etc etc
        auto t_sources = tl.AddTask(t_flux_div, Packages::AddSource, md_sub_step_init.get(), md_flux_src.get(), IndexDomain::interior);

        // Perform the update using the source term
        // Add any proportion of the step start required by the integrator (e.g., RK2)
        auto t_avg_data = tl.AddTask(t_sources, Update::WeightedSumData<std::vector<MetadataFlag>, MeshData<Real>>,
                                    std::vector<MetadataFlag>({Metadata::Independent}),
                                    md_sub_step_init.get(), md_full_step_init.get(),
                                    integrator->gam0[stage-1], integrator->gam1[stage-1],
                                    md_sub_step_final.get());
        // apply du/dt to the result
        auto t_update = tl.AddTask(t_sources, Update::WeightedSumData<std::vector<MetadataFlag>, MeshData<Real>>,
                                    std::vector<MetadataFlag>({Metadata::Independent}),
                                    md_sub_step_final.get(), md_flux_src.get(),
                                    1.0, integrator->beta[stage-1] * integrator->dt,
                                    md_sub_step_final.get());

        // UtoP needs a guess in order to converge, so we copy in md_sub_step_init
        auto t_copy_prims = t_update;
        if (integrator->nstages > 1) {
            t_copy_prims = tl.AddTask(t_none, Copy<MeshData<Real>>, std::vector<MetadataFlag>({Metadata::GetUserFlag("HD"), Metadata::GetUserFlag("Primitive")}),
                                                md_sub_step_init.get(), md_sub_step_final.get());
        }


        // Make sure the primitive values are updated.
        auto t_UtoP = tl.AddTask(t_copy_prims, Packages::MeshUtoP, md_sub_step_final.get(), IndexDomain::interior, false);

        // Apply any floors
        auto t_floors = tl.AddTask(t_UtoP, Packages::MeshApplyFloors, md_sub_step_final.get(), IndexDomain::interior);

        // Boundary sync: neighbors must be available for FixUtoP below
        KHARMADriver::AddBoundarySync(t_floors, tl, md_sub_step_final);
    }

    // Fix Region: prims/cons sync, floors, fixes, boundary conditions which need primitives
    TaskRegion &fix_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &tl = fix_region[i];
        auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], i);

        // If we're evolving the GRMHD variables explicitly, we need to fix UtoP variable inversion failures
        // Syncing bounds before calling this, and then running it over the whole domain, will make
        // behavior for different mesh breakdowns much more similar (identical?), since bad zones in
        // relevant ghost zone ranks will get to use all the same neighbors as if they were in the bulk
        auto t_fix_p = tl.AddTask(t_none, Inverter::MeshFixUtoP, md_sub_step_final.get());

        auto t_set_bc = tl.AddTask(t_fix_p, parthenon::ApplyBoundaryConditionsOnCoarseOrFineMD, md_sub_step_final, false);

        // Make sure *all* conserved vars are synchronized at step end
        auto t_ptou = tl.AddTask(t_set_bc, Flux::MeshPtoU, md_sub_step_final.get(), IndexDomain::entire, false);

        auto t_step_done = t_ptou;

        // Estimate next time step based on ctop
        if (stage == integrator->nstages) {
            auto t_new_dt =
                tl.AddTask(t_step_done, Update::EstimateTimestep<MeshData<Real>>, md_sub_step_final.get());
        }
    }

    // Second boundary sync:
    // ensure that primitive variables in ghost zones are *exactly*
    // identical to their physical counterparts, now that they have been
    // modified on each rank.
    const auto &two_sync = pkgs.at("Driver")->Param<bool>("two_sync");
    if (two_sync) {
        auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], 0);
        KHARMADriver::AddFullSyncRegion(tc, md_sub_step_final);
    }

    return tc;
}
