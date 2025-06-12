/*
 *  File: imex_step.cpp
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

#include "decs.hpp"

//Packages
#include "b_cd.hpp"
#include "b_cleanup.hpp"
#include "b_ct.hpp"
#include "b_flux_ct.hpp"
#include "electrons.hpp"
#include "inverter.hpp"
#include "ismr.hpp"
#include "grmhd.hpp"
#include "wind.hpp"
// Other headers
#include "boundaries.hpp"
#include "flux.hpp"
#include "kharma.hpp"
#include "resize_restart.hpp"
#include "implicit.hpp"

#include <parthenon/parthenon.hpp>
#include <interface/update.hpp>
#include <amr_criteria/refinement_package.hpp>

using FC = Metadata::FlagCollection;

TaskCollection KHARMADriver::MakeImExTaskCollection(BlockList_t &blocks, int stage)
{
    // Reminder that this list is created BEFORE any of the list contents are run!
    // Prints or function calls here will likely not do what you want: instead, add to the list by calling tl.AddTask()

    TaskCollection tc;
    TaskID t_none(0);

    // Which packages we've loaded affects which tasks we'll add to the list
    auto& pkgs         = blocks[0]->packages.AllPackages();
    auto& flux_pkg   = pkgs.at("Fluxes")->AllParams();
    const bool use_b_cleanup = pkgs.count("B_Cleanup");
    const bool use_b_ct = pkgs.count("B_CT");
    const bool use_electrons = pkgs.count("Electrons");
    const bool use_fofc = flux_pkg.Get<bool>("use_fofc");
    const bool use_implicit = pkgs.count("Implicit");
    const bool use_jcon = pkgs.count("Current");
    const bool use_linesearch = (use_implicit) ? pkgs.at("Implicit")->Param<bool>("linesearch") : false;
    const bool emhd_enabled = pkgs.count("EMHD");
    const bool use_ideal_guess = (emhd_enabled) ? pkgs.at("GRMHD")->Param<bool>("ideal_guess") : false;

    // Allocate/copy the things we need
    // TODO these can now be reduced by including the var lists/flags which actually need to be allocated
    // TODO except the Copy they can be run on step 1 only
    if (stage == 1) {
        auto &base = pmesh->mesh_data.Get();
        // Fluxes
        pmesh->mesh_data.Add("dUdt", base);
        for (int i = 1; i < integrator->nstages; i++)
            pmesh->mesh_data.Add(integrator->stage_name[i], base);

        if (use_jcon) {
            // Preserve state for time derivatives if we need to output current
            // Pick out the variables we need, to allocate and copy less
            static parthenon::Metadata::FlagVec preserve_flags;
            static std::vector<std::string> preserve_vars;
            if (preserve_vars.size() == 0) {
                preserve_flags = {Metadata::GetUserFlag("MHD"), Metadata::GetUserFlag("Primitive")};
                preserve_vars = KHARMA::GetVariableNames(&(pmesh->packages), FC(preserve_flags));
            }
            // Add the container once to avoid re-adding if we have lots of partitions
            //pmesh->mesh_data.Add("preserve", base, preserve_vars);
            // Ensure we copy the MHD variables every step with a task
            const int num_partitions = pmesh->DefaultNumPartitions();
            TaskRegion &copy_region = tc.AddRegion(num_partitions);
            for (int i = 0; i < num_partitions; i++) {
                auto &tl = copy_region[i];
                tl.AddTask(t_none, Copy<MeshData<Real>>, preserve_flags,
                            base.get(), pmesh->mesh_data.Add("preserve", base, preserve_vars).get());
            }
        }
        // FOFC needs to determine whether the "real" U-divF will violate floors, and needs a safe place to do it.
        // We populate it later, with each *sub-step*'s initial state
        if (use_fofc) {
            pmesh->mesh_data.Add("fofc_source", base);
            pmesh->mesh_data.Add("fofc_guess", base);
        }
        if (use_implicit) {
            // When solving, we need a temporary copy with any explicit updates,
            // but not overwriting the beginning- or mid-step values
            pmesh->mesh_data.Add("solver", base);
            if (use_linesearch) {
                // Need an additional state for linesearch
                pmesh->mesh_data.Add("linesearch", base);
            }
        }
    }

    static std::vector<std::string> sync_vars;
    if (sync_vars.size() == 0) {
        // Build the universe of variables to let Parthenon see when exchanging boundaries.
        // This is built to exclude incidental variables like B field initialization stuff, EMFs, etc.
        // "Boundaries" packs in buffers e.g. Dirichlet boundaries
        auto sync_flags = FC({Metadata::GetUserFlag("Primitive"), Metadata::Conserved,
                              Metadata::Face, Metadata::GetUserFlag("Boundaries")}, true);
        sync_vars = KHARMA::GetVariableNames(&(pmesh->packages), sync_flags);
    }

    // Flux region: calculate and apply fluxes to update conserved values
    const int num_partitions = pmesh->DefaultNumPartitions();
    TaskRegion &flux_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &tl = flux_region[i];
        // Container names:
        // '_full_step_init' refers to the fluid state at the start of the full time step (Si in iharm3d)
        // '_sub_step_init' refers to the fluid state at the start of the sub step (Ss in iharm3d)
        // '_sub_step_final' refers to the fluid state at the end of the sub step (Sf in iharm3d)
        // '_flux_src' refers to the mesh object corresponding to -divF + S
        // '_solver' refers to the fluid state passed to the Implicit solver. At the end of the solve
        // '_linesearch' refers to the fluid state updated while performing a linesearch in the solver
        // copy P and U from solver state to sub_step_final state.
        auto &md_full_step_init = pmesh->mesh_data.GetOrAdd("base", i);
        auto &md_sub_step_init  = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage - 1], i);
        auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], i);
        auto &md_flux_src       = pmesh->mesh_data.GetOrAdd("dUdt", i);
        // Normally we put explicit update in md_solver, then add implicitly-evolved variables and copy back.
        // If we're not doing an implicit solve at all, just write straight to sub_step_final
        std::shared_ptr<MeshData<Real>> &md_solver = (use_implicit) ? pmesh->mesh_data.GetOrAdd("solver", i) : md_sub_step_final;
        auto &md_sync = pmesh->mesh_data.AddShallow("sync"+integrator->stage_name[stage]+std::to_string(i), md_sub_step_final, sync_vars);

        // Start receiving flux corrections and ghost cells
        auto t_start_recv_bound = tl.AddTask(t_none, parthenon::StartReceiveBoundBufs<parthenon::BoundaryType::any>, md_sub_step_final);
        auto t_start_recv_flux = t_start_recv_bound;
        if (pmesh->multilevel || use_b_ct)
            t_start_recv_flux = tl.AddTask(t_none, parthenon::StartReceiveFluxCorrections, md_sub_step_init);

        // Calculate the flux of each variable through each face
        // This reconstructs the primitives (P) at faces and uses them to calculate fluxes
        // of the conserved variables (U) through each face.
        auto t_flux_calc = KHARMADriver::AddFluxCalculations(t_start_recv_flux, tl, md_sub_step_init.get());
        auto t_fluxes = t_flux_calc;
        if (use_fofc) {
            auto &guess_src = pmesh->mesh_data.GetOrAdd("fofc_source", i);
            auto &guess = pmesh->mesh_data.GetOrAdd("fofc_guess", i);
            auto t_fluxes = KHARMADriver::AddFOFC(t_flux_calc, tl, md_sub_step_init.get(), md_full_step_init.get(),
                                                  md_sub_step_init.get(), guess_src.get(), guess.get(), stage);
        }

        // Any package modifications to the fluxes.  e.g.:
        // 1. CT calculations for B field transport
        // 2. Zero fluxes through poles
        // etc
        auto t_fix_flux = tl.AddTask(t_fluxes, Packages::FixFlux, md_sub_step_init.get());

        // If we're in AMR, correct fluxes from neighbors
        auto t_flux_bounds = t_fix_flux;
        if (pmesh->multilevel || use_b_ct) {
            auto t_emf = t_flux_bounds;
            if (use_b_ct) {
                // Pull out a container of only EMF to synchronize
                auto &md_emf_only = pmesh->mesh_data.AddShallow("EMF", md_sub_step_init, std::vector<std::string>{"B_CT.emf"}); // TODO this gets weird if we partition
                auto t_emf_local = tl.AddTask(t_flux_bounds, B_CT::CalculateEMF, md_sub_step_init.get());
                t_emf = KHARMADriver::AddBoundarySync(t_emf_local, tl, md_emf_only);
            }
            auto t_load_send_flux = tl.AddTask(t_emf, parthenon::LoadAndSendFluxCorrections, md_sub_step_init);
            auto t_recv_flux = tl.AddTask(t_load_send_flux, parthenon::ReceiveFluxCorrections, md_sub_step_init);
            t_flux_bounds = tl.AddTask(t_recv_flux, parthenon::SetFluxCorrections, md_sub_step_init);
        }

        // Apply the fluxes to calculate a change in cell-centered values "md_flux_src"
        auto t_flux_div = tl.AddTask(t_flux_bounds, FluxDivergence, md_sub_step_init.get(), md_flux_src.get(),
                                     std::vector<MetadataFlag>{Metadata::Independent, Metadata::Cell, Metadata::WithFluxes}, 0);

        // Add any source terms: geometric \Gamma * T, wind, damping, etc etc
        auto t_sources = tl.AddTask(t_flux_div, Packages::AddSource, md_sub_step_init.get(), md_flux_src.get(), IndexDomain::interior);

        // Update explicit state with the explicit fluxes/sources
        auto t_update = KHARMADriver::AddStateUpdate(t_sources, tl, md_full_step_init.get(), md_sub_step_init.get(),
                                                     md_flux_src.get(), md_solver.get(),
                                                     std::vector<MetadataFlag>{Metadata::GetUserFlag("Explicit"), Metadata::Independent},
                                                     use_b_ct, stage);

        // Update ideal HD variables so that they can be used as a guess for the solver.
        // Only used if emhd/ideal_guess is enabled.
        // An additional `AddStateUpdate` task just for variables marked with the `IdealGuess` flag
        auto t_ideal_guess = t_update;
        if (use_ideal_guess) {
            t_ideal_guess = KHARMADriver::AddStateUpdateIdealGuess(t_sources,  tl, md_full_step_init.get(), md_sub_step_init.get(),
                                                     md_flux_src.get(), md_solver.get(),
                                                     std::vector<MetadataFlag>{Metadata::GetUserFlag("IdealGuess"), Metadata::Independent},
                                                     false, stage);
        }

        // Make sure the primitive values of *explicitly-evolved* variables are updated.
        // Packages with implicitly-evolved vars should only register BoundaryUtoP or BoundaryPtoU
        auto t_explicit_UtoP = tl.AddTask(t_ideal_guess, Packages::MeshUtoP, md_solver.get(), IndexDomain::entire, false);

        // Done with explicit update
        auto t_explicit = t_explicit_UtoP;

        auto t_implicit = t_explicit;
        if (use_implicit) {
            // Extra containers for implicit solve
            std::shared_ptr<MeshData<Real>> &md_linesearch = (use_linesearch) ? pmesh->mesh_data.GetOrAdd("linesearch", i) : md_solver;

            // Copy the current state of any implicitly-evolved vars (at least the prims) in as a guess.
            // If we aren't using the ideal solution as the guess, set md_solver = md_sub_step_init
            auto t_copy_guess       = t_explicit;
            auto t_copy_emhd_vars   = t_explicit;
            if (use_ideal_guess) {
                // This copies only q, dP for which we don't already have a guess
                t_copy_emhd_vars = tl.AddTask(t_sources, Copy<MeshData<Real>>, std::vector<MetadataFlag>({Metadata::GetUserFlag("EMHDVar")}),
                                        md_sub_step_init.get(), md_solver.get());
                t_copy_guess = t_copy_emhd_vars;
            } else {
                t_copy_guess = tl.AddTask(t_sources, Copy<MeshData<Real>>, std::vector<MetadataFlag>({Metadata::GetUserFlag("Implicit")}),
                                        md_sub_step_init.get(), md_solver.get());
            }

            auto t_guess_ready = t_explicit | t_copy_guess;

            // The `solver` MeshData object now has the implicit primitives corresponding to initial/half step and
            // explicit variables have been updated to match the current step.
            // Copy the primitives to the `linesearch` MeshData object if linesearch was enabled.
            auto t_copy_linesearch = t_guess_ready;
            if (use_linesearch) {
                t_copy_linesearch = tl.AddTask(t_guess_ready, Copy<MeshData<Real>>, std::vector<MetadataFlag>({Metadata::GetUserFlag("Primitive")}),
                                                md_solver.get(), md_linesearch.get());
            }

            // Time-step implicit variables by root-finding the residual.
            // This calculates the primitive values after the substep for all "isImplicit" variables --
            // no need for separately adding the flux divergence or calling UtoP
            auto t_implicit_step = tl.AddTask(t_copy_linesearch, Implicit::Step, md_full_step_init.get(), md_sub_step_init.get(),
                                         md_flux_src.get(), md_linesearch.get(), md_solver.get(), integrator->beta[stage-1] * integrator->dt);

            // Copy the entire solver state (everything defined on the grid, incl. our new Face variables) into the final state md_sub_step_final
            // If we're entirely explicit, we just declare these equal
            auto t_implicit_c = tl.AddTask(t_implicit_step, Copy<MeshData<Real>>, std::vector<MetadataFlag>({Metadata::Cell}),
                                    md_solver.get(), md_sub_step_final.get());
            t_implicit = tl.AddTask(t_implicit_step, WeightedSumDataFace<MetadataFlag>, std::vector<MetadataFlag>({Metadata::Face}),
                                    md_solver.get(), md_solver.get(), 1.0, 0.0, md_sub_step_final.get());
        }

        // Apply all floors & limits (GRMHD,EMHD,etc), but do *not* immediately correct UtoP failures with FixUtoP --
        // rather, we will synchronize (including pflags!) first.
        // With an extra ghost zone, this *should* still allow binary-similar evolution between numbers of mesh blocks,
        // but hasn't been tested to do so yet.
        auto t_floors = tl.AddTask(t_implicit, Packages::MeshApplyFloors, md_sub_step_final.get(), IndexDomain::interior);

        KHARMADriver::AddBoundarySync(t_floors, tl, md_sync);
    }

    // Fix Region: prims/cons sync, floors, fixes, boundary conditions which need primitives
    TaskRegion &fix_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &tl  = fix_region[i];
        auto &md_sub_step_init  = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage-1], i);
        auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], i);
        auto &md_sync = pmesh->mesh_data.AddShallow("sync"+integrator->stage_name[stage]+std::to_string(i), md_sub_step_final, sync_vars);

        // If we're evolving the GRMHD variables explicitly, we need to fix UtoP variable inversion failures.
        // If implicitly, we run a (very similar) fix for solver failures.
        // Syncing bounds before calling this, and then running it over the whole domain, will make
        // behavior for different mesh breakdowns much more similar (identical?), since bad zones in
        // relevant ghost zone ranks will get to use all the same neighbors as if they were in the bulk
        // TODO fixups as a callback?
        auto t_fix_utop = t_none;
        if (!pkgs.at("GRMHD")->Param<bool>("implicit")) {
            t_fix_utop = tl.AddTask(t_none, Inverter::MeshFixUtoP, md_sub_step_final.get());
        }
        auto t_fix_solve = t_fix_utop;
        if (use_implicit) {
            t_fix_solve = tl.AddTask(t_fix_utop, Implicit::MeshFixSolve, md_sub_step_final.get());
        }

        // Re-apply boundary conditions to reflect fixes
        auto t_set_bc = tl.AddTask(t_fix_solve, parthenon::ApplyBoundaryConditionsOnCoarseOrFineMD, md_sync, false);

        // Any package- (likely, problem-) specific source terms which must be applied to primitive variables
        // Apply these only after the final step so they're operator-split
        auto t_prim_source = t_set_bc;
        if (stage == integrator->nstages) {
            t_prim_source = tl.AddTask(t_set_bc, Packages::MeshApplyPrimSource, md_sub_step_final.get());
        }
        // Electron heating goes where it does in the KHARMA Driver, for the same reasons
        auto t_heat_electrons = t_prim_source;
        if (use_electrons) {
            t_heat_electrons = tl.AddTask(t_prim_source, Electrons::MeshApplyElectronHeating,
                                          md_sub_step_init.get(), md_sub_step_final.get(), stage == 1);
        }

        // Make sure *all* conserved vars are synchronized at step end
        auto t_ptou = tl.AddTask(t_heat_electrons, Flux::MeshPtoU, md_sub_step_final.get(), IndexDomain::entire, false);

        auto t_step_done = t_ptou;
        if (pkgs.count("ISMR")) {
            auto t_derefine_b = t_ptou;
            if (pkgs.count("B_CT"))
                t_derefine_b = tl.AddTask(t_ptou, B_CT::DerefinePoles, md_sub_step_final.get());
            t_step_done = tl.AddTask(t_derefine_b, ISMR::DerefinePoles, md_sub_step_final.get());
        }

        // Estimate next time step based on ctop
        if (stage == integrator->nstages) {
            auto t_new_dt =
                tl.AddTask(t_step_done, Update::EstimateTimestep<MeshData<Real>>, md_sub_step_final.get());

            // Update refinement
            if (pmesh->adaptive) {
                auto tag_refine = tl.AddTask(
                    t_step_done, parthenon::Refinement::Tag<MeshData<Real>>, md_sub_step_final.get());
            }
        }
    }

    // B Field cleanup: this is a separate solve so it's split out
    // It's also really slow when enabled so we don't care too much about limiting regions, etc.
    if (use_b_cleanup && (stage == integrator->nstages) && B_Cleanup::CleanupThisStep(pmesh, tm.ncycle)) {
        TaskRegion &cleanup_region = tc.AddRegion(num_partitions);
        for (int i = 0; i < num_partitions; i++) {
            auto &tl = cleanup_region[i];
            auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], i);
            tl.AddTask(t_none, B_Cleanup::CleanupDivergence, md_sub_step_final);
        }
    }

    // Second boundary sync:
    // ensure that primitive variables in ghost zones are *exactly*
    // identical to their physical counterparts, now that they have been
    // modified on each rank.
    const auto &two_sync = pkgs.at("Driver")->Param<bool>("two_sync");
    if (two_sync) {
        for (int i = 0; i < num_partitions; i++) {
            auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], i);
            auto &md_sync = pmesh->mesh_data.AddShallow("sync"+integrator->stage_name[stage]+std::to_string(i), md_sub_step_final, sync_vars);
            KHARMADriver::AddFullSyncRegion(tc, md_sync);
        }
    }

    return tc;
}

