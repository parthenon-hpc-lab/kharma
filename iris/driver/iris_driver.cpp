/*
 *  File: iris_driver.cpp
 *
 *  BSD 3-Clause License
 *
 *  Copyright (c) 2025, Iris contributors
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
#include "iris_driver.hpp"

// Iris headers
#include "cameras.hpp"
#include "model.hpp"
#include "rays.hpp"

#include <parthenon/parthenon.hpp>

// Declare class static variables
Kokkos::Timer IrisDriver::timer_output;

DriverStatus IrisDriver::Execute()
{
    PreExecute();

    DriverUtils::ConstructAndExecuteTaskLists<>(this);
    Kokkos::fence();
    std::cerr << "TRACED" << std::endl;

    timer_output.reset();
    pouts->MakeOutputs(pmesh, pinput);
    Kokkos::fence();

    PostExecute(DriverStatus::complete);

    return DriverStatus::complete; // TODO return codes based on tetrads/anticipated
                                   // accuracy?
}

template<typename T>
TaskCollection IrisDriver::MakeTaskCollection(T& blocks)
{
    TaskCollection tc;
    TaskID t_none(0);

    auto partitions = pmesh->GetDefaultBlockPartitions();
    const int num_partitions = partitions.size();
    int num_blocks = blocks.size();
    auto pkgs = pmesh->packages.AllPackages();

    const bool thin_disk =
        pmesh->packages.Get("Model")->Param<std::string>("type") == "thin_disk";
    // TODO read pol, react below
    auto cam_pkg = pmesh->packages.Get("Cameras");
    const auto camnames = cam_pkg->Param<std::vector<std::string>>("camnames");
    const auto vars = cam_pkg->Param<std::vector<std::string>>("vars");

    auto ray_pkg = pmesh->packages.Get("Rays");
    // TODO This is a very large value for iterations (which usually have many steps!)
    const int max_iter_geo = ray_pkg->Param<int>("max_nstep_geo");
    const int max_iter_rad = ray_pkg->Param<int>("max_nstep_rad");
    const auto store_paths = ray_pkg->Param<bool>("store_paths");

    TaskRegion& region1 = tc.AddRegion(num_blocks);
    for (int i = 0; i < num_blocks; i++) {
        auto& pmb = pmesh->block_list[i];
        auto t_load_file = t_none;
        // File load per-block
        if (pkgs.count("File"))
            t_load_file = region1[i].AddTask(t_none, TF(File::InitMeshBlock), pmb.get());
        // Construct cameras per-block
        region1[i].AddTask(t_load_file, TF(Cameras::InitGeodesics), pmb.get());
    }

    TaskRegion& all_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        TaskList& tl = all_region[i];
        auto& md_base = pmesh->mesh_data.GetOrAdd("base", i);

        // Sync loaded data to fill cell ghost zones once
        // This will be a no-op for non-grid problems
        auto t_initial_boundaries = AddBoundaryExchangeTasks(t_none, tl, md_base, false);

        // Rays are traced together over the whole mesh
        // 'AddTraceTasks' repeats per-block trace until rays reach stop conditions
        auto t_geodesics = Rays::AddTraceTasks<false>(t_initial_boundaries, tl,
            Rays::TraceGeodesicsBlock, md_base.get(), max_iter_geo);

        // Set any boundary conditions per-block
        // TODO meshize so we can have a single region
        if (thin_disk) {
            t_geodesics =
                tl.AddTask(t_geodesics, TF(Model::SetStokesThindisk), md_base.get());
        }

        TaskID t_trace_emission;
        if (store_paths) {
            // First template is tracing emission vs geodesics, *second* is store_paths or
            // not
            t_trace_emission = Rays::AddTraceTasks<true>(t_geodesics, tl,
                Rays::TraceEmissionBlock<true>, md_base.get(), max_iter_rad);
        } else {
            t_trace_emission = Rays::AddTraceTasks<true>(t_geodesics, tl,
                Rays::TraceEmissionBlock<false>, md_base.get(), max_iter_rad);
        }

        // Write all cameras on our rank to host arrays
        auto t_write_local_cams =
            tl.AddTask(t_trace_emission, TF(Cameras::WriteLocalCameras), md_base.get());

        // MPI reductions to the host arrays
        auto t_reduce_cams =
            tl.AddTask(t_write_local_cams, TF(Cameras::ReduceCameras), md_base.get());
        // auto t_reduce_cams = t_write_local_cams;

        // Any custom outputs, printed summaries
        auto t_write_reduced_cams =
            tl.AddTask(t_reduce_cams, TF(Cameras::WriteReducedCameras), md_base.get());
    }

    // TODO write cameras

    // TODO if verbose or something
    // std::cerr << tc << std::endl;

    return tc;
}

void IrisDriver::PostExecute(DriverStatus status)
{
    // Write total time like ipole
    std::cout << "Total wallclock time: " << elapsed_main() << " s (" << elapsed_output()
              << " s)" << std::endl
              << std::endl;
}
