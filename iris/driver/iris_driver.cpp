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

    return DriverStatus::complete; // TODO return codes based on tetrads/anticipated accuracy?
}

template <typename T>
TaskCollection IrisDriver::MakeTaskCollection(T &blocks)
{
    TaskCollection tc;
    TaskID t_none(0);

    MeshBlock *pmb = blocks[0].get(); // TODO eventually iterate over these & sync etc
    bool thin_disk = pmb->packages.Get("Model")->Param<std::string>("type") == "thin_disk";
    // TODO read pol, react below

    TaskRegion &sync_region = tc.AddRegion(1);
    auto t_load_file = t_none;
    if (!thin_disk) { // TODO more robust "if we need a file" check
        t_load_file = sync_region[0].AddTask(t_none, File::InitMeshBlock, pmb);
    }
    auto t_at_camera = sync_region[0].AddTask(t_load_file, Cameras::InitGeodesics, pmb);
    auto t_at_origins = sync_region[0].AddTask(t_at_camera, Rays::TraceGeodesicsUntilStop, pmb);
    auto t_set_stokes = t_at_origins;
    if (thin_disk) {
        t_set_stokes = sync_region[0].AddTask(t_at_origins, Model::SetStokesThindisk, pmb);
    }
    auto t_trace_rays = sync_region[0].AddTask(t_set_stokes, Rays::TraceRaysToCameras, pmb);
    auto t_write = sync_region[0].AddTask(t_trace_rays, Cameras::WriteCameras, pmb);

    return tc;
}

void IrisDriver::PostExecute(DriverStatus status)
{
    // Write total time like ipole
    std::cout << "Total wallclock time: " << elapsed_main() << " s (" << elapsed_output() << " s)" << std::endl << std::endl;
}
