/*
 *  File: solver_tasks.cpp
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
#include "b_cleanup.hpp"

#include <parthenon/parthenon.hpp>
#include <solvers/solver_base.hpp>
#include <solvers/solver_utils.hpp>

#include "b_cleanup.hpp"
#include "kharma.hpp"
#include "poisson_equation.hpp"

#if DISABLE_CLEANUP

// Do we even need to no-op this?

#else

TaskCollection B_Cleanup::MakeTaskCollection(Mesh* pmesh)
{
    using namespace parthenon;
    TaskCollection tc;
    TaskID none(0);

    auto pkg = pmesh->packages.Get("B_Cleanup");
    auto psolver =
        pkg->Param<std::shared_ptr<parthenon::solvers::SolverBase>>("solver_pointer");

    auto partitions = pmesh->GetDefaultBlockPartitions();
    const int num_partitions = partitions.size();
    TaskRegion& region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
        TaskList& tl = region[i];
        auto& md = pmesh->mesh_data.Add("base", partitions[i]);
        auto& md_u = pmesh->mesh_data.Add("u", md, {u::name()});
        auto& md_rhs = pmesh->mesh_data.Add("rhs", md, {u::name()});

        // Move the rhs variable into the rhs stage for stage based solver
        auto copy_rhs =
            tl.AddTask(none, TF(solvers::utils::between_fields::CopyData<rhs, u>), md);
        copy_rhs = tl.AddTask(
            copy_rhs, TF(solvers::utils::CopyData<parthenon::TypeList<u>>), md, md_rhs);

        // Set initial solution guess to zero
        auto zero_u = tl.AddTask(copy_rhs, TF(solvers::utils::SetToZero<u>), md_u);
        auto setup = psolver->AddSetupTasks(tl, zero_u, i, pmesh);
        auto solve = psolver->AddTasks(tl, setup, i, pmesh);

        // Move the solution back so it is output
        auto copy_back =
            tl.AddTask(solve, TF(B_Cleanup::ApplySolution), md_u.get(), md.get());
    }

    return tc;
}

#endif // DISABLE_CLEANUP
