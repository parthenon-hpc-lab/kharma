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
#include "b_cleanup_gmg.hpp"

#include <parthenon/parthenon.hpp>

#include "poisson_equation.hpp"

TaskCollection B_CleanupGMG::MakeSolverTaskCollection(std::shared_ptr<MeshData<Real>> &md)
{
    using namespace parthenon;
    TaskCollection tc;
    TaskID none(0);

    auto pmesh = md->GetMeshPointer();
    auto pkg = pmesh->packages.Get("B_CleanupGMG");
    auto solver = pkg->Param<std::string>("solver");
    auto flux_correct = pkg->Param<bool>("flux_correct");
    auto *mg_solver =
        pkg->MutableParam<parthenon::solvers::MGSolver<p, rhs, PoissonEquation>>(
            "MGsolver");
    auto *bicgstab_solver =
        pkg->MutableParam<parthenon::solvers::BiCGSTABSolver<p, rhs, PoissonEquation>>(
            "MGBiCGSTABsolver");

    auto partitions = pmesh->GetDefaultBlockPartitions();
    const int num_partitions = partitions.size();
    TaskRegion &region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
        TaskList &tl = region[i];

        // Set initial solution guess to zero
        auto t_zero_p = tl.AddTask(none, TF(solvers::utils::SetToZero<p>), md);

        auto t_solve = t_zero_p;
        if (solver == "BiCGSTAB") {
            auto t_setup = bicgstab_solver->AddSetupTasks(tl, t_zero_p, i, pmesh);
            t_solve = bicgstab_solver->AddTasks(tl, t_setup, pmesh, i);
        } else if (solver == "MG") {
            auto t_setup = mg_solver->AddSetupTasks(tl, t_zero_p, i, pmesh);
            t_solve = mg_solver->AddTasks(tl, t_setup, pmesh, i);
        } else {
            PARTHENON_FAIL("Unknown solver type.");
        }
    }

    return tc;
}