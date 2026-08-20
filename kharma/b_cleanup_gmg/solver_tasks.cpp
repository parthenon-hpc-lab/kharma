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
#include <solvers/solver_base.hpp>
#include <solvers/solver_utils.hpp>

#include "kharma.hpp"
#include "poisson_equation.hpp"

#if DISABLE_GMG_CLEANUP

// Do we even need to no-op this?

#else

TaskCollection B_CleanupGMG::MakeSolverTaskCollection(Mesh* pmesh)
{
    using namespace parthenon;
    TaskCollection tc;
    TaskID t_none(0);
    std::cerr << "1" << std::endl;

    auto pkg = pmesh->packages.Get("B_CleanupGMG");
    // auto solver_name = pkg->Param<std::string>("solver");
    auto psolver =
        pkg->Param<std::shared_ptr<parthenon::solvers::SolverBase>>("solver_pointer");
    auto poisson_eq = pkg->Param<PoissonEquation<p>>("poisson_eq");

    // List out solver-related vars for faster MPI by skipping base variables
    // using FC = Metadata::FlagCollection;
    // static std::vector<std::string> solver_vars;
    // if (solver_vars.size() == 0) {
    //     // Build the universe of variables to let Parthenon see when exchanging
    //     boundaries.
    //     // This is built to exclude incidental variables like B field initialization
    //     stuff, EMFs, etc.
    //     // "Boundaries" packs in buffers e.g. Dirichlet boundaries
    //     auto solver_flags = FC({Metadata::GetUserFlag("B_CleanupGMG")});
    //     solver_vars = KHARMA::GetVariableNames(&(pmesh->packages), solver_flags);
    // }

    auto partitions = pmesh->GetDefaultBlockPartitions();
    const int num_partitions = partitions.size();
    TaskRegion& region = tc.AddRegion(num_partitions);
    std::cerr << "2" << std::endl;
    for (int i = 0; i < num_partitions; i++) {
        auto& tl = region[i];
        auto& md = pmesh->mesh_data.Add("base", partitions[i]);
        auto& md_p = pmesh->mesh_data.Add("p", md, {p::name()});
        auto& md_rhs = pmesh->mesh_data.Add("rhs", md, {p::name()});

        // Copy RHS to p, then to "rhs" stage
        // TF() is a macro for outputting function name string + function as args
        auto t_copy_rhs =
            tl.AddTask(t_none, TF(solvers::utils::between_fields::CopyData<rhs, p>), md);
        t_copy_rhs = tl.AddTask(
            t_copy_rhs, TF(solvers::utils::CopyData<parthenon::TypeList<p>>), md, md_rhs);

        // Zero out p
        auto t_zero_p = tl.AddTask(t_copy_rhs, TF(solvers::utils::SetToZero<p>), md);
        t_zero_p = tl.AddTask(t_zero_p, TF(solvers::utils::SetToZero<p>), md_p);

        std::cerr << "ADDING SETUP TASKS" << std::endl;
        auto t_setup = psolver->AddSetupTasks(tl, t_zero_p, i, pmesh); // t_zero_p
        std::cerr << "ADDING SOLVER TASKS" << std::endl;
        auto t_solve = psolver->AddTasks(tl, t_setup, i, pmesh);

        // (Re-)Calculate and apply the fluxes directly, as they're our divB
        // `Ax` is exactly what the solver calls, i.e. applies boundaries etc.
        auto t_solve_end = poisson_eq.Ax(tl, t_solve, md, md_p, md);

        auto t_apply_dB = tl.AddTask(t_solve_end, TF(ApplyPFace), md_p.get(), md.get());
        std::cerr << "2block" << std::endl;
    }

    return tc;
}

#endif // DISABLE_GMG_CLEANUP
