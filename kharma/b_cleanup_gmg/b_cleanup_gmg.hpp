/*
 *  File: b_cleanup_gmg.hpp
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
#pragma once

#include <memory>

#include <parthenon/parthenon.hpp>
#include <solvers/bicgstab_solver.hpp>
#include <solvers/mg_solver.hpp>
#include <solvers/solver_utils.hpp>

#include "grmhd_functions.hpp"
#include "types.hpp"

using namespace parthenon;

#define VARIABLE(ns, varname)                                                            \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #ns "." #varname; }                               \
  }

/**
 * This physics package uses Parthenon's Geometric Multigrid (GMG) solver to
 * minimize magnetic field divergence.  Only useful for resizing face-centered fields.
 */
namespace B_CleanupGMG {

// New type-based variables: pre-declare variable names and get the VarMap() for free!
// All of KHARMA will be switching to these eventually...
VARIABLE(b_clean_gmg, p);
VARIABLE(b_clean_gmg, rhs);

/**
 * Declare fields, initialize parameters
 */
std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

/**
 * Single-call divergence cleanup.  Lots of MPI syncs, probably slow to use in task lists.
 */
TaskStatus CleanupDivergence(std::shared_ptr<MeshData<Real>>& md);

/**
 * Apply B -= grad(P) on faces to subtract divergence from the magnetic field
 */
TaskStatus ApplyPFace(MeshData<Real> *msolve, MeshData<Real> *md);

/**
 * Internal function to set up the task list for the solver
 */
TaskCollection MakeSolverTaskCollection(std::shared_ptr<MeshData<Real>> &md);

} // namespace B_CleanupGMG
