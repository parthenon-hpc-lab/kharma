/*
 *  File: grmhd.hpp
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

#include "decs.hpp"
#include "types.hpp"

/**
 * This physics package implements General-Relativistic Magnetohydrodynamics
 *
 * Anything specific to GRMHD (but not relating to the particular *order* of operations)
 * is implemented in this namespace, in the files grmhd.cpp, source.cpp, and fixup.cpp.
 * Many device-side functions related to GRMHD are implemented in grmhd_functions.hpp
 */
namespace GRMHD {
// For declaring variables, as well as the full intermediates we need (right & left fluxes etc)
std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

/**
 * Returns the minimum CFL timestep among all zones in the block,
 * multiplied by a proportion "cfl" for safety.
 *
 * This is just for a particular MeshBlock/package, so don't rely on it
 * Parthenon will take the minimum and put it in pmy_mesh->dt
 */
Real EstimateTimestep(MeshData<Real> *md);

// Internal version for the light phase speed crossing time of smallest zone
Real EstimateRadiativeTimestep(MeshData<Real> *md);

/**
 * Return a tag per-block indicating whether to refine it
 *
 * Criteria are very WIP
 */
AmrTag CheckRefinement(MeshBlockData<Real> *rc);

/**
 * Fill fields which are calculated only for output to file
 * Currently just the current jcon
 */
void FillOutput(MeshBlock *pmb, ParameterInput *pin);

/**
 * Diagnostics performed after each step.
 * Currently just looks for negative density/internal energy
 */
TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *rc);

/**
 * Subtract the average phi-component of velocity (U3) next to the pole,
 * simulating polar velocity advecting together
 */
void CancelBoundaryU3(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse);

/**
 * Same but for the conserved angular momentum T3
 */
void CancelBoundaryT3(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse);

/**
 * Update the signal speeds in zones affected by the above operations.
 * This is important to stay under the Courant limit at times
 */
void UpdateAveragedCtop(MeshData<Real> *md);

}
