/*
 *  File: problem.hpp
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
 * Generate the initial condition on (the physical zones of) a meshblock
 * This is the callback from Parthenon -- we apply normalization and transformation
 * afterward
 *
 * An example of running each supported problem with parameters is provided in the pars/
 * folder
 */
namespace KHARMA
{

/**
 * Generate the initial conditions inside a meshblock according to the input parameters.
 * This mostly involves including the rest of the code from the prob/ folder, and calling
 * the appropriate function based on the parameter "problem_id"
 *
 * This function also performs some initial consistency operations, such as applying
 * floors, calculating the conserved values, and synchronizing boundaries.
 *
 * Note that for some problems, this function does *not* initialize the magnetic field,
 * which is instead set in PostInitialize.  This is done if the field depends on the
 * local density rho, which may not be well-defined on the whole Mesh until after this
 * function has run over all MeshBlocks.
 */
void ProblemGenerator(MeshBlock* pmb, ParameterInput* pin);

}
