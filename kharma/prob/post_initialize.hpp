/*
 *  File: post_initialize.hpp
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

#include "b_cd.hpp"
#include "b_flux_ct.hpp"

namespace KHARMA
{

/**
 * Initialize the magnetic field (if it wasn't done in ProblemGenerator), and renormalize
 * it as is common practice for torus problems.
 *
 * Since the latter operation is global, we perform this on the whole mesh
 * once initialization of all other problem data is completed.
 */
void SeedAndNormalizeB(ParameterInput* pin, std::shared_ptr<MeshData<Real>> md);

/**
 * Functions run over the entire mesh after per-block initialization:
 * 1. Initial boundary sync to populate ghost zones
 * 2. Initialize magnetic field, which must be normalized globally to respect beta_min
 * parameter
 * 3. Any ad-hoc additions to fluid state, e.g. add hotspots etc.
 * 4. On restarts, reset any per-run parameters
 * 5. Clean up B field divergence if resizing the grid
 */
void PostInitialize(ParameterInput* pin, Mesh* pmesh, bool is_restart);

}
