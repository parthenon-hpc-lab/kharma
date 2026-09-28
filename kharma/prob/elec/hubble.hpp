/*
 *  File: hubble.hpp
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

#include <complex>

#include "decs.hpp"

#include <parthenon/parthenon.hpp>

using namespace std;
using namespace parthenon;

/**
 * Test of electron entropy/temperature evolution in 1D Hubble-type flow
 * Test of "Electrons" package
 * See Ressler+ 2015
 */
TaskStatus InitializeHubble(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin);

/**
 * Set all values on a given domain to the Hubble flow solution,
 * for both initialization and boundary conditions.
 * Use template version when posible, which just calls through to "impl" implementation
 */
TaskStatus SetHubbleImpl(
    std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse);

template<IndexDomain domain>
TaskStatus SetHubble(std::shared_ptr<MeshBlockData<Real>>& rc, bool coarse = false)
{
    return SetHubbleImpl(rc, domain, coarse);
}

/**
 * Apply the source term.  Registered as BlockApplyPrimSource, so it is called twice per
 * step, Strang-split around the transport: (t_start, dt_split) is the half-interval this
 * call must integrate Q over.
 */
void ApplyHubbleHeating(MeshBlockData<Real>* mbase, Real t_start, Real dt_split);
