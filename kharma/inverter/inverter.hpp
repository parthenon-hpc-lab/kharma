/* 
 *  File: inverter.hpp
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

// Implementation of u_to_p must be defined before instantiation below
// Additionally, invert_template contains the Type and Status enums
#include "invert_template.hpp"
#include "onedw.hpp"
#include "kastaun.hpp"

#include "pack.hpp"

using namespace parthenon;

/**
 * Recover primitive variables from conserved forms.
 * Currently can only use the 1D_W scheme of Noble et al. (2006),
 * but this is the spot for alternate implementations
 */
namespace Inverter {

std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

/**
 * Get the primitive variables
 * This just computes P, and only for the GRHD fluid varaibles rho, u, uvec.
 *
 * Defaults to entire domain, as the KHARMA algorithm relies on applying UtoP over ghost zones.
 * 
 * input: U, whatever form
 * output: U and P match down to inversion errors
 */
void BlockUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse);
inline TaskStatus MeshUtoP(MeshData<Real> *md, IndexDomain domain, bool coarse) {
    Flag("MeshUtoP");
    for (int i=0; i < md->NumBlocks(); ++i)
        BlockUtoP(md->GetBlockData(i).get(), domain, coarse);
    EndFlag();
    return TaskStatus::complete;
}

/**
 * Smooth over inversion failures, usually by averaging values of the primitive variables from each neighboring zone
 * a.k.a. Diffusion?  What diffusion?  There is no diffusion here.
 * 
 * LOCKSTEP: this function expects and should preserve P<->U
 */
TaskStatus FixUtoP(MeshBlockData<Real> *rc);
inline TaskStatus MeshFixUtoP(MeshData<Real> *md) {
    Flag("MeshFixUtoP");
    for (int i=0; i < md->NumBlocks(); ++i)
        FixUtoP(md->GetBlockData(i).get());
    EndFlag();
    return TaskStatus::complete;
}

/**
 * Ensure minimum temperature in a stable way:
 * 1. If possible, preserve total energy by stealing kinetic energy
 * 2. If not (i.e., rest energy of B exceeds total energy), add energy to
 *    ensure B and u but no velocity
 */
TaskStatus Backstop(MeshBlockData<Real> *rc);

/**
 * Count up all nonzero PFlags on md.  Used for history file reductions.
 */
int CountPFlags(MeshData<Real> *md);

/**
 * Print details of any inversion failures or fixed zones
 */
TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md);

}
