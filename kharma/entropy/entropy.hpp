/*
 *  File: entropy.hpp
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

#include "decs.hpp"
#include "kharma_package.hpp"
#include "types.hpp"

using namespace parthenon;

/**
 * This physics package tracks the fluid's total entropy Ktot, independent of any
 * package which might make use of it (e.g. Electrons, which splits the dissipation
 * it represents among several electron heating models).
 *
 * Ktot is advected as a normal passive scalar (see Flux::prim_to_flux) over the course
 * of a step, then reset at step end to the value implied by the real, current rho & u --
 * i.e., the entropy the fluid *actually* has, dissipation included.  The difference between
 * the pre-reset (purely advected, dissipation-free) value and this real value is exactly
 * the numerical dissipation incurred during the step -- see Ressler+ 2015 eq. 27 for the
 * expression this generalizes, and Electrons::ApplyElectronHeating for a consumer.
 *
 * Optionally (see "advect_entropy" below), this package also tracks a second, genuinely
 * idealized version of the entropy, Ktot_adv, which is *never* reset: it is simply
 * advected for the entire run as if there had been no dissipation at all.  Ktot - Ktot_adv
 * is then the total dissipation (numerical, or explicit if EMHD is enabled) accumulated
 * since the start of the run -- a useful global diagnostic, but not otherwise consumed
 * by KHARMA.
 */
namespace Entropy {
/**
 * Initialization: declare the fields this package will track, initialize parameters.
 */
std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

/**
 * Compute the fluid's specific total entropy from the primitives rho, u, i.e. the entropy
 * the fluid would need in order for its pressure to be exactly (gam-1)*u given its density.
 * This is the "real"/current value of Ktot, as opposed to the purely-advected value obtained
 * by evolving Ktot as a passive scalar.
 */
KOKKOS_FORCEINLINE_FUNCTION Real CalcEntropy(const Real& rho, const Real& u, const Real& gam)
{
    return (gam - 1.) * u * m::pow(rho, -gam);
}

/**
 * Set the initial values of Ktot (and Ktot_adv, if enabled) from the problem's initial rho, u.
 * Called manually at the end of problem initialization in problem.cpp, mirroring Electrons.
 */
TaskStatus InitEntropy(MeshBlockData<Real> *rc, ParameterInput *pin);
inline TaskStatus MeshInitEntropy(MeshData<Real> *md, ParameterInput *pin)
{
    Flag("MeshInitEntropy");
    for (int i=0; i < md->NumBlocks(); ++i)
        InitEntropy(md->GetBlockData(i).get(), pin);
    EndFlag();
    return TaskStatus::complete;
}

/**
 * As with Electrons::BlockUtoP, get the specific entropy primitive(s) by dividing the
 * conserved total (density-weighted) entropy by the density: Ktot/(rho*u^0).
 */
void BlockUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse=false);

/**
 * Reset Ktot, at the end of a step, to the real value implied by the step's final rho, u.
 * This must run *after* any package (e.g. Electrons) which needs to compare the real,
 * post-step entropy against the value obtained by pure advection over the step -- once this
 * runs, that comparison is no longer available, as Ktot no longer holds the advected value.
 *
 * Does not touch Ktot_adv, which needs no such per-step update: it is left to evolve via the
 * standard advection machinery for the whole run.
 */
TaskStatus ApplyEntropyUpdate(MeshBlockData<Real> *rc);
inline TaskStatus MeshUpdateEntropy(MeshData<Real> *md)
{
    Flag("MeshUpdateEntropy");
    for (int i=0; i < md->NumBlocks(); ++i)
        ApplyEntropyUpdate(md->GetBlockData(i).get());
    EndFlag();
    return TaskStatus::complete;
}

/**
 * Apply a ceiling to Ktot, moved here unmodified from the Electrons package it was
 * previously part of.  Note this is not currently wired to any callback (there, or here).
 */
void ApplyFloors(MeshBlockData<Real> *mbd, IndexDomain domain);

}
