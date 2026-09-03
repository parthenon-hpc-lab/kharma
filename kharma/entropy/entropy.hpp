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
 * i.e., the entropy the fluid *actually* has, dissipation included.  The difference
 * between the pre-reset (purely advected, dissipation-free) value and this real value is
 * exactly the numerical dissipation incurred during the step -- see Ressler+ 2015 eq. 27
 * for the expression this generalizes, and Electrons::ApplyElectronHeating for a
 * consumer.
 *
 * Optionally (see "advect_entropy" below), this package also tracks a second, genuinely
 * idealized entropy, Ktot_adv, which is *never* reset: it is simply advected for the
 * entire run as if there had been no dissipation at all.
 *
 * NOTE Ktot and Ktot_adv are not the same kind of object, and are not directly
 * comparable:
 *
 *   Ktot     = (gam-1) u / rho^gam       specific entropy, per unit mass.
 *                                        Lagrangian invariant: u^mu d_mu Ktot = 0, so it
 *                                        advects weighted by the mass flux, and its
 *                                        conserved form is sqrt(-g) rho u^t Ktot.
 *
 *   Ktot_adv = (gam-1) u / rho^(gam-1)   entropy *density*, i.e. rho * Ktot.  This is
 *                                        Noble+ 2009 (arXiv:0808.3140) eq. 20, the
 *                                        variable used by their entropy-based 
 *                                        primitive inversion scheme.
 *
 * The two conserved forms are numerically identical -- sqrt(-g) rho u^t Ktot ==
 * sqrt(-g) u^t (rho Ktot) -- which is why only the primitive normalization and the flux
 * expression differ between them.  The reason for carrying Ktot_adv in Noble's form
 * rather than as a second specific entropy is that the inversion wants p directly:
 * given rho, p = Ktot_adv * rho^(gam-1) with no root-find.
 *
 * Consequently the dissipation-free specific entropy is Ktot_adv/rho, and the
 * accumulated dissipation is Ktot - Ktot_adv/rho.
 *
 */
namespace Entropy
{
/**
 * Initialization: declare the fields this package will track, initialize parameters.
 */
std::shared_ptr<KHARMAPackage> Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages);

/**
 * Compute the fluid's specific total entropy from the primitives rho, u, i.e. the entropy
 * the fluid would need in order for its pressure to be exactly (gam-1)*u given its
 * density.  This is the "real"/current value of Ktot, as opposed to the purely-advected
 * value obtained by evolving Ktot as a passive scalar.
 */
KOKKOS_FORCEINLINE_FUNCTION Real CalcEntropy(
    const Real& rho, const Real& u, const Real& gam)
{
    return (gam - 1.) * u * m::pow(rho, -gam);
}

/**
 * The same entropy as a *density* rather than per unit mass: S = p/rho^(gam-1), i.e.
 * rho*CalcEntropy(). This is Noble+ 2009 eq. 20, the form Ktot_adv is carried in.
 * Kept as its own function rather than a multiply-by-rho so that callers cannot 
 * quietly mix the two normalizations.
 */
KOKKOS_FORCEINLINE_FUNCTION Real CalcEntropyDensity(
    const Real& rho, const Real& u, const Real& gam)
{
    return (gam - 1.) * u * m::pow(rho, 1. - gam);
}

/**
 * Set the initial values of Ktot (and Ktot_adv, if enabled) from the problem's initial
 * rho, u.  Called manually at the end of problem initialization in problem.cpp, mirroring
 * Electrons.
 */
TaskStatus InitEntropy(MeshBlockData<Real>* rc, ParameterInput* pin);
inline TaskStatus MeshInitEntropy(MeshData<Real>* md, ParameterInput* pin)
{
    Flag("MeshInitEntropy");
    for (int i = 0; i < md->NumBlocks(); ++i) InitEntropy(md->GetBlockData(i).get(), pin);
    EndFlag();
    return TaskStatus::complete;
}

/**
 * Recover the entropy primitives from their conserved forms.  The two differ:
 *   Ktot     = cons.Ktot / cons.rho                    (per mass, as Electrons::BlockUtoP)
 *   Ktot_adv = cons.Ktot_adv * prims.rho / cons.rho    (a density; the ratio here is
 *                                                       1/(gdet*u^t))
 */
void BlockUtoP(MeshBlockData<Real>* rc, IndexDomain domain, bool coarse = false);

/**
 * Reset Ktot, at the end of a (sub-)step, to the real value implied by that step's final
 * rho, u.  This must run *after* any package (e.g. Electrons) which needs to compare the
 * real, post-step entropy against the value obtained by pure advection over the step --
 * once this runs, that comparison is no longer available, as Ktot no longer holds the
 * advected value.
 *
 * Does not touch Ktot_adv, which needs no such per-step update: it is left to evolve via
 * the standard advection machinery for the whole run.  (This is also what keeps 
 * Ktot_adv usable for an entropy-based inversion: it must carry the entropy the fluid 
 * would have had *without* the dissipation.)
 */
TaskStatus ApplyEntropyUpdate(MeshBlockData<Real>* rc);
inline TaskStatus MeshUpdateEntropy(MeshData<Real>* md)
{
    Flag("MeshUpdateEntropy");
    for (int i = 0; i < md->NumBlocks(); ++i)
        ApplyEntropyUpdate(md->GetBlockData(i).get());
    EndFlag();
    return TaskStatus::complete;
}

/**
 * Apply a ceiling to Ktot, moved here unmodified from the Electrons package it was
 * previously part of.  Note this is not currently wired to any callback (there, or here).
 */
void ApplyFloors(MeshBlockData<Real>* mbd, IndexDomain domain);

}
