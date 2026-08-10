/*
 *  File: electrons.hpp
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

#include "grmhd_functions.hpp"

using namespace parthenon;

/**
 * This physics package implements self-consistent Electron heating and transport
 * as in Ressler+ 2015.
 * It tracks fluid total entropy in order to calculate stepwise changes, and electron
 * entropy fed by some fraction of dissipation.
 * It supports running any/all of several heating models (dissipation fractions) via
 * runtime parameters.
 *
 * Adapted very closely from work done for iharm3d by Cesar Diaz
 *
 * This implementation doubles as a guide to writing packages for KHARMA -- the comments
 * provide basic explanations about how they interact with KHARMA as a whole. Most package
 * functions are either registered as callbacks at the end of Initialize() (for Parthenon)
 * or added to the functions in kharma.cpp (for KHARMA).  Be sure to register all your
 * functions when first creating a package!
 */
namespace Electrons
{
/**
 * Initialization: declare any fields this package will evolve, initialize any parameters
 *
 * For electrons, this means a total entropy Ktot to track dissipation, and electron
 * entropies for each model being run.
 */
std::shared_ptr<KHARMAPackage> Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages);

/**
 * In addition to the standard functions, packages can include extras.  This is called
 * manually at the end of problem initialization in problem.cpp
 *
 * Function in this package: Initialize electron temperatures when setting up the problem.
 * Trivial.
 */
TaskStatus InitElectrons(MeshBlockData<Real>* rc, ParameterInput* pin);
inline TaskStatus MeshInitElectrons(MeshData<Real>* md, ParameterInput* pin)
{
    Flag("MeshApplyElectronHeating");
    for (int i = 0; i < md->NumBlocks(); ++i)
        InitElectrons(md->GetBlockData(i).get(), pin);
    EndFlag();
    return TaskStatus::complete;
}

/**
 * Any implementation of UtoP needs to take an IndexDomain enum and boundary "coarse"
 * boolean. This allows KHARMA to call it over the whole domain (IndexDomain::entire) or
 * just on a boundary after conserved variables have been updated.
 *
 * Usually this should default to the entire domain, as the KHARMA algorithm relies on
 * applying UtoP over ghost zones.
 *
 * Function in this package: Get the specific entropy primitive value, by dividing the
 * total entropy K/(rho*u^0)
 */
void BlockUtoP(MeshBlockData<Real>* rc, IndexDomain domain, bool coarse = false);

/**
 * This heating step is custom for this package.  It is added manually to any task list in
 * the KHARMADriver, at the very end of the step. For reasons mentioned there & above, it
 * must update *all* zones, incl. ghosts.
 *
 * The function calculates how electrons should be heated and updates their entropy
 * values, using each step's total dissipation (advected vs actual fluid entropy) It
 * applies any or all of several different esimates for this split, to each of the several
 * different primitive variables "prims.Kel_X" Finally, it checks the results against a
 * minimum and maximum temperature ratio T_protons/T_electrons
 *
 * To recap re: floors:
 * This function expects two sets of values {rho0, u0, Ktot0} from rc_old and {rho1, u1}
 * from rc, all of which obey all given floors. It produces end-of-substep values {Ktot1,
 * Kel_X1, Kel_Y1, etc}, which are also guaranteed to obey floors
 *
 * TODO this function should update fflag to reflect temperature ratio floor hits
 */
TaskStatus ApplyElectronHeating(
    MeshBlockData<Real>* rc_old, MeshBlockData<Real>* rc, bool generate_grf = false);
inline TaskStatus MeshApplyElectronHeating(
    MeshData<Real>* md_old, MeshData<Real>* md, bool generate_grf = false)
{
    Flag("MeshApplyElectronHeating");
    for (int i = 0; i < md->NumBlocks(); ++i)
        ApplyElectronHeating(
            md_old->GetBlockData(i).get(), md->GetBlockData(i).get(), generate_grf);
    EndFlag();
    return TaskStatus::complete;
}

/**
 * Apply adjustments to KTOT & e- K values based on floors.
 * Note that Kmin/max limits are applied immediately at heating,
 * *not* here.
 */
void ApplyFloors(MeshBlockData<Real>* mbd, IndexDomain domain);

/**
 * Diagnostics printed/computed after each step, called from kharma.cpp
 *
 * Function in this package: Currently nothing
 */
TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real>* rc);

/**
 * Fill fields which are calculated only for output to dump files, called from kharma.cpp
 *
 * Function in this package: Currently nothing
 */
void FillOutput(MeshBlock* pmb, ParameterInput* pin);

/**
 * KHARMA requires some method for getting conserved variables from primitives, as well.
 *
 * However, unlike UtoP, p_to_u is implemented device-side. That means that any
 * package defining new primitive/conserved vars must not only provide a prim_to_flux
 * here, but add it to the list in Flux::prim_to_flux.
 */
KOKKOS_FORCEINLINE_FUNCTION void p_to_u(const GRCoordinates& G,
    const VariablePack<Real>& P, const VarMap& m_p, const int& k, const int& j,
    const int& i, const VariablePack<Real>& flux, const VarMap m_u,
    const Loci loc = Loci::center)
{
    // Take the factor from the primitives, in case we need to reorder this to happen
    // before GRMHD::prim_to_flux later
    const Real ut =
        GRMHD::lorentz_calc(G, P, m_p, k, j, i, loc) * m::sqrt(-G.gcon(loc, j, i, 0, 0));
    const Real rho_ut = P(m_p.RHO, k, j, i) * ut * G.gdet(loc, j, i);

    flux(m_u.KTOT, k, j, i) = rho_ut * P(m_p.KTOT, k, j, i);
    if (m_p.K_CONSTANT >= 0)
        flux(m_u.K_CONSTANT, k, j, i) = rho_ut * P(m_p.K_CONSTANT, k, j, i);
    if (m_p.K_HOWES >= 0) flux(m_u.K_HOWES, k, j, i) = rho_ut * P(m_p.K_HOWES, k, j, i);
    if (m_p.K_KAWAZURA >= 0)
        flux(m_u.K_KAWAZURA, k, j, i) = rho_ut * P(m_p.K_KAWAZURA, k, j, i);
    if (m_p.K_WERNER >= 0)
        flux(m_u.K_WERNER, k, j, i) = rho_ut * P(m_p.K_WERNER, k, j, i);
    if (m_p.K_ROWAN >= 0) flux(m_u.K_ROWAN, k, j, i) = rho_ut * P(m_p.K_ROWAN, k, j, i);
    if (m_p.K_SHARMA >= 0)
        flux(m_u.K_SHARMA, k, j, i) = rho_ut * P(m_p.K_SHARMA, k, j, i);
}

}
