/*
 *  File: hubble.cpp
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
#include "hubble.hpp"

#include "pack.hpp"
#include "types.hpp"

#include <stdexcept>

TaskStatus InitializeHubble(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();

    const Real mach = pin->GetOrAddReal("hubble", "mach", 1.);
    const Real v0 = pin->GetOrAddReal("hubble", "v0", 1.e-3);
    // Whether to stop after "dyn_times" dynamical time L/max(v0*x)
    bool set_tlim = pin->GetOrAddBoolean("hubble", "set_tlim", false);
    bool cooling = pin->GetOrAddBoolean("hubble", "cooling", true);
    bool context_boundaries = pin->GetOrAddBoolean("hubble", "context_boundaries", false);
    Real dyntimes = pin->GetOrAddReal("hubble", "dyntimes", 1.0);

    // Add everything to package parameters, since they continue to be needed on
    // boundaries
    Params& g_params = pmb->packages.Get("GRMHD")->AllParams();
    const Real gam = g_params.Get<Real>("gamma");
    Real rho0 = (mach / v0) * sqrt(gam * (gam - 1));
    Real ug0 = (v0 / mach) / sqrt(gam * (gam - 1));
    if (!g_params.hasKey("rho0")) g_params.Add("rho0", rho0);
    if (!g_params.hasKey("v0")) g_params.Add("v0", v0);
    if (!g_params.hasKey("ug0")) g_params.Add("ug0", ug0);
    if (!g_params.hasKey("cooling")) g_params.Add("cooling", cooling);
    if (!g_params.hasKey("context_boundaries"))
        g_params.Add("context_boundaries", context_boundaries);

    // The electron entropy is pinned by the analytic solution (Ressler+ '15 eq. 40):
    // requiring that solution to hold at t=0 fixes ue(0)/ug(0) = (gam-2)/(game-2).
    // Nothing is free here, so refuse to run rather than let InitElectrons quietly
    // overwrite what we set below with (game-1)*fel_0*u/rho^game.
    if (pmb->packages.AllPackages().count("Electrons")) {
        const Real game = pmb->packages.Get("Electrons")->Param<Real>("gamma_e");
        const Real fel0_needed = (gam - 2) / (game - 2);
        if (pin->GetOrAddBoolean("electrons", "init_to_fel_0", true)) {
            throw std::invalid_argument("The Hubble problem sets the electron entropy "
                                        "itself: set <electrons>/init_to_fel_0 = false");
        }
        if (!g_params.hasKey("ue0")) g_params.Add("ue0", fel0_needed * ug0);
    }

    // Override end time to be 1 dynamical time L/max(v@t=0)
    if (set_tlim) {
        pin->SetReal("parthenon/time", "tlim", dyntimes / v0);
    }

    // Replace the boundary conditions
    auto bound_pkg = pmb->packages.Get<KHARMAPackage>("Boundaries");
    bound_pkg->KBoundaries[BoundaryFace::inner_x1] = SetHubble<IndexDomain::inner_x1>;
    bound_pkg->KBoundaries[BoundaryFace::outer_x1] = SetHubble<IndexDomain::outer_x1>;
    // Only the "cooling" version of the problem has a source term at all
    if (cooling) bound_pkg->BlockApplyPrimSource = ApplyHubbleHeating;

    // Then call the general function to fill the grid
    SetHubble<IndexDomain::entire>(rc);

    return TaskStatus::complete;
}

TaskStatus SetHubbleImpl(
    std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real rho0 = pmb->packages.Get("GRMHD")->Param<Real>("rho0");
    const Real v0 = pmb->packages.Get("GRMHD")->Param<Real>("v0");
    const bool cooling = pmb->packages.Get("GRMHD")->Param<bool>("cooling");
    const bool context_boundaries =
        pmb->packages.Get("GRMHD")->Param<bool>("context_boundaries");
    const Real ug0 = pmb->packages.Get("GRMHD")->Param<Real>("ug0");

    // The whole point of this test is that the analytic solution be sampled at exactly
    // the time each sub-step lands on -- otherwise the boundaries are only 1st-order
    // accurate in time and swamp what we're trying to measure.  The driver records that
    // time for us.  Before the evolution loop starts (initialization, and the boundary
    // sync for the t=0 output) we are simply at "time": note we can't fall back on
    // "dt_last" there, as it is still DBL_MAX until the first PreStepWork.
    const bool in_loop = pmb->packages.Get("Globals")->Param<bool>("in_loop");
    const Real t = (in_loop)
                       ? pmb->packages.Get("Globals")->Param<double>("time_substep_end")
                       : pmb->packages.Get("Globals")->Param<double>("time");

    const auto& G = pmb->coords;

    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);

    if (!context_boundaries || !in_loop) {
        // Setting as in equation 37
        Real toberho = rho0 / (1. + v0 * t);
        Real tobeu = ug0 / pow(1 + v0 * t, 2);
        if (!cooling) tobeu = ug0 / pow(1 + v0 * t, gam);
        pmb->par_for("hubble_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                     KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
            {
                Real X[GR_DIM];
                G.coord_embed(k, j, i, Loci::center, X);
                rho(k, j, i) = toberho;
                u(k, j, i) = tobeu;
                uvec(0, k, j, i) = v0 * X[1] / (1 + v0 * t);
                uvec(1, k, j, i) = 0.0;
                uvec(2, k, j, i) = 0.0;
            });

        if (pmb->packages.AllPackages().count("Electrons")) {
            GridScalar ktot = rc->Get("prims.Ktot").data;
            GridScalar kel_const = rc->Get("prims.Kel_Constant").data;
            const Real game = pmb->packages.Get("Electrons")->Param<Real>("gamma_e");
            const Real ue0 = pmb->packages.Get("GRMHD")->Param<Real>("ue0");
            // Equation 40: with ue0 = (gam-2)/(game-2)*ug0 this is identically
            // (gam-2)(game-1)/(game-2) * ug0/rho0^game * (1+v0*t)^(game-2)
            Real tobeke =
                (game - 1) * ue0 / pow(rho0, game) * pow(1 + v0 * t, game - 2);
            // Without cooling, the entropy of electrons should stay the same, analytic
            // solution.
            if (!cooling) tobeke = (game - 1) * ue0 / pow(rho0, game);
            // Ktot is the *total* entropy, and it is what Electrons::ApplyElectronHeating
            // differences against to find the dissipation.  Setting it to the entropy the
            // gas actually has here means these zones see zero dissipation, so they keep
            // exactly the analytic Kel we just set.
            const Real tobektot = (gam - 1) * tobeu / pow(toberho, gam);
            pmb->par_for("hubble_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                         KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
                {
                    ktot(k, j, i) = tobektot;
                    kel_const(k, j, i) = tobeke; // Since we are using fel = 1
                });
        }
    } else { // We assume the fluid is following the solution so we set the boundaries
             // from the real zones
        // Take our cue from the first physical zone just inside this boundary
        int context_index = 0;
        if (domain == IndexDomain::inner_x1)
            context_index = ib.e + 1;
        else
            context_index = ib.s - 1;

        Real context_X[GR_DIM];
        G.coord_embed(0, 0, context_index, Loci::center, context_X);
        Real context_t = (v0 * context_X[1] - uvec(0, 0, 0, context_index)) /
                         (uvec(0, 0, 0, context_index) * v0);

        pmb->par_for("hubble_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                     KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
            {
                Real X[GR_DIM];
                G.coord_embed(k, j, i, Loci::center, X);
                rho(k, j, i) = rho(k, j, context_index);
                u(k, j, i) = u(k, j, context_index);
                uvec(0, k, j, i) = v0 * X[1] / (1 + v0 * context_t);
            });
        if (pmb->packages.AllPackages().count("Electrons")) {
            GridScalar kel_const = rc->Get("prims.Kel_Constant").data;
            pmb->par_for("hubble_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                         KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
                {
                    kel_const(k, j, i) = kel_const(k, j, context_index);
                });
        }
    }
    return TaskStatus::complete;
}

// TODO(CEP) Add MeshApplySource callback & convert this
void ApplyHubbleHeating(MeshBlockData<Real>* mbase)
{
    auto pmb0 = mbase->GetBlockPointer();

    PackIndexMap prims_map;
    auto P_mbase = GRMHD::PackHDPrims(mbase, prims_map);
    const VarMap m_p(prims_map, false);

    Real Q = 0;
    // This runs once per sub-step, so treat Q as just another term of the RHS: evaluate
    // it at the time this sub-step's fluxes are evaluated, and weight it the same way the
    // integrator weights those fluxes.  The stages then combine the source exactly as
    // they combine everything else, which is what makes the heating 2nd-order in time --
    // the whole point of this test.
    const auto dt = pmb0->packages.Get("Globals")->Param<double>("dt_substep");
    const Real t = pmb0->packages.Get("Globals")->Param<double>("time_substep");
    const Real v0 = pmb0->packages.Get("GRMHD")->Param<Real>("v0");
    const Real ug0 = pmb0->packages.Get("GRMHD")->Param<Real>("ug0");
    const Real gam = pmb0->packages.Get("GRMHD")->Param<Real>("gamma");
    Q = (ug0 * v0 * (gam - 2) / pow(1 + v0 * t, 3));
    IndexDomain domain = IndexDomain::interior;
    auto ib = mbase->GetBoundsI(domain);
    auto jb = mbase->GetBoundsJ(domain);
    auto kb = mbase->GetBoundsK(domain);
    auto block = IndexRange{0, P_mbase.GetDim(5) - 1};

    pmb0->par_for("heating_substep", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                  KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            P_mbase(m_p.UU, k, j, i) += Q * dt;
        });
}
