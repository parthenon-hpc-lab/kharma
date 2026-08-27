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
    int counter = -5.0;
    Params& g_params = pmb->packages.Get("GRMHD")->AllParams();
    const Real gam = g_params.Get<Real>("gamma");
    if (!g_params.hasKey("counter")) g_params.Add("counter", counter, true);
    Real rho0 = (mach / v0) * sqrt(gam * (gam - 1));
    Real ug0 = (v0 / mach) / sqrt(gam * (gam - 1));
    if (!g_params.hasKey("rho0")) g_params.Add("rho0", rho0);
    if (!g_params.hasKey("v0")) g_params.Add("v0", v0);
    if (!g_params.hasKey("ug0")) g_params.Add("ug0", ug0);
    if (!g_params.hasKey("cooling")) g_params.Add("cooling", cooling);
    if (!g_params.hasKey("context_boundaries"))
        g_params.Add("context_boundaries", context_boundaries);

    // This is how we will initialize kel values later
    if (pmb->packages.AllPackages().count("Electrons")) {
        const Real fel0 = pmb->packages.Get("Electrons")->Param<Real>("fel_0");
        if (!g_params.hasKey("ue0")) g_params.Add("ue0", fel0 * ug0);
    }

    // Override end time to be 1 dynamical time L/max(v@t=0)
    if (set_tlim) {
        pin->SetReal("parthenon/time", "tlim", dyntimes / v0);
    }

    // Replace the boundary conditions
    auto bound_pkg = pmb->packages.Get<KHARMAPackage>("Boundaries");
    bound_pkg->KBoundaries[BoundaryFace::inner_x1] = SetHubble<IndexDomain::inner_x1>;
    bound_pkg->KBoundaries[BoundaryFace::outer_x1] = SetHubble<IndexDomain::outer_x1>;
    bound_pkg->BlockApplyPrimSource = ApplyHubbleHeating;

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
    // first time this is called in boundary conditions inside the time stepping cycle is
    // when counter == 0
    int counter = pmb->packages.Get("GRMHD")->Param<int>("counter");
    const Real tt = pmb->packages.Get("Globals")->Param<Real>("time");
    const Real dt = pmb->packages.Get("Globals")->Param<Real>("dt_last");

    Real t = tt + 0.5 * dt;
    if ((counter % 4) > 1) t = tt + dt;

    const auto& G = pmb->coords;

    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);

    if (!context_boundaries || counter < 0) {
        // Setting as in equation 37
        Real toberho = rho0 / (1. + v0 * t);
        Real tobeu = ug0 / pow(1 + v0 * t, 2);
        if (!cooling) tobeu = ug0 / pow(1 + v0 * t, gam);
        pmb->par_for("hubble_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
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
            Real tobeke = (gam - 2) * (game - 1) / (game - 2) * ue0 / pow(rho0, game) *
                          pow(1 + v0 * t, game - 2);
            // Without cooling, the entropy of electrons should stay the same, analytic
            // solution.
            if (!cooling)
                tobeke = (gam - 2) * (game - 1) / (game - 2) * ue0 / pow(rho0, game);
            pmb->par_for("hubble_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
                {
                    ktot(k, j, i) = tobeke;
                    kel_const(k, j, i) = tobeke; // Since we are using fel = 1
                });
        }
    } else { // We assume the fluid is following the solution so we set the boundaries
             // from the real zones
        // Left zone is first one to be called and counter starts at zero
        bool left_zone = !(counter % 2);
        int context_index = 0;
        if (left_zone)
            context_index = ib.e + 1;
        else
            context_index = ib.s - 1;

        Real context_X[GR_DIM];
        G.coord_embed(0, 0, context_index, Loci::center, context_X);
        Real context_t = (v0 * context_X[1] - uvec(0, 0, context_index)) /
                         (uvec(0, 0, context_index) * v0);

        pmb->par_for("hubble_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
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
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
                {
                    kel_const(k, j, i) = kel_const(k, j, context_index);
                });
        }
    }
    pmb->packages.Get("GRMHD")->UpdateParam<int>("counter", ++counter);
    return TaskStatus::complete;
}

// TODO(BSP) Add MeshApplySource callback & convert this
void ApplyHubbleHeating(MeshBlockData<Real>* mbase)
{
    auto pmb0 = mbase->GetBlockPointer();

    PackIndexMap prims_map;
    auto P_mbase = GRMHD::PackHDPrims(mbase, prims_map);
    const VarMap m_p(prims_map, false);

    Real Q = 0;
    const Real dt =
        pmb0->packages.Get("Globals")->Param<Real>("dt_last"); // Close enough?
    const Real t = pmb0->packages.Get("Globals")->Param<Real>("time") + 0.5 * dt;
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
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
        {
            P_mbase(m_p.UU, k, j, i) += Q * dt * 0.5;
        });
}
