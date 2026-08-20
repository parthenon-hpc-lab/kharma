#pragma once

#include "decs.hpp"

using namespace parthenon;

/**
 * Generic initializer for shock tubes
 * Particular problems in pars/shocks/
 *
 * Stolen directly from iharm3D
 */
TaskStatus InitializeShockTube(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;

    const auto& G = pmb->coords;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    // TODO some particular default shock
    const Real rhoL = pin->GetOrAddReal("shock", "rhoL", 0.0);
    const Real rhoR = pin->GetOrAddReal("shock", "rhoR", 0.0);
    const Real PL = pin->GetOrAddReal("shock", "PL", 0.0);
    const Real PR = pin->GetOrAddReal("shock", "PR", 0.0);
    const Real u1L = pin->GetOrAddReal("shock", "u1L", 0.0);
    const Real u1R = pin->GetOrAddReal("shock", "u1R", 0.0);
    const Real u2L = pin->GetOrAddReal("shock", "u2L", 0.0);
    const Real u2R = pin->GetOrAddReal("shock", "u2R", 0.0);
    const Real u3L = pin->GetOrAddReal("shock", "u3L", 0.0);
    const Real u3R = pin->GetOrAddReal("shock", "u3R", 0.0);

    const Real B1L = pin->GetOrAddReal("shock", "B1L", 0.0);
    const Real B1R = pin->GetOrAddReal("shock", "B1R", 0.0);
    const Real B2L = pin->GetOrAddReal("shock", "B2L", 0.0);
    const Real B2R = pin->GetOrAddReal("shock", "B2R", 0.0);
    const Real B3L = pin->GetOrAddReal("shock", "B3L", 0.0);
    const Real B3R = pin->GetOrAddReal("shock", "B3R", 0.0);

    IndexDomain domain = IndexDomain::entire;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);

    const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
    const Real x1max = pin->GetReal("parthenon/mesh", "x1max");
    const Real center = (x1min + x1max) / 2.;

    pin->GetOrAddString("b_field", "type", "shock_tube");
    pin->GetOrAddReal("b_field", "phase", center);
    pin->GetOrAddReal("b_field", "amp_B1", B1L);
    pin->GetOrAddReal("b_field", "amp_B2", B2L);
    pin->GetOrAddReal("b_field", "amp_B3", B3L);
    pin->GetOrAddReal("b_field", "amp2_B1", B1R);
    pin->GetOrAddReal("b_field", "amp2_B2", B2R);
    pin->GetOrAddReal("b_field", "amp2_B3", B3R);

    pmb->par_for("ot_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                 KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            Real X[GR_DIM];
            G.coord(k, j, i, Loci::center, X);

            const bool lhs = X[1] < center;
            rho(k, j, i) = (lhs) ? rhoL : rhoR;
            u(k, j, i) = ((lhs) ? PL : PR) / (gam - 1.);
            uvec(0, k, j, i) = (lhs) ? u1L : u1R;
            uvec(1, k, j, i) = (lhs) ? u2L : u2R;
            uvec(2, k, j, i) = (lhs) ? u3L : u3R;
        });

    return TaskStatus::complete;
}
