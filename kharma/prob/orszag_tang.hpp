#pragma once

#include "decs.hpp"
#include "types.hpp"

#include "b_ct.hpp"
#include "domain.hpp"

using namespace parthenon;

/**
 * relativistic version of the Orszag-Tang vortex
 * Orszag & Tang 1979, JFM 90, 129-143.
 * original OT problem was incompressible
 * this is based on compressible version given
 * in Toth 2000, JCP 161, 605.
 *
 * in the limit tscale -> 0 the problem is identical
 * to the nonrelativistic problem; as tscale increases
 * the problem becomes increasingly relativistic
 *
 * Originally stolen directly from iharm2d_v3,
 * now somewhat modified
 */
TaskStatus InitializeOrszagTang(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;

    const auto& G = pmb->coords;

    const Real gam = pmb->packages.Get("eos")->Param<Real>("gm1") + 1.0;
    const Real tscale = pin->GetOrAddReal("orszag_tang", "tscale", 0.05);
    // Default phase puts the current sheet in the middle of the domain
    const Real phase = pin->GetOrAddReal("orszag_tang", "phase", M_PI);

    // Set parameters for B field, which will get added differently for flux vs face
    // In a questionable decision, we allow overriding these
    pin->GetOrAddString("b_field", "type", "orszag_tang_a");
    pin->GetOrAddReal("b_field", "amp_B1", tscale);
    pin->GetOrAddReal("b_field", "amp_B2", tscale);
    pin->GetOrAddReal("b_field", "phase", phase);

    // Set the non-B values
    IndexDomain domain = IndexDomain::entire;
    IndexRange3 b = KDomain::GetRange(rc, domain);
    pmb->par_for("ot_init", b.ks, b.ke, b.js, b.je, b.is, b.ie,
                 KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            Real X[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, X);
            rho(k, j, i) = 25. / 9.;
            u(k, j, i) = 5. / (3. * (gam - 1.)) * tscale * tscale;
            uvec(0, k, j, i) = -m::sin(X[2] + phase) * tscale;
            uvec(1, k, j, i) = m::sin(X[1] + phase) * tscale;
            uvec(2, k, j, i) = 0.;
        });

    return TaskStatus::complete;
}
