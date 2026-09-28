#pragma once

#include "decs.hpp"
#include "types.hpp"

#include "b_ct.hpp"
#include "domain.hpp"

using namespace parthenon;

/**
 * Mckinney Optically Thin Radiative Pulse in 3D Cartesian Minkowsky as done in Mckinney
 * (MNRAS, 441, 4, 11 July 2024, 3177-3208)
 */
TaskStatus InitializeRadiationPulse(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();

    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    GridScalar u_rad = rc->Get("prims.u_rad").data;
    GridVector uvec_rad = rc->Get("prims.uvec_rad").data;

    Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    if (!pmb->packages.AllPackages().count("RadM1"))
        PARTHENON_FAIL("RadM1 package not loaded.");

    const auto& G = pmb->coords;

    IndexDomain domain = IndexDomain::entire;
    IndexRange3 b = KDomain::GetRange(rc, domain);
    pmb->par_for("radpulse_init", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
        {
            Real X[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, X);

            Real r2 = X[1] * X[1] + X[2] * X[2] + X[3] * X[3];
            Real T_a = 1.e6;
            Real w = 5.0;
            Real T_rad = T_a * (1.0 + 100.0 * m::exp(-r2 / (w * w)));
            Real a_rad = 8.77e-12;
            u_rad(k, j, i) = a_rad * m::pow(T_rad, 4);
            uvec_rad(0, k, j, i) = 0.;
            uvec_rad(1, k, j, i) = 0.;
            uvec_rad(2, k, j, i) = 0.;

            rho(k, j, i) = 1.0;
            u(k, j, i) = (rho(k, j, i) * T_a) / (gam - 1.0);
            uvec(0, k, j, i) = 0.;
            uvec(1, k, j, i) = 0.;
            uvec(2, k, j, i) = 0.;
        });

    return TaskStatus::complete;
}
