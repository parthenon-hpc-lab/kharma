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
TaskStatus InitializeRadiationPulse(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin)
{
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    GridScalar u_rad = rc->Get("prims.u_rad").data;
    GridVector uvec_rad = rc->Get("prims.uvec_rad").data;

    if(!pmb->packages.AllPackages().count("RadM1"))
        PARTHENON_FAIL("RadM1 package not loaded.");

    const auto& G = pmb->coords;

    IndexDomain domain = IndexDomain::entire;
    IndexRange3 b = KDomain::GetRange(rc, domain);
    pmb->par_for("radpulse_init", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            Real X[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, X);
            // Mckinney's setup 
            u_rad(k, j, i) = m::pow(8.77e-12 * 1.e6* (1 + m::exp(- (X[1] * X[1] + X[2] * X[2] + X[3] * X[3])/25. )), 4);
            uvec_rad(0, k, j, i) = 0.;
            uvec_rad(1, k, j, i) = 0.;
            uvec_rad(2, k, j, i) = 0.;


            rho(k, j, i) = 1.;
            u(k, j, i) = 0.01;
            uvec(0, k, j, i) = 0.;
            uvec(1, k, j, i) = 0.;
            uvec(2, k, j, i) = 0.;
        }
    );

    return TaskStatus::complete;
}
