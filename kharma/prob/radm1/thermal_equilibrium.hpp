#pragma once

#include "decs.hpp"
#include "types.hpp"

#include "b_ct.hpp"
#include "domain.hpp"
#include "radM1.hpp"

using namespace parthenon;

/**
 * Turner & Stone (2001) radiation-matter thermal equilibrium test, as described in
 * section 4.3 of https://iopscience.iop.org/article/10.3847/1538-4365/ab18ff/pdf .
 * Kharma can't handle a single cell, so we'll use multiple cells, but set the flux to 0.
 */
TaskStatus InitializeThermalEquilibrium(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();

    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    GridScalar u_rad = rc->Get("prims.u_rad").data;
    GridVector uvec_rad = rc->Get("prims.uvec_rad").data;

    if (!pmb->packages.AllPackages().count("RadM1"))
        PARTHENON_FAIL("RadM1 package not loaded.");

    const RadM1::UnitScales units_cgs =
        pmb->packages.Get("RadM1")->AllParams().Get<RadM1::UnitScales>("units_cgs");
    const Real mass_density_scale =
        units_cgs.mass_cgs / (units_cgs.length_cgs * units_cgs.length_cgs * units_cgs.length_cgs);
    const Real energy_density_scale = mass_density_scale * pc::c * pc::c;

    // Parameters as depicted in pluto paper is the standard values used here.
    // u_gas is set above or below the ~7e7 erg/cm^3 equilibrium value.
    const Real rho_cgs = pin->GetOrAddReal("thermal_equilibrium", "rho_cgs", 1.e-7);
    const Real Er_cgs = pin->GetOrAddReal("thermal_equilibrium", "Er_cgs", 1.e12);
    const Real u_gas_cgs = pin->GetOrAddReal("thermal_equilibrium", "u_gas_cgs", 1.e10);

    const Real rho_set = rho_cgs / mass_density_scale;
    const Real u_set = u_gas_cgs / energy_density_scale;
    const Real Erad_set = Er_cgs / energy_density_scale;

    IndexDomain domain = IndexDomain::entire;
    IndexRange3 b = KDomain::GetRange(rc, domain);
    pmb->par_for("thermal_equilibrium_init", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
        {
            rho(k, j, i) = rho_set;
            u(k, j, i) = u_set;
            uvec(0, k, j, i) = 0.;
            uvec(1, k, j, i) = 0.;
            uvec(2, k, j, i) = 0.;

            u_rad(k, j, i) = Erad_set;
            uvec_rad(0, k, j, i) = 0.;
            uvec_rad(1, k, j, i) = 0.;
            uvec_rad(2, k, j, i) = 0.;
        });

    return TaskStatus::complete;
}
