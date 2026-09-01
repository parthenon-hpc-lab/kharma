
#pragma once

#include "fits_common.hpp"
#include "maxwell_juettner_fits.hpp"

#include "decs.hpp"

#include <stdio.h>
#include <stdlib.h>

#define STOKES_DIM 4

namespace Coefficients
{

// TODO template?
KOKKOS_INLINE_FUNCTION void jar_calc(FitParams params, double& jI, double& jQ, double& jU,
    double& jV, double& aI, double& aQ, double& aU, double& aV, double& rQ, double& rU,
    double& rV)
{
    // TODO check theta_e only for thermal
    if (params.electron_density > 0 && params.nu > 0 && params.theta_e > 0 &&
        params.magnetic_field > 0 && params.observer_angle != M_PI &&
        params.observer_angle != M_PI / 2 && params.observer_angle != 0) {
        double inv_nusq = 1 / (params.nu * params.nu);
        jI = j_nu_fit<Stokes::I, ElectronDist::maxwell_juettner>(params) * inv_nusq;
        jQ = -j_nu_fit<Stokes::Q, ElectronDist::maxwell_juettner>(params) * inv_nusq;
        jU = -0.;
        jV = j_nu_fit<Stokes::V, ElectronDist::maxwell_juettner>(params) * inv_nusq;

        // Get absorptivities via Kirchoff's law
        // Already invariant, guaranteed to respect aI > aP
        // Faster than calling Symphony code since we know jS, Bnu
        double Bnuinv = Bnu_inv(params.nu, params.theta_e); // Planck function
        if (Bnuinv > 0) {
            aI = jI / Bnuinv;
            aQ = jQ / Bnuinv;
            aU = jU / Bnuinv;
            aV = jV / Bnuinv;
        } else {
            aI = 0.;
            aQ = 0.;
            aU = 0.;
            aV = 0.;
        }

        rQ = rho_nu_fit<Stokes::Q, ElectronDist::maxwell_juettner>(params) * params.nu;
        rU = 0.;
        rV = rho_nu_fit<Stokes::V, ElectronDist::maxwell_juettner>(params) * params.nu;

        // TODO checks for >100% pol, allow rotation exactly along lines, etc.
    } else {
        jI = 0.;
        jQ = 0.;
        jU = 0.;
        jV = 0.;
        aI = 0.;
        aQ = 0.;
        aU = 0.;
        aV = 0.;
        rQ = 0.;
        rU = 0.;
        rV = 0.;
    }
    // REMEMBER aI,etc are `* nu` too!!  Also -Q/U just like above
}

}
