
#pragma once

// Iris headers
#include "units.hpp"

// KHARMA headers
#include "decs.hpp"

enum class Stokes { I, Q, U, V };
enum class ElectronDist { maxwell_juettner, power_law, kappa };
enum class FitSelect { pandya, dexter, leung };

class FitParams
{
  public:
    /*USER PARAMS:*/
    double nu;               /* GHz */
    double magnetic_field;   /* Gauss */
    double electron_density; /* 1/cc */
    double observer_angle;   /* rad */

    // Options for fits
    bool approximate_bessels; // Use approximate Bessel functions for speed
    FitSelect fit_select;
    ElectronDist electrons;

    // Thermal distribution parameters
    double theta_e;

    // Power law parameters
    double power_law_p;
    double gamma_min;
    double gamma_max;

    // Kappa distribution parameters
    double kappa;
    double kappa_width;
    double kappa_interp_begin;
    double kappa_interp_end;

    void print()
    {
        printf("Fluid params: nu=%g B=%g Ne=%g theta=%g\n", nu, magnetic_field,
            electron_density, observer_angle);
        printf("Fit params: approx_bessels=%d fitselect=%d electron_dist=%d\n",
            approximate_bessels, fit_select, electrons);
        printf("Thermal dist params: theta_e=%g\n", theta_e);
    }
};

/*get_nu_c: takes in values of electron_charge, magnetic_field, mass_electron,
 *          and speed_light, and returns the cyclotron frequency nu_c.
 *
 *@params:  FitParams params, contains parameters mentioned above
 *@returns: cyclotron frequency, nu_c, for the provided parameters
 */
KOKKOS_INLINE_FUNCTION double get_nu_c(const double& B)
{
    return (EE * B) / (2. * M_PI * ME * CL);
}

KOKKOS_INLINE_FUNCTION double get_omega_p(const double& ne)
{
    const double eps0 = 1. / (4. * M_PI);
    return sqrt(ne * EE * EE / (ME * eps0));
}

KOKKOS_INLINE_FUNCTION double Bnu_inv(const double& nu, const double& Thetae)
{
    double x = HPL * nu / (ME * CL * CL * Thetae);

    if (x < 2.e-3) /* Taylor expand */
        return ((2. * HPL / (CL * CL)) / (x / 24. * (24. + x * (12. + x * (4. + x)))));
    else
        return ((2. * HPL / (CL * CL)) / (exp(x) - 1.));
}

// Templates for the fitting formulae

/*j_nu_fit: wrapper for the emissivity fitting formulae.  Takes in the
 *          same parameters as j_nu(), and passes them to the fitting formulae.
 *
 *@params: nu, magnetic_field, electron_density, observer_angle,
 *         distribution, polarization, theta_e, power_law_p,
 *         gamma_min, gamma_max, gamma_cutoff, kappa,
 *         kappa_width
 *@returns: the corresponding fitting formula (based on the distribution
 *          function) evaluated for the input parameters.
 */
template<Stokes S, ElectronDist E>
KOKKOS_INLINE_FUNCTION double j_nu_fit(const FitParams& params);

/*alpha_nu_fit: wrapper for the absorptivity fitting formulae.  Takes in the
 *              same parameters as alpha_nu(), and passes them to the fitting formulae.
 *
 *@params: nu, magnetic_field, electron_density, observer_angle,
 *         distribution, polarization, theta_e, power_law_p,
 *         gamma_min, gamma_max, gamma_cutoff, kappa,
 *         kappa_width
 *@returns: the corresponding fitting formula (based on the distribution
 *          function) evaluated for the input parameters.
 */
template<Stokes S, ElectronDist E>
KOKKOS_INLINE_FUNCTION double alpha_nu_fit(const FitParams& params);

/*rho_nu_fit: Fits to Faraday rotation/conversion coefficients; right now only has
 *            formulae for Maxwell-Juettner distribution, from Dexter (2016),
 *            and the Kappa distribution, from Marszewski+ (2021)
 *
 *@params: nu, magnetic_field, electron_density, observer_angle,
 *         distribution, polarization, theta_e, power_law_p,
 *         gamma_min, gamma_max, gamma_cutoff, kappa,
 *         kappa_width
 *@returns: the corresponding Faraday coefficient fitting formula
 *          (based on the distribution function) evaluated for
 *          the input parameters.
 */
template<Stokes S, ElectronDist E>
KOKKOS_INLINE_FUNCTION double rho_nu_fit(const FitParams& params);
