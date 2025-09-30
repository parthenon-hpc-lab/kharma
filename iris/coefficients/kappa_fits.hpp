
#include "decs.hpp"

// #include <stdio.h>
// #include <stdlib.h>

// TODO ADD THESE INTERPOLATIONS TO BELOW
  // else if(params->distribution == params->KAPPA_DIST)
  // {
  //   if (params->polarization == params->STOKES_I) {
  //     if (params->kappa < params->kappa_interp_begin) {
  //       return kappa_I(params);
  //     } else if (params->kappa >= params->kappa_interp_begin && params->kappa < params->kappa_interp_end) {
  //       return ((params->kappa_interp_end - params->kappa) * maxwell_juettner_I(params) +
  //               (params->kappa - params->kappa_interp_begin) * kappa_I(params)) /
  //               (params->kappa_interp_end - params->kappa_interp_begin);
  //     } else {
  //       return maxwell_juettner_I(params);
  //     }
  //   } else if (params->polarization == params->STOKES_Q) {
  //     if (params->kappa < params->kappa_interp_begin) {
  //       return kappa_Q(params);
  //     } else if (params->kappa >= params->kappa_interp_begin && params->kappa < params->kappa_interp_end) {
  //       return ((params->kappa_interp_end - params->kappa) * maxwell_juettner_Q(params) +
  //               (params->kappa - params->kappa_interp_begin) * kappa_Q(params)) /
  //               (params->kappa_interp_end - params->kappa_interp_begin);
  //     } else {
  //       return maxwell_juettner_Q(params);
  //     }
  //   } else if (params->polarization == params->STOKES_U) {
  //     return 0.;
  //   } else if (params->polarization == params->STOKES_V) {
  //     if (params->kappa < params->kappa_interp_begin) {
  //       return kappa_V(params);
  //     } else if (params->kappa >= params->kappa_interp_begin && params->kappa < params->kappa_interp_end) {
  //       return ((params->kappa_interp_end - params->kappa) * maxwell_juettner_V(params) +
  //               (params->kappa - params->kappa_interp_begin) * kappa_V(params)) /
  //               (params->kappa_interp_end - params->kappa_interp_begin);
  //     } else {
  //       return maxwell_juettner_V(params);
  //     }
  //   }
  // }
  // else if(params->distribution == params->KAPPA_DIST)
  // {
  //   if     (params->polarization == params->STOKES_I) {
  //     if (params->kappa < params->kappa_interp_begin) {
  //       return kappa_I_abs(params);
  //     } else if (params->kappa >= params->kappa_interp_begin && params->kappa < params->kappa_interp_end) {
  //       return ((15 - params->kappa) * maxwell_juettner_I_abs(params) +
  //               (params->kappa - params->kappa_interp_begin) * kappa_I_abs(params)) /
  //               (params->kappa_interp_end - params->kappa_interp_begin);
  //     } else {
  //       return maxwell_juettner_I_abs(params);
  //     }
  //   } else if(params->polarization == params->STOKES_Q) {
  //     if (params->kappa < params->kappa_interp_begin) {
  //       return kappa_Q_abs(params);
  //     } else if (params->kappa >= params->kappa_interp_begin && params->kappa < params->kappa_interp_end) {
  //       return ((params->kappa_interp_end - params->kappa) * maxwell_juettner_Q_abs(params) +
  //               (params->kappa - params->kappa_interp_begin) * kappa_Q_abs(params)) /
  //               (params->kappa_interp_end - params->kappa_interp_begin);
  //     } else {
  //       return maxwell_juettner_Q_abs(params);
  //     }
  //   } else if(params->polarization == params->STOKES_U) {
  //     return 0.;
  //   } else if(params->polarization == params->STOKES_V) {
  //     if (params->kappa < params->kappa_interp_begin) {
  //       return kappa_V_abs(params);
  //     } else if (params->kappa >= params->kappa_interp_begin && params->kappa < params->kappa_interp_end) {
  //       return ((params->kappa_interp_end - params->kappa) * maxwell_juettner_V_abs(params) +
  //               (params->kappa - params->kappa_interp_begin) * kappa_V_abs(params)) /
  //               (params->kappa_interp_end - params->kappa_interp_begin);
  //     } else {
  //       return maxwell_juettner_V_abs(params);
  //     }
  //   }
  // }

  // if (params->polarization == params->STOKES_Q)
  // {
  //   if (params->kappa < 3.5)
  //     return kappa35_rho_Q(params);
  //   else if (params->kappa >= 3.5 && params->kappa < 4.0)
  //     return ((4.0 - params->kappa) * kappa35_rho_Q(params) + (params->kappa - 3.5) * kappa4_rho_Q(params)) / 0.5;
  //   else if (params->kappa >= 4.0 && params->kappa < 4.5)
  //     return ((4.5 - params->kappa) * kappa4_rho_Q(params) + (params->kappa - 4.0) * kappa45_rho_Q(params)) / 0.5;
  //   else if (params->kappa >= 4.5 && params->kappa < 5.0)
  //     return ((5.0 - params->kappa) * kappa45_rho_Q(params) + (params->kappa - 4.5) * kappa5_rho_Q(params)) / 0.5;
  //   else if (params->kappa >= 5.0 && params->kappa <= 8.0)
  //     return ((8.0 - params->kappa) * kappa5_rho_Q(params) + (params->kappa - 5.0) * maxwell_juettner_rho_Q(params)) / 5.0;
  //   else if (params->kappa > 8.0)
  //     return maxwell_juettner_rho_Q(params);
  // }
  // else if(params->polarization == params->STOKES_U) return 0.;
  // else if(params->polarization == params->STOKES_V)
  // {
  //   if(params->kappa < 3.5)
  //     return kappa35_rho_V(params);
  //   else if(params->kappa >= 3.5 && params->kappa < 4.0)
  //     return ((4.0 - params->kappa) * kappa35_rho_V(params) + (params->kappa - 3.5) * kappa4_rho_V(params)) / 0.5;
  //   else if(params->kappa >= 4.0 && params->kappa < 4.5)
  //     return ((4.5 - params->kappa) * kappa4_rho_V(params) + (params->kappa - 4.0) * kappa45_rho_V(params)) / 0.5;
  //   else if(params->kappa >= 4.5 && params->kappa <= 5.0)
  //     return ((5.0 - params->kappa) * kappa45_rho_V(params) + (params->kappa - 4.5) * kappa5_rho_V(params)) / 0.5;
  //   else if (params->kappa >= 5.0 && params->kappa <= 8.0)
  //     return ((8.0 - params->kappa) * kappa5_rho_V(params) + (params->kappa - 5.0) * maxwell_juettner_rho_V(params)) / 5.0;
  //   else if (params->kappa > 8.0)
  //     return maxwell_juettner_rho_V(params);
  // }


/**
 * Stabilized version of the hypergeometric fn:
 * Use GSL's version on small domain of validity,
 * use an extension to maintain stability outside that
 *
 * Final form courtesy of Angelo Ricarte
 */
static inline double stable_hyp2f1(double a, double b, double c, double z)
{
  if (z > -1 && z < 1) {
    //fprintf(stderr, "GSL hyperg_u: %g %g %g %g\n", a, b, c, z);
    return gsl_sf_hyperg_2F1(a, b, c, z);
  } else {
    //fprintf(stderr, "GSL hyperg_p1: %g %g %g %g %g\n", a, c-b, a-b+1., 1./(1.-z), z);
    //fprintf(stderr, "GSL hyperg_p2: %g %g %g %g %g\n", b, c-a, b-a+1., 1./(1.-z), z);
    /*GSL 2F1 only works for |z| < 1; had to apply a hypergeometric function
    identity because in our case z = -kappa*w, so |z| > 1 */
    return pow(1.-z, -a) * tgamma(c) * tgamma(b-a)
                 / (tgamma(b)*tgamma(c-a))
                 * gsl_sf_hyperg_2F1(a, c-b, a-b+1., 1./(1.-z))
                 + pow(1.-z, -b) * tgamma(c) * tgamma(a-b)
                 / (tgamma(a) * tgamma(c-b))
                 * gsl_sf_hyperg_2F1(b, c-a, b-a+1., 1./(1.-z));
  }
}

/*kappa_I: fitting formula to the emissivity, in Stokes I, produced by a
 *         kappa distribution of electrons (without any exponential
 *         cutoff).  Uses eq. 29, 35, 36, 37, 38 of [1].
 *
 *@params: struct of parameters params
 *@returns: fitting formula to kappa emissivity polarized in Stokes I
 */
double kappa_I(FitParams * params)
{
  double nu_c = get_nu_c(*params);

  double nu_w = pow(params->kappa_width * params->kappa, 2.) *nu_c
                *sin(params->observer_angle);

  double X_k = params->nu/nu_w;
  if (isinf(X_k)) { return 0; }

  double prefactor = (params->electron_density 
                      * pow(EE, 2.) * nu_c 
                      * sin(params->observer_angle))
	             /CL;

  double Nlow = 4. * M_PI * tgamma(params->kappa-4./3.) 
                / (pow(3., 7./3.) * tgamma(params->kappa-2.));

  double Nhigh = (1./4.) * pow(3., (params->kappa-1.)/2.) 
                 * (params->kappa-2.) * (params->kappa-1.)
		             * tgamma(params->kappa/4.-1./3.) 
		 * tgamma(params->kappa/4.-1./3.) 
		             * tgamma(params->kappa/4.-1./3.) 
                 * tgamma(params->kappa/4.+4./3.) + SMALL_NUM;

  double x = 3. * pow(params->kappa, -3./2.);

  double ans = prefactor * Nlow * pow(X_k, 1./3.) 
               * pow(1.+pow(X_k, x * (3. * params->kappa-4.)/6.)
                     * pow(Nlow/Nhigh, x), -1./x);

#if DEBUG
  if (isnan(ans)) {
    fprintf(stderr, "NaN in kappa. X_k, prefactor, Nlow, Nhigh, x, ans: %g %g %g %g %g %g\n",
                    X_k, prefactor, Nlow, Nhigh, x, ans);
  }
#endif

  return ans;

}

/*kappa_Q: fitting formula to the emissivity, in Stokes Q, produced by a
 *         kappa distribution of electrons (without any exponential
 *         cutoff).  Uses eq. 29, 35, 36, 37, 38 of [1].
 *
 *@params: struct of parameters params
 *@returns: fitting formula to kappa emissivity polarized in Stokes Q
 */
double kappa_Q(FitParams * params)
{
  double nu_c = get_nu_c(*params);

  double nu_w = pow(params->kappa_width*params->kappa, 2.) * nu_c 
                * sin(params->observer_angle);

  double X_k = params->nu/nu_w;
  if (isinf(X_k)) { return 0; }

  double prefactor = (params->electron_density 
                      * pow(EE, 2.) 
                      * nu_c * sin(params->observer_angle))
	             /CL;

  double Nlow = -(1./2.) * 4. * M_PI * tgamma(params->kappa-4./3.)
                 /(pow(3., 7./3.) * tgamma(params->kappa-2.));

  double Nhigh = -(pow(4./5., 2)+params->kappa/50.) * (1./4.) 
                 * pow(3., (params->kappa-1.)/2.) * (params->kappa-2.) 
                 * (params->kappa-1.) * tgamma(params->kappa/4.-1./3.)
                 * tgamma(params->kappa/4.+4./3.) + SMALL_NUM;
 
  double x = (37./10.)*pow(params->kappa, -8./5.);

  double ans = prefactor * Nlow * pow(X_k, 1./3.)
              * pow(1. + pow(X_k, x * (3. * params->kappa-4.)/6.)
              * pow(Nlow/Nhigh, x), -1./x);

#if DEBUG
  if (isnan(ans)) {
    fprintf(stderr, "NaN in kappa. X_k, prefactor, Nlow, Nhigh, x, ans: %g %g %g %g %g %g\n",
                    X_k, prefactor, Nlow, Nhigh, x, ans);
  }
#endif

  return ans;

}

/*kappa_V: fitting formula to the emissivity, in Stokes V, produced by a
 *         kappa distribution of electrons (without any exponential
 *         cutoff).  Uses eq. 29, 35, 36, 37, 38 of [1].
 *
 *@params: struct of parameters params
 *@returns: fitting formula to kappa emissivity polarized in Stokes V
 */
double kappa_V(FitParams * params)
{
  double nu_c = get_nu_c(*params);

  double nu_w = pow(params->kappa_width * params->kappa, 2.) 
                * nu_c * sin(params->observer_angle);

  double X_k = params->nu/nu_w;
  if (isinf(X_k)) { return 0; }

  double prefactor = (params->electron_density 
                     * pow(EE, 2.)
                     * nu_c * sin(params->observer_angle))
                    /CL;

  double Nlow = -pow(3./4., 2.) 
                * pow(pow(sin(params->observer_angle), -12./5.)-1., 12./25.)
                * (pow(params->kappa, -66./125.) / params->kappa_width) 
                * pow(X_k, -7./20.) * 4. * M_PI 
                * tgamma(params->kappa-4./3.)
                / (pow(3., 7./3.)*tgamma(params->kappa-2.));

  double Nhigh = -pow(7./8., 2.) 
                 * pow(pow(sin(params->observer_angle), -5./2.)-1., 11./25.)
                 * (pow(params->kappa, -11./25.) / params->kappa_width) 
                 * pow(X_k, -1./2.) * (1./4.) * pow(3., (params->kappa-1.)/2.) 
                 * (params->kappa-2.) * (params->kappa-1.) 
                 * tgamma(params->kappa/4.-1./3.) 
                 * tgamma(params->kappa/4.+4./3.) + SMALL_NUM;

  double x = 3.*pow(params->kappa, -3./2.);

  double ans = (Nhigh < SMALL_NUM*SMALL_NUM) ? 0:
               prefactor * Nlow * pow(X_k, 1./3.)
               * pow(1.+pow(X_k, x * (3.*params->kappa-4.)/6.)
               * pow(Nlow/Nhigh, x), -1./x);

  /*The Stokes V absorption coefficient changes sign at observer_angle
    equals 90deg, but this formula does not.  This discrepancy is a 
    bug in this formula, and is patched by the term below.*/
  double sign_bug_patch = cos(params->observer_angle) /
                          fabs(cos(params->observer_angle));

  /*NOTE: Sign corrected; the sign in Leung et al. (2011)
    and Pandya et al. (2016) for Stokes V transfer coefficients
    does not follow the convention the papers describe (IEEE/IAU);
    the sign has been corrected here.*/

#if DEBUG
  if (isnan(ans) || isnan(sign_bug_patch)) {
    fprintf(stderr, "NaN in kappa. X_k, prefactor, Nlow, Nhigh, x, ans, sign: %g %g %g %g %g %g %g\n",
                    X_k, prefactor, Nlow, Nhigh, x, ans, sign_bug_patch);
  }
#endif

  return -ans * sign_bug_patch;

}

/*kappa_I_abs: fitting formula to the absorptivity, in Stokes I, from
 *             by a kappa distribution of electrons (without any 
 *             exponential cutoff).  Uses eq. 30, 39, 40, 41, 42 of [1].
 *
 *@params: struct of parameters params
 *@returns: fitting formula to kappa absorptivity polarized in Stokes I
 */
double kappa_I_abs(FitParams * params)
{
  double nu_c = get_nu_c(*params);

  double nu_w = pow(params->kappa_width * params->kappa, 2.) 
                * nu_c * sin(params->observer_angle);

  double X_k = params->nu/nu_w;
  if (isinf(X_k)) { return 0; }

  double prefactor = params->electron_density * EE 
                     / (params->magnetic_field * sin(params->observer_angle));

  double a = params->kappa - 1./3.;
 
  double b = params->kappa + 1.;

  double c = params->kappa + 2./3.;

  double z = -params->kappa*params->kappa_width;

  double hyp2f1 = stable_hyp2f1(a, b, c, z);
  if (fabs(hyp2f1) < 1e-200) { return 0; }

  double Nlow = pow(3., 1./6.) * (10./41.) * pow(2. * M_PI, 2.)
               / pow(params->kappa_width * params->kappa, 16./3.-params->kappa)
               * (params->kappa-2.) * (params->kappa-1.) * params->kappa
               / (3.*params->kappa-1.) * tgamma(5./3.) * hyp2f1;

  double Nhigh = 2. * pow(M_PI, 5./2.)/3. * (params->kappa-2.) 
                 * (params->kappa-1.) * params->kappa
                 / pow(params->kappa_width * params->kappa, 5.) 
                 * (2 * tgamma(2. + params->kappa/2.)
                 / (2.+params->kappa)-1.) 
                 * (pow(3./params->kappa, 19./4.) + 3./5.) + SMALL_NUM;;

  double x = pow(-7./4. + 8. * params->kappa/5., -43./50.);

  double ans = prefactor * Nlow * pow(X_k, -5./3.)
               * pow(1. + pow(X_k, x * (3. * params->kappa-1.)/6.)
               * pow(Nlow/Nhigh, x), -1./x);

#if DEBUG
  if (isnan(ans)) {
    fprintf(stderr, "NaN in kappa. X_k, prefactor: %g %g\na, b, c, z, hyp2f1: %g %g %g %g %g\nNlow, Nhigh, x, ans: %g %g %g %g\n",
                    X_k, prefactor, a, b, c, z, hyp2f1, Nlow, Nhigh, x, ans);
  }
#endif

  return ans;
}

/*kappa_Q_abs: fitting formula to the absorptivity, in Stokes Q, from
 *             by a kappa distribution of electrons (without any 
 *             exponential cutoff).  Uses eq. 30, 39, 40, 41, 42 of [1].
 *
 *@params: struct of parameters params
 *@returns: fitting formula to kappa absorptivity polarized in Stokes Q
 */
double kappa_Q_abs(FitParams * params)
{
  double nu_c = get_nu_c(*params);

  double nu_w = pow(params->kappa_width*params->kappa, 2.) 
                * nu_c * sin(params->observer_angle);

  double X_k = params->nu/nu_w;
  if (isinf(X_k)) { return 0; }

  double prefactor = params->electron_density * EE
                     / (params->magnetic_field * sin(params->observer_angle));

  double a = params->kappa - 1./3.;
 
  double b = params->kappa + 1.;

  double c = params->kappa + 2./3.;

  double z = -params->kappa * params->kappa_width;

  double hyp2f1 = stable_hyp2f1(a, b, c, z);
  if (fabs(hyp2f1) < 1e-200) { return 0; }

  double Nlow = -(25./48.) * pow(3., 1./6.) * (10./41.) 
                * pow(2. * M_PI, 2.)
                / pow(params->kappa_width*params->kappa, 16./3.-params->kappa)
                * (params->kappa-2.) * (params->kappa-1.)
                * params->kappa / (3. * params->kappa-1.) 
                * tgamma(5./3.) * hyp2f1;
  double Nhigh = -(pow(21., 2.) * pow(params->kappa, -144./25.) + 11./20.) 
                 * 2. * pow(M_PI, 5./2.)/3. * (params->kappa-2.) 
                 * (params->kappa-1.) * params->kappa 
                 / pow(params->kappa_width * params->kappa, 5.) 
                 * (2 * tgamma(2. + params->kappa/2.)
                 / (2. + params->kappa)-1.) + SMALL_NUM;

  double x = (7./5.) * pow(params->kappa, -23./20.);

  double ans = prefactor * Nlow * pow(X_k, -5./3.)
              * pow(1. + pow(X_k, x * (3. * params->kappa-1.) / 6.)
              * pow(Nlow/Nhigh, x), -1./x);

#if DEBUG
  if (isnan(ans)) {
    fprintf(stderr, "NaN in kappa. X_k, prefactor: %g %g\na, b, c, z, hyp2f1: %g %g %g %g %g\nNlow, Nhigh, x, ans: %g %g %g %g\n",
                    X_k, prefactor, a, b, c, z, hyp2f1, Nlow, Nhigh, x, ans);
  }
#endif

  return ans;
}

/*kappa_V_abs: fitting formula to the absorptivity, in Stokes V, from
 *             by a kappa distribution of electrons (without any 
 *             exponential cutoff).  Uses eq. 30, 39, 40, 41, 42 of [1].
 *
 *@params: struct of parameters params
 *@returns: fitting formula to kappa absorptivity polarized in Stokes V
 */
double kappa_V_abs(FitParams * params)
{
  double nu_c = get_nu_c(*params);

  double nu_w = pow(params->kappa_width * params->kappa, 2.) 
               * nu_c * sin(params->observer_angle);

  double X_k = params->nu/nu_w;
  if (isinf(X_k)) { return 0; }

  double prefactor = params->electron_density * EE 
                     / (params->magnetic_field * sin(params->observer_angle));

  double a = params->kappa - 1./3.;
 
  double b = params->kappa + 1.;

  double c = params->kappa + 2./3.;

  double z = -params->kappa * params->kappa_width;

  double hyp2f1 = stable_hyp2f1(a, b, c, z);
  if (fabs(hyp2f1) < 1e-200) { return 0; }

  double Nlow = -(77./(100. * params->kappa_width)) 
                * pow(pow(sin(params->observer_angle), -114./50.)
                -1., 223./500.) * pow(X_k, -7./20.) 
                * pow(params->kappa, -7./10) * pow(3., 1./6.) 
                * (10./41.) * pow(2. * M_PI, 2.)
                / pow(params->kappa_width*params->kappa, 16./3.-params->kappa)
                * (params->kappa-2.) * (params->kappa-1.)
                * params->kappa / (3. * params->kappa-1.)
                * tgamma(5./3.) * hyp2f1;

  double Nhigh = -(143./10. * pow(params->kappa_width, -116./125.))
                 * pow(pow(sin(params->observer_angle), -41./20.)-1., 1./2.)
                 * (13.*13. * pow(params->kappa, -8.) + 13./(2500.) 
                 * params->kappa - 263./5000. + 47.
                 / (200.*params->kappa)) * pow(X_k, -1./2.) 
                 * 2. * pow(M_PI, 5./2.) / 3.
                 * (params->kappa-2.) * (params->kappa-1.) * params->kappa 
                 / pow(params->kappa_width*params->kappa, 5.)
                 * (2 * tgamma(2. + params->kappa/2.)
                 / (2. + params->kappa) - 1.) + SMALL_NUM;

  double x = (61./50.)*pow(params->kappa, -142./125.)+7./1000.;

  double ans = prefactor * Nlow * pow(X_k, -5./3.)
               * pow(1.+pow(X_k, x * (3. * params->kappa-1.) / 6.)
               * pow(Nlow/Nhigh, x), -1./x);

  /*The Stokes V absorption coefficient changes sign at observer_angle
    equals 90deg, but this formula does not.  This discrepancy is a 
    bug in this formula, and is patched by the term below.*/
  double sign_bug_patch = cos(params->observer_angle) /
                          fabs(cos(params->observer_angle));

#if DEBUG
  if (isnan(ans) || isnan(sign_bug_patch)) {
    fprintf(stderr, "NaN in kappa. X_k, prefactor: %g %g\na, b, c, z, hyp2f1: %g %g %g %g %g\nNlow, Nhigh, x, ans: %g %g %g %g\n",
                    X_k, prefactor, a, b, c, z, hyp2f1, Nlow, Nhigh, x, ans);
  }
#endif

  /*NOTE: Sign corrected; the sign in Leung et al. (2011)
    and Pandya et al. (2016) for Stokes V transfer coefficients
    does not follow the convention the papers describe (IEEE/IAU);
    the sign has been corrected here.*/
  return -ans * sign_bug_patch;
}

double kappa35_rho_Q(FitParams * params)
{
  double nu_c = get_nu_c(*params);

  double nu_w = pow(params->kappa_width*params->kappa, 2.) * nu_c 
                * sin(params->observer_angle);

  double X_k = params->nu/nu_w;
  if (isinf(X_k)) { return 0; }

  double prefactor = -(params->electron_density 
                      * pow(EE, 2.) 
		      * pow(nu_c, 2.)
		      * pow(sin(params->observer_angle), 2.))
	             /(ME * CL * pow(params->nu, 3.));
	
  double w_term = (17. * params->kappa_width) 
	  - (3. * pow(params->kappa_width, .5)) 
	  + (7. * pow(params->kappa_width, .5) * exp(-5. * params->kappa_width));
	  
  double f_X = 1. - exp(-pow(X_k, .84) / 30.) 
	  - (sin(X_k / 10.) * exp(-3. * pow(X_k, .471) / 2.));

  double ans = prefactor * f_X * w_term;

#if DEBUG
 if (isnan(ans)) {
    fprintf(stderr, "\nNaN in kappa rot. X_k, prefactor, w_term, f_X, ans: %g %g %g %g %g\n",
                    X_k, prefactor, w_term, f_X, ans);
    fprintf(stderr, "kappa, kappa_width: %g %g\n", params->kappa, params->kappa_width);
 }
#endif

  return ans;
}

double kappa4_rho_Q(FitParams * params)
{
  double nu_c = get_nu_c(*params);

  double nu_w = pow(params->kappa_width*params->kappa, 2.) * nu_c 
                * sin(params->observer_angle);

  double X_k = params->nu/nu_w;
  if (isinf(X_k)) { return 0; }

  double prefactor = -(params->electron_density 
                      * pow(EE, 2.) 
		      * pow(nu_c, 2.)
		      * pow(sin(params->observer_angle), 2.))
	             /(ME * CL * pow(params->nu, 3.));
	
  double w_term = ((46./3.) * params->kappa_width) 
	  - ((5./3.) * pow(params->kappa_width, .5)) 
	  + ((17./3.) * pow(params->kappa_width, .5) * exp(-5. * params->kappa_width));
	  
  double f_X = 1. - exp(-pow(X_k, .84) / 18.) 
	  - (sin(X_k / 6.) * exp(-7. * pow(X_k, .5) / 4.));

  double ans = prefactor * f_X * w_term;

#if DEBUG
 if (isnan(ans)) {
    fprintf(stderr, "\nNaN in kappa rot. X_k, prefactor, w_term, f_X, ans: %g %g %g %g %g\n",
                    X_k, prefactor, w_term, f_X, ans);
    fprintf(stderr, "kappa, kappa_width: %g %g\n", params->kappa, params->kappa_width);
 }
#endif

  return ans;
}

double kappa45_rho_Q(FitParams * params)
{
  double nu_c = get_nu_c(*params);

  double nu_w = pow(params->kappa_width*params->kappa, 2.) * nu_c 
                * sin(params->observer_angle);

  double X_k = params->nu/nu_w;
  if (isinf(X_k)) { return 0; }

  double prefactor = -(params->electron_density 
                      * pow(EE, 2.) 
		      * pow(nu_c, 2.)
		      * pow(sin(params->observer_angle), 2.))
	             /(ME * CL * pow(params->nu, 3.));
	
  double w_term = (14. * params->kappa_width) 
	  - ((13./8.) * pow(params->kappa_width, .5)) 
	  + ((9./2.) * pow(params->kappa_width, .5) * exp(-5. * params->kappa_width));
	  
  double f_X = 1. - exp(-pow(X_k, .84) / 12.) 
	  - (sin(X_k / 4.) * exp(-2. * pow(X_k, .525)));

  double ans = prefactor * f_X * w_term;

#if DEBUG
 if (isnan(ans)) {
    fprintf(stderr, "\nNaN in kappa rot. X_k, prefactor, w_term, f_X, ans: %g %g %g %g %g\n",
                    X_k, prefactor, w_term, f_X, ans);
    fprintf(stderr, "kappa, kappa_width: %g %g\n", params->kappa, params->kappa_width);
 }
#endif

  return ans;
}

double kappa5_rho_Q(FitParams * params)
{
  double nu_c = get_nu_c(*params);

  double nu_w = pow(params->kappa_width*params->kappa, 2.) * nu_c 
                * sin(params->observer_angle);

  double X_k = params->nu/nu_w;
  if (isinf(X_k)) { return 0; }

  double prefactor = -(params->electron_density 
                      * pow(EE, 2.) 
		      * pow(nu_c, 2.)
		      * pow(sin(params->observer_angle), 2.))
	             /(ME * CL * pow(params->nu, 3.));
	
  double w_term = ((25./2.) * params->kappa_width)
	  - (pow(params->kappa_width, .5)) 
	  + (5. * pow(params->kappa_width, .5) * exp(-5. * params->kappa_width));
	  
  double f_X = 1. - exp(-pow(X_k, .84) / 8.) 
	  - (sin(3. * X_k / 8.) * exp(-9. * pow(X_k, .541) / 4.));

  double ans = prefactor * f_X * w_term;

#if DEBUG
 if (isnan(ans)) {
    fprintf(stderr, "\nNaN in kappa rot. X_k, prefactor, w_term, f_X, ans: %g %g %g %g %g\n",
                    X_k, prefactor, w_term, f_X, ans);
    fprintf(stderr, "kappa, kappa_width: %g %g\n", params->kappa, params->kappa_width);
 }
#endif

  return ans;
}

double kappa35_rho_V(FitParams * params)
{
  double nu_c = get_nu_c(*params);

  double nu_w = pow(params->kappa_width*params->kappa, 2.) * nu_c 
                * sin(params->observer_angle);

  double X_k = params->nu/nu_w;
  if (isinf(X_k)) { return 0; }

  double prefactor = 2. * (params->electron_density 
                      * pow(EE, 2.) 
		      * nu_c * cos(params->observer_angle))
	             /(ME * CL * pow(params->nu, 2.));
	
  double bessel_term = (gsl_sf_bessel_Kn(0, 1./params->kappa_width)) / (gsl_sf_bessel_Kn(2, 1./params->kappa_width) + SMALL_NUM);
	
  double w_term = (pow(params->kappa_width, 2.) + (2. * params->kappa_width) + 1.) 
		 / (((25./8.) * pow(params->kappa_width, 2.)) + (4. * params->kappa_width) + 1.);
	  
  double g_X = 1. - .17*log(1. + (.447 * pow(X_k, -.5)));

  double ans = prefactor * bessel_term * g_X * w_term;

#if DEBUG
 if (isnan(ans)) {
    fprintf(stderr, "\nNaN in kappa rot. X_k, prefactor, bessel_term, w_term, g_X, ans: %g %g %g %g %g %g\n",
                    X_k, prefactor, bessel_term, w_term, g_X, ans);
    fprintf(stderr, "kappa, kappa_width, num, denom: %g %g %g %g", params->kappa, params->kappa_width,
                    gsl_sf_bessel_Kn(0, 1./params->kappa_width), gsl_sf_bessel_Kn(2, 1./params->kappa_width) + SMALL_NUM);
 }
#endif

  return ans;
}
								       
double kappa4_rho_V(FitParams * params)
{
  double nu_c = get_nu_c(*params);

  double nu_w = pow(params->kappa_width*params->kappa, 2.) * nu_c 
                * sin(params->observer_angle);

  double X_k = params->nu/nu_w;
  if (isinf(X_k)) { return 0; }

  double prefactor = 2. * (params->electron_density 
                      * pow(EE, 2.) 
		      * nu_c * cos(params->observer_angle))
	             /(ME * CL * pow(params->nu, 2.));
	
  double bessel_term = (gsl_sf_bessel_Kn(0, 1./params->kappa_width)) / (gsl_sf_bessel_Kn(2, 1./params->kappa_width) + SMALL_NUM);
	
  double w_term = (pow(params->kappa_width, 2.) + (54. * params->kappa_width) + 50.) 
		 / (((30./11.) * pow(params->kappa_width, 2.)) + (134. * params->kappa_width) + 50.);
	  
  double g_X = 1. - .17*log(1. + (.391 * pow(X_k, -.5)));

  double ans = prefactor * bessel_term * g_X * w_term;

#if DEBUG
 if (isnan(ans)) {
    fprintf(stderr, "\nNaN in kappa rot. X_k, prefactor, bessel_term, w_term, g_X, ans: %g %g %g %g %g %g\n",
                    X_k, prefactor, bessel_term, w_term, g_X, ans);
    fprintf(stderr, "kappa, kappa_width, num, denom: %g %g %g %g", params->kappa, params->kappa_width,
                    gsl_sf_bessel_Kn(0, 1./params->kappa_width), gsl_sf_bessel_Kn(2, 1./params->kappa_width) + SMALL_NUM);
 }
#endif

  return ans;
}
								       
double kappa45_rho_V(FitParams * params)
{
  double nu_c = get_nu_c(*params);

  double nu_w = pow(params->kappa_width*params->kappa, 2.) * nu_c 
                * sin(params->observer_angle);

  double X_k = params->nu/nu_w;
  if (isinf(X_k)) { return 0; }

  double prefactor = 2. * (params->electron_density 
                      * pow(EE, 2.) 
		      * nu_c * cos(params->observer_angle))
	             /(ME * CL * pow(params->nu, 2.));
	
  double bessel_term = (gsl_sf_bessel_Kn(0, 1./params->kappa_width)) / (gsl_sf_bessel_Kn(2, 1./params->kappa_width) + SMALL_NUM);
	
  double w_term = (pow(params->kappa_width, 2.) + (43. * params->kappa_width) + 38.) 
		 / (((7./3.) * pow(params->kappa_width, 2.)) + ((185./2.) * params->kappa_width) + 38.);
	  
  double g_X = 1. - .17*log(1. + (.348 * pow(X_k, -.5)));

  double ans = prefactor * bessel_term * g_X * w_term;

#if DEBUG
 if (isnan(ans)) {
    fprintf(stderr, "\nNaN in kappa rot. X_k, prefactor, bessel_term, w_term, g_X, ans: %g %g %g %g %g %g\n",
                    X_k, prefactor, bessel_term, w_term, g_X, ans);
    fprintf(stderr, "kappa, kappa_width, num, denom: %g %g %g %g", params->kappa, params->kappa_width,
                    gsl_sf_bessel_Kn(0, 1./params->kappa_width), gsl_sf_bessel_Kn(2, 1./params->kappa_width) + SMALL_NUM);
 }
#endif

  return ans;
}
								       
double kappa5_rho_V(FitParams * params)
{
  double nu_c = get_nu_c(*params);

  double nu_w = pow(params->kappa_width*params->kappa, 2.) * nu_c 
                * sin(params->observer_angle);

  double X_k = params->nu/nu_w;
  if (isinf(X_k)) { return 0; }

  double prefactor = 2. * (params->electron_density 
                      * pow(EE, 2.) 
		      * nu_c * cos(params->observer_angle))
	             /(ME * CL * pow(params->nu, 2.));
	
  double bessel_term = (gsl_sf_bessel_Kn(0, 1./params->kappa_width)) / (gsl_sf_bessel_Kn(2, 1./params->kappa_width) + SMALL_NUM);
	
  double w_term = ((params->kappa_width) + (13./14.)) 
		 / ((2. * params->kappa_width) + (13./14.));
	  
  double g_X = 1. - .17*log(1. + (.313 * pow(X_k, -.5)));

  double ans = prefactor * bessel_term * g_X * w_term;

#if DEBUG
 if (isnan(ans)) {
    fprintf(stderr, "\nNaN in kappa rot. X_k, prefactor, bessel_term, w_term, g_X, ans: %g %g %g %g %g %g\n",
                    X_k, prefactor, bessel_term, w_term, g_X, ans);
    fprintf(stderr, "kappa, kappa_width, num, denom: %g %g %g %g", params->kappa, params->kappa_width,
                    gsl_sf_bessel_Kn(0, 1./params->kappa_width), gsl_sf_bessel_Kn(2, 1./params->kappa_width) + SMALL_NUM);
 }
#endif

  return ans;
}								       
