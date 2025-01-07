/*
 *  File: ipolarray.hpp
 *
 *  BSD 3-Clause License
 *
 *  Copyright (c) 2025, Iris contributors
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
#pragma once

// Iris headers
#include "coefficients.hpp"

// Define where to switch from 1-exp() to
// Taylor-expanded quantities in the optical
// depth
#define CUT_SMALL_OPTICAL_DEPTH 1e-5

// Smallest double that prevents NaNs on inverses
#define CUT_PREVENT_NAN 1e-80

/*
 * must be a stable, approximate solution to radiative transfer
 * that runs between points w/ initial intensity I, emissivity
 * ji, opacity ki, and ends with emissivity jf, opacity kf.
 *
 * return final intensity
 */
KOKKOS_INLINE_FUNCTION double approximate_solve(const double &Ii, const double &ji, const double &ki, const double &jf,
                        const double &kf, const double &dl, double &tau)
{
    double javg = (ji + jf) / 2.;
    double kavg = (ki + kf) / 2.;

    double dtau = dl * kavg;
    tau += dtau;

    double If;
    if (dtau < 1.e-3) {
        If = Ii + (javg - Ii * kavg) * dl * (1. - (dtau / 2.) * (1. - dtau / 3.));
    } else {
        double efac = exp(-dtau);
        If = Ii * efac + (javg / kavg) * (1. - efac);
    }

    return If;
}

KOKKOS_INLINE_FUNCTION void evolve_stokes(double &SI, double &SQ, double &SU, double &SV,
    const double &jI, const double &jQ, const double &jU, const double &jV,
    const double &aI, const double &aQ, const double &aU, const double &aV,
    const double &rQ, const double &rU, const double &rV,
    const double &dlam)
{
    double SI0 = SI, SQ0 = SQ, SU0 = SU, SV0 = SV;
    double SI1, SQ1, SU1, SV1, SI2, SQ2, SU2, SV2;

  // Apply the Faraday rotation solution for a half step
  double x = dlam * 0.5;
  double rho2 = rQ * rQ + rU * rU + rV * rV;
  if (rho2 > CUT_PREVENT_NAN) {
    double rho = sqrt(rho2);
    double rdS = rQ * SQ0 + rU * SU0 + rV * SV0;
    double c = cos(rho * x);
    double s = sin(rho * x);
    double sh = sin(0.5 * rho * x);
    SI1 = SI0;
    SQ1 = SQ0 * c + 2 * rQ * rdS / rho2 * sh * sh + (rU * SV0 - rV * SU0) / rho * s;
    SU1 = SU0 * c + 2 * rU * rdS / rho2 * sh * sh + (rV * SQ0 - rQ * SV0) / rho * s;
    SV1 = SV0 * c + 2 * rV * rdS / rho2 * sh * sh + (rQ * SU0 - rU * SQ0) / rho * s;
  } else {
    SI1 = SI0;
    SQ1 = SQ0 + (-rV * SU0 + rU * SV0) * x;
    SU1 = SU0 + ( rV * SQ0 - rQ * SV0) * x;
    SV1 = SV0 + (-rU * SQ0 + rQ * SU0) * x;
  }

  // Apply full absorption/emission step
  x = dlam;
  double aP2 = aQ * aQ + aU * aU + aV * aV;
  if (aP2 > CUT_PREVENT_NAN) { // If 1/(aP2) will not return NaN...
    double aP = sqrt(aP2);
    double tauP = aP*x;
    double tauI = aI*x;
    double ads0 = aQ * SQ1 + aU * SU1 + aV * SV1;
    double adj = aQ * jQ + aU * jU + aV * jV;
    // appearance of factor 1/(aI2 - aP2) suggests that polarized and unpolarized
    // transfer will differ by O(aP2/aI2)
    double efacm = exp(-tauI + tauP);
    double efacp = exp(-tauI - tauP);
    double efac  = exp(-tauI);

    /* The formal solution for polarized transfer with emission and absorption
    * but no faraday rotation has 3 distinct eigenvalues (the first is degenerate):
    * aI, aI+aP, and aI-aP, see Moscibrodzka & Gammie 2017 appendix A2.
    *
    *  The general solution contains terms like (1 - exp(-\lambda/\lambda_i)),
    *  which can suffer loss of precision for small lambda.
    */

    /* this is the piece that captures effect of absorption on initial stokes vector,
    * safe for all optical depths
    *
    * These expressions assume that aP < aI and all jS, aS > 0
    */
    SI2 = efacm*(SI1/2. - ads0/(2.*aP)) +
          efacp*(SI1/2. + ads0/(2.*aP));

    SQ2 = efacm*(-SI1*aQ*aP + ads0*aQ)/(2.*aP2) +
          efacp*(SI1*aQ*aP + ads0*aQ)/(2.*aP2) +
          efac*(SQ1 - ads0*aQ/(aP2));

    SU2 = efacm*(-SI1*aU*aP + ads0*aU)/(2.*aP2) +
          efacp*(SI1*aU*aP + ads0*aU)/(2.*aP2) +
          efac*(SU1 - ads0*aU/(aP2));

    SV2 = efacm*(-SI1*aV*aP + ads0*aV)/(2.*aP2) +
          efacp*(SI1*aV*aP + ads0*aV)/(2.*aP2) +
          efac*(SV1 - ads0*aV/(aP2));

    /* In the limit of large optical depth Kirchoff's law is satisfied
    * for thermal transfer coefficients, i.e. the solution reduces to
    * {I,Q,U,V} = {B_\nu, 0, 0, 0}, provided j_S = a_S B_\nu (S = IQUV)
    * Notice that there is a potential loss of precision since the Q,U,V
    * terms vanish for thermal eDFs.
    */

    // Taylor series expansion at small optical depth to avoid loss of precision
    double afacm = 1. - efacm;
    double afac  = 1. - efac;
    double afacp = 1. - efacp;
    // Try to be clever by nesting the conditions.  Of questionable use.
    if (tauI - tauP <= CUT_SMALL_OPTICAL_DEPTH) {
      double e = tauI - tauP;
      afacm = e*(1. - (e/2.)*(1. - e/3.));

      if (tauI <= CUT_SMALL_OPTICAL_DEPTH) {
        e = tauI;
        afac = e*(1. - (e/2.)*(1. - e/3.));

        if (tauI + tauP <= CUT_SMALL_OPTICAL_DEPTH) {
          e = tauI + tauP;
          afacp = e*(1. - (e/2.)*(1. - e/3.));
        }
      }
    }
    // piece proportional to afac
    SQ2 += afac*(jQ/aI - aQ*adj/(aI*aP2));
    SU2 += afac*(jU/aI - aU*adj/(aI*aP2));
    SV2 += afac*(jV/aI - aV*adj/(aI*aP2));
    // pieces proportional to afacm
    SI2 += afacm*(aP*jI - adj)/(2.*aP*(aI - aP));
    SQ2 += afacm*aQ*(-aP*jI + adj)/(2.*aP2*(aI - aP));
    SU2 += afacm*aU*(-aP*jI + adj)/(2.*aP2*(aI - aP));
    SV2 += afacm*aV*(-aP*jI + adj)/(2.*aP2*(aI - aP));
    // pieces proportional to afacp
    SI2 += afacp*(aP*jI + adj)/(2.*aP*(aI + aP));
    SQ2 += afacp*aQ*(aP*jI + adj)/(2.*aP2*(aI + aP));
    SU2 += afacp*aU*(aP*jI + adj)/(2.*aP2*(aI + aP));
    SV2 += afacp*aV*(aP*jI + adj)/(2.*aP2*(aI + aP));
  } else {
    // Since we can now rely on aP==0, this is less questionable
    double tau_fake = 0;
    SI2 = approximate_solve(SI1, jI, aI, jI, aI, x, tau_fake);
    SQ2 = approximate_solve(SQ1, jQ, aI, jQ, aI, x, tau_fake);
    SU2 = approximate_solve(SU1, jU, aI, jU, aI, x, tau_fake);
    SV2 = approximate_solve(SV1, jV, aI, jV, aI, x, tau_fake);
  }

  // Apply the second rotation half-step
  x = dlam * 0.5;
  rho2 = rQ * rQ + rU * rU + rV * rV;
  if (rho2 > CUT_PREVENT_NAN) {
    double rdS = rQ * SQ2 + rU * SU2 + rV * SV2;
    double rho = sqrt(rho2);
    double c = cos(rho * x);
    double s = sin(rho * x);
    double sh = sin(0.5 * rho * x);
    SI = SI2;
    SQ = SQ2 * c + 2 * rQ * rdS / rho2 * sh * sh + (rU * SV2 - rV * SU2) / rho * s;
    SU = SU2 * c + 2 * rU * rdS / rho2 * sh * sh + (rV * SQ2 - rQ * SV2) / rho * s;
    SV = SV2 * c + 2 * rV * rdS / rho2 * sh * sh + (rQ * SU2 - rU * SQ2) / rho * s;
  } else {
    SI = SI2;
    SQ = SQ2 + (-rV * SU2 + rU * SV2) * x;
    SU = SU2 + ( rV * SQ2 - rQ * SV2) * x;
    SV = SV2 + (-rU * SQ2 + rQ * SU2) * x;
  }

  if (false) { // TODO OPTION
    // Optionally correct the resulting Stokes parameters to guarantee:
    // 1. I > 0
    // 2. sqrt(Q^2 + U^2 + V^2) < I
    // This is ensured of the emissivities by default,
    // but can be additionally enforced here.
    if (SI < 0) {
      SI = 0;
      SQ = 0;
      SU = 0;
      SV = 0;
    } else {
      double pol_frac = sqrt(SQ*SQ + SU*SU + SV*SV) / SI;
      if ( pol_frac > 1. ) {
        SQ /= pol_frac;
        SU /= pol_frac;
        SV /= pol_frac;
      }
    }
  }
}