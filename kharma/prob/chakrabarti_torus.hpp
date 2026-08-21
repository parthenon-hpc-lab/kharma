// Chakrabarti torus initialization functions
#pragma once

#include "decs.hpp"
#include "types.hpp"

/**
 * Initialize a wide variety of different Chakrabarti torii.
 *
 * @param rin is the torus innermost radius, in r_g
 * @param rmax is the radius of maximum density of the F-M torus in r_g
 */
TaskStatus InitializeChakrabartiTorus(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin);

/**
 * Compute angular momentum density
 */
KOKKOS_INLINE_FUNCTION Real l_calc(
    const GReal a, const GReal r, const GReal sth, const GReal cc, const GReal nn)
{
    // Compute BL metric components
    Real SS = (r * r) + (a * a) * (1.0 - (sth * sth));

    Real gcov_tt = -1.0 + 2.0 * r / SS;
    Real gcov_tphi = -2.0 * a * r / SS * sth;
    Real gcov_phiphi =
        ((r * r) + (a * a) + 2.0 * (a * a) * r / SS * (sth * sth) * (sth * sth));

    // Perform Bisection
    Real l_min = 1.0;
    Real l_max = 100.0;
    Real l_val = 0.5 * (l_min + l_max);

    int max_iterations = 25;
    Real tol_rel = 1.0e-8;

    for (int n = 0; n < max_iterations; ++n) {
        Real error_rel = 0.5 * (l_max - l_min) / l_val;
        if (error_rel < tol_rel) {
            break;
        }
        Real residual = pow(l_val / cc, 2.0 / nn) +
                        (l_val * gcov_phiphi + (l_val * l_val) * gcov_tphi) /
                            (gcov_tphi + l_val * gcov_tt);
        if (residual < 0.0) {
            l_min = l_val;
            l_val = 0.5 * (l_min + l_max);
        } else if (residual > 0.0) {
            l_max = l_val;
            l_val = 0.5 * (l_min + l_max);
        } else if (residual == 0.0) {
            break;
        }
    }

    return l_val;
}

/**
 * Compute time component of the covariant four velocity in BL
 */
KOKKOS_INLINE_FUNCTION Real u_t_calc(
    const GReal a, const GReal r, const GReal sth, const GReal l)
{
    // Compute BL metric components
    Real a2 = a * a;
    Real SS = (r * r) + (a * a) * (1.0 - (sth * sth));

    Real gcov_tt = -(1.0 - (2.0 * r / SS));
    Real gcov_tphi = -2.0 * a * r / SS * sth * sth;
    Real gcov_phiphi = (SS + (1.0 + (2.0 * r / SS)) * a2 * sth * sth) * sth * sth;

    // Compute time component of covariant four velocity
    Real u_t =
        -m::sqrt(m::max(((gcov_tphi * gcov_tphi) - (gcov_tt * gcov_phiphi)) /
                            (gcov_phiphi + (2.0 * l * gcov_tphi) + (l * l * gcov_tt)),
            0.0));
    return u_t;
}

/**
 * Torus solution for ln h, see AthenaK problem generator
 */
KOKKOS_INLINE_FUNCTION Real lnh_calc(const GReal a, const GReal rin, const GReal r,
    const GReal sth, const GReal cc, const GReal nn)
{

    // The energy per baryon (hu_t) is constant in the disk (for isentropic fluid with
    // constant angular momentum density)
    Real l = l_calc(a, r, sth, cc, nn);
    Real ut = u_t_calc(a, r, sth, l);
    Real l_in = l_calc(a, rin, 1.0, cc, nn);
    Real ut_in = u_t_calc(a, rin, 1.0, l_in);
    Real h = ut_in / ut;

    if (nn == 1.0) {
        h *= m::pow(l_in / l, (cc * cc) / (cc * cc - 1.0));
    } else {
        Real pow_c = 2.0 / nn;
        Real pow_l = 2.0 - (2.0 / nn);
        Real pow_abs = nn / (2.0 - (2.0 * nn));

        h *= (m::pow(std::fabs(1.0 - m::pow(cc, pow_c) * m::pow(l, pow_l)), pow_abs) *
              m::pow(std::fabs(1.0 - m::pow(cc, pow_c) * m::pow(l_in, pow_l)),
                  -1.0 * pow_abs));
    }

    if (isfinite(h) && h >= 1.0) {
        return m::log(h);
    } else {
        return -1.0;
    }
}

/**
 * This function computes the c, n parmameters that set the
 * angular momentum density profile in the Chakrabarti torus
 */
KOKKOS_INLINE_FUNCTION void cn_calc(
    const GReal a, const GReal rin, const GReal rmax, GReal* cc, GReal* nn)
{
    Real a_sq = a * a;
    Real rin_sq = rin * rin;
    Real rin_sqrt = m::sqrt(rin);
    Real rmax_sq = rmax * rmax;
    Real rmax_sqrt = m::sqrt(rmax);

    Real l_in = ((rin_sq + a_sq - (2.0 * a * rin_sqrt)) / ((rin_sqrt * (rin - 2.0)) + a));
    Real l_max =
        ((rmax_sq + a_sq - (2.0 * a * rmax_sqrt)) / ((rmax_sqrt * (rmax - 2.0)) + a));

    Real lambda_in =
        m::sqrt((l_in * ((-2.0 * a * l_in) + (rin_sq * rin) + a_sq * (2.0 + rin))) /
                ((2.0 * a) + (l_in * (rin - 2.0))));
    Real lambda_max =
        m::sqrt((l_max * ((-2.0 * a * l_max) + (rmax_sq * rmax) + a_sq * (2.0 + rmax))) /
                ((2.0 * a) + (l_max * (rmax - 2.0))));

    *nn = m::log(l_max / l_in) / m::log(lambda_max / lambda_in);
    *cc = l_in * m::pow(lambda_in, -(*nn));
}
