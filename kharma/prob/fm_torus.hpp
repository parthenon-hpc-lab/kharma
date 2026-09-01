// Fishbone-Moncrief torus initialization functions
#pragma once

#include "decs.hpp"
#include "types.hpp"

/**
 * Initialize a wide variety of different fishbone-moncrief torii.
 *
 * @param rin is the torus innermost radius, in r_g
 * @param rmax is the radius of maximum density of the F-M torus in r_g
 */
TaskStatus InitializeFMTorus(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin);

/**
 * Torus solution for ln h, See Fishbone and Moncrief eqn. 3.6.
 */
KOKKOS_INLINE_FUNCTION Real lnh_calc(
    const GReal a, const Real l, const GReal rin, const GReal r, const GReal th)
{
    // TODO this isn't faster than splitting into two evaluations of a sub-function,
    // and it doesn't matter anyway.  Make it clearer
    Real sth = m::sin(th);
    Real cth = m::cos(th);

    Real r2 = r * r;
    Real a2 = a * a;
    // Metric
    Real DD = r2 - 2. * r + a2;
    Real AA = m::pow(r2 + a2, 2) - DD * a2 * sth * sth;
    Real SS = r2 + a2 * cth * cth;

    Real thin = M_PI / 2.;
    Real sthin = m::sin(thin);
    Real cthin = m::cos(thin);

    Real rin2 = m::pow(rin, 2);
    Real DDin = rin2 - 2. * rin + a2;
    Real AAin = m::pow(rin2 + a2, 2) - DDin * a2 * sthin * sthin;
    Real SSin = rin2 + a2 * cthin * cthin;

    if (r >= rin) {
        return 0.5 * m::log((1. + m::sqrt(1. + 4. * (l * l * SS * SS) * DD /
                                                   (AA * AA * sth * sth))) /
                            (SS * DD / AA)) -
               0.5 * m::sqrt(1. + 4. * (l * l * SS * SS) * DD / (AA * AA * sth * sth)) -
               2. * a * r * l / AA -
               (0.5 * m::log((1. + m::sqrt(1. + 4. * (l * l * SSin * SSin) * DDin /
                                                    (AAin * AAin * sthin * sthin))) /
                             (SSin * DDin / AAin)) -
                   0.5 * m::sqrt(1. + 4. * (l * l * SSin * SSin) * DDin /
                                          (AAin * AAin * sthin * sthin)) -
                   2. * a * rin * l / AAin);
    } else {
        return 1.;
    }
}

/**
 * This function calculates specific the angular momentum of the
 * Fishbone-Moncrief solution in the midplane, as a function of radius.
 * (see Fishbone & Moncrief eqn. 3.8)
 * It improves on (3.8) by requiring no sign changes for
 * co-rotating (a > 0) vs counter-rotating (a < 0) disks.
 */
KOKKOS_INLINE_FUNCTION Real lfish_calc(const GReal a, const GReal r)
{
    GReal sqtr = m::sqrt(r);
    return ((a * a - 2. * a * sqtr + r * r) *
               ((-2. * a * r * (a * a - 2. * a * sqtr + r * r)) /
                       m::sqrt(2. * a * sqtr + (-3. + r) * r) +
                   ((a + (-2. + r) * sqtr) * (r * r * r + a * a * (2. + r))) /
                       m::sqrt(1 + (2. * a) / m::pow(r, 1.5) - 3. / r))) /
           (r * r * r * m::sqrt(2. * a * sqtr + (-3. + r) * r) * (a * a + (-2. + r) * r));
}

/**
 * Torus solution for density at a given location.
 *
 * This function is *not* used for the actual initialization (where rho is calculated
 * alongside the other primitive variables).  Rather, it is for:
 * 1. Normalization, in which the max of this function over the domain is calculated.
 * 2. B field initialization, which requires density of the untilted disk for simplicity
 */
KOKKOS_INLINE_FUNCTION Real fm_torus_rho(const GReal a, const GReal rin, const GReal rmax,
    const Real gam, const Real kappa, const GReal r, const GReal th)
{
    Real l = lfish_calc(a, rmax);
    Real lnh = lnh_calc(a, l, rin, r, th);
    if (lnh >= 0. && r >= rin) {
        // Calculate rho
        Real hm1 = m::exp(lnh) - 1.;
        return m::pow(hm1 * (gam - 1.) / (kappa * gam), 1. / (gam - 1.));
    } else {
        return 0;
    }
}
