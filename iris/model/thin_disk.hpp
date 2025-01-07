/*
 *  File: thin_disk.hpp
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
#include "chandra_table.hpp"
#include "model_utils.hpp"
#include "tetrads.hpp"
#include "units.hpp"

// KHARMA headers
#include "decs.hpp"
#include "types.hpp"

namespace ThinDisk {

KOKKOS_INLINE_FUNCTION double krolikc(const double &r, const double &a, const double &r_isco)
{
    double y = m::sqrt(r);
    double yms = m::sqrt(r_isco);
    double y1 = 2. * m::cos(1. / 3. * (m::acos(a) - M_PI));
    double y2 = 2. * m::cos(1. / 3. * (m::acos(a) + M_PI));
    double y3 = -2. * m::cos(1. / 3. * m::acos(a));
    double arg1 = 3. * a / (2. * y);
    double arg2 = 3. * m::pow(y1 - a, 2) / (y * y1 * (y1 - y2) * (y1 - y3));
    double arg3 = 3. * m::pow(y2 - a, 2) / (y * y2 * (y2 - y1) * (y2 - y3));
    double arg4 = 3. * m::pow(y3 - a, 2) / (y * y3 * (y3 - y1) * (y3 - y2));

    return 1. - yms / y - arg1 * m::log(y / yms) - arg2 * m::log((y - y1) / (yms - y1))
        - arg3 * m::log((y - y2) / (yms - y2)) - arg4 * m::log((y - y3) / (yms - y3));
}

// Only supports midplane!
KOKKOS_INLINE_FUNCTION void thindisk_vals(const CoordinateEmbedding &coords, const double &r, const double &T0, double &T, double &omega)
{
    constexpr double th = M_PI_2;
    const double a = coords.get_a();
    const double r_isco = coords.get_isco();

    const double b = 1. - 3. / r + 2. * a / m::pow(r, 3. / 2.);
    const double d = r * r - 2. * r + a * a;
    const double lc = (r_isco * r_isco - 2. * a * m::sqrt(r_isco) + a * a)
        / (m::pow(r_isco, 1.5) - 2. * m::sqrt(r_isco) + a);
    const double hc = (2. * r - a * lc) / d;

    const double ar = m::pow(r * r + a * a, 2.) - a * a * d * m::pow(m::sin(th), 2.);
    const double om = 2. * a * r / ar;

    // Start the disk at r_isco, the marginally stable orbit which N-K take as an inner boundary condition.
    // End it eventually.
    if (r > r_isco) {
        omega = m::max(1. / (m::pow(r, 3. / 2.) + a), om);
    } else {
        omega = m::max((lc + a * hc) / (r * r + 2. * r * (1. + hc)), om);
    }

    if (r > r_isco) { //&& r < Rout
        T = T0 * pow(krolikc(r, a, r_isco) / b / pow(r, 3), 1. / 4.);
    } else {
        T = T0 / 1e5;
    }
}

//// SUPPORT: Emissivities ////

// Blackbody function B_nu(theta_e)
// c.f. Bnu_inv, same function but different interface. TODO unify
KOKKOS_INLINE_FUNCTION double bnu(const double &nu, const double &T) {
    return 2 * HPL * nu*nu*nu / (CL * CL) / (exp(HPL * nu / (KBOL * T)) - 1);
}

/*
 * Set photon wavevector for each radial zone of thin disk (polarized emission)
 */
KOKKOS_INLINE_FUNCTION void fbbpolemis(const double &nu, const double &T, const double &cosne, double &SI, double &SQ)
{
    double f = 1.8;
    SI = m::pow(f, -4.) * bnu(nu, T * f);

    // assumes Chandrasekhar electron scattering from semi-infinite atmosphere
    double interpI, interpdel;
    interp_chandra(cosne, &interpI, &interpdel);
    SI = SI * interpI;
    SQ = SI * interpdel;

    // Return invariant intensity & polarization
    SI = SI / (nu*nu*nu);
    SQ = SQ / (nu*nu*nu);
}

//// SUPPORT: Tetrads ////

// This sure is a vector.
KOKKOS_INLINE_FUNCTION void calc_polvec(const CoordinateEmbedding &coords, const double X[GR_DIM], const double Kcon[GR_DIM], double fourf[GR_DIM])
{
    double fourf_bl[GR_DIM];
    fourf_bl[0] = 0;
    fourf_bl[1] = 0;
    fourf_bl[2] = 1;
    fourf_bl[3] = 0;

    // Then transform to KS and to native coords
    coords.bl_vec_to_native(X, fourf_bl, fourf);
    //vec_to_ks(X, fourf_ks, fourf);

    // Now normalize
    double gcov[GR_DIM][GR_DIM];
    coords.gcov_native(X, gcov);
    normalize(fourf, gcov);
}

//// PUBLIC INTERFACE ////
KOKKOS_INLINE_FUNCTION void get_model_fourv(const CoordinateEmbedding &coords, const double X[GR_DIM], const double Kcon[GR_DIM], const double &T0,
                                            double Ucon[GR_DIM], double Ucov[GR_DIM], double Bcon[GR_DIM], double Bcov[GR_DIM])
{
    double r = coords.X1_to_embed(X[1]);
    double T, omega;
    thindisk_vals(coords, r, T0, T, omega);
    double gcov[GR_DIM][GR_DIM];
    coords.gcov_native(X, gcov);

    // normal observer velocity for Ucon/Ucov
    Ucon[0] =
        sqrt(-1. / (gcov[0][0] + 2. * gcov[0][3] * omega
                    + gcov[3][3] * omega * omega));
    Ucon[1] = 0.;
    Ucon[2] = 0.;
    Ucon[3] = omega * Ucon[0];

    flip_index(Ucon, gcov, Ucov);

    // B is handled in native coordinates here
    calc_polvec(coords, X, Kcon, Bcon);

    // Flip B
    flip_index(Bcon, gcov, Bcov);
}

KOKKOS_INLINE_FUNCTION void get_model_stokes(const CoordinateEmbedding &coords, const double X[GR_DIM], const double Kcon[GR_DIM],
                      const double &T0, double &SI, double &SQ, double &SU, double &SV)
{
    double r = coords.X1_to_embed(X[1]);
    if (r > coords.get_horizon()) {
        double T, omega;
        thindisk_vals(coords, r, T0, T, omega);

        double Ucon[GR_DIM], Ucov[GR_DIM], Bcon[GR_DIM], Bcov[GR_DIM];
        get_model_fourv(coords, X, Kcon, T0, Ucon, Ucov, Bcon, Bcov);

        double B = m::sqrt(dot(Bcon, Bcov));
        double mu = m::abs(get_bk_mu(Kcon, Ucov, Bcov, B));
        double nu = get_fluid_nu(Kcon, Ucov);
        fbbpolemis(nu, T, mu, SI, SQ);

    } else {
      SI = 0;
      SQ = 0;
    }

    SU = 0;
    SV = 0;
}

}
