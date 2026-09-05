/*
 *  File: seed_B.hpp
 *
 *  BSD 3-Clause License
 *
 *  Copyright (c) 2020, AFD Group at UIUC
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

#include "decs.hpp"
#include "types.hpp"

TaskStatus SeedBField(MeshData<Real>* md, ParameterInput* pin);

TaskStatus NormalizeBField(MeshData<Real>* md, ParameterInput* pin);

/*
 * B field initializations.
 * TO ADD A FIELD:
 * 1. add its internal name to the enum below
 * 2. Implement the template specialization for your field, either from seed_a<> or
 * seed_b<>
 * 3. Add your specialization to the `if` statements in SeedBField
 * 4. If you used seed_b<>, add your case where SeedBFieldType<> selects direct
 * initialization
 * 5. If you added arguments, make sure the calls in SeedBFieldType<> are up-to-date
 */

// Internal representation of the field initialization preference, used for templating
enum BSeedType {
    constant,
    monopole,
    orszag_tang,
    orszag_tang_a,
    wave,
    shock_tube,
    sane,
    mad,
    mad_quadrupole,
    mcaf,
    r3s3,
    r3s3_min_rho,
    r5s5,
    gaussian,
    bz_monopole,
    split_monopole,
    split_monopole_const,
    vertical,
    vertical_chakrabarti,
    r1s2
};

#define SEEDA_ARGS                                                                       \
    GReal *x, const GReal *dxc, double rho, double rin, double min_A, double A0,         \
        double arg1, double rb, bool in_torus, double rho_max, double potential_rho_pow, \
        double potential_falloff, double potential_r_pow

// This will also act as the default implementation for unspecified types,
// which should all be filled as B field by seed_b below.
// So, we want to set it to something dramatic.
template<BSeedType T>
KOKKOS_INLINE_FUNCTION Real seed_a(SEEDA_ARGS)
{
    return 0. / 0.;
}

// EHT comparison SANE
template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::sane>(SEEDA_ARGS)
{
    return m::max(rho - min_A, 0.);
}

// used in testing to exactly agree with harmpi
template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::bz_monopole>(SEEDA_ARGS)
{
    return 1. - m::cos(x[2]);
}

template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::split_monopole>(SEEDA_ARGS)
{
    return 1. - m::abs(m::cos(x[2]));
}

// BR's smoothed poloidal in-torus, EHT standard MAD
template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::mad>(SEEDA_ARGS)
{
    return m::max(
        m::pow(x[1] / rin, 3) * m::pow(sin(x[2]), 3) * m::exp(-x[1] / 400) * rho - min_A,
        0.);
}

// MAD, but turned into a quadrupole
template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::mad_quadrupole>(SEEDA_ARGS)
{
    return m::max(pow(x[1] / rin, 3) * m::pow(sin(x[2]), 3) * m::exp(-x[1] / 400) * rho -
                      min_A,
               0.) *
           m::cos(x[2]);
}

// Magnetically "choked" from McKinney+2012 (when paired with 10/100 torus)
// Just an r^2 sin^2 th term, but cut off subtly differently
// Also technically should be pegged to u, not rho
template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::mcaf>(SEEDA_ARGS)
{
    return m::max(m::pow(x[1] / rin, 2) * m::pow(m::sin(x[2]), 2) * (rho - min_A), 0.);
}

// Just the r^3 sin^3 th term
template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::r3s3>(SEEDA_ARGS)
{
    return m::max(m::pow(x[1] / rin, 3) * m::pow(m::sin(x[2]), 3) * rho - min_A, 0.);
}

// Just the r^3 sin^3 th term, but cut on rho > 0.2 rather than A_phi > 0.2
template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::r3s3_min_rho>(SEEDA_ARGS)
{
    return m::max(m::pow(x[1] / rin, 3) * m::pow(m::sin(x[2]), 3) * (rho - min_A), 0.);
}

// Bump power to r^5 sin^5 th term, quieter MAD
template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::r5s5>(SEEDA_ARGS)
{
    return m::max(m::pow(x[1] / rin, 5) * m::pow(m::sin(x[2]), 5) * rho - min_A, 0.);
}

// Pure vertical threaded field of gaussian strength with FWHM 2*rin (i.e. HM@rin)
// centered at BH center
template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::gaussian>(SEEDA_ARGS)
{
    const Real xf = (x[1] / rin) * m::sin(x[2]);
    const Real sigma = 2 / m::sqrt(2 * m::log(2));
    const Real u = xf / m::abs(sigma);
    return (1 / (m::sqrt(2 * M_PI) * m::abs(sigma))) * m::exp(-u * u / 2);
}

template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::vertical>(SEEDA_ARGS)
{
    // return A0 * x[1] * m::sin(x[2]) / 2.;
    return A0 * (x[1] * m::sin(x[2])) * (x[1] * m::sin(x[2])) / 2.;
}

template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::r1s2>(SEEDA_ARGS)
{
    return A0 * (x[1] * x[1] / 2. + x[1] * rb / 2.) * m::sin(x[2]) * m::sin(x[2]);
}

template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::orszag_tang_a>(SEEDA_ARGS)
{
    return A0 * (-0.5 * std::cos(2 * x[1] + arg1) + std::cos(x[2] + arg1));
}

template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::vertical_chakrabarti>(SEEDA_ARGS)
{
    Real r = x[1];
    Real th = x[2];
    Real sth = sin(x[2]);
    Real svarth = std::fabs(sin(th));

    Real cyl_radius = r * svarth;
    Real rcyl_in = rin;
    Real rcyl_falloff = potential_falloff;

    Real Aphi = m::pow(cyl_radius / rcyl_in, potential_r_pow);
    if (potential_falloff != 0) {
        Aphi *= m::exp(-cyl_radius / rcyl_falloff);
    }
    Real Aphi_offset = exp(-rcyl_in / rcyl_falloff);
    if (cyl_radius < rcyl_in) {
        Aphi = 0.0;
    } else {
        Aphi -= Aphi_offset;
    }
    if (potential_rho_pow != 0.0) {
        if (in_torus) {
            Aphi *= m::pow(rho / rho_max, potential_rho_pow);
        } else {
            Aphi = 0.0;
        }
    }

    return Aphi;
}

#undef SEEDA_ARGS
#define SEEDB_ARGS                                                                       \
    GReal *x, GReal gdet, double k1, double k2, double k3, double phase, double amp_B1,  \
        double amp_B2, double amp_B3, double amp2_B1, double amp2_B2, double amp2_B3,    \
        double &B1, double &B2, double &B3

template<BSeedType T>
KOKKOS_INLINE_FUNCTION void seed_b(SEEDB_ARGS)
{
    B1 = 0. / 0.;
    B2 = 0. / 0.;
    B3 = 0. / 0.;
}

// Constant field of B10, B20, B30 is always set
template<>
KOKKOS_INLINE_FUNCTION void seed_b<BSeedType::constant>(SEEDB_ARGS)
{}

// Reduce set constant radial component by the cube of radius
template<>
KOKKOS_INLINE_FUNCTION void seed_b<BSeedType::monopole>(SEEDB_ARGS)
{
    B1 /= (x[1] * x[1] * x[1]);
}
// template<>
// KOKKOS_INLINE_FUNCTION void seed_b<BSeedType::split_monopole>(SEEDB_ARGS)
// {
//     if (x[2] < phase) {
//         B1 /= (x[1]*x[1]*x[1]);
//     } else {
//         B1 /= -(x[1]*x[1]*x[1]);
//     }
// }
template<>
KOKKOS_INLINE_FUNCTION void seed_b<BSeedType::split_monopole_const>(SEEDB_ARGS)
{
    B1 = (x[2] < phase) ? B1 : -B1;
}

// For mhdmodes or linear waves tests
template<>
KOKKOS_INLINE_FUNCTION void seed_b<BSeedType::wave>(SEEDB_ARGS)
{
    const Real smode = m::sin(k1 * x[1] + k2 * x[2] + k3 * x[3] + phase);
    const Real cmode = m::cos(k1 * x[1] + k2 * x[2] + k3 * x[3] + phase);
    B1 += amp_B1 * cmode + amp2_B1 * smode;
    B2 += amp_B2 * cmode + amp2_B2 * smode;
    B3 += amp_B3 * cmode + amp2_B3 * smode;
}

// Shock tube init
template<>
KOKKOS_INLINE_FUNCTION void seed_b<BSeedType::shock_tube>(SEEDB_ARGS)
{
    const bool lhs = x[1] < phase;
    B1 += (lhs) ? amp_B1 : amp2_B1;
    B2 += (lhs) ? amp_B2 : amp2_B2;
    B3 += (lhs) ? amp_B3 : amp2_B3;
}

// For Orszag-Tang vortex
template<>
KOKKOS_INLINE_FUNCTION void seed_b<BSeedType::orszag_tang>(SEEDB_ARGS)
{
    B1 -= amp_B1 * m::sin(x[2] + phase);
    B2 += amp_B2 * m::sin(2. * (x[1] + phase));
}

#undef SEEDB_ARGS
