/*
 *  File: coordinate_utils.hpp
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
#include "matrix.hpp"

/**
 * Rotate a set of coordinates 'Xin' by 'angle' about the *y-axis*
 * (chosen so the slice at phi=0 in output will show the desired tilt)
 */
KOKKOS_INLINE_FUNCTION void rotate_polar(const GReal Xin[GR_DIM], const GReal angle,
    GReal Xout[GR_DIM], const bool spherical = true)
{
    // Make sure we don't break the trivial case
    if (m::abs(angle) < 1e-20) {
        DLOOP1
            Xout[mu] = Xin[mu];
        return;
    }

    // There are clever ways to do this, but this way is more flexible and understandable
    // Like everything else in this file, it is not necessarily very fast

    // Convert to cartesian
    GReal Xin_cart[GR_DIM] = {0};
    if (spherical) {
        Xin_cart[1] = Xin[1] * sin(Xin[2]) * cos(Xin[3]);
        Xin_cart[2] = Xin[1] * sin(Xin[2]) * sin(Xin[3]);
        Xin_cart[3] = Xin[1] * cos(Xin[2]);
    } else {
        DLOOP1
            Xin_cart[mu] = Xin[mu];
    }

    // Rotate about the y axis
    GReal R[GR_DIM][GR_DIM] = {0};
    R[0][0] = 1;
    R[1][1] = cos(angle);
    R[1][3] = sin(angle);
    R[2][2] = 1;
    R[3][1] = -sin(angle);
    R[3][3] = cos(angle);

    GReal Xout_cart[GR_DIM] = {0};
    DLOOP2
        Xout_cart[mu] += R[mu][nu] * Xin_cart[nu];

    // Convert back
    if (spherical) {
        Xout[0] = Xin[0];
        // This transformation preserves r, we keep the accurate version
        Xout[1] = Xin[1]; // m::sqrt(Xout_cart[1]*Xout_cart[1] + Xout_cart[2]*Xout_cart[2]
                          // + Xout_cart[3]*Xout_cart[3]);
        Xout[2] = acos(Xout_cart[3] / Xout[1]);
        if (m::isnan(Xout[2])) { // GCC has some trouble with ~acos(-1)
            if (Xout_cart[3] / Xout[1] < 0)
                Xout[2] = M_PI;
            else
                Xout[2] = 0.0;
        }
        Xout[3] = atan2(Xout_cart[2], Xout_cart[1]);
    } else {
        DLOOP1
            Xout[mu] = Xout_cart[mu];
    }
}

/**
 * Set the transformation matrix dXdx for converting vectors from spherical to Cartesian
 * coordinates, including rotation *and* normalization! There exists an analytic inverse,
 * of course, but we just take numerical inverses because they are easy
 */
KOKKOS_INLINE_FUNCTION void set_dXdx_sph2cart(
    const GReal X[GR_DIM], GReal dXdx[GR_DIM][GR_DIM])
{
    const GReal &r = X[1], &th = X[2], &phi = X[3];
    dXdx[0][0] = 1;
    dXdx[1][1] = sin(th) * cos(phi);
    dXdx[1][2] = r * cos(th) * cos(phi);
    dXdx[1][3] = -r * sin(th) * sin(phi);
    dXdx[2][1] = sin(th) * sin(phi);
    dXdx[2][2] = r * cos(th) * sin(phi);
    dXdx[2][3] = r * sin(th) * cos(phi);
    dXdx[3][1] = cos(th);
    dXdx[3][2] = -r * sin(th);
    dXdx[3][3] = 0;
}

/**
 * Same as rotate_polar but for vectors: rotate about the y-axis
 */
KOKKOS_INLINE_FUNCTION void rotate_polar_vec(const GReal Xin[GR_DIM],
    const GReal vin[GR_DIM], const GReal angle, const GReal Xout[GR_DIM],
    GReal vout[GR_DIM], const bool spherical = true)
{
    // Make sure we don't break the trivial case
    if (m::abs(angle) < 1e-20) {
        DLOOP1
            vout[mu] = vin[mu];
        return;
    }

    // Again, there are clever ways to do this by mapping to a spherical surface, etc
    // But this seems much more straightforward, and this is more flexible in letting us
    // define any rotation or translation we want in Cartesian coordinates.

    // Convert to Cartesian
    GReal vin_cart[GR_DIM] = {0};
    if (spherical) {
        // Note we use the *inverse* matrix here
        GReal dXdx[GR_DIM][GR_DIM] = {0};
        set_dXdx_sph2cart(Xin, dXdx);
        DLOOP2
            vin_cart[mu] += dXdx[mu][nu] * vin[nu];
    } else {
        DLOOP1
            vin_cart[mu] = vin[mu];
    }

    // Rotate about the y axis
    GReal R[GR_DIM][GR_DIM] = {0};
    R[0][0] = 1;
    R[1][1] = cos(angle);
    R[1][3] = sin(angle);
    R[2][2] = 1;
    R[3][1] = -sin(angle);
    R[3][3] = cos(angle);

    GReal vout_cart[GR_DIM] = {0};
    DLOOP2
        vout_cart[mu] += R[mu][nu] * vin_cart[nu];

    // Convert back
    if (spherical) {
        // We have to clear vout since it's passed in
        GReal dXdx[GR_DIM][GR_DIM] = {0}, dxdX[GR_DIM][GR_DIM] = {0};
        set_dXdx_sph2cart(Xout, dXdx);
        invert(&dXdx[0][0], &dxdX[0][0]);
        DLOOP1
            vout[mu] = 0;
        DLOOP2
            vout[mu] += dxdX[mu][nu] * vout_cart[nu];
    } else {
        DLOOP1
            vout[mu] = vout_cart[mu];
    }
}

/**
 * Set time component for a consistent 4-velocity given a 3-velocity
 */
KOKKOS_INLINE_FUNCTION void set_ut(const Real gcov[GR_DIM][GR_DIM], Real ucon[GR_DIM])
{
    Real AA, BB, CC;

    AA = gcov[0][0];
    BB = 2. * (gcov[0][1] * ucon[1] + gcov[0][2] * ucon[2] + gcov[0][3] * ucon[3]);
    CC = 1. + gcov[1][1] * ucon[1] * ucon[1] + gcov[2][2] * ucon[2] * ucon[2] +
         gcov[3][3] * ucon[3] * ucon[3] +
         2. * (gcov[1][2] * ucon[1] * ucon[2] + gcov[1][3] * ucon[1] * ucon[3] +
                  gcov[2][3] * ucon[2] * ucon[3]);

    Real discr = BB * BB - 4. * AA * CC;
    ucon[0] = (-BB - m::sqrt(discr)) / (2. * AA);
}

/**
 * Make primitive velocities u-twiddle out of 4-velocity.  See Gammie '04
 *
 * This function and set_ut together can turn any desired 3-velocity into a
 * form usable to initialize uvec in KHARMA; see bondi.hpp for usage.
 */
KOKKOS_INLINE_FUNCTION void fourvel_to_prim(
    const Real gcon[GR_DIM][GR_DIM], const Real ucon[GR_DIM], Real u_prim[NVEC])
{
    Real alpha2 = -1.0 / gcon[0][0];
    // Note gamma/alpha is ucon[0]
    u_prim[0] = ucon[1] + ucon[0] * alpha2 * gcon[0][1];
    u_prim[1] = ucon[2] + ucon[0] * alpha2 * gcon[0][2];
    u_prim[2] = ucon[3] + ucon[0] * alpha2 * gcon[0][3];
}
