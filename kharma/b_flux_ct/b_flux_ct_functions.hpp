/*
 *  File: b_flux_ct_functions.hpp
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

#include "b_double.hpp"

#include "decs.hpp"
#include "grmhd_functions.hpp"
#include "reductions.hpp"
#include "types.hpp"

#include <memory>

/**
 * Device functions for Flux-CT
 */
namespace B_FluxCT
{

/**
 * ND divergence, averaging to cell corners
 * TODO likely better templated, as with all ND stuff
 */
template<typename Global>
KOKKOS_FORCEINLINE_FUNCTION double corner_div(const GRCoordinates& G, const Global& B_U,
    const int& k, const int& j, const int& i, const bool& do_3D)
{
    const double norm = (do_3D) ? 0.25 : 0.5;
    // 2D divergence, averaging to corners
    double term1 = B_U(V1, k, j, i) - B_U(V1, k, j, i - 1) + B_U(V1, k, j - 1, i) -
                   B_U(V1, k, j - 1, i - 1);
    double term2 = B_U(V2, k, j, i) - B_U(V2, k, j - 1, i) + B_U(V2, k, j, i - 1) -
                   B_U(V2, k, j - 1, i - 1);
    double term3 = 0.;
    if (do_3D) {
        // Average to corners in 3D, add 3rd flux
        term1 += B_U(V1, k - 1, j, i) + B_U(V1, k - 1, j - 1, i) -
                 B_U(V1, k - 1, j, i - 1) - B_U(V1, k - 1, j - 1, i - 1);
        term2 += B_U(V2, k - 1, j, i) + B_U(V2, k - 1, j, i - 1) -
                 B_U(V2, k - 1, j - 1, i) - B_U(V2, k - 1, j - 1, i - 1);
        term3 = B_U(V3, k, j, i) + B_U(V3, k, j - 1, i) + B_U(V3, k, j, i - 1) +
                B_U(V3, k, j - 1, i - 1) - B_U(V3, k - 1, j, i) -
                B_U(V3, k - 1, j - 1, i) - B_U(V3, k - 1, j, i - 1) -
                B_U(V3, k - 1, j - 1, i - 1);
    }
    return norm * term1 / G.Dxc<1>(i) + norm * term2 / G.Dxc<2>(j) +
           norm * term3 / G.Dxc<3>(k);
}

template<typename EMF, typename Flux>
KOKKOS_FORCEINLINE_FUNCTION double ct_flux(const EMF& emf, const Flux& F, const int& dir,
    const int& v, const int& k, const int& j, const int& i, const int& ndim,
    const bool& use_emf)
{
    if (!use_emf || (ndim < 3 && v == V3)) return F.flux(dir, v, k, j, i);
    if (v == dir - 1) return 0.;
    if (dir == X1DIR) {
        return (v == V2)
                   ? 0.5 * (static_cast<double>(emf(V3, k, j, i)) + emf(V3, k, j + 1, i))
                   : -0.5 * (static_cast<double>(emf(V2, k, j, i)) + emf(V2, k + 1, j, i));
    } else if (dir == X2DIR) {
        return (v == V1)
                   ? -0.5 * (static_cast<double>(emf(V3, k, j, i)) + emf(V3, k, j, i + 1))
                   : 0.5 * (static_cast<double>(emf(V1, k, j, i)) + emf(V1, k + 1, j, i));
    } else {
        return (v == V1)
                   ? 0.5 * (static_cast<double>(emf(V2, k, j, i)) + emf(V2, k, j, i + 1))
                   : -0.5 * (static_cast<double>(emf(V1, k, j, i)) + emf(V1, k, j + 1, i));
    }
}

/**
 * 2D or 3D gradient, averaging to cell centers from corners.
 * Note this is forward-difference, while previous def is backward
 */
template<typename Global>
KOKKOS_FORCEINLINE_FUNCTION void center_grad(const GRCoordinates& G, const Global& P,
    const int& k, const int& j, const int& i, const bool& do_3D, double& B1, double& B2,
    double& B3)
{
    const double norm = (do_3D) ? 0.25 : 0.5;
    // 2D gradient, averaging to centers
    double term1 =
        P(0, k, j + 1, i + 1) + P(0, k, j, i + 1) - P(0, k, j + 1, i) - P(0, k, j, i);
    double term2 =
        P(0, k, j + 1, i + 1) + P(0, k, j + 1, i) - P(0, k, j, i + 1) - P(0, k, j, i);
    double term3 = 0.;
    if (do_3D) {
        // Average to centers in 3D, add 3rd flux
        term1 += P(0, k + 1, j + 1, i + 1) + P(0, k + 1, j, i + 1) -
                 P(0, k + 1, j + 1, i) - P(0, k + 1, j, i);
        term2 += P(0, k + 1, j + 1, i + 1) + P(0, k + 1, j + 1, i) -
                 P(0, k + 1, j, i + 1) - P(0, k + 1, j, i);
        term3 = P(0, k + 1, j + 1, i + 1) + P(0, k + 1, j, i + 1) +
                P(0, k + 1, j + 1, i) + P(0, k + 1, j, i) - P(0, k, j + 1, i + 1) -
                P(0, k, j, i + 1) - P(0, k, j + 1, i) - P(0, k, j, i);
    }
    B1 = norm * term1 / G.Dxc<1>(i);
    B2 = norm * term2 / G.Dxc<2>(j);
    B3 = norm * term3 / G.Dxc<3>(k);
}

template<typename APack, typename BPack>
KOKKOS_FORCEINLINE_FUNCTION void averaged_curl_3D(const GRCoordinates& G,
    const APack& A, const BPack& B_U, const int& k, const int& j, const int& i)
{
    // Take a flux-ct step from the corner potentials.
    // This needs to be 3D because post-tilt A may not point in the phi direction only

    // A3,2 derivative
    const double A3c2f = (static_cast<double>(A(V3, k, j + 1, i)) + A(V3, k, j + 1, i + 1) +
                           A(V3, k + 1, j + 1, i) + A(V3, k + 1, j + 1, i + 1)) /
                       4;
    const double A3c2b = (static_cast<double>(A(V3, k, j, i)) + A(V3, k, j, i + 1) + A(V3, k + 1, j, i) +
                           A(V3, k + 1, j, i + 1)) /
                       4;
    // A2,3 derivative
    const double A2c3f = (static_cast<double>(A(V2, k + 1, j, i)) + A(V2, k + 1, j, i + 1) +
                           A(V2, k + 1, j + 1, i) + A(V2, k + 1, j + 1, i + 1)) /
                       4;
    const double A2c3b = (static_cast<double>(A(V2, k, j, i)) + A(V2, k, j, i + 1) + A(V2, k, j + 1, i) +
                           A(V2, k, j + 1, i + 1)) /
                       4;
    BDouble::Store(B_U, (A3c2f - A3c2b) / G.Dxc<2>(j) - (A2c3f - A2c3b) / G.Dxc<3>(k), V1, k, j, i);

    // A1,3 derivative
    const double A1c3f = (static_cast<double>(A(V1, k + 1, j, i)) + A(V1, k + 1, j, i + 1) +
                           A(V1, k + 1, j + 1, i) + A(V1, k + 1, j + 1, i + 1)) /
                       4;
    const double A1c3b = (static_cast<double>(A(V1, k, j, i)) + A(V1, k, j, i + 1) + A(V1, k, j + 1, i) +
                           A(V1, k, j + 1, i + 1)) /
                       4;
    // A3,1 derivative
    const double A3c1f = (static_cast<double>(A(V3, k, j, i + 1)) + A(V3, k + 1, j, i + 1) +
                           A(V3, k, j + 1, i + 1) + A(V3, k + 1, j + 1, i + 1)) /
                       4;
    const double A3c1b = (static_cast<double>(A(V3, k, j, i)) + A(V3, k + 1, j, i) + A(V3, k, j + 1, i) +
                           A(V3, k + 1, j + 1, i)) /
                       4;
    BDouble::Store(B_U, (A1c3f - A1c3b) / G.Dxc<3>(k) - (A3c1f - A3c1b) / G.Dxc<1>(i), V2, k, j, i);

    // A2,1 derivative
    const double A2c1f = (static_cast<double>(A(V2, k, j, i + 1)) + A(V2, k, j + 1, i + 1) +
                           A(V2, k + 1, j, i + 1) + A(V2, k + 1, j + 1, i + 1)) /
                       4;
    const double A2c1b = (static_cast<double>(A(V2, k, j, i)) + A(V2, k, j + 1, i) + A(V2, k + 1, j, i) +
                           A(V2, k + 1, j + 1, i)) /
                       4;
    // A1,2 derivative
    const double A1c2f = (static_cast<double>(A(V1, k, j + 1, i)) + A(V1, k, j + 1, i + 1) +
                           A(V1, k + 1, j + 1, i) + A(V1, k + 1, j + 1, i + 1)) /
                       4;
    const double A1c2b = (static_cast<double>(A(V1, k, j, i)) + A(V1, k, j, i + 1) + A(V1, k + 1, j, i) +
                           A(V1, k + 1, j, i + 1)) /
                       4;
    BDouble::Store(B_U, (A2c1f - A2c1b) / G.Dxc<1>(i) - (A1c2f - A1c2b) / G.Dxc<2>(j), V3, k, j, i);
}

template<typename APack, typename BPack>
KOKKOS_FORCEINLINE_FUNCTION void averaged_curl_2D(const GRCoordinates& G,
    const APack& A, const BPack& B_U, const int& k, const int& j, const int& i)
{
    // A3,2 derivative
    const double A3c2f = (static_cast<double>(A(V3, k, j + 1, i)) + A(V3, k, j + 1, i + 1)) / 2;
    const double A3c2b = (static_cast<double>(A(V3, k, j, i)) + A(V3, k, j, i + 1)) / 2;
    BDouble::Store(B_U, (A3c2f - A3c2b) / G.Dxc<2>(j), V1, k, j, i);

    // A3,1 derivative
    const double A3c1f = (static_cast<double>(A(V3, k, j, i + 1)) + A(V3, k, j + 1, i + 1)) / 2;
    const double A3c1b = (static_cast<double>(A(V3, k, j, i)) + A(V3, k, j + 1, i)) / 2;
    BDouble::Store(B_U, -(A3c1f - A3c1b) / G.Dxc<1>(i), V2, k, j, i);

    BDouble::Store(B_U, 0, V3, k, j, i);
}

}
