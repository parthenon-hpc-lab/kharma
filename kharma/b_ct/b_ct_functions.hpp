/*
 *  File: b_ct_functions.hpp
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
#include "domain.hpp"
#include "grmhd_functions.hpp"
#include "matrix.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

// Device functions for B_CT package
namespace B_CT
{

template<typename Global>
KOKKOS_INLINE_FUNCTION Real face_div(const GRCoordinates& G, Global& v, const int& ndim,
    const int& k, const int& j, const int& i)
{
    Real du = (v(F1, 0, k, j, i + 1) - v(F1, 0, k, j, i)) / G.Dxc<1>(k, j, i);
    if (ndim > 1) du += (v(F2, 0, k, j + 1, i) - v(F2, 0, k, j, i)) / G.Dxc<2>(k, j, i);
    if (ndim > 2) du += (v(F3, 0, k + 1, j, i) - v(F3, 0, k, j, i)) / G.Dxc<3>(k, j, i);
    return du;
}

/**
 * Gradient on one face, denoted by DIR. Split vs Flux-CT version because generally
 * this will be needed in separate loops setting F1, F2, F3, with separate bounds
 * Note this is backward-difference: face N borders cells N-1 & N
 */
template<CoordinateDirection DIR>
KOKKOS_FORCEINLINE_FUNCTION double face_grad(const GRCoordinates& G,
    const VariablePack<Real>& P, const int& k, const int& j, const int& i)
{
    // Backward gradient to cell faces
    if constexpr (DIR == X1DIR) {
        return (P(0, k, j, i) - P(0, k, j, i - 1)) / G.Dxc<1>(k, j, i);
    } else if constexpr (DIR == X2DIR) {
        return (P(0, k, j, i) - P(0, k, j - 1, i)) / G.Dxc<2>(k, j, i);
    } else if constexpr (DIR == X3DIR) {
        return (P(0, k, j, i) - P(0, k - 1, j, i)) / G.Dxc<3>(k, j, i);
    }
}

template<TE el, int NDIM>
KOKKOS_INLINE_FUNCTION void edge_curl(const GRCoordinates& G, const GridVector& A,
    const VariablePack<Real>& B_U, const int& k, const int& j, const int& i)
{
    if constexpr (NDIM == 2) {
        if constexpr (el == TE::F1) {
            // A3,2 derivative
            B_U(F1, 0, k, j, i) = (A(V3, k, j + 1, i) - A(V3, k, j, i)) / G.Dxc<2>(j);
        } else if constexpr (el == TE::F2) {
            // A3,1 derivative;
            B_U(F2, 0, k, j, i) = -(A(V3, k, j, i + 1) - A(V3, k, j, i)) / G.Dxc<1>(i);
        } else if constexpr (el == TE::F3) {
            B_U(F3, 0, k, j, i) = 0.;
        }
    } else if constexpr (NDIM == 3) {
        // This version is only needed for tilted disks, i.e. where |A| != A_phi
        // TODO TODO test a tilted disk using this code
        if constexpr (el == TE::F1) {
            // A3,2 derivative
            const Real A3c2f = (A(V3, k, j + 1, i) + A(V3, k + 1, j + 1, i)) / 2;
            const Real A3c2b = (A(V3, k, j, i) + A(V3, k + 1, j, i)) / 2;
            // A2,3 derivative
            const Real A2c3f = (A(V2, k + 1, j, i) + A(V2, k + 1, j + 1, i)) / 2;
            const Real A2c3b = (A(V2, k, j, i) + A(V2, k, j + 1, i)) / 2;
            B_U(F1, 0, k, j, i) =
                (A3c2f - A3c2b) / G.Dxc<2>(j) - (A2c3f - A2c3b) / G.Dxc<3>(k);
        } else if constexpr (el == TE::F2) {
            // A1,3 derivative
            const Real A1c3f = (A(V1, k + 1, j, i) + A(V1, k + 1, j, i + 1)) / 2;
            const Real A1c3b = (A(V1, k, j, i) + A(V1, k, j, i + 1)) / 2;
            // A3,1 derivative
            const Real A3c1f = (A(V3, k, j, i + 1) + A(V3, k + 1, j, i + 1)) / 2;
            const Real A3c1b = (A(V3, k, j, i) + A(V3, k + 1, j, i)) / 2;
            B_U(F2, 0, k, j, i) =
                (A1c3f - A1c3b) / G.Dxc<3>(k) - (A3c1f - A3c1b) / G.Dxc<1>(i);
        } else if constexpr (el == TE::F3) {
            // A2,1 derivative
            const Real A2c1f = (A(V2, k, j, i + 1) + A(V2, k, j + 1, i + 1)) / 2;
            const Real A2c1b = (A(V2, k, j, i) + A(V2, k, j + 1, i)) / 2;
            // A1,2 derivative
            const Real A1c2f = (A(V1, k, j + 1, i) + A(V1, k, j + 1, i + 1)) / 2;
            const Real A1c2b = (A(V1, k, j, i) + A(V1, k, j, i + 1)) / 2;
            B_U(F3, 0, k, j, i) =
                (A2c1f - A2c1b) / G.Dxc<1>(i) - (A1c2f - A1c2b) / G.Dxc<2>(j);
        }
    }
}

// Only through formatting has the following been made even a little comprehensible.
// TODO move it into Parthenon!

template<int diff_face, int diff_side, int offset, int DIM>
KOKKOS_FORCEINLINE_FUNCTION Real F(const ParArrayND<Real, VariableState>& fine,
    const Coordinates_t& coords, int l, int m, int n, int fk, int fj, int fi)
{
    // Trivial directions
    if constexpr (diff_face + 1 > DIM) return 0.;
    // TODO compile-time error on misuse? (diff_face == diff_side etc)
    constexpr int df_is_k = 2 * (diff_face == V3 && DIM > 2);
    constexpr int df_is_j = 2 * (diff_face == V2 && DIM > 1);
    constexpr int df_is_i = 2 * (diff_face == V1 && DIM > 0);
    constexpr int ds_is_k = (diff_side == V3 && DIM > 2);
    constexpr int ds_is_j = (diff_side == V2 && DIM > 1);
    constexpr int ds_is_i = (diff_side == V1 && DIM > 0);
    constexpr int of_is_k = (offset == V3 && DIM > 2);
    constexpr int of_is_j = (offset == V2 && DIM > 1);
    constexpr int of_is_i = (offset == V1 && DIM > 0);
    return fine(diff_face, l, m, n, fk + df_is_k + ds_is_k + of_is_k,
               fj + df_is_j + ds_is_j + of_is_j, fi + df_is_i + ds_is_i + of_is_i) *
               coords.FaceArea<diff_face + 1>(fk + df_is_k + ds_is_k + of_is_k,
                   fj + df_is_j + ds_is_j + of_is_j, fi + df_is_i + ds_is_i + of_is_i) -
           fine(diff_face, l, m, n, fk + ds_is_k + of_is_k, fj + ds_is_j + of_is_j,
               fi + ds_is_i + of_is_i) *
               coords.FaceArea<diff_face + 1>(fk + ds_is_k + of_is_k,
                   fj + ds_is_j + of_is_j, fi + ds_is_i + of_is_i) -
           fine(diff_face, l, m, n, fk + df_is_k + of_is_k, fj + df_is_j + of_is_j,
               fi + df_is_i + of_is_i) *
               coords.FaceArea<diff_face + 1>(fk + df_is_k + of_is_k,
                   fj + df_is_j + of_is_j, fi + df_is_i + of_is_i) +
           fine(diff_face, l, m, n, fk + of_is_k, fj + of_is_j, fi + of_is_i) *
               coords.FaceArea<diff_face + 1>(fk + of_is_k, fj + of_is_j, fi + of_is_i);
}

struct ProlongateInternalOlivares
{
    static constexpr bool OperationRequired(
        TopologicalElement fel, TopologicalElement cel)
    {
        // We will always be filling some locations of fine element fel with others of the
        // same element. However, the chosen coarse element cel defines our *domain*
        return IsSubmanifold(fel, cel);
    }

    template<int DIM, TopologicalElement fel = TopologicalElement::CC,
        TopologicalElement cel = TopologicalElement::CC>
    KOKKOS_FORCEINLINE_FUNCTION static void Do(const int l, const int m, const int n,
        const int k, const int j, const int i, const IndexRange& ckb,
        const IndexRange& cjb, const IndexRange& cib, const IndexRange& kb,
        const IndexRange& jb, const IndexRange& ib, const Coordinates_t& coords,
        const Coordinates_t& coarse_coords, const ParArrayND<Real, VariableState>*,
        const ParArrayND<Real, VariableState>* pfine)
    {

        // Definitely exit on what we can't handle
        // This is never hit as currently compiled in KHARMA
        if constexpr (fel != TE::F1 && fel != TE::F2 && fel != TE::F3) return;

        // Handle permutations "naturally."
        // Olivares et al. is fond of listing x1 versions which permute,
        // this makes translating/checking those easier
        constexpr int me = static_cast<int>(fel) % 3;
        constexpr int next = (me + 1) % 3;
        constexpr int third = (me + 2) % 3;

        // Fine array, indices
        // Note the boundaries are *always the interior*
        auto& fine = *pfine;
        const int fi = (DIM > 0) ? (i - cib.s) * 2 + ib.s : ib.s;
        const int fj = (DIM > 1) ? (j - cjb.s) * 2 + jb.s : jb.s;
        const int fk = (DIM > 2) ? (k - ckb.s) * 2 + kb.s : kb.s;

        // Coefficients selecting a particular formula (see Olivares et al. 2019)
        // TODO options here. There are 3 presented:
        // 1. Zeros (Cunningham)
        // 2. differences of squares of zone dimesnions (Toth)
        // 3. heuristic based on flux difference of top vs bottom halves (Olivares)
        // constexpr Real a[3] = {0., 0., 0.};
        const Real a[3] = {(SQR(coords.Dxc<2>(fj)) - SQR(coords.Dxc<3>(fk))) /
                               (SQR(coords.Dxc<2>(fj)) + SQR(coords.Dxc<3>(fk))),
            (SQR(coords.Dxc<3>(fk)) - SQR(coords.Dxc<1>(fi))) /
                (SQR(coords.Dxc<3>(fk)) + SQR(coords.Dxc<1>(fi))),
            (SQR(coords.Dxc<1>(fi)) - SQR(coords.Dxc<2>(fj))) /
                (SQR(coords.Dxc<1>(fi)) + SQR(coords.Dxc<2>(fj)))};

        // Coefficients for each term evaluating the four sub-faces
        const Real coeff[4][4] = {{3 + a[next], 1 - a[next], 3 - a[third], 1 + a[third]},
            {3 + a[next], 1 - a[next], 1 + a[third], 3 - a[third]},
            {1 - a[next], 3 + a[next], 3 - a[third], 1 + a[third]},
            {1 - a[next], 3 + a[next], 1 + a[third], 3 - a[third]}};

        constexpr int diff_k = (me == V3 && DIM > 2), diff_j = (me == V2 && DIM > 1),
                      diff_i = (me == V1 && DIM > 0);

        // Iterate through the 4 sub-faces
        for (int elem = 0; elem < 4; elem++) {
            // Make sure we can offset in other directions before doing so, though
            const int off_i =
                (DIM > 0) ? (elem % 2) * (me == V2) + (elem / 2) * (me == V3) + (me == V1)
                          : 0;
            const int off_j =
                (DIM > 1) ? (elem % 2) * (me == V3) + (elem / 2) * (me == V1) + (me == V2)
                          : 0;
            const int off_k =
                (DIM > 2) ? (elem % 2) * (me == V1) + (elem / 2) * (me == V2) + (me == V3)
                          : 0;

            fine(me, l, m, n, fk + off_k, fj + off_j, fi + off_i) =
                (
                    // Average faces on either side of us in selected direction (diff), on
                    // each of the 4 sub-faces (off)
                    0.5 * (fine(me, l, m, n, fk + off_k - diff_k, fj + off_j - diff_j,
                               fi + off_i - diff_i) *
                                  coords.Volume<fel>(fk + off_k - diff_k,
                                      fj + off_j - diff_j, fi + off_i - diff_i) +
                              fine(me, l, m, n, fk + off_k + diff_k, fj + off_j + diff_j,
                                  fi + off_i + diff_i) *
                                  coords.Volume<fel>(fk + off_k + diff_k,
                                      fj + off_j + diff_j, fi + off_i + diff_i)) +
                    1. / 16 *
                        (coeff[elem][0] *
                                F<next, me, -1, DIM>(fine, coords, l, m, n, fk, fj, fi) +
                            coeff[elem][1] * F<next, me, third, DIM>(
                                                 fine, coords, l, m, n, fk, fj, fi) +
                            coeff[elem][2] *
                                F<third, me, -1, DIM>(fine, coords, l, m, n, fk, fj, fi) +
                            coeff[elem][3] * F<third, me, next, DIM>(
                                                 fine, coords, l, m, n, fk, fj, fi))) /
                coords.Volume<fel>(fk + off_k, fj + off_j, fi + off_i);
        }
    }
};

}
