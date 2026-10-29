/*
 *  File: slices.hpp
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

// Thin per-zone slices of the component Kokkos Views in a Parthenon MeshBlockPack
// Components must be packed row-major: component index = row*nvars + col.
namespace Slices
{
template <typename pack_t>
struct PackMatrix
{
    pack_t pack;
    int b, k, j, i, n;
    KOKKOS_INLINE_FUNCTION Real& operator()(int r, int c)
    {
        return pack(b, r * n + c, k, j, i);
    }
    KOKKOS_INLINE_FUNCTION const Real& operator()(int r, int c) const
    {
        return pack(b, r * n + c, k, j, i);
    }
};
template <typename pack_t>
KOKKOS_FORCEINLINE_FUNCTION int GetNrows(const PackMatrix<pack_t>& m)
{
    return m.n;
}
template <typename pack_t>
KOKKOS_FORCEINLINE_FUNCTION int GetNcols(const PackMatrix<pack_t>& m)
{
    return m.n;
}

// Per-zone vector view: component c at zone (b,k,j,i) of the pack
template <typename pack_t>
struct PackVector
{
    pack_t pack;
    int b, k, j, i;
    KOKKOS_INLINE_FUNCTION Real& operator[](int c) const { return pack(b, c, k, j, i); }
};

// View evaluating to the *negated* pack components, so the solve's right-hand
// side b = -residual requires no copy or in-place negation
template <typename pack_t>
struct NegatedPackVector
{
    pack_t pack;
    int b, k, j, i;
    KOKKOS_INLINE_FUNCTION Real operator[](int c) const { return -pack(b, c, k, j, i); }
};

template <typename pack_t>
KOKKOS_INLINE_FUNCTION PackMatrix<pack_t> pack_matrix(
    pack_t pack, int b, int k, int j, int i, int n)
{
    return PackMatrix<pack_t>{pack, b, k, j, i, n};
}
template <typename pack_t>
KOKKOS_INLINE_FUNCTION PackVector<pack_t> pack_vector(pack_t pack, int b, int k, int j, int i)
{
    return PackVector<pack_t>{pack, b, k, j, i};
}
template <typename pack_t>
KOKKOS_INLINE_FUNCTION NegatedPackVector<pack_t> negated_pack_vector(
    pack_t pack, int b, int k, int j, int i)
{
    return NegatedPackVector<pack_t>{pack, b, k, j, i};
}
} // anonymous namespace