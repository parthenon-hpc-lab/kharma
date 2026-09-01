/*
 *  File: domain.hpp
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

#include "boundaries.hpp"

namespace KDomain
{

/**
 * Functions for checking boundaries in 3D.
 */
KOKKOS_INLINE_FUNCTION bool outside(const int& k, const int& j, const int& i,
    const IndexRange& kb, const IndexRange& jb, const IndexRange& ib)
{
    return (i < ib.s) || (i > ib.e) || (j < jb.s) || (j > jb.e) || (k < kb.s) ||
           (k > kb.e);
}
KOKKOS_INLINE_FUNCTION bool outside(
    const int& k, const int& j, const int& i, const IndexRange3& b)
{
    return (i < b.is) || (i > b.ie) || (j < b.js) || (j > b.je) || (k < b.ks) ||
           (k > b.ke);
}
KOKKOS_INLINE_FUNCTION bool inside(const int& k, const int& j, const int& i,
    const IndexRange& kb, const IndexRange& jb, const IndexRange& ib)
{
    // This is faster in the case that the point is outside
    return !outside(k, j, i, kb, jb, ib);
}
KOKKOS_INLINE_FUNCTION bool inside(
    const int& k, const int& j, const int& i, const IndexRange3& b)
{
    return !outside(k, j, i, b);
}

// TODO(CEP) these really should be in Parthenon
// There's a templated way to do it I forget, but this would be easier
template<typename T>
inline const int& GetNDim(MeshBlockData<T>* rc)
{
    return rc->GetBlockPointer()->pmy_mesh->ndim;
}
template<typename T>
inline const int& GetNDim(std::shared_ptr<MeshBlockData<T>> rc)
{
    return rc->GetBlockPointer()->pmy_mesh->ndim;
}
template<typename T>
inline const int& GetNDim(MeshData<T>* md)
{
    return md->GetMeshPointer()->ndim;
}
template<typename T>
inline const int& GetNDim(std::shared_ptr<MeshData<T>> md)
{
    return md->GetMeshPointer()->ndim;
}

template<typename T>
inline const IndexShape& GetCellbounds(MeshBlockData<T>* rc, bool coarse = false)
{
    return (coarse) ? rc->GetBlockPointer()->c_cellbounds
                    : rc->GetBlockPointer()->cellbounds;
}
template<typename T>
inline const IndexShape& GetCellbounds(
    std::shared_ptr<MeshBlockData<T>> rc, bool coarse = false)
{
    return GetCellbounds(rc.get());
}
template<typename T>
inline const IndexShape& GetCellbounds(MeshData<T>* md, bool coarse = false)
{
    return (coarse) ? md->GetBlockData(0)->GetBlockPointer()->c_cellbounds
                    : md->GetBlockData(0)->GetBlockPointer()->cellbounds;
}
template<typename T>
inline const IndexShape& GetCellbounds(
    std::shared_ptr<MeshData<T>> md, bool coarse = false)
{
    return GetCellbounds(md.get());
}

/**
 * Get the actual indices corresponding to an IndexDomain, optionally with some halo.
 * Note both "halo" values are *added*, i.e. measured to the *right*.  That is, the
 * size of GetRange(rc, interior, -1, 1) is [N1+2, N2+2, N3+2].
 * This seemed more natural for people coming from for loops.
 */
template<typename T>
inline IndexRange3 GetRange(T data, IndexDomain domain, TopologicalElement el = CC,
    int left_halo = 0, int right_halo = 0, bool coarse = false)
{
    // TODO also offsets for e.g. PtoU_Send?
    // Get sizes
    const auto& cellbounds = GetCellbounds(data, coarse);
    const IndexRange ib = cellbounds.GetBoundsI(domain, el);
    const IndexRange jb = cellbounds.GetBoundsJ(domain, el);
    const IndexRange kb = cellbounds.GetBoundsK(domain, el);
    // Compute sizes with specified halo zones included in non-trivial dimensions
    // TODO support arbitrary trivial directions
    const int& ndim = GetNDim(data);
    const IndexRange il = IndexRange{ib.s + left_halo, ib.e + right_halo};
    const IndexRange jl =
        (ndim > 1) ? IndexRange{jb.s + left_halo, jb.e + right_halo} : jb;
    const IndexRange kl =
        (ndim > 2) ? IndexRange{kb.s + left_halo, kb.e + right_halo} : kb;
    // Bounds of entire domain, we never mean to go beyond these
    const IndexRange ibe = cellbounds.GetBoundsI(IndexDomain::entire, el);
    const IndexRange jbe = cellbounds.GetBoundsJ(IndexDomain::entire, el);
    const IndexRange kbe = cellbounds.GetBoundsK(IndexDomain::entire, el);
    return IndexRange3{m::max(il.s, ibe.s), m::min(il.e, ibe.e), m::max(jl.s, jbe.s),
        m::min(jl.e, jbe.e), m::max(kl.s, kbe.s), m::min(kl.e, kbe.e)};
}
template<typename T>
inline IndexRange3 GetRange(
    T data, IndexDomain domain, int left_halo, int right_halo, bool coarse = false)
{
    return GetRange(data, domain, CC, left_halo, right_halo, coarse);
}
template<typename T>
inline IndexRange3 GetRange(T data, IndexDomain domain, bool coarse)
{
    return GetRange(data, domain, CC, 0, 0, coarse);
}

/**
 * Special range to include domain faces when computing boundaries,
 * as KHARMA's physical boundary conditions set these
 */
template<typename T>
inline IndexRange3 GetBoundaryRange(
    T data, IndexDomain domain, TopologicalElement el = CC, bool coarse = false)
{
    using KBoundaries::BoundaryDirection;
    using KBoundaries::BoundaryIsInner;
    const int bdir = BoundaryDirection(domain);
    if (el == FaceOf(bdir) || (el == E1 && (bdir == X2DIR || bdir == X3DIR)) ||
        (el == E2 && (bdir == X1DIR || bdir == X3DIR)) ||
        (el == E3 && (bdir == X1DIR || bdir == X2DIR))) {
        const int binner = BoundaryIsInner(domain);
        return GetRange(data, domain, el, (binner) ? 0 : -1, (binner) ? 1 : 0, coarse);
    } else {
        return GetRange(data, domain, el, 0, 0, coarse);
    }
}

/**
 * Get zones which are inside the physical domain, i.e. set by computation or MPI halo
 * sync, not by problem boundary conditions.
 */
template<typename T>
inline IndexRange3 GetPhysicalRange(MeshBlockData<T>* rc)
{
    using KBoundaries::IsPhysicalBoundary;

    const auto& bounds = GetCellbounds(rc);
    const auto pmb = rc->GetBlockPointer();

    return IndexRange3{IsPhysicalBoundary(pmb, BoundaryFace::inner_x1)
                           ? bounds.is(IndexDomain::interior)
                           : bounds.is(IndexDomain::entire),
        IsPhysicalBoundary(pmb, BoundaryFace::outer_x1) ? bounds.ie(IndexDomain::interior)
                                                        : bounds.ie(IndexDomain::entire),
        IsPhysicalBoundary(pmb, BoundaryFace::inner_x2) ? bounds.js(IndexDomain::interior)
                                                        : bounds.js(IndexDomain::entire),
        IsPhysicalBoundary(pmb, BoundaryFace::outer_x2) ? bounds.je(IndexDomain::interior)
                                                        : bounds.je(IndexDomain::entire),
        IsPhysicalBoundary(pmb, BoundaryFace::inner_x3) ? bounds.ks(IndexDomain::interior)
                                                        : bounds.ks(IndexDomain::entire),
        IsPhysicalBoundary(pmb, BoundaryFace::outer_x3) ? bounds.ke(IndexDomain::interior)
                                                        : bounds.ke(IndexDomain::entire)};
}

template<typename T>
inline IndexSize3 GetBlockSize(T data, IndexDomain domain = IndexDomain::entire)
{
    // Get sizes
    const auto& cellbounds = GetCellbounds(data);
    const uint n1 = cellbounds.ncellsi(domain);
    const uint n2 = cellbounds.ncellsj(domain);
    const uint n3 = cellbounds.ncellsk(domain);
    return IndexSize3{n1, n2, n3};
}

}
