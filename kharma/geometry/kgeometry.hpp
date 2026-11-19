/*
 *  File: kgeometry.hpp
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

#include <parthenon/parthenon.hpp>

/**
 * Geometry stored as package fields: filled from CoordinateEmbedding (for now),
 * read through GeomReader (later). Replaces the caches in GRCoordinates.
 */
namespace KGeometry
{

/**
 * Declare the geometry fields and read options from <coordinates> and <kgeometry>
 */
std::shared_ptr<KHARMAPackage> Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages);

// Flat component indices into the kg.* fields.  The fields mirror the GRCoordinates
// caches, with the location folded into the component index:
//   kg.gcov, kg.gcon : (loc, mu, nu)   ->  gcov_direct(loc, j, i, mu, nu)
//   kg.gdet          : (loc)           ->  gdet_direct(loc, j, i)
//   kg.conn, kg.gdet_conn : (mu, nu, lam) -> conn_direct(j, i, mu, nu, lam)

KOKKOS_FORCEINLINE_FUNCTION int metric_idx(const Loci loc, const int mu, const int nu)
{
    return (static_cast<int>(loc) * GR_DIM + mu) * GR_DIM + nu;
}

KOKKOS_FORCEINLINE_FUNCTION int gdet_idx(const Loci loc) { return static_cast<int>(loc); }

KOKKOS_FORCEINLINE_FUNCTION int conn_idx(const int mu, const int nu, const int lam)
{
    return (mu * GR_DIM + nu) * GR_DIM + lam;
}

/**
 * Out adapter for GeomFill: writes the kg.* package fields of one block.
 * Fields have one k plane, so k is always 0.
 */
struct FieldOut {
    parthenon::VariablePack<Real> pack;
    int i_gcov, i_gcon, i_gdet, i_conn, i_gdet_conn;
    KOKKOS_FORCEINLINE_FUNCTION Real& gcov(const Loci loc, const int j, const int i,
        const int mu, const int nu) const
    { return pack(i_gcov + metric_idx(loc, mu, nu), 0, j, i); }
    KOKKOS_FORCEINLINE_FUNCTION Real& gcon(const Loci loc, const int j, const int i,
        const int mu, const int nu) const
    { return pack(i_gcon + metric_idx(loc, mu, nu), 0, j, i); }
    KOKKOS_FORCEINLINE_FUNCTION Real& gdet(const Loci loc, const int j, const int i) const
    { return pack(i_gdet + gdet_idx(loc), 0, j, i); }
    KOKKOS_FORCEINLINE_FUNCTION Real& conn(const int j, const int i, const int mu,
        const int nu, const int lam) const
    { return pack(i_conn + conn_idx(mu, nu, lam), 0, j, i); }
    KOKKOS_FORCEINLINE_FUNCTION Real& gdet_conn(const int j, const int i, const int mu,
        const int nu, const int lam) const
    { return pack(i_gdet_conn + conn_idx(mu, nu, lam), 0, j, i); }
};

/**
 * Fill one block's geometry fields.  Registered as Parthenon's InitMeshBlockUserData,
 * so it runs at initialization (before problem generation), restart, and for every
 * block after each remesh/load balance.
 */
void SetGeometryBlock(MeshBlock* pmb, ParameterInput* pin);

/**
 * Compare one block's fields against its GRCoordinates caches.  Throws on mismatch.
 */
void VerifyBlock(MeshBlock* pmb);

} //namespace KGeometry
