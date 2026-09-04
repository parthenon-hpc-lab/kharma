/*
 *  File: grmhd_functions.hpp
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

// phoebus includes
#include "microphysics/eos_kharma/eos_kharma.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "phoebus_utils/variables.hpp"

#include "gr_coordinates.hpp"
#include "kharma_utils.hpp"
#include "types.hpp"



/**
 * This namespace is solely for calc_tensor.
 * GRMHD::calc_4vecs intelligently skips the bcon calculation if B field is not present
 */
namespace GRHD
{
/**
 * Get a row of the hydrodynamic stress-energy tensor with first index up, second index
 * down.
 */
KOKKOS_INLINE_FUNCTION void calc_tensor(const Real& rho, const Real& u, const Real& pgas,
    const FourVectors& D, const int dir, Real hd[GR_DIM])
{
    const Real eta = pgas + rho + u;
    DLOOP1 {
        hd[mu] = eta * D.ucon[dir] * D.ucov[mu] + pgas * (dir == mu);
    }
}

}

/**
 * Device-side GR(M)HD functions
 * Anything reasonably specific to doing GRHD/GRMHD, which will not change:
 * lorentz factor, stress-energy tensor, 4-vectors ucon/bcon
 *
 * These functions mostly have several overloads, related to local vs global variables.
 * Many also have a form for split variables rho, uvec, etc, and one for a full array of
 * primitive variables P. Where all 4 combinations are used, we get 4 overloads.
 *
 * Local full-primitives versions are templated, to accept Slices/Scratch/etc equivalently
 */
namespace GRMHD
{

/**
 * Find gamma-factor of the fluid w.r.t. normal observer
 */
KOKKOS_INLINE_FUNCTION Real lorentz_calc(const GRCoordinates& G, const GridVector uvec,
    const int& k, const int& j, const int& i, const Loci loc)
{

    const Real qsq =
        G.gcov(loc, j, i, 1, 1) * uvec(V1, k, j, i) * uvec(V1, k, j, i) +
        G.gcov(loc, j, i, 2, 2) * uvec(V2, k, j, i) * uvec(V2, k, j, i) +
        G.gcov(loc, j, i, 3, 3) * uvec(V3, k, j, i) * uvec(V3, k, j, i) +
        2. * (G.gcov(loc, j, i, 1, 2) * uvec(V1, k, j, i) * uvec(V2, k, j, i) +
                 G.gcov(loc, j, i, 1, 3) * uvec(V1, k, j, i) * uvec(V3, k, j, i) +
                 G.gcov(loc, j, i, 2, 3) * uvec(V2, k, j, i) * uvec(V3, k, j, i));

    return m::sqrt(1. + qsq);
}
KOKKOS_INLINE_FUNCTION Real lorentz_calc(const GRCoordinates& G, const Real uv[NVEC],
    const int& k, const int& j, const int& i, const Loci loc)
{
    const Real qsq = G.gcov(loc, j, i, 1, 1) * uv[V1] * uv[V1] +
                     G.gcov(loc, j, i, 2, 2) * uv[V2] * uv[V2] +
                     G.gcov(loc, j, i, 3, 3) * uv[V3] * uv[V3] +
                     2. * (G.gcov(loc, j, i, 1, 2) * uv[V1] * uv[V2] +
                              G.gcov(loc, j, i, 1, 3) * uv[V1] * uv[V3] +
                              G.gcov(loc, j, i, 2, 3) * uv[V2] * uv[V3]);

    return m::sqrt(1. + qsq);
}
// Versions for full primitives array
KOKKOS_INLINE_FUNCTION Real lorentz_calc(const GRCoordinates& G,
    const VariablePack<Real>& P, const VarMap& m, const int& k, const int& j,
    const int& i, const Loci& loc = Loci::center)
{
    const Real qsq =
        G.gcov(loc, j, i, 1, 1) * P(m.U1, k, j, i) * P(m.U1, k, j, i) +
        G.gcov(loc, j, i, 2, 2) * P(m.U2, k, j, i) * P(m.U2, k, j, i) +
        G.gcov(loc, j, i, 3, 3) * P(m.U3, k, j, i) * P(m.U3, k, j, i) +
        2. * (G.gcov(loc, j, i, 1, 2) * P(m.U1, k, j, i) * P(m.U2, k, j, i) +
                 G.gcov(loc, j, i, 1, 3) * P(m.U1, k, j, i) * P(m.U3, k, j, i) +
                 G.gcov(loc, j, i, 2, 3) * P(m.U2, k, j, i) * P(m.U3, k, j, i));

    return m::sqrt(1. + qsq);
}

/**
 * Get a row of the MHD stress-energy tensor with first index up, second index down.
 * A factor of m::sqrt(4 pi) is absorbed into the definition of b.
 * See Gammie & McKinney '04.
 *
 * Entirely local!
 */
KOKKOS_INLINE_FUNCTION void calc_tensor(const Real& rho, const Real& u, const Real& pgas,
    const FourVectors& D, const int dir, Real mhd[GR_DIM])
{
    const Real bsq = dot(D.bcon, D.bcov);
    const Real eta = pgas + rho + u + bsq;
    const Real ptot = pgas + 0.5 * bsq;

    DLOOP1 {
        mhd[mu] = eta * D.ucon[dir] * D.ucov[mu] + ptot * (dir == mu) -
                  D.bcon[dir] * D.bcov[mu];
    }
}

/**
 * Calculate the 4-velocities ucon, ucov, and 4-fields bcon, bcov from primitive versions
 *
 * First two versions are for local stack variables and split global variables,
 * respectively, as we sometimes want the 4-vectors without having assembled the full
 * primitives list or anything.
 *
 * The latter are the usual Local/Global versions for primitives arrays
 */
KOKKOS_INLINE_FUNCTION void calc_4vecs(const GRCoordinates& G, const Real uvec[NVEC],
    const Real B_P[NVEC], const int& k, const int& j, const int& i, const Loci loc,
    FourVectors& D)
{
    const Real gamma = lorentz_calc(G, uvec, k, j, i, loc);
    const Real alpha = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));

    D.ucon[0] = gamma / alpha;
    VLOOP
        D.ucon[v + 1] = uvec[v] - gamma * alpha * G.gcon(loc, j, i, 0, v + 1);

    G.lower(D.ucon, D.ucov, k, j, i, loc);

    // This fn is guaranteed to have B values
    D.bcon[0] = 0;
    VLOOP
        D.bcon[0] += B_P[v] * D.ucov[v + 1];
    VLOOP
        D.bcon[v + 1] = (B_P[v] + D.bcon[0] * D.ucon[v + 1]) / D.ucon[0];

    G.lower(D.bcon, D.bcov, k, j, i, loc);
}
KOKKOS_INLINE_FUNCTION void calc_4vecs(const GRCoordinates& G, const GridVector uvec,
    const GridVector B_P, const int& k, const int& j, const int& i, const Loci loc,
    FourVectors& D)
{
    const Real gamma = lorentz_calc(G, uvec, k, j, i, loc);
    const Real alpha = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));

    D.ucon[0] = gamma / alpha;
    VLOOP
        D.ucon[v + 1] = uvec(v, k, j, i) - gamma * alpha * G.gcon(loc, j, i, 0, v + 1);

    G.lower(D.ucon, D.ucov, k, j, i, loc);

    // This fn is guaranteed to have B values
    D.bcon[0] = 0;
    VLOOP
        D.bcon[0] += B_P(v, k, j, i) * D.ucov[v + 1];
    VLOOP
        D.bcon[v + 1] = (B_P(v, k, j, i) + D.bcon[0] * D.ucon[v + 1]) / D.ucon[0];

    G.lower(D.bcon, D.bcov, k, j, i, loc);
}
// Primitive/VarMap versions of calc_4vecs for kernels that use "packed" primitives
KOKKOS_INLINE_FUNCTION void calc_4vecs(const GRCoordinates& G,
    const VariablePack<Real>& P, const VarMap& m, const int& k, const int& j,
    const int& i, const Loci loc, FourVectors& D)
{
    const Real gamma = lorentz_calc(G, P, m, k, j, i, loc);
    const Real alpha = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));

    D.ucon[0] = gamma / alpha;
    VLOOP
        D.ucon[v + 1] =
            P(m.U1 + v, k, j, i) - gamma * alpha * G.gcon(loc, j, i, 0, v + 1);

    G.lower(D.ucon, D.ucov, k, j, i, loc);

    if (m.B1 >= 0) {
        D.bcon[0] = 0;
        VLOOP
            D.bcon[0] += P(m.B1 + v, k, j, i) * D.ucov[v + 1];
        VLOOP
            D.bcon[v + 1] =
                (P(m.B1 + v, k, j, i) + D.bcon[0] * D.ucon[v + 1]) / D.ucon[0];

        G.lower(D.bcon, D.bcov, k, j, i, loc);
    } else {
        DLOOP1
            D.bcon[mu] = D.bcov[mu] = 0.;
    }
}

/**
 * Just the velocity 4-vector, in the first two styles of calc_4vecs.  For various
 * corners.
 */
KOKKOS_INLINE_FUNCTION void calc_ucon(const GRCoordinates& G, const GridVector uvec,
    const int& k, const int& j, const int& i, const Loci loc, Real ucon[GR_DIM])
{
    const Real gamma = lorentz_calc(G, uvec, k, j, i, loc);
    const Real alpha = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));

    ucon[0] = gamma / alpha;
    VLOOP
        ucon[v + 1] = uvec(v, k, j, i) - gamma * alpha * G.gcon(loc, j, i, 0, v + 1);
}
KOKKOS_INLINE_FUNCTION void calc_ucon(const GRCoordinates& G, const Real uvec[NVEC],
    const int& k, const int& j, const int& i, const Loci loc, Real ucon[GR_DIM])
{
    const Real gamma = lorentz_calc(G, uvec, k, j, i, loc);
    const Real alpha = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));

    ucon[0] = gamma / alpha;
    VLOOP
        ucon[v + 1] = uvec[v] - gamma * alpha * G.gcon(loc, j, i, 0, v + 1);
}
template<typename Global>
KOKKOS_INLINE_FUNCTION void calc_ucon(const GRCoordinates& G, const Global& P,
    const VarMap& m, const int& k, const int& j, const int& i, const Loci loc,
    Real ucon[GR_DIM])
{
    const Real gamma = lorentz_calc(G, P, m, k, j, i, loc);
    const Real alpha = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));

    ucon[0] = gamma / alpha;
    VLOOP
        ucon[v + 1] = P(m.U1 + v, k, j, i) - gamma * alpha * G.gcon(loc, j, i, 0, v + 1);
}

/**
 * Global GRMHD-only "p_to_u" call: for areas where nonideal terms are *always* 0!
 */
template<typename Global>
KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates& G, const Global& P, const VarMap& m_p, const Microphysics::EOS::EOS& eos, const int& k, const int& j, const int& i,
    const Global& U, const VarMap& m_u, const Loci& loc = Loci::center)
{
    Real gdet = G.gdet(loc, j, i);
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, loc, Dtmp); // TODO switch GRHD/GRMHD
    // Particle number flux
    U(m_u.RHO, k, j, i) = P(m_p.RHO, k, j, i) * Dtmp.ucon[0] * gdet;

    Real pg = eos.PressureFromDensityInternalEnergy(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i) / P(m_p.RHO, k, j, i));
    if (m_u.B1 >= 0) {
        // MHD stress-energy tensor w/ first index up, second index down
        Real mhd[GR_DIM];
        GRMHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i),
            pg, Dtmp, 0, mhd);
        U(m_u.UU, k, j, i) = mhd[0] * gdet + U(m_u.RHO, k, j, i);
        U(m_u.U1, k, j, i) = mhd[1] * gdet;
        U(m_u.U2, k, j, i) = mhd[2] * gdet;
        U(m_u.U3, k, j, i) = mhd[3] * gdet;
    } else {
        // HD stress-energy tensor w/ first index up, second index down
        Real hd[GR_DIM];
        GRHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i),
            pg, Dtmp, 0, hd);
        U(m_u.UU, k, j, i) = hd[0] * gdet + U(m_u.RHO, k, j, i);
        U(m_u.U1, k, j, i) = hd[1] * gdet;
        U(m_u.U2, k, j, i) = hd[2] * gdet;
        U(m_u.U3, k, j, i) = hd[3] * gdet;
    }
}

/**
 * Special all-local "p_to_u" call for just MHD variables, used in fluid frame floors &
 * wind source.
 */
KOKKOS_INLINE_FUNCTION void p_to_u_mhd(const GRCoordinates& G, const Real& rho,
    const Real& u, const Real uvec[NVEC], const Real B_P[NVEC], const Microphysics::EOS::EOS& eos,
    const int& k, const int& j, const int& i, Real& rho_ut, Real T[GR_DIM],
    const Loci loc = Loci::center)
{
    Real gdet = G.gdet(loc, j, i);

    FourVectors Dtmp;
    calc_4vecs(G, uvec, B_P, k, j, i, loc, Dtmp);

    // Particle number flux
    rho_ut = rho * Dtmp.ucon[0] * gdet;

    // MHD stress-energy tensor w/ first index up, second index down
    Real mhd[GR_DIM];
    Real pg = eos.PressureFromDensityInternalEnergy(rho, u/rho);
    calc_tensor(rho, u, pg, Dtmp, 0, mhd);

    T[0] = mhd[0] * gdet + rho_ut;
    VLOOP
        T[1 + v] = mhd[1 + v] * gdet;
}

}
