/* 
 *  File: reductions_variables.hpp
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

#include "emhd.hpp"
#include "flux_functions.hpp"

// Arguments to computing any variable defined below
#define REDUCE_FUNCTION_ARGS const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p, \
                        const VariableFluxPack<Real>& U, const VarMap& m_u, \
                        const VariablePack<Real>& cmax, const VariablePack<Real>& cmin,\
                        const EMHD::EMHD_parameters& emhd_params, const Real& gam, const int& k, const int& j, const int& i
// Call for passing a particular block's values
#define REDUCE_FUNCTION_CALL G, P(b), m_p, U(b), m_u, cmax(b), cmin(b), emhd_params, gam, k, j, i

using namespace parthenon;

namespace Reductions {

// Add any new reduction variables to this list, then implementations below
// Not elegant, but fast & portable.
// HIPCC doesn't like passing function pointers as we used to do,
// and it doesn't vectorize anyway. Look forward to more of this pattern in the code
enum class Var{phi, bsq, gas_pressure, beta, rhou0, mix_T00, mix_T01, mix_T02, mix_T03,
               mdot, edot, ldot, mdot_flux, edot_flux, ldot_flux, eht_lum, jet_lum,
               nan_ctop, zero_ctop, neg_rho, neg_u, neg_rhout};

// Function template for all reductions.
template<Var T>
KOKKOS_INLINE_FUNCTION Real reduction_var(REDUCE_FUNCTION_ARGS);

// Can also sum the hemispheres independently to be fancy (TODO?)
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::phi>(REDUCE_FUNCTION_ARGS)
{
    // \Phi == \int |*F^1^0| * gdet * dx2 * dx3 == \int |B1| * gdet * dx2 * dx3
    return 0.5 * m::abs(U(m_u.B1, k, j, i)); // factor of gdet already in cons.B
}

// Basic variables
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::bsq>(REDUCE_FUNCTION_ARGS)
{
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
    return dot(Dtmp.bcon, Dtmp.bcov);
}
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::gas_pressure>(REDUCE_FUNCTION_ARGS)
{
    return (gam - 1) * P(m_p.UU, k, j, i);
}
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::beta>(REDUCE_FUNCTION_ARGS)
{
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
    return ((gam - 1) * P(m_p.UU, k, j, i))/(0.5*(dot(Dtmp.bcon, Dtmp.bcov) + SMALL_NUM));
}

// Stuff that should be conserved
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::rhou0>(REDUCE_FUNCTION_ARGS)
{
    return U(m_u.RHO, k, j, i);
}
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::mix_T00>(REDUCE_FUNCTION_ARGS)
{
    return U(m_u.UU, k, j, i);
}
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::mix_T01>(REDUCE_FUNCTION_ARGS)
{
    return U(m_u.U1, k, j, i);
}
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::mix_T02>(REDUCE_FUNCTION_ARGS)
{
    return U(m_u.U2, k, j, i);
}
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::mix_T03>(REDUCE_FUNCTION_ARGS)
{
    return U(m_u.U3, k, j, i);
}

// Accretion rates: return a zone's contribution to the surface integral
// forming each rate measurement.
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::mdot>(REDUCE_FUNCTION_ARGS)
{
    Real ucon[GR_DIM];
    GRMHD::calc_ucon(G, P, m_p, k, j, i, Loci::center, ucon);
    // \dot{M} == \int rho * u^1 * gdet * dx2 * dx3
    return -P(m_p.RHO, k, j, i) * ucon[X1DIR] * G.gdet(Loci::center, j, i);
}
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::edot>(REDUCE_FUNCTION_ARGS)
{
    FourVectors Dtmp;
    Real T1[GR_DIM];
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
    Flux::calc_tensor(P, m_p, Dtmp, emhd_params, gam, k, j, i, X1DIR, T1);
    // \dot{E} == \int - T^1_0 * gdet * dx2 * dx3
    return -T1[X0DIR] * G.gdet(Loci::center, j, i);
}
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::ldot>(REDUCE_FUNCTION_ARGS)
{
    FourVectors Dtmp;
    Real T1[GR_DIM];
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
    Flux::calc_tensor(P, m_p, Dtmp, emhd_params, gam, k, j, i, X1DIR, T1);
    // \dot{L} == \int T^1_3 * gdet * dx2 * dx3
    return T1[X3DIR] * G.gdet(Loci::center, j, i);
}

// Then we can define the same with fluxes.
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::mdot_flux>(REDUCE_FUNCTION_ARGS)
{
    return -U.flux(X1DIR, m_u.RHO, k, j, i);
}
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::edot_flux>(REDUCE_FUNCTION_ARGS)
{
    return (U.flux(X1DIR, m_u.UU, k, j, i) - U.flux(X1DIR, m_u.RHO, k, j, i));
}
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::ldot_flux>(REDUCE_FUNCTION_ARGS)
{
    return U.flux(X1DIR, m_u.U3, k, j, i);
}

// Luminosity proxy from (for example) Porth et al 2019.
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::eht_lum>(REDUCE_FUNCTION_ARGS)
{
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
    Real rho = P(m_p.RHO, k, j, i);
    Real Pg = (gam - 1.) * P(m_p.UU, k, j, i);
    Real Bmag = m::sqrt(dot(Dtmp.bcon, Dtmp.bcov));
    Real j_eht = rho*rho*rho/Pg/Pg * m::exp(-0.2 * m::cbrt(rho * rho / (Bmag * Pg * Pg)));
    return j_eht;
}

// Example of checking extra conditions before adding local results:
// sums total jet power only at exactly r=radius, for areas with sig > 1
// TODO version w/E&M power only.  Needs "calc_tensor_EM"
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::jet_lum>(REDUCE_FUNCTION_ARGS)
{
    FourVectors Dtmp;
    Real T1[GR_DIM];
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
    Flux::calc_tensor(P, m_p, Dtmp, emhd_params, gam, k, j, i, X1DIR, T1);
    // If sigma > 1...
    if ((dot(Dtmp.bcon, Dtmp.bcov) / P(m_p.RHO, k, j, i)) > 1.) {
        // Energy flux, like at EH
        return -T1[X0DIR];
    } else {
        return 0.;
    }
}

// Diagnostics.  Still have to return Real so we get creative.
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::zero_ctop>(REDUCE_FUNCTION_ARGS)
{
    Real is_zero = 0;
    VLOOP {
        if(m::max(cmax(v, k, j, i), cmin(v, k, j, i)) <= 0.) {
            is_zero = 1.; // once per zone
#if DEBUG
#ifndef KOKKOS_ENABLE_SYCL
            printf("ctop zero at %d %d %d along dir %d\n", i, j, k, v+1);
#endif
#endif
        }
    }

    return is_zero;
}
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::nan_ctop>(REDUCE_FUNCTION_ARGS)
{
    Real is_nan = 0.;
    VLOOP {
        if(m::isnan(m::max(cmax(v, k, j, i), cmin(v, k, j, i)))) {
            is_nan = 1.;
#if DEBUG
#ifndef KOKKOS_ENABLE_SYCL
            printf("ctop NaN at %d %d %d along dir %d\n", i, j, k, v+1);
#endif
#endif
        }
    }

    return is_nan;
}

template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::neg_rhout>(REDUCE_FUNCTION_ARGS)
{
    Real is_neg = 0.;
    if (U(m_u.RHO, k, j, i) < 0.) {
        is_neg = 1.;
#if DEBUG
#ifndef KOKKOS_ENABLE_SYCL
        printf("Negative rho*u^0 (cons.rho) at %d %d %d\n", i, j, k);
#endif
#endif
    }
    return is_neg;
}
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::neg_u>(REDUCE_FUNCTION_ARGS)
{
    Real is_neg = 0.;
    if (P(m_p.UU, k, j, i) < 0.) {
        is_neg = 1.;
#if DEBUG
#ifndef KOKKOS_ENABLE_SYCL
        printf("Negative internal energy (prims.u) at %d %d %d\n", i, j, k);
#endif
#endif
    }
    return is_neg;
}
template <>
KOKKOS_INLINE_FUNCTION Real reduction_var<Var::neg_rho>(REDUCE_FUNCTION_ARGS)
{
    Real is_neg = 0.;
    if (P(m_p.RHO, k, j, i) < 0.) {
        is_neg = 1.;
#if DEBUG
#ifndef KOKKOS_ENABLE_SYCL
        printf("Negative density (prims.rho) at %d %d %d\n", i, j, k);
#endif
#endif
    }
    return is_neg;
}

}

#undef REDUCE_FUNCTION_ARGS
