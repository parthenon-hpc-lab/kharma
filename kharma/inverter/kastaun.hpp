/*
 *  File: kastaun.hpp
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
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
// © 2021-2023. Triad National Security, LLC. All rights reserved.  This
// program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.
#pragma once

// Robust primitive variable recovery as described in Kastaun et al. (2020)
// IMPORTANT: The following functions are stolen directly from:
// Phoebus: https://github.com/lanl/phoebus (con2prim_robust.hpp)
// AthenaK: https://gitlab.com/theias/hpc/jmstone/athena-parthenon/athenak
// (ideal_c2p_mhd.hpp) They have been lightly adapted to fit into KHARMA, and hopefully
// original authors should be clear from comments

// General template
// We define a specialization based on the Inverter::Type parameter
#include "invert_template.hpp"

#include "coordinate_utils.hpp"
// #include "floors_functions.hpp"
#include "grmhd_functions.hpp"
#include "kharma_utils.hpp"

// phoebus includes
#include "microphysics/eos_kharma/eos_kharma.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "phoebus_utils/variables.hpp"



// This isn't a vecloop, also it takes an argument.
// Left it in since it's useful and all over Phoebus, maybe we'll adopt it
#define SPACELOOP(i) for (int i = 0; i < 3; i++)
#define SPACELOOP2(i, j) SPACELOOP(i) SPACELOOP(j)
#define SPACETIMELOOP(mu) for (int mu = 0; mu < GR_DIM; mu++)

namespace Inverter
{

/**
 * Residual class from Phoebus.
 * Caches function arguments which won't change during solve
 */
class KastaunResidual
{
  public:
    KOKKOS_FUNCTION
    KastaunResidual(const Real& D, const Real& q, const Real& bsq, const Real& bsq_rpsq,
        const Real& rsq, const Real& rbsq, const Real& v0sq, const Microphysics::EOS::EOS& eos)
        : D_(D)
        , q_(q)
        , bsq_(bsq)
        , bsq_rpsq_(bsq_rpsq)
        , rsq_(rsq)
        , rbsq_(rbsq)
        , v0sq_(v0sq)
        , eos_(eos)
    {}

    KOKKOS_FORCEINLINE_FUNCTION
    Real x_mu(const Real mu) { return 1.0 / (1.0 + mu * bsq_); }
    KOKKOS_FORCEINLINE_FUNCTION
    Real rbarsq_mu(const Real mu, const Real x)
    {
        return x * (x * rsq_ + mu * (1.0 + x) * rbsq_);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Real qbar_mu(const Real mu, const Real x)
    {
        const Real mux = mu * x;
        return q_ - 0.5 * (bsq_ + mux * mux * bsq_rpsq_);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Real vhatsq_mu(const Real mu, const Real rbarsq)
    {
        return std::min(mu * mu * rbarsq, v0sq_);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Real iWhat_mu(const Real vhatsq) { return std::sqrt(1.0 - vhatsq); }
    KOKKOS_FORCEINLINE_FUNCTION
    Real rhohat_mu(const Real iWhat) { return D_ * iWhat; }
    KOKKOS_FORCEINLINE_FUNCTION
    Real ehat_mu(const Real mu, const Real qbar, const Real rbarsq, const Real vhatsq,
        const Real What)
    {
        return What * (qbar - mu * rbarsq) + vhatsq * What * What / (1.0 + What);
    }

    // Evaluate residual at a value of mu.
    // Kastaun eqn 44
    KOKKOS_INLINE_FUNCTION
    Real operator()(const Real mu)
    {
        const Real x = x_mu(mu);
        const Real rbarsq = rbarsq_mu(mu, x);
        const Real qbar = qbar_mu(mu, x);
        const Real vhatsq = vhatsq_mu(mu, rbarsq);
        const Real iWhat = iWhat_mu(vhatsq);
        const Real What = 1.0 / iWhat;
        // TODO technically we should only limit P>0, and allow returning negative u
        const Real rhohat = std::max(rhohat_mu(iWhat), 0.);
        const Real ehat = std::max(ehat_mu(mu, qbar, rbarsq, vhatsq, What), 0.);
        const Real Phat = eos_.PressureFromDensityInternalEnergy(rhohat, ehat);
        //TODO_EOS: ahat general or ideal-only?
        const Real ahat = Phat / (rhohat * (1.0 + ehat));

        const Real nua = (1.0 + ahat) * (1.0 + ehat) * iWhat;
        const Real nub = (1.0 + ahat) * (1.0 + qbar - mu * rbarsq);
        const Real nuhat = std::max(nua, nub);

        const Real muhat = 1.0 / (nuhat + mu * rbarsq);
        return mu - muhat;
    }

    // Residual for finding bracket values
    // Kastaun eqn 49
    KOKKOS_FORCEINLINE_FUNCTION
    Real aux_func(const Real mu)
    {
        const Real x = 1.0 / (1.0 + mu * bsq_);
        const Real rbarsq = x * (rsq_ * x + mu * (1.0 + x) * rbsq_);
        return mu * std::sqrt(1.0 + rbarsq) - 1.0;
    }

  private:
    const Real D_, q_, bsq_, bsq_rpsq_, rsq_, rbsq_, v0sq_;
    const Microphysics::EOS::EOS& eos_;
};

/**
 * Robust inversion scheme from Kastaun et al. 2020
 * Unholy mashup of the transformation/equations from Phoebus (which are
 * coordinate-general), and the solver from AthenaK (which is easier to read and
 * precomputes the bracket)
 * TODO keep mu between calls to speed up convergence
 * TODO better returns: be explicit about pre- and post-inversion floors, cat neg_input
 * too
 */
template<>
KOKKOS_INLINE_FUNCTION int u_to_p<Type::kastaun>(const GRCoordinates& G,
    const VariablePack<Real>& U, const VarMap& m_u, const Microphysics::EOS::EOS& eos, const int& k,
    const int& j, const int& i, const VariablePack<Real>& P, const VarMap& m_p,
    const Loci& loc, const int& max_iterations, const Real& tol)
{
    // Shouldn't need this, KHARMA should die on NaN
    // But it's here for debugging
    // int num_nans = std::isnan(U(m_u.RHO, k, j, i)) + std::isnan(U(m_u.U1, k, j, i)) +
    // std::isnan(U(m_u.UU, k, j, i)); if (num_nans > 0) return
    // static_cast<int>(Status::neg_input);

    // This exists only to keep the math stable on first call,
    // so we can add floors instead of failing outright
    if (U(m_u.RHO, k, j, i) < 1e-20) {
        U(m_u.RHO, k, j, i) = 1e-20;
    }

    // Transform GRMHD variables for the SRMHD Kastaun solver
    const Real alpha = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));
    const Real a_over_g = alpha / G.gdet(loc, j, i);

    const Real& Urho = U(m_u.RHO, k, j, i);
    const Real D = Urho * a_over_g;

    Real Qcov[GR_DIM] = {(U(m_u.UU, k, j, i) - Urho) * a_over_g,
        U(m_u.U1, k, j, i) * a_over_g, U(m_u.U2, k, j, i) * a_over_g,
        U(m_u.U3, k, j, i) * a_over_g};

    const Real ncov[GR_DIM] = {(Real)-alpha, 0., 0., 0.};
    Real ncon[GR_DIM];
    G.raise(ncov, ncon, k, j, i, loc);
    const Real q = (-dot(Qcov, ncon) - D) / D; // TODO floor on this?

    // r_i
    Real rcov[3] = {
        U(m_u.U1, k, j, i) / Urho, U(m_u.U2, k, j, i) / Urho, U(m_u.U3, k, j, i) / Urho};
    Real rcon[3];
    Real gupper[GR_DIM][GR_DIM];
    G.gcon(loc, j, i, gupper);
    // Ripped from AthenaK's "TransformToSRMHD,"
    // since we don't use the spatial metric anywhere else.  Original comment:
    // Gourghoulon says: g^ij = gamma^ij - beta^i beta^j/alpha^2
    //       g^0i = beta^i/alpha^2
    //       g^00 = -1/ alpha^2
    // Hence gamma^ij =  g^ij - g^0i g^0j/g^00
    rcon[0] = ((gupper[1][1] - gupper[0][1] * gupper[0][1] / gupper[0][0]) * rcov[0] +
               (gupper[1][2] - gupper[0][1] * gupper[0][2] / gupper[0][0]) * rcov[1] +
               (gupper[1][3] - gupper[0][1] * gupper[0][3] / gupper[0][0]) *
                   rcov[2]); // (C26)

    rcon[1] = ((gupper[2][1] - gupper[0][2] * gupper[0][1] / gupper[0][0]) * rcov[0] +
               (gupper[2][2] - gupper[0][2] * gupper[0][2] / gupper[0][0]) * rcov[1] +
               (gupper[2][3] - gupper[0][2] * gupper[0][3] / gupper[0][0]) *
                   rcov[2]); // (C26)

    rcon[2] = ((gupper[3][1] - gupper[0][3] * gupper[0][1] / gupper[0][0]) * rcov[0] +
               (gupper[3][2] - gupper[0][3] * gupper[0][2] / gupper[0][0]) * rcov[1] +
               (gupper[3][3] - gupper[0][3] * gupper[0][3] / gupper[0][0]) *
                   rcov[2]); // (C26)

    Real rsq = 0.0;
    SPACELOOP(ii) rsq += rcon[ii] * rcov[ii];

    Real bsq = 0.0;
    Real bsq_rpsq = 0.0;
    Real rbsq = 0.0;
    Real bdotr = 0.0;
    Real bu[] = {0.0, 0.0, 0.0};
    // If the magnetic field is being evolved at all...
    if (m_u.B1 >= 0) {
        const Real sD = 1.0 / m::sqrt(D);
        // b^i
        SPACELOOP(ii)
        {
            bu[ii] = (U(m_u.B1 + ii, k, j, i) * a_over_g) * sD;
            bdotr += bu[ii] * rcov[ii];
        }
        SPACELOOP2(ii, jj) bsq += G.gcov(loc, j, i, ii + 1, jj + 1) * bu[ii] * bu[jj];
        bsq = std::max(0.0, bsq);

        rbsq = bdotr * bdotr;
        bsq_rpsq = bsq * rsq - rbsq;
    }
    // const Real zsq = rsq / h0sq_; // h0sq_ normalization set to 1 in Phoebus
    const Real zsq = rsq;
    const Real v0sq = std::min(zsq / (1.0 + zsq), 1.0 - 1.0 / SQR(51.));

    // residual object. Caches most arguments/floors so calls are single-argument
    KastaunResidual res(D, q, bsq, bsq_rpsq, rsq, rbsq, v0sq, eos);

    // SOLVE
    // TODO(CEP) better or faster solver?  (Optionally) skip bracketing?
    // Need to find initial bracket. Requires separate solve
    Real zm = 0.;
    Real zp = 1.; // This is the lowest specific enthalpy admitted by the EOS

    // Evaluate master function (eq 49) at bracket values
    Real fm = res.aux_func(zm);
    Real fp = res.aux_func(zp);

    // For simplicity on the GPU, find roots using the false position method
    int iterations = max_iterations;
    // If bracket within tolerances, don't bother doing any iterations
    if ((m::abs(zm - zp) < tol) || ((m::abs(fm) + m::abs(fp)) < 2.0 * tol)) {
        iterations = -1;
    }
    Real z = 0.5 * (zm + zp);

    int iter;
    for (iter = 0; iter < iterations; ++iter) {
        z = (zm * fp - zp * fm) / (fp - fm); // linear interpolation to point f(z)=0
        Real f = res.aux_func(z);
        // Quit if convergence reached
        // NOTE(@ermost): both z and f are of order unity
        if ((m::abs(zm - zp) < tol) || (m::abs(f) < tol)) {
            break;
        }
        // assign zm-->zp if root bracketed by [z,zp]
        if (f * fp < 0.0) {
            zm = zp;
            fm = fp;
            zp = z;
            fp = f;
        } else { // assign zp-->z if root bracketed by [zm,z]
            fm =
                0.5 * fm; // 1/2 comes from "Illinois algorithm" to accelerate convergence
            zp = z;
            fp = f;
        }
    }

    // Found brackets. Now find solution in bounded interval, again using the
    // false position method
    zm = 0.;
    zp = z;

    // Evaluate master function (eq 44) at bracket values
    fm = res(zm);
    fp = res(zp);

    iterations = max_iterations;
    if ((m::abs(zm - zp) < tol) || ((m::abs(fm) + m::abs(fp)) < 2.0 * tol)) {
        iterations = -1;
    }
    z = 0.5 * (zm + zp);

    for (iter = 0; iter < iterations; ++iter) {
        z = (zm * fp - zp * fm) / (fp - fm); // linear interpolation to point f(z)=0
        Real f = res(z);
        // Quit if convergence reached
        // NOTE: both z and f are of order unity
        if ((m::abs(zm - zp) < tol) || (m::abs(f) < tol)) {
            break;
        }
        // assign zm-->zp if root bracketed by [z,zp]
        if (f * fp < 0.0) {
            zm = zp;
            fm = fp;
            zp = z;
            fp = f;
        } else { // assign zp-->z if root bracketed by [zm,z]
            fm =
                0.5 * fm; // 1/2 comes from "Illinois algorithm" to accelerate convergence
            zp = z;
            fp = f;
        }
    }
    // TODO(CEP) make sure we really don't want to set P for this case
    // is it filled with last step, or zero?
    if (iter >= max_iterations) return static_cast<int>(Status::max_iter);

    // Just unwrap everything into primitive vars as-is
    const Real mu = z;
    const Real x = res.x_mu(mu);
    const Real rbarsq = res.rbarsq_mu(mu, x);
    const Real vsq = res.vhatsq_mu(mu, rbarsq);
    const Real iW = res.iWhat_mu(vsq);
    const Real W = 1.0 / iW;
    const Real qbar = res.qbar_mu(mu, x);
    // These values should be as *raw* as possible, whether or not they respect the floors
    // (or even physics).  We will add material and try again if they're bad
    Real rho = res.rhohat_mu(iW);
    P(m_p.RHO, k, j, i) = m::max(rho, 0.);
    Real u = res.ehat_mu(mu, qbar, rbarsq, vsq, W) * P(m_p.RHO, k, j, i);
    P(m_p.UU, k, j, i) = m::max(u, 0.);
    // Latter part is a vector/signed quantity, don't set a minimum at 0
    Real mag_vel = W * mu * x;
    SPACELOOP(ii)
    P(m_p.U1 + ii, k, j, i) = std::max(mag_vel, 0.) * (rcon[ii] + mu * bdotr * bu[ii]);

    if (rho <= 0.) {
        return static_cast<int>(Status::neg_rho);
    } else if (u <= 0.) {
        return static_cast<int>(Status::neg_u);
    } else if (mag_vel <= 0.) {
        return static_cast<int>(Status::bad_gamma);
    }

    // Mark for fix only if convergence is not established within max_iterations (should
    // be *extremely* rare)
    return static_cast<int>(Status::success);
}

}
