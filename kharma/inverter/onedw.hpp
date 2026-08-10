/*
 *  File: onedw.hpp
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

// General template
// We define a specialization based on the Inverter::Type parameter
#include "invert_template.hpp"

#include "grmhd_functions.hpp"
#include "kharma_utils.hpp"

namespace Inverter
{

// Could put support fns in their own namespace, but I'm lazy
/**
 * Fluid relativistic factor gamma in terms of inversion state variables of the Noble 1D_W
 * inverter
 */
KOKKOS_INLINE_FUNCTION Real lorentz_calc_w(
    const Real& Bsq, const Real& D, const Real& QdB, const Real& Qtsq, const Real& Wp)
{
    const Real QdBsq = QdB * QdB;
    const Real W = Wp + D;
    const Real W2 = W * W;
    const Real WB = W + Bsq;

    // This is basically inversion of eq. A7 of Mignone & McKinney
    const Real utsq =
        -((W + WB) * QdBsq + W2 * Qtsq) / (QdBsq * (W + WB) + W2 * (Qtsq - WB * WB));

    // Catch utsq < 0 and YELL
    // Latter number should be about ~1e3*GAMMAMAX^2, but obvs doesn't need to be precise
    if (utsq < -1.e-15 || utsq > 1.e7) {
        return -1.; // This will trigger caller to return an error immediately
    } else {
        return m::sqrt(1. + m::abs(utsq));
    }
}

/**
 * Error metric for Newton-Raphson step in Noble 1D_W inverter
 */
KOKKOS_INLINE_FUNCTION Real err_eqn(const Real& gam, const Real& Bsq, const Real& D,
    const Real& Ep, const Real& QdB, const Real& Qtsq, const Real& Wp, Status& eflag)
{
    const Real W = Wp + D;
    const Real gamma = lorentz_calc_w(Bsq, D, QdB, Qtsq, Wp);
    if (gamma < 1) eflag = Status::bad_ut;
    const Real w = W / (gamma * gamma);
    const Real rho = D / gamma;
    const Real p = (w - rho) * (gam - 1) / gam;

    return -Ep + Wp - p + 0.5 * Bsq + 0.5 * (Bsq * Qtsq - QdB * QdB) / SQR(Bsq + W);
}

/**
 * 1D_W inverter from Ressler et al. 2006.
 */
template<>
KOKKOS_INLINE_FUNCTION int u_to_p<Type::onedw>(const GRCoordinates& G,
    const VariablePack<Real>& U, const VarMap& m_u, const Real& gam, const int& k,
    const int& j, const int& i, const VariablePack<Real>& P, const VarMap& m_p,
    const Loci& loc, const int& max_iterations, const Real& tol,
    const bool recover_velocity)
{
    // TODO try inline floors in the old 1Dw?  Probably not relevant anymore
    // Catch negative density
    // TODO(BSP) if we start to see this hit, fix it here by returning P=P_floors
    if (U(m_u.RHO, k, j, i) <= 0.) {
        return static_cast<int>(Status::neg_input);
    }

    // Convert from conserved variables to four-vectors
    const Real alpha = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));
    const Real gdet = G.gdet(loc, j, i);
    const Real a_over_g = alpha / gdet;
    const Real D = U(m_u.RHO, k, j, i) * a_over_g;

    Real Bcon[GR_DIM] = {0};
    if (m_u.B1 >= 0) {
        Bcon[1] = U(m_u.B1, k, j, i) * a_over_g;
        Bcon[2] = U(m_u.B2, k, j, i) * a_over_g;
        Bcon[3] = U(m_u.B3, k, j, i) * a_over_g;
    }

    const Real Qcov[GR_DIM] = {(U(m_u.UU, k, j, i) - U(m_u.RHO, k, j, i)) * a_over_g,
        U(m_u.U1, k, j, i) * a_over_g, U(m_u.U2, k, j, i) * a_over_g,
        U(m_u.U3, k, j, i) * a_over_g};

    const Real ncov[GR_DIM] = {(Real)-alpha, 0., 0., 0.};

    // TODO faster with on-the-fly gcon/cov?
    Real Bcov[GR_DIM], Qcon[GR_DIM], ncon[GR_DIM];
    G.lower(Bcon, Bcov, k, j, i, loc);
    G.raise(Qcov, Qcon, k, j, i, loc);
    G.raise(ncov, ncon, k, j, i, loc);

    const Real Bsq = dot(Bcon, Bcov);
    const Real QdB = dot(Bcon, Qcov);
    const Real Qdotn = dot(Qcon, ncov);

    // TODO(BSP) check, test if this gets hit unlike above
    // if (U(m_u.UU, k, j, i) <= gdet*(1e-8) + gdet*Bsq/2) {

    // }

    Real Qtcon[GR_DIM];
    DLOOP1
        Qtcon[mu] = Qcon[mu] + ncon[mu] * Qdotn;
    const Real Qtsq = dot(Qcon, Qcov) + Qdotn * Qdotn;

    // Set up eqtn for W'; this is the energy density
    const Real Ep = -Qdotn - D;

    // Numerical rootfinding

    // Accumulator for errors in err_eqn
    Status eflag = Status::success;

    // Initial guess from primitives:
    Real Wp, err;
    {
        const Real gamma = GRMHD::lorentz_calc(G, P, m_p, k, j, i, loc);
        if (gamma < 1) return static_cast<int>(Status::bad_ut);
        const Real rho = P(m_p.RHO, k, j, i), u = P(m_p.UU, k, j, i);

        Wp = (rho + u + (gam - 1) * u) * gamma * gamma - rho * gamma;
        err = err_eqn(gam, Bsq, D, Ep, QdB, Qtsq, Wp, eflag);
    }

    Real dW;
    {
        // Step around the guess & evaluate errors
        // TODO take stepsize as arg?
        const Real Wpm = (1. - 1.e-5) * Wp; // heuristic
        const Real h = Wp - Wpm;
        const Real Wpp = Wp + h;
        const Real errm = err_eqn(gam, Bsq, D, Ep, QdB, Qtsq, Wpm, eflag);
        const Real errp = err_eqn(gam, Bsq, D, Ep, QdB, Qtsq, Wpp, eflag);

        // Attempt a Halley/Muller/Bailey/Press step
        const Real dedW = (errp - errm) / (Wpp - Wpm);
        const Real dedW2 = (errp - 2. * err + errm) / (h * h);
        // TODO look into changing these clipped values?
        const Real f = clip(0.5 * err * dedW2 / (dedW * dedW), -0.3, 0.3);

        dW = clip(-err / dedW / (1. - f), -0.5 * Wp, 2.0 * Wp);
    }

    // Take the first step
    Real Wp1 = Wp;
    Real err1 = err;
    Wp += dW;
    err = err_eqn(gam, Bsq, D, Ep, QdB, Qtsq, Wp, eflag);

    // Not good enough?  apply secant method
    int iter = 0;
    for (iter = 0; iter < max_iterations; iter++) {
        dW = clip((Wp1 - Wp) * err / (err - err1), (Real)-0.5 * Wp, (Real)2.0 * Wp);

        Wp1 = Wp;
        err1 = err;

        Wp += dW;

        if (m::abs(dW / Wp) < tol) break;

        err = err_eqn(gam, Bsq, D, Ep, QdB, Qtsq, Wp, eflag);

        if (m::abs(err / Wp) < tol) break;
    }
    // Return failure to converge
    if (iter == max_iterations) return static_cast<int>(Status::max_iter);

    // Find utsq, gamma, rho from Wp
    // Note we only throw an error if *this* velocity calculation is bad
    const Real gamma = lorentz_calc_w(Bsq, D, QdB, Qtsq, Wp);
    if (gamma < 1) return static_cast<int>(Status::bad_ut);

    const Real rho = D / gamma;
    const Real W = Wp + D;
    const Real w = W / (gamma * gamma);
    const Real p = (w - rho) * (gam - 1) / gam;
    const Real u = w - (rho + p);

    // Return without updating non-B primitives
    if (rho < 0 && u < 0)
        return static_cast<int>(Status::neg_rhou);
    else if (rho < 0)
        return static_cast<int>(Status::neg_rho);
    else if (u < 0)
        return static_cast<int>(Status::neg_u);

    // Set primitives
    P(m_p.RHO, k, j, i) = rho;
    P(m_p.UU, k, j, i) = u;

    // Find u(tilde); Eqn. 31 of Noble et al.
    const Real pre = (gamma / (W + Bsq));
    P(m_p.U1, k, j, i) = pre * (Qtcon[1] + QdB * Bcon[1] / W);
    P(m_p.U2, k, j, i) = pre * (Qtcon[2] + QdB * Bcon[2] / W);
    P(m_p.U3, k, j, i) = pre * (Qtcon[3] + QdB * Bcon[3] / W);

    return static_cast<int>(Status::success);
}

} // namespace Inverter
