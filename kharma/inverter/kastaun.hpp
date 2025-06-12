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
// AthenaK: https://gitlab.com/theias/hpc/jmstone/athena-parthenon/athenak (ideal_c2p_mhd.hpp)
// They have been lightly adapted to fit into KHARMA,
// and hopefully original authors should be clear from comments

// General template
// We define a specialization based on the Inverter::Type parameter
#include "invert_template.hpp"

#include "coordinate_utils.hpp"
#include "floors_functions.hpp"
#include "grmhd_functions.hpp"
#include "kharma_utils.hpp"

// This isn't a vecloop, also it takes an argument.
// Left it in since it's useful and all over Phoebus, maybe we'll adopt it
#define SPACELOOP(i) for (int i = 0; i < 3; i++)
#define SPACELOOP2(i, j) SPACELOOP(i) SPACELOOP(j)
#define SPACETIMELOOP(mu) for (int mu = 0; mu < GR_DIM; mu++)

namespace Inverter {

/**
 * Residual class from Phoebus, allowing caching of:
 * 1. Function arguments other than solution var "mu"
 * 2. Floors/ceilings and tracking of floor hits
 * Also handles translating mu->primitive variables
 */
class KastaunResidual {
    public:
        KOKKOS_FUNCTION
        KastaunResidual(const Real &D, const Real &q, const Real &bsq, const Real &bsq_rpsq,
                const Real &rsq, const Real &rbsq, const Real &v0sq, const Real &gam,
                const Real &rho_floor, const Real &e_floor,
                const Real &gamma_max, const Real &e_max)
            : D_(D), q_(q), bsq_(bsq), bsq_rpsq_(bsq_rpsq),
            rsq_(rsq), rbsq_(rbsq), v0sq_(v0sq), gam_(gam),
            rho_floor_(rho_floor), e_floor_(e_floor),
            gamma_max_(gamma_max), e_max_(e_max) {}

        KOKKOS_FORCEINLINE_FUNCTION
        Real x_mu(const Real mu)
        {
            return 1.0 / (1.0 + mu * bsq_);
        }
        KOKKOS_FORCEINLINE_FUNCTION
        Real rbarsq_mu(const Real mu, const Real x) {
            return x * (x * rsq_ + mu * (1.0 + x) * rbsq_);
        }
        KOKKOS_FORCEINLINE_FUNCTION
        Real qbar_mu(const Real mu, const Real x) {
            const Real mux = mu * x;
            return q_ - 0.5 * (bsq_ + mux * mux * bsq_rpsq_);
        }
        KOKKOS_FORCEINLINE_FUNCTION
        Real vhatsq_mu(const Real mu, const Real rbarsq) {
            const Real vsq_trial = mu * mu * rbarsq;
            if (vsq_trial > v0sq_) {
                used_gamma_max_ = true;
                return v0sq_;
            } else {
                used_gamma_max_ = false;
                return vsq_trial;
            }
        }
        KOKKOS_FORCEINLINE_FUNCTION
        Real iWhat_mu(const Real vhatsq)
        {
            return std::sqrt(1.0 - vhatsq);
        }
        KOKKOS_FORCEINLINE_FUNCTION
        Real rhohat_mu(const Real iWhat) {
            const Real rho_trial = D_ * iWhat;
            if (rho_trial <= rho_floor_) {
                used_density_floor_ = true;
                return rho_floor_;
            } else {
                used_density_floor_ = false;
                return rho_trial;
            }
        }
        KOKKOS_FORCEINLINE_FUNCTION
        Real ehat_mu(const Real mu, const Real qbar, const Real rbarsq, const Real vhatsq,
                    const Real What)
        {
            const Real ehat_trial =
                    What * (qbar - mu * rbarsq) + vhatsq * What * What / (1.0 + What);
            // Note this floor is approximate, since we haven't landed on a density
            used_energy_floor_ = false;
            used_energy_max_ = false;
            if (ehat_trial <= e_floor_) {
                used_energy_floor_ = true;
                return e_floor_;
            } else if (ehat_trial > e_max_) {
                used_energy_max_ = true;
                return e_max_;
            } else {
                return ehat_trial;
            }
        }

        // Evaluate residual at a value of mu.
        // Kastaun eqn 44
        KOKKOS_INLINE_FUNCTION
        Real operator()(const Real mu) {
            const Real x = x_mu(mu);
            const Real rbarsq = rbarsq_mu(mu, x);
            const Real qbar = qbar_mu(mu, x);
            const Real vhatsq = vhatsq_mu(mu, rbarsq);
            const Real iWhat = iWhat_mu(vhatsq);
            const Real What = 1.0 / iWhat;
            const Real rhohat = rhohat_mu(iWhat);
            const Real ehat = ehat_mu(mu, qbar, rbarsq, vhatsq, What);
            // TODO this is ideal-only
            const Real Phat = ehat * rhohat * (gam_ - 1.0);
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
        Real aux_func(const Real mu) {
            const Real x = 1.0 / (1.0 + mu * bsq_);
            const Real rbarsq = x * (rsq_ * x + mu * (1.0 + x) * rbsq_);
            return mu * std::sqrt(1.0 + rbarsq) - 1.0;
        }

        // Query floors
        // TODO fold into single int w/FFlag?  That's what we return anyway
        KOKKOS_INLINE_FUNCTION
        bool used_density_floor() const { return used_density_floor_; }
        KOKKOS_INLINE_FUNCTION
        bool used_energy_floor() const { return used_energy_floor_; }
        KOKKOS_INLINE_FUNCTION
        bool used_energy_max() const { return used_energy_max_; }
        KOKKOS_INLINE_FUNCTION
        bool used_gamma_max() const { return used_gamma_max_; }

    private:
        const Real D_, q_, bsq_, bsq_rpsq_, rsq_, rbsq_, v0sq_, gam_;
        const Real rho_floor_, e_floor_, gamma_max_, e_max_;
        bool used_density_floor_, used_energy_floor_, used_energy_max_, used_gamma_max_;
};

/**
 * Robust inversion scheme from Kastaun et al. 2020
 * Unholy mashup of the transformation/equations from Phoebus (which are coordinate-general),
 * and the solver from AthenaK (which is easier to read and precomputes the bracket)
 * TODO keep mu between calls to speed up convergence
 * TODO better returns: be explicit about pre- and post-inversion floors, cat neg_input too
 */
template <>
KOKKOS_INLINE_FUNCTION int u_to_p<Type::kastaun>(const GRCoordinates& G, const VariablePack<Real>& U, const VarMap& m_u,
                                              const Real& gam, const int& k, const int& j, const int& i,
                                              const VariablePack<Real>& P, const VarMap& m_p,
                                              const Loci& loc, const Floors::Prescription& floors,
                                              const int& max_iterations, const Real& tol)
{
    // Shouldn't need this, KHARMA should die on NaN
    // But it's here for debugging
    // int num_nans = std::isnan(U(m_u.RHO, k, j, i)) + std::isnan(U(m_u.U1, k, j, i)) + std::isnan(U(m_u.UU, k, j, i));
    // if (num_nans > 0) return static_cast<int>(Status::neg_input);

    // Conserved particle number rho*u0 > 0.  Just floor instead of yelling?
    if (U(m_u.RHO, k, j, i) <= 0.) {
        return static_cast<int>(Status::neg_input);
    }

    // Transform GRMHD variables for the SRMHD Kastaun solver
    const Real alpha  = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));
    const Real a_over_g = alpha / G.gdet(loc, j, i);

    const Real &Urho = U(m_u.RHO, k, j, i);
    const Real D = Urho * a_over_g;

    Real Qcov[GR_DIM] = {(U(m_u.UU, k, j, i) - Urho) * a_over_g,
                    U(m_u.U1, k, j, i) * a_over_g,
                    U(m_u.U2, k, j, i) * a_over_g,
                    U(m_u.U3, k, j, i) * a_over_g};

    const Real ncov[GR_DIM] = {(Real) -alpha, 0., 0., 0.};
    Real ncon[GR_DIM];
    G.raise(ncov, ncon, k, j, i, loc);
    const Real q = (-dot(Qcov, ncon) - D) / D;

    // r_i
    Real rcov[3] = {U(m_u.U1, k, j, i) / Urho,
                    U(m_u.U2, k, j, i) / Urho,
                    U(m_u.U3, k, j, i) / Urho};
    Real rcon[3];
    Real gupper[GR_DIM][GR_DIM];
    G.gcon(loc, j, i, gupper);
    // Ripped from AthenaK's "TransformToSRMHD,"
    // since we don't use the spatial metric anywhere else.  Original comment:
    // Gourghoulon says: g^ij = gamma^ij - beta^i beta^j/alpha^2
    //       g^0i = beta^i/alpha^2
    //       g^00 = -1/ alpha^2
    // Hence gamma^ij =  g^ij - g^0i g^0j/g^00
    rcon[0] = ((gupper[1][1] - gupper[0][1]*gupper[0][1]/gupper[0][0])*rcov[0] +
                (gupper[1][2] - gupper[0][1]*gupper[0][2]/gupper[0][0])*rcov[1] +
                (gupper[1][3] - gupper[0][1]*gupper[0][3]/gupper[0][0])*rcov[2]);  // (C26)

    rcon[1] = ((gupper[2][1] - gupper[0][2]*gupper[0][1]/gupper[0][0])*rcov[0] +
                (gupper[2][2] - gupper[0][2]*gupper[0][2]/gupper[0][0])*rcov[1] +
                (gupper[2][3] - gupper[0][2]*gupper[0][3]/gupper[0][0])*rcov[2]);  // (C26)

    rcon[2] = ((gupper[3][1] - gupper[0][3]*gupper[0][1]/gupper[0][0])*rcov[0] +
                (gupper[3][2] - gupper[0][3]*gupper[0][2]/gupper[0][0])*rcov[1] +
                (gupper[3][3] - gupper[0][3]*gupper[0][3]/gupper[0][0])*rcov[2]);  // (C26)

    Real rsq = 0.0;
    SPACELOOP(ii) rsq += rcon[ii]*rcov[ii];

    Real bsq = 0.0;
    Real bsq_rpsq = 0.0;
    Real rbsq = 0.0;
    Real bdotr = 0.0;
    Real bu[] = {0.0, 0.0, 0.0};
    if (m_u.B1 >= 0) {
        const Real sD = 1.0 / m::sqrt(D);
        // b^i
        SPACELOOP(ii) {
            bu[ii] = (U(m_u.B1 + ii, k, j, i) * a_over_g) * sD;
            bdotr += bu[ii] * rcov[ii];
        }
        SPACELOOP2(ii, jj) bsq += G.gcov(loc, j, i, ii + 1, jj + 1) * bu[ii] * bu[jj];
        bsq = std::max(0.0, bsq);

        rbsq = bdotr * bdotr;
        bsq_rpsq = bsq * rsq - rbsq;
    }
    //const Real zsq = rsq / h0sq_; // h0sq_ normalization set to 1 in Phoebus
    const Real zsq = rsq;
    const Real v0sq = std::min(zsq / (1.0 + zsq), 1.0 - 1.0 / SQR(floors.gamma_max));

    // residual object. Caches most arguments/floors so calls are single-argument
    // TODO second arg is technically floor on specific energy, should be split...
    KastaunResidual res(D, q, bsq, bsq_rpsq, rsq, rbsq, v0sq, gam,
                        floors.rho_min_const, floors.u_min_const,
                        floors.gamma_max, floors.u_over_rho_max);

    // SOLVE
    // TODO(BSP) better or faster solver?  (Optionally) skip bracketing?
    // Need to find initial bracket. Requires separate solve
    Real zm = 0.;
    Real zp = 1.; // This is the lowest specific enthalpy admitted by the EOS

    // Evaluate master function (eq 49) at bracket values
    Real fm = res.aux_func(zm);
    Real fp = res.aux_func(zp);

    // For simplicity on the GPU, find roots using the false position method
    int iterations = max_iterations;
    // If bracket within tolerances, don't bother doing any iterations
    if ((m::abs(zm-zp) < tol) || ((m::abs(fm) + m::abs(fp)) < 2.0*tol)) {
        iterations = -1;
    }
    Real z = 0.5*(zm + zp);

    int iter;
    for (iter = 0; iter < iterations; ++iter) {
        z =  (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
        Real f = res.aux_func(z);
        // Quit if convergence reached
        // NOTE(@ermost): both z and f are of order unity
        if ((m::abs(zm-zp) < tol) || (m::abs(f) < tol)) {
            break;
        }
        // assign zm-->zp if root bracketed by [z,zp]
        if (f*fp < 0.0) {
            zm = zp;
            fm = fp;
            zp = z;
            fp = f;
        } else {  // assign zp-->z if root bracketed by [zm,z]
            fm = 0.5*fm; // 1/2 comes from "Illinois algorithm" to accelerate convergence
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
    if ((m::abs(zm-zp) < tol) || ((m::abs(fm) + m::abs(fp)) < 2.0*tol)) {
        iterations = -1;
    }
    z = 0.5*(zm + zp);

    for (iter = 0; iter < iterations; ++iter) {
        z = (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
        Real f = res(z);
        // Quit if convergence reached
        // NOTE: both z and f are of order unity
        if ((m::abs(zm-zp) < tol) || (m::abs(f) < tol)) {
            break;
        }
        // assign zm-->zp if root bracketed by [z,zp]
        if (f*fp < 0.0) {
            zm = zp;
            fm = fp;
            zp = z;
            fp = f;
        } else {  // assign zp-->z if root bracketed by [zm,z]
            fm = 0.5*fm; // 1/2 comes from "Illinois algorithm" to accelerate convergence
            zp = z;
            fp = f;
        }
    }
    // TODO keep track of max iter

    // check if convergence is established within max_iterations.  If not, return
    // failure without replacing prims, for consistency w/1Dw solver.
    // We generally replace failed zones with neighbors or atmosphere later, at user option
    if (iter == max_iterations) {
        return static_cast<int>(Status::max_iter);
    }

    int fflag = 0;
    // Now unwrap everything into primitive vars...
    Real mu = z;
    Real x = res.x_mu(mu);
    Real rbarsq = res.rbarsq_mu(mu, x);
    Real vsq = res.vhatsq_mu(mu, rbarsq);
    Real iW = res.iWhat_mu(vsq);
    Real W = 1.0 / iW;
    P(m_p.RHO, k, j, i) = res.rhohat_mu(iW);

    // Correct mu if we used rho floor
    if (res.used_density_floor()) {
        // Mark the flag
        fflag |= Floors::FFlag::INVERTER_RHO;
        // New W determined from rho
        W = D / P(m_p.RHO, k, j, i);
        iW = 1.0 / W;
        vsq = 1.0 - SQR(iW);
        // Analytic solution for x, mu given new W
        double A = rsq - rbsq / bsq;
        double B = rbsq / bsq;
        double C = vsq * bsq;
        double coeff = -A*A + 3*A*B + 3*A*C;
        double coeff3 = coeff * coeff * coeff;
        double coeff_2 = -2*A*A*A - 18*A*A*B + 9*A*A*C;
        double det = m::sqrt(4 * coeff3 + SQR(coeff_2));
        double cbrt_det = m::cbrt(coeff_2 + det);
        x = m::cbrt(2) * coeff / (3 * A * cbrt_det)
        - cbrt_det / (3 * A * m::cbrt(2)) + 1/3;
        mu = (1./x - 1) / bsq;
        rbarsq = res.rbarsq_mu(mu, x);
    }

    const Real qbar = res.qbar_mu(mu, x);
    // Use the floored density in energy calc, we corrected mu for it
    P(m_p.UU, k, j, i) = res.ehat_mu(mu, qbar, rbarsq, vsq, W) * P(m_p.RHO, k, j, i);
    // We applied this floor to the ratio u/rho, but not the value u
    if(P(m_p.UU, k, j, i) < floors.u_min_const) {
        fflag |= Floors::FFlag::INVERTER_U;
        P(m_p.UU, k, j, i) = floors.u_min_const;
    }

    // Try to correct mu if we used the e floor
    bool ret_momentum_failure = false;
    if (res.used_energy_floor() || (fflag & Floors::FFlag::INVERTER_U)) {
        // Record flag
        fflag |= Floors::FFlag::INVERTER_U;

        // Function e(mu) - e_actual so we can rootfind
        auto f = [&] (Real mu) {
            Real x = res.x_mu(mu);
            Real rbarsq = res.rbarsq_mu(mu, x);
            Real vsq = res.vhatsq_mu(mu, rbarsq);
            Real W = 1. / res.iWhat_mu(vsq);
            Real qbar = res.qbar_mu(mu, x);
            return (W * (qbar - mu * rbarsq) + vsq * W * W /
                        (1.0 + W)) * P(m_p.RHO, k, j, i)
                    - P(m_p.UU, k, j, i);
        };
        // Rootfind for mu that would have produced our floored e
        Real mum = 0., mup = 1.;
        if (f(mum) * f(mup) > 0.) {
            mu = 0.;
            x = 0.;
            W = 0.;
            ret_momentum_failure = true;
        } else {
            while (1) {
                Real muc = (mum + mup) / 2.;
                Real resv = m::abs(f(muc));
                if (resv < 1e-8 || m::abs((mup - mum) / 2) < 1e-10) {
                    mu = muc;
                    if (resv < 1e-8) {
                        // If we're solved set the solution
                        x = res.x_mu(mu);
                        rbarsq = res.rbarsq_mu(mu, x);
                        vsq = res.vhatsq_mu(mu, rbarsq);
                        iW = res.iWhat_mu(vsq);
                        W = 1. / iW;
                    } else {
                        // Else defaults
                        x = 0.;
                        W = 0.;
                        // Mark momentum failure
                        ret_momentum_failure = true;
                    }
                    // printf("Solution mu %g x %g rsq %g rbsq %g rbarsq %g vsq %g W %g\n",
                    //         mu, x, rsq, rbsq, rbarsq, vsq, W);
                    // SPACELOOP(ii)
                    //     printf("New U%d: %g\n", ii, W * mu * x * (rcon[ii] + mu * bdotr * bu[ii]));
                    break;
                }
                // Same sign as left side -> center now left side
                if (f(muc) * f(mum) > 0.)
                    mum = muc;
                else
                    mup = muc;
            }
        }
    }

    // Set velocity with corrected values from rho+e floors
    SPACELOOP(ii) P(m_p.U1 + ii, k, j, i) = W * mu * x * (rcon[ii] + mu * bdotr * bu[ii]);

    // Record other flags
    if (res.used_gamma_max())     fflag |= Floors::FFlag::INVERTER_GAMMA;
    if (res.used_energy_max())    fflag |= Floors::FFlag::INVERTER_U_MAX;

    return fflag + ((ret_momentum_failure) ? static_cast<int>(Status::bad_velocity) : 0);
}

}
