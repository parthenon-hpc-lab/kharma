/*
 *  File: radM1_solvers.hpp
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

#include "radM1.hpp"

#include "entropy.hpp"
#include "inverter.hpp"

#define RAD_LARGE (0.1 * std::numeric_limits<Real>::max())
#define RAD_SMALL (10.0 * std::numeric_limits<Real>::min())
#define RAD_EPS (10.0 * std::numeric_limits<Real>::epsilon())

namespace RadM1
{

KOKKOS_INLINE_FUNCTION Real compute_y_max(const Real GAMMAMAX)
{
    const Real target = GAMMAMAX * GAMMAMAX;
    Real y_old = 0.9998;
    Real E_old =
        target - (2.0 - y_old + m::sqrt(4.0 - 3.0 * y_old)) / (4.0 - 4.0 * y_old);
    for (int n = 0; n < 40; ++n) {
        Real dEdy = (0.375 * y_old - 0.25 * m::sqrt(4.0 - 3.0 * y_old) - 0.625) /
                    (m::sqrt(4.0 - 3.0 * y_old) * (1.0 - y_old) * (1.0 - y_old));
        Real y_new = m::min(y_old - E_old / dEdy, 0.99999999999999);
        Real E_new =
            target - (2.0 - y_new + m::sqrt(4.0 - 3.0 * y_new)) / (4.0 - 4.0 * y_new);
        y_old = y_new;
        if (m::abs(E_new) / target <= 1.e-9) break;
        E_old = E_new;
    }
    return y_old;
}

KOKKOS_INLINE_FUNCTION StatusRadiationInversion u_to_p_rad(const GRCoordinates& G,
    const Real U_rad[4], Real P_rad[4], const int k, const int j, const int i,
    bool* used_normal_out = nullptr, RadLimiterType type = RadLimiterType::TYPE2)
{
    Real gdet = G.gdet(Loci::center, j, i);

    if (U_rad[0] == 0.0 && U_rad[1] == 0.0 && U_rad[2] == 0.0 && U_rad[3] == 0.0) {
        P_rad[0] = 0.0;
        P_rad[1] = 0.0;
        P_rad[2] = 0.0;
        P_rad[3] = 0.0;
        if (used_normal_out != nullptr) *used_normal_out = true;
        return StatusRadiationInversion::success;
    }

    Real gcon_tt = G.gcon(Loci::center, j, i, 0, 0);
    Real alpha = m::sqrt(-1.0 / gcon_tt);

    Real Ucov_zamo[4];
    for (int mu = 0; mu < 4; ++mu) Ucov_zamo[mu] = alpha * U_rad[mu] / gdet;

    if (!m::isfinite(Ucov_zamo[0]) || !m::isfinite(Ucov_zamo[1]) ||
        !m::isfinite(Ucov_zamo[2]) || !m::isfinite(Ucov_zamo[3])) {
        P_rad[0] = 1.e-300;
        P_rad[1] = 0.0;
        P_rad[2] = 0.0;
        P_rad[3] = 0.0;
        if (used_normal_out != nullptr) *used_normal_out = false;
        return StatusRadiationInversion::not_finite;
    }

    Real Ucon_zamo[4];
    G.raise(Ucov_zamo, Ucon_zamo, k, j, i, Loci::center);

    Real eta_cov0 = -alpha;
    Real eta_con[4];
    for (int mu = 0; mu < 4; ++mu)
        eta_con[mu] = G.gcon(Loci::center, j, i, 0, mu) * eta_cov0;

    Real U_dot_eta = Ucon_zamo[0] * eta_cov0;

    Real Utilde_con[4] = {0.0, 0.0, 0.0, 0.0};
    for (int mu = 1; mu < 4; ++mu)
        Utilde_con[mu] = Ucon_zamo[mu] + eta_con[mu] * U_dot_eta;

    Real Utilde_sq = 0.0;
    for (int mu = 1; mu < 4; ++mu)
        for (int nu = 1; nu < 4; ++nu)
            Utilde_sq +=
                G.gcov(Loci::center, j, i, mu, nu) * Utilde_con[mu] * Utilde_con[nu];

    if (Utilde_sq < 0.0) {
        Utilde_sq = 0.0;
        Utilde_con[1] = 0.0;
        Utilde_con[2] = 0.0;
        Utilde_con[3] = 0.0;
    }

    const Real GAMMAMAX = 50.0;
    const Real y_max = compute_y_max(GAMMAMAX);

    Real y = Utilde_sq / (U_dot_eta * U_dot_eta + 1.e-150);
    Real gamma_rad_sq = (2.0 - y + m::sqrt(m::max(0.0, 4.0 - 3.0 * y))) / (4.0 - 4.0 * y);

    Real p_rad = -U_dot_eta / (4.0 * gamma_rad_sq - 1.0);
    Real Erf = p_rad * 3.0;

    Real uvec_rad[4] = {0.0, 0.0, 0.0, 0.0};
    for (int mu = 1; mu < 4; ++mu)
        uvec_rad[mu] =
            m::sqrt(gamma_rad_sq) * Utilde_con[mu] / (4.0 * p_rad * gamma_rad_sq);

    bool failed = (y > y_max) || (y < 0.0) || !m::isfinite(U_dot_eta) ||
                  (U_dot_eta > 0.0) || !m::isfinite(uvec_rad[1]) ||
                  !m::isfinite(uvec_rad[2]) || !m::isfinite(uvec_rad[3]);

    bool used_normal = !failed;
    if (used_normal_out != nullptr) *used_normal_out = used_normal;

    if (failed) {
        if (type == RadLimiterType::BASIC) {
            if (U_dot_eta > 0.0) {
                // Erad <= 0
                P_rad[0] = 1.e-300;
                P_rad[1] = 0.0;
                P_rad[2] = 0.0;
                P_rad[3] = 0.0;
                return StatusRadiationInversion::urad_negative;
            } else if (y < 0.0 && y > -RAD_EPS) {
                // 0 > y > -m: zero the velocity only, leave Erf as already computed above
                if (!m::isfinite(Erf)) Erf = 1.e-300;
                P_rad[0] = Erf;
                P_rad[1] = 0.0;
                P_rad[2] = 0.0;
                P_rad[3] = 0.0;
                return StatusRadiationInversion::negative_gamma;
            } else {
                // y > y_max: rescale directly to y = y_max with no intermediate
                // Uabs/GAMMAMAX step
                Real U_dot_eta_basic = -(1.e-30 + m::sqrt(m::abs(Utilde_sq) / y_max));
                Real gamma_rad_sq_basic =
                    (2.0 - y_max + m::sqrt(4.0 - 3.0 * y_max)) / (4.0 - 4.0 * y_max);
                Real p_rad_basic = -U_dot_eta_basic / (4.0 * gamma_rad_sq_basic - 1.0);
                Erf = p_rad_basic * 3.0;
                for (int mu = 1; mu < 4; ++mu)
                    uvec_rad[mu] = m::sqrt(gamma_rad_sq_basic) * Utilde_con[mu] /
                                   (4.0 * p_rad_basic * gamma_rad_sq_basic);

                if (!m::isfinite(Erf)) Erf = 1.e-300;
                if (!m::isfinite(uvec_rad[1])) uvec_rad[1] = 0.0;
                if (!m::isfinite(uvec_rad[2])) uvec_rad[2] = 0.0;
                if (!m::isfinite(uvec_rad[3])) uvec_rad[3] = 0.0;

                P_rad[0] = Erf;
                P_rad[1] = uvec_rad[1];
                P_rad[2] = uvec_rad[2];
                P_rad[3] = uvec_rad[3];
                return StatusRadiationInversion::maximum_gamma;
            }
        } else {
            // Do TYPE2 fallback
            Real Uabs = 0.5 * (m::abs(U_dot_eta) + m::sqrt(m::abs(Utilde_sq)) + 1.e-150);
            for (int mu = 1; mu < 4; ++mu)
                uvec_rad[mu] = GAMMAMAX * Utilde_con[mu] / Uabs;

            Real qsq =
                G.gcov(Loci::center, j, i, 1, 1) * uvec_rad[1] * uvec_rad[1] +
                G.gcov(Loci::center, j, i, 2, 2) * uvec_rad[2] * uvec_rad[2] +
                G.gcov(Loci::center, j, i, 3, 3) * uvec_rad[3] * uvec_rad[3] +
                2.0 * G.gcov(Loci::center, j, i, 1, 2) * uvec_rad[1] * uvec_rad[2] +
                2.0 * G.gcov(Loci::center, j, i, 1, 3) * uvec_rad[1] * uvec_rad[3] +
                2.0 * G.gcov(Loci::center, j, i, 2, 3) * uvec_rad[2] * uvec_rad[3];
            if (qsq < 0.0 || m::abs(qsq) < 1.e-10) qsq = 1.e-10;
            Real gamma_rad_sq_fb = 1.0 + qsq;

            Real f = m::sqrt((GAMMAMAX * GAMMAMAX - 1.0) / (gamma_rad_sq_fb - 1.0));
            uvec_rad[1] *= f;
            uvec_rad[2] *= f;
            uvec_rad[3] *= f;

            Real U_dot_eta_tp2 = -(1.e-30 + m::sqrt(m::abs(Utilde_sq) / y_max));
            Real gamma_rad_sq_tp2 =
                (2.0 - y_max + m::sqrt(4.0 - 3.0 * y_max)) / (4.0 - 4.0 * y_max);
            Real p_rad_tp2 = -U_dot_eta_tp2 / (4.0 * gamma_rad_sq_tp2 - 1.0);
            Erf = p_rad_tp2 * 3.0;
            for (int mu = 1; mu < 4; ++mu)
                uvec_rad[mu] = m::sqrt(gamma_rad_sq_tp2) * Utilde_con[mu] /
                               (4.0 * p_rad_tp2 * gamma_rad_sq_tp2);
            if (!m::isfinite(Erf)) Erf = 1.e-300;
            if (!m::isfinite(uvec_rad[1])) uvec_rad[1] = 0.0;
            if (!m::isfinite(uvec_rad[2])) uvec_rad[2] = 0.0;
            if (!m::isfinite(uvec_rad[3])) uvec_rad[3] = 0.0;
            P_rad[0] = Erf;
            P_rad[1] = uvec_rad[1];
            P_rad[2] = uvec_rad[2];
            P_rad[3] = uvec_rad[3];
            return StatusRadiationInversion::type2;
        }
    }

    if (!m::isfinite(Erf)) Erf = 1.e-300;
    if (!m::isfinite(uvec_rad[1])) uvec_rad[1] = 0.0;
    if (!m::isfinite(uvec_rad[2])) uvec_rad[2] = 0.0;
    if (!m::isfinite(uvec_rad[3])) uvec_rad[3] = 0.0;

    P_rad[0] = Erf;
    P_rad[1] = uvec_rad[1];
    P_rad[2] = uvec_rad[2];
    P_rad[3] = uvec_rad[3];

    return StatusRadiationInversion::success;
}

KOKKOS_INLINE_FUNCTION void compute_covariant_fourforce(const GRCoordinates& G,
    const Real P_mhd[4], const Real P_rad[4], const Real rho, const Real B_P[NVEC],
    const Microphysics::EOS::EOS& eos, const RadOpac& rad_opac, const int k, const int j,
    const int i, Real dS[4], Real* kappa_tot_out = nullptr, Real* coupling_out = nullptr,
    Real kap[3] = nullptr, const bool kap_frozen = false)
{

    if (static_cast<RadM1::OpacityType>(rad_opac.opacity_type) ==
        RadM1::OpacityType::Transparent) {
        dS[0] = 0.0;
        dS[1] = 0.0;
        dS[2] = 0.0;
        dS[3] = 0.0;
        return;
    }

    Real uvec_mhd[3] = {P_mhd[1], P_mhd[2], P_mhd[3]};
    Real ucon_mhd[4], ucov_mhd[4];
    GRMHD::calc_ucon(G, uvec_mhd, k, j, i, Loci::center, ucon_mhd);
    G.lower(ucon_mhd, ucov_mhd, k, j, i, Loci::center);

    Real Erf = P_rad[0];
    Real ucon_rad[4], ucov_rad[4];
    RadM1::calc_ucon_rad(G, P_rad, j, i, ucon_rad);
    G.lower(ucon_rad, ucov_rad, k, j, i, Loci::center);

    Real gamma_rel = -(ucon_rad[0] * ucov_mhd[0] + ucon_rad[1] * ucov_mhd[1] +
                       ucon_rad[2] * ucov_mhd[2] + ucon_rad[3] * ucov_mhd[3]);
    gamma_rel = m::max(gamma_rel, 1.0);

    Real E_hat = Erf * ((4.0 / 3.0) * gamma_rel * gamma_rel - (1.0 / 3.0));
    Real F_hat_cov[4];
    for (int mu = 0; mu < 4; ++mu) {
        F_hat_cov[mu] = (4.0 / 3.0) * Erf * gamma_rel * ucov_rad[mu] -
                        ((1.0 / 3.0) * Erf + E_hat) * ucov_mhd[mu];
    }

    Real Tg = eos.TemperatureFromDensityInternalEnergy(rho, P_mhd[0] / rho);

    // kap = {absorption (phi at Trad), emission (phi at Tg, Kirchhoff), scattering}: used
    // as given if kap_frozen (Jacobian columns), else computed and returned
    Real kappa_abs_rad, kappa_emit_gas, kappa_sc, bsq = 0.;
    if (kap_frozen) {
        kappa_abs_rad = kap[0];
        kappa_emit_gas = kap[1];
        kappa_sc = kap[2];
    } else {
        // b^2 and Trad only enter through the opacities
        Real bcon[4] = {0., 0., 0., 0.};
        Real bcov[4] = {0., 0., 0., 0.};
        VLOOP
            bcon[0] += B_P[v] * ucov_mhd[v + 1];
        VLOOP
            bcon[v + 1] = (B_P[v] + bcon[0] * ucon_mhd[v + 1]) / ucon_mhd[0];
        G.lower(bcon, bcov, k, j, i, Loci::center);
        bsq = dot(bcov, bcon);
        const Real Trad = rad_opac.Trad(E_hat, rho, bsq);
        rad_opac.kappa_a_rad_gas(rho, Tg, bsq, Trad, kappa_abs_rad, kappa_emit_gas);
        kappa_sc = RadM1::calc_kscattering(rho, Tg, Trad, bsq, rad_opac);
        if (kap != nullptr) {
            kap[0] = kappa_abs_rad;
            kap[1] = kappa_emit_gas;
            kap[2] = kappa_sc;
        }
    }
    Real JBB_val = rad_opac.JBB(Tg, rho, bsq);

    Real kappa_tot = kappa_abs_rad + kappa_sc;

    Real coupling_term = kappa_emit_gas * JBB_val - kappa_abs_rad * E_hat;
    dS[0] = coupling_term * ucov_mhd[0] - kappa_tot * F_hat_cov[0];
    dS[1] = coupling_term * ucov_mhd[1] - kappa_tot * F_hat_cov[1];
    dS[2] = coupling_term * ucov_mhd[2] - kappa_tot * F_hat_cov[2];
    dS[3] = coupling_term * ucov_mhd[3] - kappa_tot * F_hat_cov[3];
    if (kappa_tot_out != nullptr)
        *kappa_tot_out = kappa_sc + m::max(kappa_abs_rad, kappa_emit_gas);
    if (coupling_out != nullptr) *coupling_out = coupling_term;
}

KOKKOS_INLINE_FUNCTION Real calculate_energy_residual(const GRCoordinates& G,
    const Real u_trial, const Real uvec_frozen[NVEC], const Real B_P[NVEC],
    const Real U_mhd_0[4], const Real U_rad_0[4], const Real rho,
    const Microphysics::EOS::EOS& eos, const RadOpac& rad_opac, const Real dt,
    const Real gdet, const int k, const int j, const int i, Real U_mhd_trial_out[4],
    Real U_rad_trial_out[4], Real P_rad_trial_out[4], Real dS_trial_out[4],
    bool& rad_recovery_ok)
{
    Real uvec[NVEC] = {uvec_frozen[0], uvec_frozen[1], uvec_frozen[2]};
    Real rho_new;
    GRMHD::p_to_u_mhd(
        G, rho, u_trial, uvec, B_P, eos, k, j, i, rho_new, U_mhd_trial_out, Loci::center);

    for (int n = 0; n < 4; n++) {
        U_rad_trial_out[n] = U_rad_0[n] - (U_mhd_trial_out[n] - U_mhd_0[n]);
    }

    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, uvec, B_P, k, j, i, Loci::center, Dtmp);
    rho_new = rho_new / (gdet * Dtmp.ucon[0]);

    auto status = u_to_p_rad(G, U_rad_trial_out, P_rad_trial_out, k, j, i);
    rad_recovery_ok = (status == StatusRadiationInversion::success);

    Real P_mhd_trial[4] = {u_trial, uvec_frozen[0], uvec_frozen[1], uvec_frozen[2]};
    compute_covariant_fourforce(G, P_mhd_trial, P_rad_trial_out, rho_new, B_P, eos,
        rad_opac, k, j, i, dS_trial_out);
    for (int n = 0; n < 4; n++) dS_trial_out[n] = gdet * dS_trial_out[n];

    Real resid = (U_mhd_trial_out[0] - U_mhd_0[0]) + dt * dS_trial_out[0];
    Real scale = m::max(RAD_SMALL,
        m::abs(U_mhd_trial_out[0]) + m::abs(U_mhd_0[0]) + m::abs(dt * dS_trial_out[0]));
    return resid / scale;
}

KOKKOS_INLINE_FUNCTION StatusImplicitStep solve_radiation_1d(const GRCoordinates& G,
    const VariablePack<Real> P_init, const VarMap m_p, const VarMap m_u,
    const Microphysics::EOS::EOS& eos, const RadOpac& rad_opac, const int k, const int j,
    const int i, const Real dt, const double tol, const int maxiter,
    const VariablePack<Real> pflag, const VariablePack<Real> rinvflag,
    const Real U_entry[8], Real dS_final[5])
{

    const Real gdet = G.gdet(Loci::center, j, i);
    const Real uvec_frozen[NVEC] = {
        P_init(m_p.U1, k, j, i), P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};
    Real B_P[NVEC] = {0.};
    if (m_p.B1 >= 0) {
        B_P[0] = P_init(m_p.B1, k, j, i);
        B_P[1] = P_init(m_p.B2, k, j, i);
        B_P[2] = P_init(m_p.B3, k, j, i);
    }
    const Real U_mhd_0[4] = {U_entry[0], U_entry[1], U_entry[2], U_entry[3]};
    const Real U_rad_0[4] = {U_entry[4], U_entry[5], U_entry[6], U_entry[7]};
    const Real rho_init = P_init(m_p.RHO, k, j, i);
    const Real u_init = P_init(m_p.UU, k, j, i);

    Real U_mhd_trial[4], U_rad_trial[4], P_rad_trial[4], dS_trial[4];
    bool rad_ok;

    Real u_lo = 1.e-2 * u_init;
    Real u_hi = 1.e2 * u_init;
    Real f_lo = calculate_energy_residual(G, u_lo, uvec_frozen, B_P, U_mhd_0, U_rad_0,
        rho_init, eos, rad_opac, dt, gdet, k, j, i, U_mhd_trial, U_rad_trial, P_rad_trial,
        dS_trial, rad_ok);
    Real f_hi = calculate_energy_residual(G, u_hi, uvec_frozen, B_P, U_mhd_0, U_rad_0,
        rho_init, eos, rad_opac, dt, gdet, k, j, i, U_mhd_trial, U_rad_trial, P_rad_trial,
        dS_trial, rad_ok);

    bool bracketed = (f_lo * f_hi < 0.0);
    Real rebracket_fac = 10.0;
    int n_rebracket = 0;
    const int MAX_REBRACKET = 4;
    while (!bracketed && n_rebracket < MAX_REBRACKET) {
        u_lo = (1.e-1 / rebracket_fac) * u_init;
        u_hi = (1.e1 * rebracket_fac) * u_init;
        f_lo = calculate_energy_residual(G, u_lo, uvec_frozen, B_P, U_mhd_0, U_rad_0,
            rho_init, eos, rad_opac, dt, gdet, k, j, i, U_mhd_trial, U_rad_trial,
            P_rad_trial, dS_trial, rad_ok);
        f_hi = calculate_energy_residual(G, u_hi, uvec_frozen, B_P, U_mhd_0, U_rad_0,
            rho_init, eos, rad_opac, dt, gdet, k, j, i, U_mhd_trial, U_rad_trial,
            P_rad_trial, dS_trial, rad_ok);
        bracketed = (f_lo * f_hi < 0.0);
        rebracket_fac *= 10.0;
        n_rebracket++;
    }

    if (!bracketed) {
        return StatusImplicitStep::failure;
    }

    Real u_root = 0.5 * (u_lo + u_hi);
    bool converged = false;
    int stuck_lo = 0, stuck_hi = 0;
    for (int iter = 0; iter < maxiter; iter++) {
        u_root = (u_lo * f_hi - u_hi * f_lo) / (f_hi - f_lo);
        Real f_root = calculate_energy_residual(G, u_root, uvec_frozen, B_P, U_mhd_0,
            U_rad_0, rho_init, eos, rad_opac, dt, gdet, k, j, i, U_mhd_trial, U_rad_trial,
            P_rad_trial, dS_trial, rad_ok);

        if (!rad_ok) {
            if (f_root * f_lo > 0.0) {
                u_lo = u_root;
                f_lo = f_root;
                stuck_hi++;
                stuck_lo = 0;
            } else {
                u_hi = u_root;
                f_hi = f_root;
                stuck_lo++;
                stuck_hi = 0;
            }
            continue;
        }

        if (m::abs(f_root) < tol || m::abs(u_hi - u_lo) < tol * u_init) {
            converged = true;
            break;
        }

        if (f_root * f_lo > 0.0) {
            u_lo = u_root;
            f_lo = f_root;
            stuck_hi++;
            stuck_lo = 0;
            if (stuck_hi >= 2) {
                f_hi *= 0.5;
                stuck_hi = 0;
            }
        } else {
            u_hi = u_root;
            f_hi = f_root;
            stuck_lo++;
            stuck_hi = 0;
            if (stuck_lo >= 2) {
                f_lo *= 0.5;
                stuck_lo = 0;
            }
        }
    }

    if (!converged) {
        return StatusImplicitStep::onedfallback_failure;
    }

    calculate_energy_residual(G, u_root, uvec_frozen, B_P, U_mhd_0, U_rad_0, rho_init,
        eos, rad_opac, dt, gdet, k, j, i, U_mhd_trial, U_rad_trial, P_rad_trial, dS_trial,
        rad_ok);
    if (!rad_ok) {
        return StatusImplicitStep::onedfallback_failure;
    }

    dS_final[0] = dS_trial[0];
    dS_final[1] = dS_trial[1];
    dS_final[2] = dS_trial[2];
    dS_final[3] = dS_trial[3];

    {
        Real ucon_final[4];
        GRMHD::calc_ucon(G, uvec_frozen, k, j, i, Loci::center, ucon_final);
        Real Gdotu = dS_trial[0] * ucon_final[0] + dS_trial[1] * ucon_final[1] +
                     dS_trial[2] * ucon_final[2] + dS_trial[3] * ucon_final[3];
        dS_final[4] = Gdotu;
    }

    return StatusImplicitStep::success;
}

KOKKOS_INLINE_FUNCTION Real calculate_error(const Real resid[4],
    const Real U_mhd_guess[4], const Real U_mhd_0[4], const Real dtdS[4])
{
    constexpr Real KORAL_SMALL = 1.e-80;

    Real err[4];

    // Energy component (koral's err[0], MHD-lab-frame energy case):
    // err[0] = |f[0]| / (|uu[UU]| + |uu0[UU]| + |dt*gdetu*Gi[0]|)
    if (m::abs(resid[0]) > KORAL_SMALL) {
        err[0] = m::abs(resid[0]) /
                 (m::abs(U_mhd_guess[0]) + m::abs(U_mhd_0[0]) + m::abs(dtdS[0]));
    } else {
        err[0] = 0.0;
    }

    /// Momentum components normalized by their own history plus the energy
    // component, so near-zero momentum does not turn round-off into large errors
    for (int n = 1; n < 4; n++) {
        if (m::abs(resid[n]) > KORAL_SMALL) {
            err[n] = m::abs(resid[n]) / (m::abs(U_mhd_guess[0]) + m::abs(U_mhd_guess[n]) +
                                            m::abs(U_mhd_0[n]) + m::abs(dtdS[n]));
        } else {
            err[n] = 0.0;
        }
    }

    Real err_max = 0.0;
    for (int n = 0; n < 4; n++) {
        err_max = m::max(err_max, err[n]);
    }
    return err_max;
}

KOKKOS_INLINE_FUNCTION Real rho_from_D(const GRCoordinates& G, const Real D,
    const Real uvec[NVEC], const int k, const int j, const int i)
{
    Real uv[NVEC] = {uvec[0], uvec[1], uvec[2]};
    Real ucon[GR_DIM];
    GRMHD::calc_ucon(G, uv, k, j, i, Loci::center, ucon);
    return D / ucon[0];
}

// Ideal-gas inversion keeping D, S_i and entropy Kent (Kastaun master function in mu)
KOKKOS_INLINE_FUNCTION bool u_to_p_entropy(const GRCoordinates& G, const Real Urho,
    const Real Ucov[NVEC], const Real Bcons[NVEC], const Real Kent, const Real gam,
    const int k, const int j, const int i, Real& rho, Real& u, Real uvec[NVEC])
{
    if (!(Urho > 0.) || !(Kent > 0.) || !m::isfinite(Kent)) return false;
    const Real alpha = 1. / m::sqrt(-G.gcon(Loci::center, j, i, 0, 0));
    const Real D = Urho * alpha / G.gdet(Loci::center, j, i);
    const Real rcov[NVEC] = {Ucov[0] / Urho, Ucov[1] / Urho, Ucov[2] / Urho};
    Real gup[GR_DIM][GR_DIM];
    G.gcon(Loci::center, j, i, gup);
    Real rcon[NVEC], bu[NVEC], rsq = 0., bsq = 0., bdotr = 0.;
    for (int a = 0; a < NVEC; a++) {
        rcon[a] = 0.;
        for (int b = 0; b < NVEC; b++)
            rcon[a] +=
                (gup[a + 1][b + 1] - gup[0][a + 1] * gup[0][b + 1] / gup[0][0]) * rcov[b];
        rsq += rcon[a] * rcov[a];
        bu[a] = Bcons[a] * alpha / G.gdet(Loci::center, j, i) / m::sqrt(D);
        bdotr += bu[a] * rcov[a];
    }
    for (int a = 0; a < NVEC; a++)
        for (int b = 0; b < NVEC; b++)
            bsq += G.gcov(Loci::center, j, i, a + 1, b + 1) * bu[a] * bu[b];
    bsq = m::max(bsq, 0.);
    const Real rbsq = bdotr * bdotr;
    const Real v0sq = m::min(rsq / (1. + rsq), 1. - 1. / (51. * 51.));
    auto resid = [&](const Real mu, Real& x, Real& iW)
    {
        x = 1. / (1. + mu * bsq);
        const Real rbarsq = x * (x * rsq + mu * (1. + x) * rbsq);
        iW = m::sqrt(1. - m::min(mu * mu * rbarsq, v0sq));
        const Real eps = Kent * m::pow(D * iW, gam - 1.) / (gam - 1.);
        return mu - 1. / ((1. + gam * eps) * iW + mu * rbarsq);
    };
    // f(0) < 0 <= f(1) for any h >= 1, so bisect on [0, 1]
    Real lo = 0., hi = 1., x, iW;
    for (int it = 0; it < 60; it++) {
        const Real mid = 0.5 * (lo + hi);
        if (resid(mid, x, iW) < 0.)
            lo = mid;
        else
            hi = mid;
    }
    const Real mu = 0.5 * (lo + hi);
    resid(mu, x, iW);
    rho = D * iW;
    u = Kent * m::pow(rho, gam) / (gam - 1.);
    for (int a = 0; a < NVEC; a++) uvec[a] = mu * x / iW * (rcon[a] + mu * bdotr * bu[a]);
    return m::isfinite(rho) && m::isfinite(u) && m::isfinite(uvec[0]) &&
           m::isfinite(uvec[1]) && m::isfinite(uvec[2]) && rho > 0. && u > 0.;
}

KOKKOS_INLINE_FUNCTION int solve_4d_pmhd(const GRCoordinates& G,
    const VariablePack<Real>& P_init, const VarMap m_p, const VarMap m_u, const int k,
    const int j, const int i, const Real dt, const Microphysics::EOS::EOS& eos,
    const double src_rootfind_eps, const double src_rootfind_tol,
    const int src_rootfind_maxiter, const RadOpac& rad_opac,
    const VariablePack<Real>& pflag, const VariablePack<Real>& rinvflag,
    const Real U_entry[8], Real dS_final[5], int* niter_out = nullptr,
    const bool star_guess = false)
{
    // Read the entry state from U_entry instead of copying it (fewer live registers)
    const Real* U_mhd_0 = U_entry;
    const Real* U_rad_0 = U_entry + 4;

    const Real rho_init = P_init(m_p.RHO, k, j, i);

    Real B_P[NVEC] = {0.};
    if (m_p.B1 >= 0) {
        B_P[V1] = P_init(m_p.B1, k, j, i);
        B_P[V2] = P_init(m_p.B2, k, j, i);
        B_P[V3] = P_init(m_p.B3, k, j, i);
    }
    Real P_mhd_guess[4] = {P_init(m_p.UU, k, j, i), P_init(m_p.U1, k, j, i),
        P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};

    Real uvec[NVEC] = {
        P_init(m_p.U1, k, j, i), P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};
    const Real D_rho = rho_init / rho_from_D(G, 1.0, uvec, k, j, i);
    const Real gdet = G.gdet(Loci::center, j, i);
    // Check if radiation inversion reached maximum gamma. 
    const bool beam_ok = static_cast<int>(rinvflag(0, k, j, i)) ==
                         static_cast<int>(StatusRadiationInversion::maximum_gamma);

    // Iteration 0
    Real U_mhd_guess[4] = {U_mhd_0[0], U_mhd_0[1], U_mhd_0[2], U_mhd_0[3]};

    // A star-repair guess doesn't invert U_mhd_0, so use its own U. So we replace U_mhd_guess. 
    if (star_guess) {
        Real rho_ut;
        GRMHD::p_to_u_mhd(G, rho_init, P_mhd_guess[0], uvec, B_P, eos, k, j, i, rho_ut,
            U_mhd_guess, Loci::center);
    }
    Real dS_guess[4];
    
    // Opacities at the current iterate, frozen in the Jacobian columns
    Real kap_guess[3];

    // If the inversion took different branches, can we trust the guess? This is what this variable represents.
    bool used_normal_guess = true;

    // Only E_rad of the radiation primitives is needed after the loop
    Real E_rad_guess;
    Real tol_eff;

    // Measuring what is the effective tolerance for the iteration.
    // You see, very stiff flows will have very large dS. Such that the truncation error can be larger than the tolerance.
    // So we scale the tolerance by the magnitude of dS, and the time step.
    {
        Real P_rad_guess[4], U_rad_e[4];
        for (int n = 0; n < 4; n++)
            U_rad_e[n] = U_rad_0[n] - (U_mhd_guess[n] - U_mhd_0[n]);
        u_to_p_rad(G, U_rad_e, P_rad_guess, k, j, i, &used_normal_guess);
        Real kappa_tot_entry = 0.0;
        compute_covariant_fourforce(G, P_mhd_guess, P_rad_guess, rho_init, B_P, eos,
            rad_opac, k, j, i, dS_guess, &kappa_tot_entry, nullptr, kap_guess);
        E_rad_guess = P_rad_guess[0];
        // Checking what is this effective tolerance.
        tol_eff = m::max(static_cast<Real>(src_rootfind_tol),
            10.0 * std::numeric_limits<Real>::epsilon() * kappa_tot_entry * dt);
    }

    for (int n = 0; n < 4; n++) dS_guess[n] = gdet * dS_guess[n];

    Real resid[4];
    DLOOP1 {
        resid[mu] = U_mhd_guess[mu] - U_mhd_0[mu] + dt * dS_guess[mu];
    }

    // Compute err here and do a convergence check
    Real err;
    {
        Real dtdS_0[4];
        for (int n = 0; n < 4; n++) dtdS_0[n] = dt * dS_guess[n];
        err = calculate_error(resid, U_mhd_guess, U_mhd_0, dtdS_0);
    }
    int niter = 0;
    bool bad_guess = false;

    Real rho_iter = rho_init;

    // FD Jacobian when needed, else a Broyden update from the last step
    Real jac[4][4], P_prev[4], resid_prev[4], err_prev = err;
    bool fresh_jac = true;
    do {
        if (err <= tol_eff) {
            break;
        }
        // Iterations started, for the solver_count diagnostic
        if (niter_out != nullptr) ++*niter_out;

        // Find minimum non-zero magnitude from P_mhd_guess to scale FD step safely
        Real P_mhd_mag_min = RAD_LARGE;
        for (int m = 0; m < 4; m++) {
            if (m::abs(P_mhd_guess[m]) > 0.) {
                P_mhd_mag_min = m::min(P_mhd_mag_min, m::abs(P_mhd_guess[m]));
            }
        }

        // Broyden in variables scaled by their magnitude, so u and u^i weigh alike
        // It doesn't enter here at the first pass.
        if (!fresh_jac) {
            Real dP[4], Jdp[4], dPsq = 0.;
            for (int m = 0; m < 4; m++) {
                const Real sc = m::max(m::abs(P_prev[m]), P_mhd_mag_min);
                dP[m] = (P_mhd_guess[m] - P_prev[m]) / (sc * sc);
                dPsq += (P_mhd_guess[m] - P_prev[m]) * dP[m];
            }
            for (int n = 0; n < 4; n++) {
                Jdp[n] = 0.;
                for (int m = 0; m < 4; m++)
                    Jdp[n] += jac[n][m] * (P_mhd_guess[m] - P_prev[m]);
                for (int m = 0; m < 4; m++)
                    jac[n][m] += (resid[n] - resid_prev[n] - Jdp[n]) * dP[m] / dPsq;
            }
        }

        // One-sided differences. We try forward,however if the forward support point is invalid, we try backward. If both are invalid, we give up and return failure.
        for (int m = 0; m < 4 && fresh_jac && !bad_guess; m++) {
            const Real fd_step = m::max(src_rootfind_eps * P_mhd_mag_min,
                src_rootfind_eps * m::abs(P_mhd_guess[m]));
            bool ok = false;

            // side here changes the forward -> backward (1 -> -1)
            for (int side = 1; side >= -1 && !ok; side -= 2) {
                Real P_mhd_p[4] = {
                    P_mhd_guess[0], P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};
                P_mhd_p[m] += side * fd_step;
                uvec[0] = P_mhd_p[1];
                uvec[1] = P_mhd_p[2];
                uvec[2] = P_mhd_p[3];
                const Real rho_p = rho_from_D(G, D_rho, uvec, k, j, i);
                Real rho_ut, U_mhd_p[4], U_rad_p[4], P_rad_p[4], dS_p[4];
                GRMHD::p_to_u_mhd(G, rho_p, P_mhd_p[0], uvec, B_P, eos, k, j, i, rho_ut,
                    U_mhd_p, Loci::center);
                // Conservation law: Delta U_rad = - Delta U_mhd
                for (int n = 0; n < 4; n++)
                    U_rad_p[n] = U_rad_0[n] - (U_mhd_p[n] - U_mhd_0[n]);
                bool used_normal_p;
                const auto st_p =
                    u_to_p_rad(G, U_rad_p, P_rad_p, k, j, i, &used_normal_p);
                ok = (st_p == StatusRadiationInversion::success ||
                         (beam_ok && st_p == StatusRadiationInversion::type2)) &&
                     used_normal_p == used_normal_guess;
                // if it fails, try the other side. If both fail, we give up and return failure.
                if (!ok) continue;
                compute_covariant_fourforce(G, P_mhd_p, P_rad_p, rho_p, B_P, eos,
                    rad_opac, k, j, i, dS_p, nullptr, nullptr, kap_guess, true);
                for (int n = 0; n < 4; n++)
                    jac[n][m] =
                        (U_mhd_p[n] - U_mhd_0[n] + dt * gdet * dS_p[n] - resid[n]) /
                        (P_mhd_p[m] - P_mhd_guess[m]);
            }
            if (!ok) bad_guess = true;
        }
        if (bad_guess) break;

        Real delta[4];
        {
            Real jacinv[4][4];
            invert(&jac[0][0], &jacinv[0][0]);
            for (int m = 0; m < 4; m++) {
                delta[m] = 0.;
                for (int n = 0; n < 4; n++) delta[m] += jacinv[m][n] * resid[n];
            }
        }

        // Here, we do a damped Newton step, which means that, we take the full step, and if it violates any of the checks
        // below, shrink it by 3 and retry, at most 15 times or to scaling_factor 1.e-8, these hardcoded numbers came from Koral.
        // PNM: Change hardcoded values?
        Real P_mhd_pre[4] = {
            P_mhd_guess[0], P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};
        Real scaling_factor = 1.0;
        int track = 0;
        bool step_ok = false;
        do {
            for (int m = 0; m < 4; m++)
                P_mhd_guess[m] = P_mhd_pre[m] - scaling_factor * delta[m];

            // if energy is negative, shrink step and retry.
            if (P_mhd_guess[0] <= 0.0) {
                scaling_factor /= 3.0;
                track++;
                // But, if we have tried 15 times, or the scaling factor is too small, we give up and return failure.
                if (track > 15 || scaling_factor < 1.e-8) break;
                continue;
            }

            // Re-evaluate residual with updated guess
            uvec[0] = P_mhd_guess[1];
            uvec[1] = P_mhd_guess[2];
            uvec[2] = P_mhd_guess[3];
            const Real rho_iter_next = rho_from_D(G, D_rho, uvec, k, j, i);
            Real rho_ut;
            GRMHD::p_to_u_mhd(G, rho_iter_next, P_mhd_guess[0], uvec, B_P, eos, k, j, i,
                rho_ut, U_mhd_guess, Loci::center);

            Real U_rad_guess[4], P_rad_guess[4];
            for (int n = 0; n < 4; n++) {
                U_rad_guess[n] = U_rad_0[n] - (U_mhd_guess[n] - U_mhd_0[n]);
            }

            auto status =
                u_to_p_rad(G, U_rad_guess, P_rad_guess, k, j, i, &used_normal_guess);
            if (status != StatusRadiationInversion::success &&
                !(beam_ok && status == StatusRadiationInversion::type2)) {
                scaling_factor /= 3.0;
                track++;
                if (track > 15 || scaling_factor < 1.e-8) break;
                continue;
            }
            compute_covariant_fourforce(G, P_mhd_guess, P_rad_guess, rho_iter_next, B_P,
                eos, rad_opac, k, j, i, dS_guess, nullptr, nullptr, kap_guess);
            for (int n = 0; n < 4; n++) dS_guess[n] = gdet * dS_guess[n];

            E_rad_guess = P_rad_guess[0];
            step_ok = true;
            rho_iter = rho_iter_next;
        } while (!step_ok && track <= 15);

        if (track > 15 || scaling_factor < 1.e-8) {
            bad_guess = true;
            break;
        }

        // Update residuals
        for (int n = 0; n < 4; n++) {
            P_prev[n] = P_mhd_pre[n];
            resid_prev[n] = resid[n];
            resid[n] = U_mhd_guess[n] - U_mhd_0[n] + dt * dS_guess[n];

            if (!m::isfinite(resid[n])) {
                bad_guess = true;
                break;
            }
        }

        // This is needed since the previous bad_guess = true would only break out of the
        // for loop
        if (bad_guess) {
            break;
        }

        // Calculate error now
        {
            Real dtdS_iter[4];
            for (int n = 0; n < 4; n++) dtdS_iter[n] = dt * dS_guess[n];
            err = calculate_error(resid, U_mhd_guess, U_mhd_0, dtdS_iter);
        }
        // Fresh FD Jacobian next time unless this step at least halved the error
        fresh_jac = !(err < 0.5 * err_prev);
        err_prev = err;

        niter++;
    } while (err > tol_eff && niter < src_rootfind_maxiter);

    // U_rad of the last iterate, from the conservation law
    Real U_rad_guess[4];
    for (int n = 0; n < 4; n++)
        U_rad_guess[n] = U_rad_0[n] - (U_mhd_guess[n] - U_mhd_0[n]);

    if (err > tol_eff || !m::isfinite(U_rad_guess[0]) || !m::isfinite(U_rad_guess[1]) ||
        !m::isfinite(U_rad_guess[2]) || !m::isfinite(U_rad_guess[3]) || bad_guess) {

        return static_cast<int>(StatusImplicitStep::failure);
    }

    {
        Real uvec_g[NVEC] = {P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};
        FourVectors D_g;
        GRMHD::calc_4vecs(G, uvec_g, B_P, k, j, i, Loci::center, D_g);
        const Real w_gas =
            rho_iter + P_mhd_guess[0] +
            eos.PressureFromDensityInternalEnergy(rho_iter, P_mhd_guess[0] / rho_iter) +
            dot(D_g.bcon, D_g.bcov);
        const bool e_rad = E_rad_guess > P_mhd_guess[0];
        const bool m_rad = (4.0 / 3.0) * E_rad_guess > w_gas;
        for (int n = 0; n < 4; n++) {
            const bool solved_state = (n == 0) ? e_rad : m_rad;
            dS_final[n] = solved_state ? (U_rad_guess[n] - U_rad_0[n]) / dt : dS_guess[n];
        }
    }

    {
        Real uvec_final[NVEC] = {P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};
        Real ucon_final[4];
        GRMHD::calc_ucon(G, uvec_final, k, j, i, Loci::center, ucon_final);
        Real Gdotu = dS_guess[0] * ucon_final[0] + dS_guess[1] * ucon_final[1] +
                     dS_guess[2] * ucon_final[2] + dS_guess[3] * ucon_final[3];
        dS_final[4] = Gdotu;
    }

    return static_cast<int>(StatusImplicitStep::success);
}


KOKKOS_INLINE_FUNCTION Real u_equilibrium(const GRCoordinates& G, const Real P_mhd[4],
    const Real P_rad[4], const Real rho, const Microphysics::EOS::EOS& eos,
    const RadOpac& rad_opac, const int k, const int j, const int i)
{
    // E_hat as in compute_covariant_fourforce
    Real uvec[NVEC] = {P_mhd[1], P_mhd[2], P_mhd[3]};
    Real ucon[GR_DIM], ucov[GR_DIM], ucon_rad[GR_DIM];
    GRMHD::calc_ucon(G, uvec, k, j, i, Loci::center, ucon);
    G.lower(ucon, ucov, k, j, i, Loci::center);
    RadM1::calc_ucon_rad(G, P_rad, j, i, ucon_rad);
    Real gamma_rel = 0.;
    for (int mu = 0; mu < GR_DIM; mu++) gamma_rel -= ucon_rad[mu] * ucov[mu];
    gamma_rel = m::max(gamma_rel, 1.0);
    const Real E_hat = P_rad[0] * ((4.0 / 3.0) * gamma_rel * gamma_rel - (1.0 / 3.0));
    const Real E_tot = P_mhd[0] + E_hat;
    // bsq only enters opacities, not Trad or JBB
    Real T = m::max(eos.TemperatureFromDensityInternalEnergy(rho, P_mhd[0] / rho),
        rad_opac.Trad(E_hat, rho, 0.0));
    for (int it = 0; it < 30; it++) {
        const Real e_gas = rho * eos.InternalEnergyFromDensityTemperature(rho, T);
        const Real e_rad = rad_opac.JBB(T, rho, 0.0);
        const Real dT = (e_gas + e_rad - E_tot) * T / (e_gas + 4. * e_rad);
        T -= dT;
        if (!(T > 0.)) return -1.;
        if (m::abs(dT) <= 1.e-12 * T) break;
    }
    return rho * eos.InternalEnergyFromDensityTemperature(rho, T);
}

// Fallback for solve_4d_pmhd -> fluid-frame gas energy residual normalized by u.
KOKKOS_INLINE_FUNCTION int solve_4d_pmhd_ff(const GRCoordinates& G,
    const VariablePack<Real> P_init, const VarMap m_p, const int k, const int j,
    const int i, const Real dt, const Microphysics::EOS::EOS& eos,
    const double src_rootfind_tol, const int src_rootfind_maxiter,
    const RadOpac& rad_opac, const Real U_entry[8], Real dS_final[5],
    int* niter_out = nullptr, int* why_out = nullptr, const bool eq_guess = false,
    const Real accept_tol = 0.)
{
    const Real gdet = G.gdet(Loci::center, j, i);
    Real B_P[NVEC] = {0.};
    if (m_p.B1 >= 0)
        for (int v = 0; v < NVEC; v++) B_P[v] = P_init(m_p.B1 + v, k, j, i);

    const Real u_star = P_init(m_p.UU, k, j, i);
    Real x[4] = {u_star, P_init(m_p.U1, k, j, i), P_init(m_p.U2, k, j, i),
        P_init(m_p.U3, k, j, i)};

    // If the initial guess (reconstructed in the implicit solver function) has already a negative energy, let's bounce off of here already.
    if (!(u_star > 0.)) { if (why_out != nullptr) *why_out = 1; return static_cast<int>(StatusImplicitStep::failure); }

    const Real D_rho = P_init(m_p.RHO, k, j, i) / rho_from_D(G, 1.0, &x[1], k, j, i);
    const Real U_mhd_0[4] = {U_entry[0], U_entry[1], U_entry[2], U_entry[3]};
    const Real U_rad_0[4] = {U_entry[4], U_entry[5], U_entry[6], U_entry[7]};

    // Minimum velocity magnitude for the rootfinder, in covariant notation.
    const Real vel_min[3] = {1.e-8 / m::sqrt(G.gcov(Loci::center, j, i, 1, 1)),
        1.e-8 / m::sqrt(G.gcov(Loci::center, j, i, 2, 2)),
        1.e-8 / m::sqrt(G.gcov(Loci::center, j, i, 3, 3))};

    // We are constructing a lambda evaluation function within this function. That will basically get rid of redundant code
    // and since this will not be called from any other function, I think it's fine.
    // From the current guess, compute the residuals and the error metrics err and errm.
    auto eval = [&](const Real xx[4], Real f[4], Real& err, Real& errm, Real& rho,
                    Real P_rad[4], Real dS[4], Real& Gu)
    {
        // From the guess primitives, compute the conserved variables.
        Real uvec[NVEC] = {xx[1], xx[2], xx[3]};
        rho = rho_from_D(G, D_rho, uvec, k, j, i);
        Real rho_ut, U_mhd[4], U_rad[4];
        GRMHD::p_to_u_mhd(G, rho, xx[0], uvec, B_P, eos, k, j, i, rho_ut, U_mhd);

        // do delta R = delta U
        for (int n = 0; n < 4; n++) U_rad[n] = U_rad_0[n] - (U_mhd[n] - U_mhd_0[n]);
        if (u_to_p_rad(G, U_rad, P_rad, k, j, i) != StatusRadiationInversion::success)
            return false;

        compute_covariant_fourforce(G, xx, P_rad, rho, B_P, eos, rad_opac, k, j, i, dS);
        Real ucon[GR_DIM];
        GRMHD::calc_ucon(G, uvec, k, j, i, Loci::center, ucon);
        Gu = 0.;
        for (int n = 0; n < 4; n++) {
            dS[n] *= gdet;
            // Project Gu onto the fluid frame.
            Gu += dS[n] * ucon[n];
        }

        const Real heat = dt / ucon[0] * Gu / gdet;
        // This comes from the first law of thermodynamics
        // Basically du/dtau = −(u+p) divu + G^mu u_mu
        f[0] = xx[0] - u_star - heat;
        errm = 0.;
        for (int n = 1; n < 4; n++) {
            // The other residuals are just the same...
            f[n] = U_mhd[n] - U_mhd_0[n] + dt * dS[n];
            errm =
                m::max(errm, m::abs(f[n]) / (m::abs(U_mhd[0]) + m::abs(U_mhd[n]) +
                                                m::abs(U_mhd_0[n]) + m::abs(dt * dS[n])));
        }
        err =
            m::max(errm, m::abs(f[0]) / (m::abs(xx[0]) + m::abs(u_star) + m::abs(heat)));
        return m::isfinite(err);
    };

    Real f[4], err, errm, rho, P_rad[4], dS[4], Gu;
    // if the first evaluation already produces a non-finite error, we return failure. This is a check for the initial guess.
    if (!eval(x, f, err, errm, rho, P_rad, dS, Gu))
        { if (why_out != nullptr) *why_out = 2; return static_cast<int>(StatusImplicitStep::failure); }
    bool converged = err <= src_rootfind_tol;

    // If you choose to try equilibrium guess first, 
    // It gives the FF solver a second starting point before Newton begins, and keeps whichever start is better
    // Given the energy of the plasma and the energy of the radiation, it computes the equilibrium T for optically thick flows.
    if (eq_guess && !converged) {
        Real xe[4] = {
            u_equilibrium(G, x, P_rad, rho, eos, rad_opac, k, j, i), x[1], x[2], x[3]};
        Real fe[4], ee, eme, re, Pe[4], dSe[4], Ge;
        if (xe[0] > 0. && eval(xe, fe, ee, eme, re, Pe, dSe, Ge) && ee < err) {
            for (int n = 0; n < 4; n++) {
                x[n] = xe[n];
                f[n] = fe[n];
                P_rad[n] = Pe[n];
                dS[n] = dSe[n];
            }
            err = ee;
            errm = eme;
            rho = re;
            Gu = Ge;
            converged = err <= src_rootfind_tol;
        }
    }
    // Best valid state so far. If the solve then fails but this reached accept_tol, it is
    // accepted (relaxed_success), with the exchange computed as for a converged solve
    Real x_best[4] = {x[0], x[1], x[2], x[3]}, rho_best = rho, Gu_best = Gu, err_best = err;

    // Lambda function to compare states and keep the best one.
    auto keep_best = [&]()
    {
        if (!(err < err_best)) return;
        for (int n = 0; n < 4; n++) x_best[n] = x[n];
        rho_best = rho;
        Gu_best = Gu;
        err_best = err;
    };
    // Exit of a failed solve that had a valid state, it decides if that state is accepted if it reaches accept_tol
    auto fail_or_accept = [&]() -> int
    {
        if (!(err_best <= accept_tol)) return static_cast<int>(StatusImplicitStep::failure);
        Real uvec_b[NVEC] = {x_best[1], x_best[2], x_best[3]}, rho_ut_b, U_mhd_b[4];
        GRMHD::p_to_u_mhd(
            G, rho_best, x_best[0], uvec_b, B_P, eos, k, j, i, rho_ut_b, U_mhd_b);
        for (int n = 0; n < 4; n++) dS_final[n] = -(U_mhd_b[n] - U_mhd_0[n]) / dt;
        dS_final[4] = Gu_best;
        return static_cast<int>(StatusImplicitStep::relaxed_success);
    };


    // Here we do a bisection in ug. so basically, If Newton's ug would leave the interval [u_lo, u_hi], just the u part of the step is
    // replaced by bisection. The velocity part keeps its Newton value.
    Real u_lo = (f[0] < 0.) ? x[0] : 0., u_hi = (f[0] >= 0.) ? x[0] : 0.;
    for (int nb = 0; nb < 30 && !converged && !(u_lo > 0. && u_hi > 0.); nb++) {
        Real xb[4] = {(u_hi > 0.) ? 0.1 * u_hi : 10. * u_lo, x[1], x[2], x[3]};
        Real fb[4], eb, emb, rb, Pb[4], dSb[4], Gb;
        if (!eval(xb, fb, eb, emb, rb, Pb, dSb, Gb))
            { if (why_out != nullptr) *why_out = 3; return fail_or_accept(); }
        if (fb[0] >= 0.)
            u_hi = xb[0];
        else
            u_lo = xb[0];
    }
    // if the bracket did not converge after 30 tries, or if it is not positive, the solve fails.
    if (!converged && !(u_lo > 0. && u_hi > 0.))
        { if (why_out != nullptr) *why_out = 4; return fail_or_accept(); }

    // Restart from the bracket end nearest u*, so damped steps stay inside the bracket
    if (!converged && (x[0] < u_lo || x[0] > u_hi)) {
        x[0] = (x[0] > u_hi) ? u_hi : u_lo;
        if (!eval(x, f, err, errm, rho, P_rad, dS, Gu))
            { if (why_out != nullptr) *why_out = 5; return fail_or_accept(); }
        converged = err <= src_rootfind_tol;
        keep_best();
    }

    bool bisect = false;
    for (int niter = 0; niter < src_rootfind_maxiter && !converged; niter++) {
        if (niter_out != nullptr) ++*niter_out;
        // Forward differences, flipped to backward if the perturbed state is invalid
        // same thing as PMHD
        Real jac[4][4], jacinv[4][4];
        for (int m = 0; m < 4; m++) {
            const Real del =
                (m == 0) ? 1.e-6 * x[0] : 1.e-6 * m::max(vel_min[m - 1], m::abs(x[m]));
            bool ok = false;
            Real fp[4], ep, emp, rp, Pp[4], dSp[4], Gp;
            for (int side = 0; side < 2 && !ok; side++) {
                Real xp[4] = {x[0], x[1], x[2], x[3]};
                xp[m] += (side == 0) ? del : -del;
                ok = eval(xp, fp, ep, emp, rp, Pp, dSp, Gp);
                if (ok)
                    for (int n = 0; n < 4; n++)
                        jac[n][m] = (fp[n] - f[n]) / (xp[m] - x[m]);
            }
            if (!ok) { if (why_out != nullptr) *why_out = 6; return fail_or_accept(); }
        }
        const Real det = invert(&jac[0][0], &jacinv[0][0]);
        if (!m::isfinite(det) || det == 0.)
            { if (why_out != nullptr) *why_out = 7; return fail_or_accept(); }
        Real dx[4];
        for (int m = 0; m < 4; m++) {
            dx[m] = 0.;
            for (int n = 0; n < 4; n++) dx[m] -= jacinv[m][n] * f[n];
        }
        // The u bracket was found at earlier velocities, and moving them moves the root of
        // f0(u). If Newton leaves the bracket, check that f0 still changes sign across it
        // at the current velocities; if not, rebuild it around u by x10 steps, as at the
        // start, before deciding to bisect. This has shown some improvements in the runs I tested in Aurora.
        if (!(x[0] + dx[0] > u_lo && x[0] + dx[0] < u_hi)) {
            Real fb[4], eb, emb, rb, Pb[4], dSb[4], Gb;
            const Real xlo[4] = {u_lo, x[1], x[2], x[3]};
            const Real xhi[4] = {u_hi, x[1], x[2], x[3]};
            bool stale = false;
            if (eval(xlo, fb, eb, emb, rb, Pb, dSb, Gb)) {
                const Real f0_lo = fb[0];
                if (eval(xhi, fb, eb, emb, rb, Pb, dSb, Gb))
                    stale = !(f0_lo < 0. && fb[0] >= 0.);
            }
            // If the bracket is stale, we need to rebuild it.
            if (stale) {
                Real lo = (f[0] < 0.) ? x[0] : 0., hi = (f[0] >= 0.) ? x[0] : 0.;
                for (int nb = 0; nb < 30 && !(lo > 0. && hi > 0.); nb++) {
                    Real xb[4] = {(hi > 0.) ? 0.1 * hi : 10. * lo, x[1], x[2], x[3]};
                    if (!eval(xb, fb, eb, emb, rb, Pb, dSb, Gb)) break;
                    if (fb[0] >= 0.)
                        hi = xb[0];
                    else
                        lo = xb[0];
                }
                if (lo > 0. && hi > 0.) {
                    u_lo = lo;
                    u_hi = hi;
                }
            }
        }
        // Geometric bisection if Newton leaves the bracket or did not halve |f0|
        if (bisect || !(x[0] + dx[0] > u_lo && x[0] + dx[0] < u_hi))
            dx[0] = m::sqrt(u_lo * u_hi) - x[0];
        // Damp by 3 until T_gas and E_rad each change by at most x10
        Real scale = 1., xn[4], fn[4], errn, errmn, rhon, P_radn[4], dSn[4], Gun;
        bool step_ok = false;
        for (int track = 0; track <= 15 && !step_ok; track++, scale /= 3.) {
            for (int m = 0; m < 4; m++) xn[m] = x[m] + scale * dx[m];
            if (!eval(xn, fn, errn, errmn, rhon, P_radn, dSn, Gun)) continue;
            const Real Tr = (xn[0] / rhon) / (x[0] / rho);
            const Real Er = P_radn[0] / P_rad[0];
            step_ok = Tr > 0.1 && Tr < 10. && Er > 0.1 && Er < 10.;
        }
        if (!step_ok) { if (why_out != nullptr) *why_out = 8; return fail_or_accept(); }
        if (fn[0] >= 0.)
            u_hi = m::min(u_hi, xn[0]);
        else
            u_lo = m::max(u_lo, xn[0]);
        bisect = m::abs(fn[0]) > 0.5 * m::abs(f[0]);
        Real relchange = m::abs(xn[0] - x[0]) / x[0];
        for (int m = 1; m < 4; m++)
            relchange = m::max(
                relchange, m::abs(xn[m] - x[m]) / m::max(vel_min[m - 1], m::abs(x[m])));
        for (int m = 0; m < 4; m++) {
            x[m] = xn[m];
            f[m] = fn[m];
            P_rad[m] = P_radn[m];
            dS[m] = dSn[m];
        }
        err = errn;
        errm = errmn;
        rho = rhon;
        Gu = Gun;
        keep_best();
        // Converged, stagnated with a small residual, or the u bracket has collapsed
        converged = err <= src_rootfind_tol || (relchange < 1.e-6 && err <= 1.e-5) ||
                    (u_hi - u_lo <= 1.e-10 * u_hi && errm <= src_rootfind_tol);
    }
    if (!converged) { if (why_out != nullptr) *why_out = 9; return fail_or_accept(); }

    // Exchange exactly the solved change, so the updated gas is the solved state
    Real uvec[NVEC] = {x[1], x[2], x[3]}, rho_ut, U_mhd[4];
    GRMHD::p_to_u_mhd(G, rho, x[0], uvec, B_P, eos, k, j, i, rho_ut, U_mhd);
    for (int n = 0; n < 4; n++) dS_final[n] = -(U_mhd[n] - U_mhd_0[n]) / dt;
    dS_final[4] = Gu;
    return static_cast<int>(StatusImplicitStep::success);
}

KOKKOS_INLINE_FUNCTION int prad_comoving_energy_residual(const GRCoordinates& G,
    const VariablePack<Real> U_init, const VariablePack<Real> P_init, const VarMap m_p,
    const VarMap m_u, const int k, const int j, const int i,
    const Microphysics::EOS::EOS& eos, const RadOpac& rad_opac, const Real B_P[NVEC],
    const Real dt, const Real gdet, const Real U_mhd_0[4], const Real U_rad_0[4],
    Real P_rad[4], Real& f, Real& scale)
{
    for (int it = 0; it < 40; it++) {
        Real U_rad[4];
        RadM1::calc_tensor(G, P_rad, 0, j, i, U_rad);
        for (int n = 0; n < 4; n++) U_rad[n] *= gdet;
        U_init(m_u.UU, k, j, i) = U_mhd_0[0] - (U_rad[0] - U_rad_0[0]);
        U_init(m_u.U1, k, j, i) = U_mhd_0[1] - (U_rad[1] - U_rad_0[1]);
        U_init(m_u.U2, k, j, i) = U_mhd_0[2] - (U_rad[2] - U_rad_0[2]);
        U_init(m_u.U3, k, j, i) = U_mhd_0[3] - (U_rad[3] - U_rad_0[3]);
        const int status = Inverter::u_to_p<Inverter::Type::kastaun>(
            G, U_init, m_u, eos, k, j, i, P_init, m_p, Loci::center, 25, 1e-12);
        if (status != static_cast<int>(Inverter::Status::success)) return status;
        const Real rho_g = P_init(m_p.RHO, k, j, i);
        Real P_mhd[4] = {P_init(m_p.UU, k, j, i), P_init(m_p.U1, k, j, i),
            P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};
        Real uvec_g[NVEC] = {P_mhd[1], P_mhd[2], P_mhd[3]};
        FourVectors D_g;
        GRMHD::calc_4vecs(G, uvec_g, B_P, k, j, i, Loci::center, D_g);
        const Real w_g = rho_g + P_mhd[0] +
                         eos.PressureFromDensityInternalEnergy(rho_g, P_mhd[0] / rho_g) +
                         dot(D_g.bcon, D_g.bcov);
        const Real lw_g = w_g * GRMHD::lorentz_calc(G, uvec_g, k, j, i, Loci::center);
        const Real lw_r =
            (4.0 / 3.0) * P_rad[0] * RadM1::lorentz_calc_rad(G, P_rad, j, i);
        Real uvec_new[NVEC];
        Real dv = 0.0, vn = 1.0;
        for (int v = 0; v < NVEC; v++) {
            uvec_new[v] = (lw_g * uvec_g[v] + lw_r * P_rad[v + 1]) / (lw_g + lw_r);
            dv = m::max(dv, m::abs(uvec_new[v] - P_rad[v + 1]));
            vn = m::max(vn, m::abs(uvec_new[v]));
        }
        if (dv <= 1.e-13 * vn) {
            Real dS[4], coupling;
            compute_covariant_fourforce(G, P_mhd, P_rad, rho_g, B_P, eos, rad_opac, k, j,
                i, dS, nullptr, &coupling);
            Real ucon_g[GR_DIM];
            GRMHD::calc_ucon(G, uvec_g, k, j, i, Loci::center, ucon_g);
            const Real src = dt * gdet * coupling;
            f = src;
            scale = m::abs(src);
            for (int mu = 0; mu < GR_DIM; mu++) {
                f += ucon_g[mu] * (U_rad[mu] - U_rad_0[mu]);
                scale +=
                    m::abs(ucon_g[mu] * U_rad[mu]) + m::abs(ucon_g[mu] * U_rad_0[mu]);
            }
            return status;
        }
        for (int v = 0; v < NVEC; v++) P_rad[v + 1] = uvec_new[v];
    }
    return static_cast<int>(Inverter::Status::max_iter);
}

KOKKOS_INLINE_FUNCTION int prad_residual_at(const GRCoordinates& G,
    const VariablePack<Real> U_init, const VariablePack<Real> P_init, const VarMap m_p,
    const VarMap m_u, const int k, const int j, const int i,
    const Microphysics::EOS::EOS& eos, const RadOpac& rad_opac, const Real B_P[NVEC],
    const Real dt, const Real gdet, const Real U_mhd_0[4], const Real U_rad_0[4],
    const Real P_rad[4], Real resid[4], Real U_rad[4], Real dtdS[4], Real P_mhd[4],
    Real dS[4])
{
    RadM1::calc_tensor(G, P_rad, 0, j, i, U_rad);
    for (int n = 0; n < 4; n++) U_rad[n] *= gdet;
    U_init(m_u.UU, k, j, i) = U_mhd_0[0] - (U_rad[0] - U_rad_0[0]);
    U_init(m_u.U1, k, j, i) = U_mhd_0[1] - (U_rad[1] - U_rad_0[1]);
    U_init(m_u.U2, k, j, i) = U_mhd_0[2] - (U_rad[2] - U_rad_0[2]);
    U_init(m_u.U3, k, j, i) = U_mhd_0[3] - (U_rad[3] - U_rad_0[3]);
    const int status = Inverter::u_to_p<Inverter::Type::kastaun>(
        G, U_init, m_u, eos, k, j, i, P_init, m_p, Loci::center, 25, 1e-12);
    if (status != static_cast<int>(Inverter::Status::success)) return status;
    P_mhd[0] = P_init(m_p.UU, k, j, i);
    P_mhd[1] = P_init(m_p.U1, k, j, i);
    P_mhd[2] = P_init(m_p.U2, k, j, i);
    P_mhd[3] = P_init(m_p.U3, k, j, i);
    compute_covariant_fourforce(
        G, P_mhd, P_rad, P_init(m_p.RHO, k, j, i), B_P, eos, rad_opac, k, j, i, dS);
    for (int n = 0; n < 4; n++) {
        dS[n] *= gdet;
        dtdS[n] = dt * dS[n];
        resid[n] = U_rad[n] - U_rad_0[n] - dtdS[n];
    }
    return status;
}

// Residual at a Jacobian support point of solve_4d_prad; false if the gas inversion fails
KOKKOS_INLINE_FUNCTION bool prad_fd_point(const GRCoordinates& G,
    const VariablePack<Real>& U_init, const VariablePack<Real>& P_init, const VarMap m_p,
    const VarMap m_u, const int k, const int j, const int i,
    const Microphysics::EOS::EOS& eos, const RadOpac& rad_opac, const Real B_P[NVEC],
    const Real dt, const Real gdet, const Real U_mhd_0[4], const Real U_rad_0[4],
    const Real P_rad[4], Real f[4])
{
    Real U_rad[4];
    RadM1::calc_tensor(G, P_rad, 0, j, i, U_rad);
    for (int n = 0; n < 4; n++) U_rad[n] *= gdet;
    U_init(m_u.UU, k, j, i) = U_mhd_0[0] - (U_rad[0] - U_rad_0[0]);
    U_init(m_u.U1, k, j, i) = U_mhd_0[1] - (U_rad[1] - U_rad_0[1]);
    U_init(m_u.U2, k, j, i) = U_mhd_0[2] - (U_rad[2] - U_rad_0[2]);
    U_init(m_u.U3, k, j, i) = U_mhd_0[3] - (U_rad[3] - U_rad_0[3]);
    if (Inverter::u_to_p<Inverter::Type::kastaun>(G, U_init, m_u, eos, k, j, i, P_init,
            m_p, Loci::center, 25, 1e-12) != static_cast<int>(Inverter::Status::success))
        return false;
    const Real P_mhd[4] = {P_init(m_p.UU, k, j, i), P_init(m_p.U1, k, j, i),
        P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};
    Real dS[4];
    compute_covariant_fourforce(
        G, P_mhd, P_rad, P_init(m_p.RHO, k, j, i), B_P, eos, rad_opac, k, j, i, dS);
    for (int n = 0; n < 4; n++) f[n] = U_rad[n] - U_rad_0[n] - dt * (gdet * dS[n]);
    return true;
}

KOKKOS_INLINE_FUNCTION int solve_4d_prad(const GRCoordinates& G,
    const VariablePack<Real>& U_init, const VariablePack<Real>& P_init, const VarMap m_p,
    const VarMap m_u, const int k, const int j, const int i, const Real dt,
    const Microphysics::EOS::EOS& eos, const double src_rootfind_eps,
    const double src_rootfind_tol, const int src_rootfind_maxiter,
    const RadOpac& rad_opac, const VariablePack<Real>& pflag,
    const VariablePack<Real>& rinvflag, const Real U_entry[8], Real dS_final[5],
    int* niter_out = nullptr)
{
    // Read the entry state from U_entry instead of copying it (fewer live registers)
    const Real* U_mhd_0 = U_entry;
    const Real* U_rad_0 = U_entry + 4;

    const Real rho_init = P_init(m_p.RHO, k, j, i);

    Real B_P[NVEC] = {0.};
    if (m_p.B1 >= 0) {
        B_P[V1] = P_init(m_p.B1, k, j, i);
        B_P[V2] = P_init(m_p.B2, k, j, i);
        B_P[V3] = P_init(m_p.B3, k, j, i);
    }
    const Real gdet = G.gdet(Loci::center, j, i);

    // State carried into the Newton loop
    Real P_rad_guess[4], U_rad_guess[4], dS_guess[4], resid[4];
    Real err, tol_eff;
    // Gas internal energy of the current iterate (FD step limit)
    Real u_gas_guess;

    // Initial guess. Everything declared in here is dead before the Newton loop.
    {
        Real P_mhd_guess[4] = {P_init(m_p.UU, k, j, i), P_init(m_p.U1, k, j, i),
            P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};
        for (int n = 0; n < 4; n++) U_rad_guess[n] = U_rad_0[n];

        // Convert the newly guessed U_rad to P_rad
        // This will determine which closure branch the inversion took
        bool used_normal_guess = true;
        const auto rad_status_entry =
            u_to_p_rad(G, U_rad_guess, P_rad_guess, k, j, i, &used_normal_guess);
        Real kappa_tot_entry = 0.0;
        compute_covariant_fourforce(G, P_mhd_guess, P_rad_guess, rho_init, B_P, eos,
            rad_opac, k, j, i, dS_guess, &kappa_tot_entry);
        tol_eff = m::max(static_cast<Real>(src_rootfind_tol),
            10.0 * std::numeric_limits<Real>::epsilon() * kappa_tot_entry * dt);

        bool comoving_guess = false;
        if (kappa_tot_entry * dt > 1.0 ||
            rad_status_entry != StatusRadiationInversion::success) {
            Real uvec_gas[NVEC] = {P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};
            Real ucon_gas[GR_DIM], ucov_gas[GR_DIM];
            GRMHD::calc_ucon(G, uvec_gas, k, j, i, Loci::center, ucon_gas);
            G.lower(ucon_gas, ucov_gas, k, j, i, Loci::center);
            Real ucon_rad_in[GR_DIM];
            RadM1::calc_ucon_rad(G, P_rad_guess, j, i, ucon_rad_in);
            FourVectors D_gas;
            GRMHD::calc_4vecs(G, uvec_gas, B_P, k, j, i, Loci::center, D_gas);
            const Real w_gas = rho_init + P_mhd_guess[0] +
                               eos.PressureFromDensityInternalEnergy(
                                   rho_init, P_mhd_guess[0] / rho_init) +
                               dot(D_gas.bcon, D_gas.bcov);
            const Real lw_gas =
                w_gas * GRMHD::lorentz_calc(G, uvec_gas, k, j, i, Loci::center);
            const Real lw_rad = (4.0 / 3.0) * P_rad_guess[0] *
                                RadM1::lorentz_calc_rad(G, P_rad_guess, j, i);
            Real uvec_com[NVEC];
            for (int v = 0; v < NVEC; v++)
                uvec_com[v] = (lw_gas * uvec_gas[v] + lw_rad * P_rad_guess[v + 1]) /
                              (lw_gas + lw_rad);
            Real ucon_com[GR_DIM], ucov_com[GR_DIM];
            GRMHD::calc_ucon(G, uvec_com, k, j, i, Loci::center, ucon_com);
            G.lower(ucon_com, ucov_com, k, j, i, Loci::center);
            Real gamma_rel_com = 0.0;
            for (int mu = 0; mu < GR_DIM; mu++)
                gamma_rel_com -= ucon_rad_in[mu] * ucov_com[mu];
            const Real Erf_com =
                P_rad_guess[0] *
                ((4.0 / 3.0) * gamma_rel_com * gamma_rel_com - 1.0 / 3.0);
            const Real Erf_gas =
                U_rad_0[0] / gdet / ((4.0 / 3.0) * ucon_gas[0] * ucov_gas[0] + 1.0 / 3.0);
            const Real candidates[3][4] = {
                {Erf_com, uvec_com[0], uvec_com[1], uvec_com[2]},
                {Erf_gas, uvec_gas[0], uvec_gas[1], uvec_gas[2]},
                {P_init(m_p.UU_RAD, k, j, i), P_init(m_p.U1_RAD, k, j, i),
                    P_init(m_p.U2_RAD, k, j, i), P_init(m_p.U3_RAD, k, j, i)}};
            const Real P_save[5] = {P_init(m_p.RHO, k, j, i), P_init(m_p.UU, k, j, i),
                P_init(m_p.U1, k, j, i), P_init(m_p.U2, k, j, i),
                P_init(m_p.U3, k, j, i)};
            const Real U_save[4] = {U_init(m_u.UU, k, j, i), U_init(m_u.U1, k, j, i),
                U_init(m_u.U2, k, j, i), U_init(m_u.U3, k, j, i)};
            for (int c = 0; c < 3; c++) {
                if (!(candidates[c][0] > 0.0) || !m::isfinite(candidates[c][0])) continue;
                Real P_rad_c[4] = {candidates[c][0], candidates[c][1], candidates[c][2],
                    candidates[c][3]};
                Real U_rad_c[4], U_mhd_c[4];
                RadM1::calc_tensor(G, P_rad_c, 0, j, i, U_rad_c);
                for (int n = 0; n < 4; n++) {
                    U_rad_c[n] *= gdet;
                    U_mhd_c[n] = U_mhd_0[n] - (U_rad_c[n] - U_rad_0[n]);
                }
                U_init(m_u.UU, k, j, i) = U_mhd_c[0];
                U_init(m_u.U1, k, j, i) = U_mhd_c[1];
                U_init(m_u.U2, k, j, i) = U_mhd_c[2];
                U_init(m_u.U3, k, j, i) = U_mhd_c[3];
                auto status_c = Inverter::u_to_p<Inverter::Type::kastaun>(
                    G, U_init, m_u, eos, k, j, i, P_init, m_p, Loci::center, 25, 1e-12);
                if (status_c == static_cast<int>(Inverter::Status::success)) {
                    for (int n = 0; n < 4; n++) {
                        P_rad_guess[n] = P_rad_c[n];
                        U_rad_guess[n] = U_rad_c[n];
                    }
                    P_mhd_guess[0] = P_init(m_p.UU, k, j, i);
                    P_mhd_guess[1] = P_init(m_p.U1, k, j, i);
                    P_mhd_guess[2] = P_init(m_p.U2, k, j, i);
                    P_mhd_guess[3] = P_init(m_p.U3, k, j, i);
                    compute_covariant_fourforce(G, P_mhd_guess, P_rad_guess,
                        P_init(m_p.RHO, k, j, i), B_P, eos, rad_opac, k, j, i, dS_guess);
                    comoving_guess = true;
                    break;
                }
                P_init(m_p.RHO, k, j, i) = P_save[0];
                P_init(m_p.UU, k, j, i) = P_save[1];
                P_init(m_p.U1, k, j, i) = P_save[2];
                P_init(m_p.U2, k, j, i) = P_save[3];
                P_init(m_p.U3, k, j, i) = P_save[4];
                U_init(m_u.UU, k, j, i) = U_save[0];
                U_init(m_u.U1, k, j, i) = U_save[1];
                U_init(m_u.U2, k, j, i) = U_save[2];
                U_init(m_u.U3, k, j, i) = U_save[3];
            }
        }
        for (int n = 0; n < 4; n++) dS_guess[n] = gdet * dS_guess[n];

        DLOOP1 {
            resid[mu] = U_rad_guess[mu] - U_rad_0[mu] - dt * dS_guess[mu];
        }

        // Compute err here and do a convergence check
        {
            Real dtdS_0[4];
            for (int n = 0; n < 4; n++) dtdS_0[n] = dt * dS_guess[n];
            err = calculate_error(resid, U_rad_guess, U_rad_0, dtdS_0);
        }

        // Here, if the comoving guess was used (in case of very optically thick gas) and
        // err is still large, we can make a better gas by trying to improve the initial
        // guess, this is done by checking the sign of the energy transfer, and
        // increase/decrease radiation and gas energy's initial guess. The search works in
        // s = ln (E). We do a bracket method.
        if (comoving_guess && err > 1.e-3) {

            // We save the gas state because this will overwrite P_init/U_init while
            // testing.
            const Real P_keep[5] = {P_init(m_p.RHO, k, j, i), P_init(m_p.UU, k, j, i),
                P_init(m_p.U1, k, j, i), P_init(m_p.U2, k, j, i),
                P_init(m_p.U3, k, j, i)};
            const Real U_keep[4] = {U_init(m_u.UU, k, j, i), U_init(m_u.U1, k, j, i),
                U_init(m_u.U2, k, j, i), U_init(m_u.U3, k, j, i)};

            constexpr Real LN10 = 2.302585092994046;
            const int ok = static_cast<int>(Inverter::Status::success);
            Real Pr[4] = {P_rad_guess[0], P_rad_guess[1], P_rad_guess[2], P_rad_guess[3]};
            Real v_last[3] = {Pr[1], Pr[2], Pr[3]};
            Real f_a, sc;
            Real s_a = m::log(P_rad_guess[0]);
            bool have_bracket = false;
            Real s_lo = 0., s_hi = 0., f_lo = 0., f_hi = 0.;

            // Bracket search branch. the sign of f at the current E says whether E should
            // go up or down. Step E by x10 (or /10) until f changes sign. If the gas
            // inversion fails, decrease the step. v_last keeps the last converged
            // velocity.
            if (prad_comoving_energy_residual(G, U_init, P_init, m_p, m_u, k, j, i, eos,
                    rad_opac, B_P, dt, gdet, U_mhd_0, U_rad_0, Pr, f_a, sc) == ok) {
                for (int v = 0; v < 3; v++) v_last[v] = Pr[v + 1];
                const Real dir = (f_a > 0.0) ? 1.0 : -1.0;
                Real step = LN10;
                for (int b = 0; b < 60 && !have_bracket; b++) {
                    const Real s_b = s_a + dir * step;
                    Pr[0] = m::exp(s_b);
                    for (int v = 0; v < 3; v++) Pr[v + 1] = v_last[v];
                    Real f_b;
                    const int st_b = prad_comoving_energy_residual(G, U_init, P_init, m_p,
                        m_u, k, j, i, eos, rad_opac, B_P, dt, gdet, U_mhd_0, U_rad_0, Pr,
                        f_b, sc);
                    if (st_b != ok) {
                        step *= 0.5;
                        if (step < 1.e-6) break;
                        continue;
                    }
                    for (int v = 0; v < 3; v++) v_last[v] = Pr[v + 1];
                    if (f_b * f_a <= 0.0) {
                        have_bracket = true;
                        if (s_a < s_b) {
                            s_lo = s_a;
                            f_lo = f_a;
                            s_hi = s_b;
                            f_hi = f_b;
                        } else {
                            s_lo = s_b;
                            f_lo = f_b;
                            s_hi = s_a;
                            f_hi = f_a;
                        }
                    } else {
                        s_a = s_b;
                        f_a = f_b;
                    }
                }
            }

            // Once we find the bracket, we can just approximate them by a line, with the
            // slope and point values, we can approximate which point it will cross E = 0.
            if (have_bracket) {
                Real s_c = s_lo;
                int side = 0;
                for (int it = 0; it < 60; it++) {
                    s_c = (s_lo * f_hi - s_hi * f_lo) / (f_hi - f_lo);
                    if (!(s_c > s_lo && s_c < s_hi)) s_c = 0.5 * (s_lo + s_hi);
                    Pr[0] = m::exp(s_c);
                    for (int v = 0; v < 3; v++) Pr[v + 1] = v_last[v];
                    Real f_c;
                    const int st_c = prad_comoving_energy_residual(G, U_init, P_init, m_p,
                        m_u, k, j, i, eos, rad_opac, B_P, dt, gdet, U_mhd_0, U_rad_0, Pr,
                        f_c, sc);
                    if (st_c != ok) {
                        s_hi = s_c;
                        f_hi = -m::abs(f_lo);
                        continue;
                    }
                    for (int v = 0; v < 3; v++) v_last[v] = Pr[v + 1];
                    if (m::abs(f_c) <= 1.e-3 * tol_eff * sc || s_hi - s_lo < 1.e-13)
                        break;
                    if (f_c * f_lo > 0.0) {
                        s_lo = s_c;
                        f_lo = f_c;
                        if (side == -1) f_hi *= 0.5;
                        side = -1;
                    } else {
                        s_hi = s_c;
                        f_hi = f_c;
                        if (side == 1) f_lo *= 0.5;
                        side = 1;
                    }
                }
                // Check the result in the full 4D problem, use it as Newton's starting
                // point only if the error is smaller than before.
                Real f_c;
                Pr[0] = m::exp(s_c);
                for (int v = 0; v < 3; v++) Pr[v + 1] = v_last[v];
                const int st_f =
                    prad_comoving_energy_residual(G, U_init, P_init, m_p, m_u, k, j, i,
                        eos, rad_opac, B_P, dt, gdet, U_mhd_0, U_rad_0, Pr, f_c, sc);
                if (st_f == ok) {
                    Real U_rad_c[4], U_mhd_c[4], dS_c[4], res_c[4], dtdS_c[4];
                    RadM1::calc_tensor(G, Pr, 0, j, i, U_rad_c);
                    for (int n = 0; n < 4; n++) {
                        U_rad_c[n] *= gdet;
                        U_mhd_c[n] = U_mhd_0[n] - (U_rad_c[n] - U_rad_0[n]);
                    }
                    U_init(m_u.UU, k, j, i) = U_mhd_c[0];
                    U_init(m_u.U1, k, j, i) = U_mhd_c[1];
                    U_init(m_u.U2, k, j, i) = U_mhd_c[2];
                    U_init(m_u.U3, k, j, i) = U_mhd_c[3];
                    const int st_c = Inverter::u_to_p<Inverter::Type::kastaun>(G, U_init,
                        m_u, eos, k, j, i, P_init, m_p, Loci::center, 25, 1e-12);
                    if (st_c == ok) {
                        Real P_mhd_c[4] = {P_init(m_p.UU, k, j, i),
                            P_init(m_p.U1, k, j, i), P_init(m_p.U2, k, j, i),
                            P_init(m_p.U3, k, j, i)};
                        compute_covariant_fourforce(G, P_mhd_c, Pr,
                            P_init(m_p.RHO, k, j, i), B_P, eos, rad_opac, k, j, i, dS_c);
                        for (int n = 0; n < 4; n++) {
                            dS_c[n] *= gdet;
                            dtdS_c[n] = dt * dS_c[n];
                            res_c[n] = U_rad_c[n] - U_rad_0[n] - dtdS_c[n];
                        }
                        const Real err_c =
                            calculate_error(res_c, U_rad_c, U_rad_0, dtdS_c);
                        if (err_c < err) {
                            for (int n = 0; n < 4; n++) {
                                P_rad_guess[n] = Pr[n];
                                U_rad_guess[n] = U_rad_c[n];
                                dS_guess[n] = dS_c[n];
                                resid[n] = res_c[n];
                                P_mhd_guess[n] = P_mhd_c[n];
                            }
                            err = err_c;
                        }
                    }
                }
            }
            // If the improved guess was not kept, restore the saved gas state
            if (P_init(m_p.UU, k, j, i) != P_mhd_guess[0] ||
                P_init(m_p.U1, k, j, i) != P_mhd_guess[1]) {
                P_init(m_p.RHO, k, j, i) = P_keep[0];
                P_init(m_p.UU, k, j, i) = P_keep[1];
                P_init(m_p.U1, k, j, i) = P_keep[2];
                P_init(m_p.U2, k, j, i) = P_keep[3];
                P_init(m_p.U3, k, j, i) = P_keep[4];
                U_init(m_u.UU, k, j, i) = U_keep[0];
                U_init(m_u.U1, k, j, i) = U_keep[1];
                U_init(m_u.U2, k, j, i) = U_keep[2];
                U_init(m_u.U3, k, j, i) = U_keep[3];
            }
        }
        u_gas_guess = P_mhd_guess[0];
    }

    Real best_err = err;
    Real best_P[4] = {P_rad_guess[0], P_rad_guess[1], P_rad_guess[2], P_rad_guess[3]};
    int niter = 0;
    bool bad_guess = false;

    do {
        if (err <= tol_eff) {
            break;
        }
        if (niter_out != nullptr) ++*niter_out;

        // FD Jacobian dU_rad/dP_rad: central differences, or one-sided from the guess
        // when one support point's gas inversion fails
        Real jac[4][4];
        {
            // Find minimum non-zero magnitude from P_rad_guess to scale FD step safely
            Real P_rad_mag_min = RAD_LARGE;
            for (int m = 0; m < 4; m++) {
                if (m::abs(P_rad_guess[m]) > 0.) {
                    P_rad_mag_min = m::min(P_rad_mag_min, m::abs(P_rad_guess[m]));
                }
            }
            for (int m = 0; m < 4 && !bad_guess; m++) {
                Real fd_step = m::max(src_rootfind_eps * P_rad_mag_min,
                    src_rootfind_eps * m::abs(P_rad_guess[m]));
                if (m == 0)
                    fd_step =
                        m::max(fd_step, src_rootfind_eps * m::abs(U_mhd_0[0]) / gdet);
                {
                    const Real gas_energy_margin = 0.1 * m::abs(u_gas_guess);
                    fd_step = m::min(fd_step, m::max(gas_energy_margin, RAD_SMALL));
                }
                Real P_rad_s[4] = {
                    P_rad_guess[0], P_rad_guess[1], P_rad_guess[2], P_rad_guess[3]};
                P_rad_s[m] -= fd_step;
                const Real p_m = P_rad_s[m];
                Real f_m[4];
                const bool ok_m = prad_fd_point(G, U_init, P_init, m_p, m_u, k, j, i, eos,
                    rad_opac, B_P, dt, gdet, U_mhd_0, U_rad_0, P_rad_s, f_m);
                P_rad_s[m] = P_rad_guess[m] + fd_step;
                const Real p_p = P_rad_s[m];
                Real f_p[4];
                const bool ok_p = prad_fd_point(G, U_init, P_init, m_p, m_u, k, j, i, eos,
                    rad_opac, B_P, dt, gdet, U_mhd_0, U_rad_0, P_rad_s, f_p);
                if (ok_m && ok_p) {
                    for (int n = 0; n < 4; n++)
                        jac[n][m] = (f_p[n] - f_m[n]) / (p_p - p_m);
                } else if (ok_p) {
                    for (int n = 0; n < 4; n++)
                        jac[n][m] = (f_p[n] - resid[n]) / (p_p - P_rad_guess[m]);
                } else if (ok_m) {
                    for (int n = 0; n < 4; n++)
                        jac[n][m] = (resid[n] - f_m[n]) / (P_rad_guess[m] - p_m);
                } else {
                    bad_guess = true;
                }
            }
        }
        if (bad_guess) break;

        // Newton direction J^-1 resid, so jacinv is dead before the step evaluations
        Real delta[4];
        {
            Real jacinv[4][4];
            invert(&jac[0][0], &jacinv[0][0]);
            for (int m = 0; m < 4; m++) {
                delta[m] = 0.;
                for (int n = 0; n < 4; n++) delta[m] += jacinv[m][n] * resid[n];
            }
        }

        // Update guess via a damped/backtracking Newton step: take the full step, and if
        // it violates either check below, shrink the step (Koral's fixed factor of 3)
        // and retry, up to 15 shrinks.
        Real P_rad_pre[4] = {
            P_rad_guess[0], P_rad_guess[1], P_rad_guess[2], P_rad_guess[3]};
        Real scaling_factor = 1.0;
        int track = 0;
        bool step_ok = false;
        do {
            for (int m = 0; m < 4; m++)
                P_rad_guess[m] = P_rad_pre[m] - scaling_factor * delta[m];

            if (P_rad_guess[0] <= 0.0) {
                scaling_factor /= 3.0;
                track++;
                if (track > 15 || scaling_factor < 1.e-8) break;
                continue;
            }

            RadM1::calc_tensor(G, P_rad_guess, 0, j, i, U_rad_guess);
            for (int n = 0; n < 4; n++) U_rad_guess[n] *= gdet;

            U_init(m_u.UU, k, j, i) = U_mhd_0[0] - (U_rad_guess[0] - U_rad_0[0]);
            U_init(m_u.U1, k, j, i) = U_mhd_0[1] - (U_rad_guess[1] - U_rad_0[1]);
            U_init(m_u.U2, k, j, i) = U_mhd_0[2] - (U_rad_guess[2] - U_rad_0[2]);
            U_init(m_u.U3, k, j, i) = U_mhd_0[3] - (U_rad_guess[3] - U_rad_0[3]);

            auto mhd_inverter_status = Inverter::u_to_p<Inverter::Type::kastaun>(
                G, U_init, m_u, eos, k, j, i, P_init, m_p, Loci::center, 25, 1e-12);

            if (mhd_inverter_status != static_cast<int>(Inverter::Status::success)) {
                scaling_factor /= 3.0;
                track++;
                if (track > 15 || scaling_factor < 1.e-8) break;
                continue;
            }

            step_ok = true;
        } while (!step_ok && track <= 15);

        if (track > 15 || scaling_factor < 1.e-8) {
            bad_guess = true;
            break;
        }

        {
            const Real P_mhd_guess[4] = {P_init(m_p.UU, k, j, i), P_init(m_p.U1, k, j, i),
                P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};
            u_gas_guess = P_mhd_guess[0];
            compute_covariant_fourforce(G, P_mhd_guess, P_rad_guess,
                P_init(m_p.RHO, k, j, i), B_P, eos, rad_opac, k, j, i, dS_guess);
        }

        for (int n = 0; n < 4; n++) dS_guess[n] = gdet * dS_guess[n];

        // Update residuals
        for (int n = 0; n < 4; n++) {
            resid[n] = U_rad_guess[n] - U_rad_0[n] - dt * dS_guess[n];

            if (!m::isfinite(resid[n])) {
                bad_guess = true;
                break;
            }
        }

        // This is needed since the previous bad_guess = true would only break out of the
        // for loop
        if (bad_guess) {
            break;
        }

        // Calculate error now
        {
            Real dtdS_iter[4];
            for (int n = 0; n < 4; n++) dtdS_iter[n] = dt * dS_guess[n];
            err = calculate_error(resid, U_rad_guess, U_rad_0, dtdS_iter);
        }

        niter++;
        if (err < best_err) {
            best_err = err;
            for (int n = 0; n < 4; n++) best_P[n] = P_rad_guess[n];
        }
    } while (err > tol_eff && niter < src_rootfind_maxiter);

    // U_mhd of the last iterate, from the conservation law
    bool failed = err > tol_eff || bad_guess;
    for (int n = 0; n < 4; n++)
        failed = failed || !m::isfinite(U_mhd_0[n] - (U_rad_guess[n] - U_rad_0[n]));

    // Newton failed, but its best attempt may already be as accurate as double
    // precision allows. Accept it if its error is no bigger than the round-off noise
    if (failed && best_err <= 1.e-4) {
        Real r0[4], ur0[4], dd0[4], pm0[4], ds0[4];
        if (prad_residual_at(G, U_init, P_init, m_p, m_u, k, j, i, eos, rad_opac, B_P, dt,
                gdet, U_mhd_0, U_rad_0, best_P, r0, ur0, dd0, pm0,
                ds0) == static_cast<int>(Inverter::Status::success)) {
            const Real err0 = calculate_error(r0, ur0, U_rad_0, dd0);

            // Measure the round-off noise, nudge the unknowns by 4 ULPs (one at a time,
            // all down, then alternating signs) and record how much the error changes
            const Real ee = 4.0 * std::numeric_limits<Real>::epsilon();
            Real noise = 0.0;
            for (int q = 0; q < 6 && err0 <= 1.e-4 && err0 > tol_eff; q++) {
                Real Pq[4] = {best_P[0], best_P[1], best_P[2], best_P[3]};
                for (int n = 0; n < 4; n++) {
                    if (q < 4)
                        Pq[n] *= (n == q) ? (1.0 + ee) : 1.0;
                    else
                        Pq[n] *= (q == 4) ? (1.0 - ee) : (1.0 + ((n % 2) ? ee : -ee));
                }
                Real r1[4], ur1[4], dd1[4], pm1[4], ds1[4], dr[4];
                if (prad_residual_at(G, U_init, P_init, m_p, m_u, k, j, i, eos, rad_opac,
                        B_P, dt, gdet, U_mhd_0, U_rad_0, Pq, r1, ur1, dd1, pm1,
                        ds1) != static_cast<int>(Inverter::Status::success))
                    continue;
                for (int n = 0; n < 4; n++) dr[n] = r1[n] - r0[n];
                noise = m::max(noise, calculate_error(dr, ur0, U_rad_0, dd0));
            }

            // Accept if the error is within tolerance or within 10x the noise. Evaluate
            // once more at best_P so U_init/P_init match the accepted state.
            if (err0 <= 1.e-4 && (err0 <= tol_eff || err0 <= 10.0 * noise) &&
                prad_residual_at(G, U_init, P_init, m_p, m_u, k, j, i, eos, rad_opac, B_P,
                    dt, gdet, U_mhd_0, U_rad_0, best_P, r0, ur0, dd0, pm0,
                    ds0) == static_cast<int>(Inverter::Status::success)) {
                // Use best attempt as the solution.
                for (int n = 0; n < 4; n++) {
                    U_rad_guess[n] = ur0[n];
                    dS_guess[n] = ds0[n];
                }
                failed = false;
            }
        }
    }

    if (failed) {

        return static_cast<int>(StatusImplicitStep::failure);
    }

    for (int n = 0; n < 4; n++) dS_final[n] = (U_rad_guess[n] - U_rad_0[n]) / dt;

    {
        // P_init holds the accepted gas state here
        Real uvec_final[NVEC] = {
            P_init(m_p.U1, k, j, i), P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};
        Real ucon_final[4];
        GRMHD::calc_ucon(G, uvec_final, k, j, i, Loci::center, ucon_final);
        Real Gdotu = dS_guess[0] * ucon_final[0] + dS_guess[1] * ucon_final[1] +
                     dS_guess[2] * ucon_final[2] + dS_guess[3] * ucon_final[3];
        dS_final[4] = Gdotu;
    }

    return static_cast<int>(StatusImplicitStep::success);
}

} // namespace RadM1
