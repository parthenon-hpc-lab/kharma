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

#include "inverter.hpp"


#define RAD_LARGE (0.1 * std::numeric_limits<Real>::max())
#define RAD_SMALL (10.0 * std::numeric_limits<Real>::min())
#define RAD_EPS (10.0 * std::numeric_limits<Real>::epsilon())

namespace RadM1
{

KOKKOS_INLINE_FUNCTION void ApplyColdClosureFix(const GRCoordinates& G,
    const Real R_t_cov_orig[GR_DIM], const double gammarel2_fixed, const int& j,
    const int& i, Real& new_R_t_t, Real& Erf)
{

    Real gcon_tt = G.gcon(Loci::center, j, i, 0, 0);

    // Time-Space cross term: g^{ti} R_i
    Real dot_t_i = G.gcon(Loci::center, j, i, 0, 1) * R_t_cov_orig[1] +
                   G.gcon(Loci::center, j, i, 0, 2) * R_t_cov_orig[2] +
                   G.gcon(Loci::center, j, i, 0, 3) * R_t_cov_orig[3];

    // Purely Spatial term: g^{ij} R_i R_j
    Real dot_spatial =
        G.gcon(Loci::center, j, i, 1, 1) * (R_t_cov_orig[1] * R_t_cov_orig[1]) +
        2.0 * G.gcon(Loci::center, j, i, 1, 2) * (R_t_cov_orig[1] * R_t_cov_orig[2]) +
        G.gcon(Loci::center, j, i, 2, 2) * (R_t_cov_orig[2] * R_t_cov_orig[2]) +
        2.0 * G.gcon(Loci::center, j, i, 1, 3) * (R_t_cov_orig[1] * R_t_cov_orig[3]) +
        2.0 * G.gcon(Loci::center, j, i, 2, 3) * (R_t_cov_orig[2] * R_t_cov_orig[3]) +
        G.gcon(Loci::center, j, i, 3, 3) * (R_t_cov_orig[3] * R_t_cov_orig[3]);

    Real utsq = -gammarel2_fixed * gcon_tt;

    Real radical_inside = (dot_t_i * dot_t_i - gcon_tt * dot_spatial) * utsq *
                          (gcon_tt + utsq) * m::pow(gcon_tt + 4.0 * utsq, 2);

    Real radical = m::sqrt(m::max(0.0, radical_inside));

    new_R_t_t = 0.25 * (-4.0 * dot_t_i * utsq * (gcon_tt + utsq) + radical) /
                (gcon_tt * utsq * (gcon_tt + utsq));

    Erf = 0.75 * radical / (utsq * (gcon_tt + utsq) * (gcon_tt + 4.0 * utsq));
}

KOKKOS_INLINE_FUNCTION double calculate_gamma_rel2(const GRCoordinates& G,
    const Real R_t_cov[GR_DIM], const int& j, const int& i)
{
    Real gcon_tt = G.gcon(Loci::center, j, i, 0, 0);
    Real R_t_t = R_t_cov[0];

    // Time-Space cross term: g^{ti} R^t_i
    Real dot_t_i = G.gcon(Loci::center, j, i, 0, 1) * R_t_cov[1] +
                   G.gcon(Loci::center, j, i, 0, 2) * R_t_cov[2] +
                   G.gcon(Loci::center, j, i, 0, 3) * R_t_cov[3];

    // Purely Spatial term: g^{ij} R^t_i R^t_j
    Real dot_spatial =
        G.gcon(Loci::center, j, i, 1, 1) * (R_t_cov[1] * R_t_cov[1]) +
        2.0 * G.gcon(Loci::center, j, i, 1, 2) * (R_t_cov[1] * R_t_cov[2]) +
        G.gcon(Loci::center, j, i, 2, 2) * (R_t_cov[2] * R_t_cov[2]) +
        2.0 * G.gcon(Loci::center, j, i, 1, 3) * (R_t_cov[1] * R_t_cov[3]) +
        2.0 * G.gcon(Loci::center, j, i, 2, 3) * (R_t_cov[2] * R_t_cov[3]) +
        G.gcon(Loci::center, j, i, 3, 3) * (R_t_cov[3] * R_t_cov[3]);


    Real invariant_scalar = gcon_tt * (R_t_t * R_t_t) + 2.0 * R_t_t * dot_t_i + dot_spatial;

    // Calculate Roots for (u^t_R)^2
    Real radical_inside = 4.0 * (gcon_tt * gcon_tt) * (R_t_t * R_t_t) +
                          (dot_t_i * dot_t_i) +
                          gcon_tt * (8.0 * R_t_t * dot_t_i + 3.0 * dot_spatial);
    Real radical = m::sqrt(m::max(0.0, radical_inside));

    Real num_a = 2.0 * (gcon_tt * gcon_tt) * (R_t_t * R_t_t) +
                 dot_t_i * (dot_t_i + radical) +
                 gcon_tt * (4.0 * R_t_t * dot_t_i + dot_spatial + R_t_t * radical);

    Real num_b = -2.0 * (gcon_tt * gcon_tt) * (R_t_t * R_t_t) -
                 gcon_tt * (4.0 * R_t_t * dot_t_i + dot_spatial) +
                 gcon_tt * R_t_t * radical + dot_t_i * (-dot_t_i + radical);

    Real gamma2a = -0.25 * num_a / invariant_scalar;
    Real gamma2b = 0.25 * num_b / invariant_scalar;

    // TODO (PNM): Check if the ill_conditoned boolean here is necessary.
    const Real natural_scale = m::abs(gcon_tt * R_t_t * R_t_t) +
                                m::abs(2.0 * R_t_t * dot_t_i) + m::abs(dot_spatial);
    const bool ill_conditioned =
        m::abs(invariant_scalar) < 1.e-3 * natural_scale;

    Real gamma2 = gamma2a;
    if (ill_conditioned || gamma2a < (1.0 - 1e-10) || m::isnan(gamma2a) || m::isinf(gamma2a)) {
        gamma2 = gamma2b;
    }

    Real alpha_sq = -1.0 / gcon_tt;
    Real gammarel2 = gamma2 * alpha_sq;

    // Hard floor for physical bounds (Lorentz factor squared MUST be >= 1.0)
    // TODO: FLOOR! CHANGE THIS
    Real GAMMA_SMALL_LIMIT = (1.0 - 1e-10);
    if (gammarel2 < 1.0 && gammarel2 > GAMMA_SMALL_LIMIT) {
        gammarel2 = 1.0;
    }

    return gammarel2;
}

KOKKOS_INLINE_FUNCTION StatusRadiationInversion u_to_p_rad(const GRCoordinates& G,
    const Real U_rad[4], Real P_rad[4], const int k, const int j, const int i)
{
    Real gdet = G.gdet(Loci::center, j, i);

    // Exact-zero radiation field: for testing cases. We don't wanna go down this entire path, otherwise, it will break due to division by 0.
    // Sometimes we might want to run with radM1 on, but no source term just to check if something is broken, it's nice to have the possibility to do so.
    if (U_rad[0] == 0.0 && U_rad[1] == 0.0 && U_rad[2] == 0.0 && U_rad[3] == 0.0) {
        P_rad[0] = 0.0;
        P_rad[1] = 0.0;
        P_rad[2] = 0.0;
        P_rad[3] = 0.0;
        return StatusRadiationInversion::success;
    }

    // Recover R^t_mu from conserved state
    Real R_t_cov[4] = {
        U_rad[0] / gdet, U_rad[1] / gdet, U_rad[2] / gdet, U_rad[3] / gdet};

    // Raise index to get R^{t\mu}
    Real R_t_con[4];
    G.raise(R_t_cov, R_t_con, k, j, i, Loci::center);

    // Calculate gamma^2 for the radiation frame
    Real gammarel2 = calculate_gamma_rel2(G, R_t_cov, j, i);

    // Pre-calculate alpha bounds and rest-frame energy E_rf
    Real alpha_sq = -1.0 / G.gcon(Loci::center, j, i, 0, 0);
    Real alpha = m::sqrt(alpha_sq);
    Real E_rf = (3.0 * R_t_con[0] * alpha_sq) / (4.0 * gammarel2 - 1.0);

    // Limits
    const Real min_erad = 1.e-30;
    const Real GAMMAMAX = 50.0;
    const Real GAMMA_TOL = 1.0 - 1e-10; // matches koral's GAMMASMALLLIMIT
    int flag1 = (gammarel2 >= 1.0);
    int flag2 = (E_rf > min_erad);
    int flag3 = (gammarel2 <= (GAMMAMAX * GAMMAMAX) / (GAMMA_TOL * GAMMA_TOL));

    int nonfailure = flag1 && flag2 && flag3;

    // Our primitives is saving uvec_rad as the eulerian frame velocity.
    // $\tilde{u}^i = \gamma v^i$
    Real uvec_radframe_con[4] = {0};

    // Evaluate valid primitives or apply cold closure fix
    bool used_normal = false;
    int flag4 = 1;
    if (nonfailure) {
        for (int mu = 0; mu < 4; ++mu) {
            uvec_radframe_con[mu] = alpha * (R_t_con[mu] + 1. / 3. * E_rf * G.gcon(Loci::center, j, i, 0, mu) * (4.0 * gammarel2 - 1.0)) / (4. / 3. * E_rf * m::sqrt(gammarel2));
        }

        // After calculating uvec_radframe_con, we can still have trouble with gamma > gammamax, so let's recheck
        Real qsq = G.gcov(Loci::center, j, i, 1, 1) * uvec_radframe_con[1] * uvec_radframe_con[1]
             + G.gcov(Loci::center, j, i, 2, 2) * uvec_radframe_con[2] * uvec_radframe_con[2]
             + G.gcov(Loci::center, j, i, 3, 3) * uvec_radframe_con[3] * uvec_radframe_con[3]
             + 2.0 * G.gcov(Loci::center, j, i, 1, 2) * uvec_radframe_con[1] * uvec_radframe_con[2]
             + 2.0 * G.gcov(Loci::center, j, i, 1, 3) * uvec_radframe_con[1] * uvec_radframe_con[3]
             + 2.0 * G.gcov(Loci::center, j, i, 2, 3) * uvec_radframe_con[2] * uvec_radframe_con[3];
        Real gammarel2_out = 1.0 + qsq;

        // self consistent is comparing the gammarel2 we obtained with the gmmarel2_out
        // They should match I think, but we are checking here 1e-6 is hard coded, we should get rid of this
        // TODO (PNM): Maybe get rid of this?
        const bool self_consistent =
            m::abs(gammarel2_out - gammarel2) <= 1.e-6 * (gammarel2_out + gammarel2);

        flag4 = gammarel2_out <= (GAMMAMAX * GAMMAMAX) / (GAMMA_TOL * GAMMA_TOL);
        used_normal = std::isfinite(E_rf) && std::isfinite(uvec_radframe_con[1]) && std::isfinite(uvec_radframe_con[2]) && std::isfinite(uvec_radframe_con[3])
        && flag4 && self_consistent;
    }

    if (!used_normal) {
        // Attempt Cold Closure
        //gammarel2_slow should definitely not be this, should be way smaller
        // TODO (PNM): See if this is necessary
        Real gammarel2_slow = m::pow(1.0 + 1.0e-4, 2.0);
        Real gammarel2_fast = GAMMAMAX * GAMMAMAX;

        Real R_t_t_slow, Erf_slow;
        ApplyColdClosureFix(G, R_t_cov, gammarel2_slow, j, i, R_t_t_slow, Erf_slow);

        Real R_t_t_fast, Erf_fast;
        ApplyColdClosureFix(G, R_t_cov, gammarel2_fast, j, i, R_t_t_fast, Erf_fast);

        Real R_t_t_new, gammarel2_new;

        if (m::abs(R_t_t_slow - R_t_cov[0]) > m::abs(R_t_t_fast - R_t_cov[0])) {
            R_t_t_new = R_t_t_fast;
            E_rf = Erf_fast;
            gammarel2_new = gammarel2_fast;
        } else {
            R_t_t_new = R_t_t_slow;
            E_rf = Erf_slow;
            gammarel2_new = gammarel2_slow;
        }

        // If even the closure fix yields a non-positive energy, this step is a failure.
        if (E_rf <= 0.0) {
            P_rad[0] = min_erad;
            P_rad[1] = 0.0;
            P_rad[2] = 0.0;
            P_rad[3] = 0.0;
            if (!flag1){
                return StatusRadiationInversion::gammarel2_low;
            }else if(!flag2){
                return StatusRadiationInversion::urad_below_floor;
            }else if(!flag3 || !flag4){
                return StatusRadiationInversion::gammarel2_high;
            }else{
                return StatusRadiationInversion::division_nonfinite;
            }
        }

        Real R_t_cov_new[4] = {R_t_t_new, R_t_cov[1], R_t_cov[2], R_t_cov[3]};
        Real R_t_con_new[4];
        G.raise(R_t_cov_new, R_t_con_new, k, j, i, Loci::center);

        for (int mu = 0; mu < 4; ++mu) {
            uvec_radframe_con[mu] = alpha * (R_t_con_new[mu] + 1. / 3. * E_rf * G.gcon(Loci::center, j, i, 0, mu) * (4.0 * gammarel2_new - 1.0)) / (4. / 3. * E_rf * m::sqrt(gammarel2_new));
        }
        

        // The cold-closure fallback also divides by E_rf, so check it too.
        if (!std::isfinite(E_rf) || !std::isfinite(uvec_radframe_con[1]) || !std::isfinite(uvec_radframe_con[2]) || !std::isfinite(uvec_radframe_con[3])) {
            P_rad[0] = min_erad;
            P_rad[1] = 0.0;
            P_rad[2] = 0.0;
            P_rad[3] = 0.0;
            return StatusRadiationInversion::cold_closure_nonfinite;
        }
    }

    P_rad[0] = E_rf;
    P_rad[1] = uvec_radframe_con[1];
    P_rad[2] = uvec_radframe_con[2];
    P_rad[3] = uvec_radframe_con[3];

    return StatusRadiationInversion::success;
}

KOKKOS_INLINE_FUNCTION void compute_covariant_fourforce(const GRCoordinates& G,
    const Real P_mhd[4], const Real P_rad[4], const Real Gas_Rho, const Real gam,
    const int opacity_model, const Real shocktube_sigma_rad,
    const Real shocktube_kappa_rho, const Real shocktube_kappa_scat,
    const UnitScales& units_cgs, const Microphysics::Opacities& opacities, const int k, const int j,
    const int i, Real dS[4])
{
    Real uvec_mhd[3] = {P_mhd[1], P_mhd[2], P_mhd[3]};
    Real ucon_mhd[4], ucov_mhd[4];
    GRMHD::calc_ucon(G, uvec_mhd, k, j, i, Loci::center, ucon_mhd);
    G.lower(ucon_mhd, ucov_mhd, k, j, i, Loci::center);

    Real Erf = P_rad[0];
    Real uvec_rad[3] = {P_rad[1], P_rad[2], P_rad[3]};
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

    Real Tg = (gam - 1.0) * (P_mhd[0] / Gas_Rho);

    Real kappa_a = RadM1::calc_kabs(
        Gas_Rho, Tg, opacity_model, shocktube_kappa_rho, units_cgs, opacities);
    Real kappa_sc = RadM1::calc_kscattering(
        Gas_Rho, Tg, opacity_model, shocktube_kappa_scat, units_cgs, opacities);

    Real JBB;
    if (opacity_model == (int)RadM1::OpacityModel::ShocktubeConstant) {
        Real sigma_rad = shocktube_sigma_rad;
        JBB = 4.0 * sigma_rad * (Tg * Tg * Tg * Tg);
    } else if (opacity_model == (int)RadM1::OpacityModel::Bondi) {
        //This sigma_rad will actually be 	arad = ARAD/(MASS_DENSITY_SCALE*C_CGS*C_CGS/pow(MMW*MH_CGS*C_CGS*C_CGS/BOLTZ_CGS,4.));
        Real energy_density_scale = units_cgs.energy_cgs / (units_cgs.length_cgs * units_cgs.length_cgs * units_cgs.length_cgs);
        Real mass_density_scale = units_cgs.mass_cgs / (units_cgs.length_cgs * units_cgs.length_cgs * units_cgs.length_cgs);
        Real sigma_rad = 5.670374419e-5/pc::c/(energy_density_scale/pow(pc::mp*pc::c*pc::c/pc::kb,4.0));
        JBB = 4.0 * sigma_rad * (Tg * Tg * Tg * Tg);
    } else if (opacity_model == (int)RadM1::OpacityModel::ThermalEquilibrium) {
        Real sigma_rad = shocktube_sigma_rad;
        JBB = 4.0 * sigma_rad * (Tg * Tg * Tg * Tg);
    } else {
        const Real temp_arg = m::abs(Tg) * units_cgs.mu * pc::mp * pc::c * pc::c;
        JBB = opacities.EnergyDensityFromTemperature(temp_arg);
    }

    Real kappa_tot = kappa_a + kappa_sc;

    if (kappa_tot == 0.0) {
        dS[0] = 0.0;
        dS[1] = 0.0;
        dS[2] = 0.0;
        dS[3] = 0.0;
        return;
    }

    Real coupling_term = kappa_a * (JBB - E_hat);

    dS[0] = coupling_term * ucov_mhd[0] - kappa_tot * F_hat_cov[0];
    dS[1] = coupling_term * ucov_mhd[1] - kappa_tot * F_hat_cov[1];
    dS[2] = coupling_term * ucov_mhd[2] - kappa_tot * F_hat_cov[2];
    dS[3] = coupling_term * ucov_mhd[3] - kappa_tot * F_hat_cov[3];
}

KOKKOS_INLINE_FUNCTION Real calculate_energy_residual(const GRCoordinates& G,
    const Real u_trial, const Real uvec_frozen[NVEC], const Real B_P[NVEC],
    const Real U_mhd_0[4], const Real U_rad_0[4], const Real Gas_Rho, const Real gam,
    const int opacity_model, const Real shocktube_sigma_rad,
    const Real shocktube_kappa_rho, const Real shocktube_kappa_scat,
    const UnitScales& units_cgs, const Microphysics::Opacities& opacities, const Real dt,
    const Real gdet, const int k,
    const int j, const int i, Real U_mhd_trial_out[4], Real U_rad_trial_out[4],
    Real P_rad_trial_out[4], Real dS_trial_out[4], bool& rad_recovery_ok)
{
    Real uvec[NVEC] = {uvec_frozen[0], uvec_frozen[1], uvec_frozen[2]};
    Real rho_ut_dummy;
    GRMHD::p_to_u_mhd(G, Gas_Rho, u_trial, uvec, B_P, gam, k, j, i, rho_ut_dummy,
        U_mhd_trial_out, Loci::center);

    for (int n = 0; n < 4; n++) {
        U_rad_trial_out[n] = U_rad_0[n] - (U_mhd_trial_out[n] - U_mhd_0[n]);
    }

    auto status = u_to_p_rad(G, U_rad_trial_out, P_rad_trial_out, k, j, i);
    rad_recovery_ok = (status == StatusRadiationInversion::success);

    Real P_mhd_trial[4] = {u_trial, uvec_frozen[0], uvec_frozen[1], uvec_frozen[2]};
    compute_covariant_fourforce(G, P_mhd_trial, P_rad_trial_out, Gas_Rho, gam,
        opacity_model, shocktube_sigma_rad, shocktube_kappa_rho, shocktube_kappa_scat, units_cgs, opacities, k, j, i, dS_trial_out);
    for (int n = 0; n < 4; n++) dS_trial_out[n] = gdet * dS_trial_out[n];

    Real resid = (U_mhd_trial_out[0] - U_mhd_0[0]) + dt * dS_trial_out[0];
    Real scale = m::max(RAD_SMALL,
        m::abs(U_mhd_trial_out[0]) + m::abs(U_mhd_0[0]) + m::abs(dt * dS_trial_out[0]));
    return resid / scale;
}

KOKKOS_INLINE_FUNCTION StatusImplicitStep solve_radiation_1d(const GRCoordinates& G,
    const Real U_mhd_0[4], const Real U_rad_0[4], const Real P_mhd_init[4],
    const Real B_P[NVEC], const Real Gas_Rho, const Real gam, const int opacity_model,
    const Real shocktube_sigma_rad, const Real shocktube_kappa_rho,
    const Real shocktube_kappa_scat, const UnitScales& units_cgs, const Microphysics::Opacities& opacities, const int k,
    const int j, const int i, const Real dt, const double tol, const int maxiter,
    Real P_mhd_new[4], Real dcov_rad[4])
{
    const Real gdet = G.gdet(Loci::center, j, i);
    const Real uvec_frozen[NVEC] = {P_mhd_init[1], P_mhd_init[2], P_mhd_init[3]};
    const Real u_init = P_mhd_init[0];

    Real U_mhd_trial[4], U_rad_trial[4], P_rad_trial[4], dS_trial[4];
    bool rad_ok;

    Real u_lo = 1.e-2 * u_init;
    Real u_hi = 1.e2 * u_init;
    Real f_lo = calculate_energy_residual(G, u_lo, uvec_frozen, B_P, U_mhd_0, U_rad_0,
        Gas_Rho, gam, opacity_model, shocktube_sigma_rad, shocktube_kappa_rho, shocktube_kappa_scat, units_cgs, opacities, dt, gdet,
        k, j, i, U_mhd_trial, U_rad_trial, P_rad_trial, dS_trial, rad_ok);
    Real f_hi = calculate_energy_residual(G, u_hi, uvec_frozen, B_P, U_mhd_0, U_rad_0,
        Gas_Rho, gam, opacity_model, shocktube_sigma_rad, shocktube_kappa_rho, shocktube_kappa_scat, units_cgs, opacities, dt, gdet,
        k, j, i, U_mhd_trial, U_rad_trial, P_rad_trial, dS_trial, rad_ok);

    bool bracketed = (f_lo * f_hi < 0.0);
    Real rebracket_fac = 10.0;
    int n_rebracket = 0;
    const int MAX_REBRACKET = 4;
    while (!bracketed && n_rebracket < MAX_REBRACKET) {
        u_lo = (1.e-1 / rebracket_fac) * u_init;
        u_hi = (1.e1 * rebracket_fac) * u_init;
        f_lo = calculate_energy_residual(G, u_lo, uvec_frozen, B_P, U_mhd_0, U_rad_0,
            Gas_Rho, gam, opacity_model, shocktube_sigma_rad, shocktube_kappa_rho, shocktube_kappa_scat, units_cgs, opacities, dt,
            gdet, k, j, i, U_mhd_trial, U_rad_trial, P_rad_trial, dS_trial, rad_ok);
        f_hi = calculate_energy_residual(G, u_hi, uvec_frozen, B_P, U_mhd_0, U_rad_0,
            Gas_Rho, gam, opacity_model, shocktube_sigma_rad, shocktube_kappa_rho, shocktube_kappa_scat, units_cgs, opacities, dt,
            gdet, k, j, i, U_mhd_trial, U_rad_trial, P_rad_trial, dS_trial, rad_ok);
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
        Real f_root =
            calculate_energy_residual(G, u_root, uvec_frozen, B_P, U_mhd_0, U_rad_0,
                Gas_Rho, gam, opacity_model, shocktube_sigma_rad, shocktube_kappa_rho, shocktube_kappa_scat, units_cgs, opacities, dt,
                gdet, k, j, i, U_mhd_trial, U_rad_trial, P_rad_trial, dS_trial, rad_ok);

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
        return StatusImplicitStep::failure;
    }

    calculate_energy_residual(G, u_root, uvec_frozen, B_P, U_mhd_0, U_rad_0, Gas_Rho, gam,
        opacity_model, shocktube_sigma_rad, shocktube_kappa_rho, shocktube_kappa_scat, units_cgs, opacities, dt, gdet, k, j, i,
        U_mhd_trial, U_rad_trial, P_rad_trial, dS_trial, rad_ok);
    if (!rad_ok) {
        return StatusImplicitStep::failure;
    }

    P_mhd_new[0] = u_root;
    P_mhd_new[1] = uvec_frozen[0];
    P_mhd_new[2] = uvec_frozen[1];
    P_mhd_new[3] = uvec_frozen[2];

    for (int n = 0; n < 4; n++) {
        dcov_rad[n] = U_rad_trial[n] - U_rad_0[n];
    }

    return StatusImplicitStep::success;
}

KOKKOS_INLINE_FUNCTION int solve_radiation_4d(const GRCoordinates& G,
    const VariablePack<Real> U_init, const VariablePack<Real> P_init,
    VariablePack<Real> P_new, VariablePack<Real> U_new, const VarMap m_p,
    const VarMap m_u, const int k, const int j, const int i, const Real dt,
    const Real gam, const double src_rootfind_eps, const double src_rootfind_tol,
    const int src_rootfind_maxiter, const int opacity_model,
    const Real shocktube_sigma_rad, const Real shocktube_kappa_rho,
    const Real shocktube_kappa_scat, const UnitScales& units_cgs, const Microphysics::Opacities& opacities,
    const VariablePack<Real> pflag, const VariablePack<Real> rinvflag)
{
    const Real Gas_Rho = P_init(m_p.RHO, k, j, i);

    Real B_P[NVEC] = {0.};
    if (m_p.B1 >= 0) {
        B_P[V1] = P_init(m_p.B1, k, j, i);
        B_P[V2] = P_init(m_p.B2, k, j, i);
        B_P[V3] = P_init(m_p.B3, k, j, i);
    }
    Real P_mhd_guess[4] = {P_init(m_p.UU, k, j, i), P_init(m_p.U1, k, j, i),
        P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};

    Real U_rad_0[4] = {U_init(m_u.UU_RAD, k, j, i), U_init(m_u.U1_RAD, k, j, i),
        U_init(m_u.U2_RAD, k, j, i), U_init(m_u.U3_RAD, k, j, i)};

    Real resid[4];

    Real U_mhd_0[4];

    Real uvec[NVEC] = {
        P_init(m_p.U1, k, j, i), P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};


    Real U_mhd_guess[4];
    Real U_rad_guess[4];
    Real P_rad_guess[4];
    Real dS_guess[4];
    Real dcov_rad[4] = {0., 0., 0., 0.};
    
    U_mhd_0[0] = U_init(m_u.UU, k, j, i);
    U_mhd_0[1] = U_init(m_u.U1, k, j, i);
    U_mhd_0[2] = U_init(m_u.U2, k, j, i);
    U_mhd_0[3] = U_init(m_u.U3, k, j, i);

    // Iteration 0
    U_mhd_guess[0] = U_mhd_0[0];
    U_mhd_guess[1] = U_mhd_0[1];
    U_mhd_guess[2] = U_mhd_0[2];
    U_mhd_guess[3] = U_mhd_0[3];

    // Conservation law: Delta U_rad = - Delta U_mhd
    DLOOP1 U_rad_guess[mu] = U_rad_0[mu];
    Real gdet = G.gdet(Loci::center, j, i);

    // Convert the newly guessed U_rad to P_rad
    u_to_p_rad(G, U_rad_guess, P_rad_guess, k, j, i);
    compute_covariant_fourforce(G, P_mhd_guess, P_rad_guess, Gas_Rho, gam, opacity_model,
        shocktube_sigma_rad, shocktube_kappa_rho, shocktube_kappa_scat, units_cgs, opacities, k, j, i, dS_guess);

    for (int n = 0; n < 4; n++) dS_guess[n] = gdet * dS_guess[n];

    DLOOP1{
        resid[mu] = U_mhd_guess[mu] - U_mhd_0[mu] + dt * dS_guess[mu];
    }

    //Compute err here and do a convergence check
    Real err = RAD_SMALL;
    Real max_divisor = RAD_SMALL;

    for (int n = 0; n < 4; n++) {
        max_divisor = std::max<Real>(max_divisor, std::fabs(U_mhd_guess[n]) +
                                                        std::fabs(U_mhd_0[n]) +
                                                        std::fabs(dt * dS_guess[n]));
    }

    for (int n = 0; n < 4; n++) {
        Real suberr = std::fabs(resid[n]) / max_divisor;
        if (suberr > err) {
            err = suberr;
        }
    }

    int niter = 0;
    bool bad_guess = false;

    do {
        if (err <= src_rootfind_tol) {
            break;
        }

        Real P_rad_m[4];
        Real P_rad_p[4];
        Real U_mhd_m[4];
        Real U_mhd_p[4];
        Real U_rad_m[4];
        Real U_rad_p[4];
        Real dS_m[4];
        Real dS_p[4];

        Real jac[4][4] = {0};

        // Find minimum non-zero magnitude from P_mhd_guess to scale FD step safely
        Real P_mhd_mag_min = RAD_LARGE;
        for (int m = 0; m < 4; m++) {
            if (m::abs(P_mhd_guess[m]) > 0.) {
                P_mhd_mag_min = m::min(P_mhd_mag_min, m::abs(P_mhd_guess[m]));
            }
        }

        bool bad_guess_m = false;
        bool bad_guess_p = false;

        // Loop over the 4 fluid variables to perturb each one
        for (int m = 0; m < 4; m++) {
            Real P_mhd_m[4] = {
                P_mhd_guess[0], P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};
            Real P_mhd_p[4] = {
                P_mhd_guess[0], P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};

            const Real fd_step = m::max(src_rootfind_eps * P_mhd_mag_min,
                src_rootfind_eps * m::abs(P_mhd_guess[m]));
            P_mhd_m[m] -= fd_step;
            P_mhd_p[m] += fd_step;

            // Evaluate minus perturbation
            Real rho_ut_dummy_m;
            uvec[0] = P_mhd_m[1];
            uvec[1] = P_mhd_m[2];
            uvec[2] = P_mhd_m[3];
            GRMHD::p_to_u_mhd(G, Gas_Rho, P_mhd_m[0], uvec, B_P, gam, k, j, i,
                rho_ut_dummy_m, U_mhd_m, Loci::center);

            // Conservation law: Delta U_rad = - Delta U_mhd
            for (int n = 0; n < 4; n++) {
                U_rad_m[n] = U_rad_0[n] - (U_mhd_m[n] - U_mhd_0[n]);
            }

            // Recover rad primitives
            auto status_m = u_to_p_rad(G, U_rad_m, P_rad_m, k, j, i);
            if (status_m != StatusRadiationInversion::success) {
                bad_guess_m = true;
            }

            // If a bad guess has already been found before, we can't really skip the whole Jacobian evaluation.
            // We need to keep evaluating the other blocks to figure out if, during the next m's, the other side (plus/minus) will also go bad, trigerring a bad_guess_m == true && bad_guess_p == true
            if (!bad_guess_m && !bad_guess_p) {
                compute_covariant_fourforce(G, P_mhd_m, P_rad_m, Gas_Rho, gam, opacity_model,
                    shocktube_sigma_rad, shocktube_kappa_rho, shocktube_kappa_scat, units_cgs, opacities, k, j, i, dS_m);
                for (int n = 0; n < 4; n++) dS_m[n] = gdet * dS_m[n];
            }
            // Evaluate plus perturbation
            Real rho_ut_dummy_p;
            uvec[0] = P_mhd_p[1];
            uvec[1] = P_mhd_p[2];
            uvec[2] = P_mhd_p[3];
            GRMHD::p_to_u_mhd(G, Gas_Rho, P_mhd_p[0], uvec, B_P, gam, k, j, i,
                rho_ut_dummy_p, U_mhd_p, Loci::center);

            for (int n = 0; n < 4; n++) {
                U_rad_p[n] = U_rad_0[n] - (U_mhd_p[n] - U_mhd_0[n]);
            }
            //TODO (PNM): Change name of KOKKOS kernel to lower snake case
            auto status_p = u_to_p_rad(G, U_rad_p, P_rad_p, k, j, i);
            if (status_p != StatusRadiationInversion::success) {
                bad_guess_p = true;
            }

            // If a bad guess has already been found before, we can't really skip the whole Jacobian evaluation.
            // We need to keep evaluating the other blocks to figure out if, during the next m's, the other side (plus/minus) will also go bad, trigerring a bad_guess_m == true && bad_guess_p == true
            if (!bad_guess_m && !bad_guess_p) {
                compute_covariant_fourforce(G, P_mhd_p, P_rad_p, Gas_Rho, gam, opacity_model,
                    shocktube_sigma_rad, shocktube_kappa_rho, shocktube_kappa_scat, units_cgs, opacities, k, j, i, dS_p);
                for (int n = 0; n < 4; n++) dS_p[n] = gdet * dS_p[n];

                // Populate Jacobian
                for (int n = 0; n < 4; n++) {
                    Real fp = U_mhd_p[n] - U_mhd_0[n] + dt * dS_p[n];
                    Real fm = U_mhd_m[n] - U_mhd_0[n] + dt * dS_m[n];
                    // Jacobian here is dU_rad/dP_mhd
                    // Since div R^mu_nu = G_nu
                    // and div T^mu_nu = -G_nu
                    jac[n][m] = (fp - fm) / (P_mhd_p[m] - P_mhd_m[m]);
                }
            }
        }
        // TODO (PNM): Separate these if/elses into different kernels.
        if (bad_guess_m == true && bad_guess_p == true) {
            bad_guess = true;
            break; // Exit the iteration loop if both perturbations yield bad guesses
        } else if (bad_guess_m == true) {
            // If only - finite difference support point is bad, do one-sided
            // difference with + support point

            for (int m = 0; m < 4; m++) {
                Real P_mhd_p[4] = {
                    P_mhd_guess[0], P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};
                P_mhd_p[m] += std::max(src_rootfind_eps * P_mhd_mag_min,
                    src_rootfind_eps * m::abs(P_mhd_p[m]));

                Real rho_ut_dummy_p;
                uvec[0] = P_mhd_p[1];
                uvec[1] = P_mhd_p[2];
                uvec[2] = P_mhd_p[3];
                // TODO (PNM): This is stupid since we have already computed the plus perturbation above.
                // PNM: Actually, I don't know if that's actually stupid, we would need to save a lot of 4x4 matrices to get this working. it's a trade-off between register pressure and doing a few more calculations, which
                // I think would be more efficient.
                GRMHD::p_to_u_mhd(G, Gas_Rho, P_mhd_p[0], uvec, B_P, gam, k, j, i,
                    rho_ut_dummy_p, U_mhd_p, Loci::center);
                for (int n = 0; n < 4; n++) {
                    U_rad_p[n] = U_rad_0[n] - (U_mhd_p[n] - U_mhd_0[n]);
                }
                auto status_p = u_to_p_rad(G, U_rad_p, P_rad_p, k, j, i);
                compute_covariant_fourforce(G, P_mhd_p, P_rad_p, Gas_Rho, gam,
                    opacity_model, shocktube_sigma_rad, shocktube_kappa_rho, shocktube_kappa_scat, units_cgs, opacities, k, j, i,
                    dS_p);
                for (int n = 0; n < 4; n++) dS_p[n] = gdet * dS_p[n];

                PARTHENON_REQUIRE(status_p == StatusRadiationInversion::success,
                    "This inversion should have already worked!");

                for (int n = 0; n < 4; n++) {
                    Real fp = U_mhd_p[n] - U_mhd_0[n] + dt * dS_p[n];
                    Real fguess = U_mhd_guess[n] - U_mhd_0[n] + dt * dS_guess[n];
                    // Jacobian here is dU_rad/dP_mhd
                    // Since div R^mu_nu = G_nu
                    // and div T^mu_nu = -G_nu
                    jac[n][m] = (fp - fguess) / (P_mhd_p[m] - P_mhd_guess[m]);
                }
            }
        } else if (bad_guess_p == true) {
            // If only + finite difference support point is bad, do one-sided
            // difference with - support point

            for (int m = 0; m < 4; m++) {
                Real P_mhd_m[4] = {
                    P_mhd_guess[0], P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};
                P_mhd_m[m] -= std::max(src_rootfind_eps * P_mhd_mag_min,
                    src_rootfind_eps * m::abs(P_mhd_m[m]));

                Real rho_ut_dummy_m;
                uvec[0] = P_mhd_m[1];
                uvec[1] = P_mhd_m[2];
                uvec[2] = P_mhd_m[3];
                GRMHD::p_to_u_mhd(G, Gas_Rho, P_mhd_m[0], uvec, B_P, gam, k, j, i,
                    rho_ut_dummy_m, U_mhd_m, Loci::center);
                for (int n = 0; n < 4; n++) {
                    U_rad_m[n] = U_rad_0[n] - (U_mhd_m[n] - U_mhd_0[n]);
                }
                auto status_m = u_to_p_rad(G, U_rad_m, P_rad_m, k, j, i);
                compute_covariant_fourforce(G, P_mhd_m, P_rad_m, Gas_Rho, gam,
                    opacity_model, shocktube_sigma_rad, shocktube_kappa_rho, shocktube_kappa_scat, units_cgs, opacities, k, j, i,
                    dS_m);
                for (int n = 0; n < 4; n++) dS_m[n] = gdet * dS_m[n];
                PARTHENON_REQUIRE(status_m == StatusRadiationInversion::success,
                    "This inversion should have already worked!");
                for (int n = 0; n < 4; n++) {
                    Real fm = U_mhd_m[n] - U_mhd_0[n] + dt * dS_m[n];
                    Real fguess = U_mhd_guess[n] - U_mhd_0[n] + dt * dS_guess[n];
                    jac[n][m] = (fguess - fm) / (P_mhd_guess[m] - P_mhd_m[m]);
                }
            }
        }

        Real jacinv[4][4];
        // Inverting the 4x4 matrix;
        invert(&jac[0][0], &jacinv[0][0]);

        // We already broke from here if the guess was bad.
        Real ug0 = P_mhd_guess[0];
        Real ur0 = P_rad_guess[0];

        // update guess
        for (int m = 0; m < 4; m++) {
            for (int n = 0; n < 4; n++) {
                P_mhd_guess[m] -= jacinv[m][n] * resid[n];
            }
        }

        // Check if the new gas energy is negative, if it is, do a reflect positivity divided by 1/2
        if (P_mhd_guess[0] < 0.0) {
            P_mhd_guess[0] = 0.5 * m::abs(P_mhd_guess[0]);
        }

        // Re-evaluate residual with updated guess
        Real rho_ut_dummy_new;
        uvec[0] = P_mhd_guess[1];
        uvec[1] = P_mhd_guess[2];
        uvec[2] = P_mhd_guess[3];
        GRMHD::p_to_u_mhd(G, Gas_Rho, P_mhd_guess[0], uvec, B_P, gam, k, j, i,
            rho_ut_dummy_new, U_mhd_guess, Loci::center);
        for (int n = 0; n < 4; n++) {
            U_rad_guess[n] = U_rad_0[n] - (U_mhd_guess[n] - U_mhd_0[n]);
        }
        auto status = u_to_p_rad(G, U_rad_guess, P_rad_guess, k, j, i);
        compute_covariant_fourforce(G, P_mhd_guess, P_rad_guess, Gas_Rho, gam,
            opacity_model, shocktube_sigma_rad, shocktube_kappa_rho, shocktube_kappa_scat, units_cgs, opacities, k, j, i,
            dS_guess);

        for (int n = 0; n < 4; n++) dS_guess[n] = gdet * dS_guess[n];

        // Line search if rad prim had a bad inversion. Maybe reducing the step will help.
        if (status != StatusRadiationInversion::success) {
            constexpr Real umin = 1.e-12;
            constexpr Real Emin = 1.e-60;
            const Real gamma_max_sq = 1.e6; // Corresponds to Gamma_max = 1000

            Real scaling_factor = 0.0;

            // Check Gas Internal Energy Violation
            if (P_mhd_guess[0] < umin) {
                // If it went negative or too small, calculate relative overstep
                scaling_factor = m::max(
                    scaling_factor, (ug0 - umin) / (ug0 - P_mhd_guess[0] + 1e-20));
            }

            // Check Radiation Rest-Frame Energy Violation
            if (P_rad_guess[0] < Emin) {
                scaling_factor = m::max(
                    scaling_factor, (ur0 - Emin) / (ur0 - P_rad_guess[0] + 1e-20));
            }

            // Check Velocity / Lorentz Factor Violation
            // Re-calculate the Lorentz factor squared of the guess to see if it went
            // superluminal
            Real R_t_cov_guess[4] = {U_rad_guess[0] / gdet, U_rad_guess[1] / gdet,
                U_rad_guess[2] / gdet, U_rad_guess[3] / gdet};

            Real gamma_sq_guess =
                calculate_gamma_rel2(G, R_t_cov_guess, j, i);

            if (gamma_sq_guess > gamma_max_sq || gamma_sq_guess < 1.0) {
                // If velocity exploded, aggressively damp the step (e.g., cut it in
                // half)
                scaling_factor = m::max(scaling_factor, 0.5);
            }

            // Verify the scaling factor is sane
            if (!(scaling_factor > 0.0 && scaling_factor <= 1.0)) {
                bad_guess = true;
                break; // Step is completely unrecoverable, abort to avoid NaN
                        // cascading
            }

            // Retain a 50% safety buffer away from the boundary edge
            scaling_factor *= 0.5;

            // Roll back to old guess, then take the scaled/damped step
            for (int m = 0; m < 4; m++) {
                for (int n = 0; n < 4; n++) {
                    // Notice the += here!
                    P_mhd_guess[m] +=
                        (1.0 - scaling_factor) * jacinv[m][n] * resid[n];
                }
            }

            // Ensure the new gas energy is positive
            if (P_mhd_guess[0] < 0.0) {
                P_mhd_guess[0] = 0.5 * m::abs(P_mhd_guess[0]);
            }

            // Re-evaluate unperturbed state with the newly dampened primitives
            uvec[0] = P_mhd_guess[1];
            uvec[1] = P_mhd_guess[2];
            uvec[2] = P_mhd_guess[3];
            GRMHD::p_to_u_mhd(G, Gas_Rho, P_mhd_guess[0], uvec, B_P, gam, k, j, i,
                rho_ut_dummy_new, U_mhd_guess, Loci::center);
            for (int n = 0; n < 4; n++)
                U_rad_guess[n] = U_rad_0[n] - (U_mhd_guess[n] - U_mhd_0[n]);

            status = u_to_p_rad(G, U_rad_guess, P_rad_guess, k, j, i);
            compute_covariant_fourforce(G, P_mhd_guess, P_rad_guess, Gas_Rho, gam,
                opacity_model, shocktube_sigma_rad, shocktube_kappa_rho, shocktube_kappa_scat, units_cgs, opacities, k, j, i,
                dS_guess);
            for (int n = 0; n < 4; n++) dS_guess[n] = gdet * dS_guess[n];

            // If the scaled step lands in a physically valid regime, we cleared the
            // error flag!
            if (status != StatusRadiationInversion::success || P_mhd_guess[0] < umin) {
                bad_guess = true;
                break;
            }
        }

        // Update residuals
        for (int n = 0; n < 4; n++) {
            resid[n] = U_mhd_guess[n] - U_mhd_0[n] + dt * dS_guess[n];

            if (std::isnan(resid[n])) {
                bad_guess = true;
                break;
            }
        }

        // This is needed since the previous bad_guess = true would only break out of the for loop
        if (bad_guess) {
            break;
        }


        // Calculate error now
        err = RAD_SMALL;
        Real max_divisor = RAD_SMALL;

        for (int n = 0; n < 4; n++) {
            max_divisor = std::max<Real>(max_divisor, std::fabs(U_mhd_guess[n]) +
                                                          std::fabs(U_mhd_0[n]) +
                                                          std::fabs(dt * dS_guess[n]));
        }

        for (int n = 0; n < 4; n++) {
            Real suberr = std::fabs(resid[n]) / max_divisor;
            if (suberr > err) {
                err = suberr;
            }
        }

        niter++;
    } while (err > src_rootfind_tol && niter < src_rootfind_maxiter);

    // isnan is no-ops for GPU code and for fast-math cpu code (default intel compiler). Careful, these isnans might not trigger.
    bool used_1d_fallback = false;
    if (niter == src_rootfind_maxiter || err > src_rootfind_tol ||
        m::isnan(U_rad_guess[0]) || m::isnan(U_rad_guess[1]) ||
        m::isnan(U_rad_guess[2]) || m::isnan(U_rad_guess[3]) || bad_guess) {
        used_1d_fallback = true;
        Real P_mhd_init[4] = {P_init(m_p.UU, k, j, i), P_init(m_p.U1, k, j, i),
            P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};
        // TODO (PNM): Make P_mhd_new_1d just a scalar
        // Currently, we don't need P_mhd_new_1d at all, since from here, we will only make a u_to_p transf for the fluid.
        //However, eventually, I hope we add an option to use oned solver only, that would make it require a P_mhd_new_1d, so I will leave it here for now.
        Real P_mhd_new_1d[4];
        auto status_1d = solve_radiation_1d(G, U_mhd_0, U_rad_0, P_mhd_init, B_P, Gas_Rho,
            gam, opacity_model, shocktube_sigma_rad, shocktube_kappa_rho, shocktube_kappa_scat, units_cgs, opacities, k, j, i, dt,
            src_rootfind_tol, src_rootfind_maxiter, P_mhd_new_1d, dcov_rad);

        // We don't update velocity since the 1D update is only for temperature.
        if (status_1d != StatusImplicitStep::success) {
            // It failed the 1d too!
            //Let's try reverting to the initial state and assume that the source term was zero for this step;
            U_new(m_u.UU_RAD, k , j, i) = U_init(m_u.UU_RAD, k , j, i);
            U_new(m_u.U1_RAD, k , j, i) = U_init(m_u.U1_RAD, k , j, i);
            U_new(m_u.U2_RAD, k , j, i) = U_init(m_u.U2_RAD, k , j, i);
            U_new(m_u.U3_RAD, k , j, i) = U_init(m_u.U3_RAD, k , j, i);
            U_new(m_u.UU, k , j, i) = U_init(m_u.UU, k , j, i);
            U_new(m_u.U1, k , j, i) = U_init(m_u.U1, k , j, i);
            U_new(m_u.U2, k , j, i) = U_init(m_u.U2, k , j, i);
            U_new(m_u.U3, k , j, i) = U_init(m_u.U3, k , j, i);

            // Just invert both as if the source term was zero;
            auto mhd_inverter_status = Inverter::u_to_p<Inverter::Type::kastaun>(
            G, U_new, m_u, gam, k, j, i, P_new, m_p, Loci::center, 25, 1e-12);
            pflag(0, k, j, i) = mhd_inverter_status;
            // Now since the u2p for MHD was successful, do it for radiation:
            Real U_rad_final[4] = {U_new(m_u.UU_RAD, k, j, i), U_new(m_u.U1_RAD, k, j, i),
                U_new(m_u.U2_RAD, k, j, i), U_new(m_u.U3_RAD, k, j, i)};
            Real P_rad_final[4];

            auto rad_status = u_to_p_rad(G, U_rad_final, P_rad_final, k, j, i);

            P_new(m_p.UU_RAD, k, j, i) = P_rad_final[0];
            P_new(m_p.U1_RAD, k, j, i) = P_rad_final[1];
            P_new(m_p.U2_RAD, k, j, i) = P_rad_final[2];
            P_new(m_p.U3_RAD, k, j, i) = P_rad_final[3];
    
            return static_cast<int>(StatusImplicitStep::onedfallback_failure);
        }
    } else {
        dcov_rad[0] = U_rad_guess[0] - U_rad_0[0];
        dcov_rad[1] = U_rad_guess[1] - U_rad_0[1];
        dcov_rad[2] = U_rad_guess[2] - U_rad_0[2];
        dcov_rad[3] = U_rad_guess[3] - U_rad_0[3];
    }


    bool successful_prim_recovery = false;

    U_new(m_u.UU_RAD, k, j, i) += dcov_rad[0];
    U_new(m_u.U1_RAD, k, j, i) += dcov_rad[1];
    U_new(m_u.U2_RAD, k, j, i) += dcov_rad[2];
    U_new(m_u.U3_RAD, k, j, i) += dcov_rad[3];
    U_new(m_u.UU, k, j, i) -= dcov_rad[0];
    U_new(m_u.U1, k, j, i) -= dcov_rad[1];
    U_new(m_u.U2, k, j, i) -= dcov_rad[2];
    U_new(m_u.U3, k, j, i) -= dcov_rad[3];

    auto mhd_inverter_status = Inverter::u_to_p<Inverter::Type::kastaun>(
        G, U_new, m_u, gam, k, j, i, P_new, m_p, Loci::center, 25, 1e-12);

    // Refresh pflag each step so a stale failure doesn't linger.
    pflag(0, k, j, i) = static_cast<int>(Inverter::Status::success);
    rinvflag(0, k, j, i) = static_cast<int>(StatusRadiationInversion::success);

    if (mhd_inverter_status != static_cast<int>(Inverter::Status::success)) {
        successful_prim_recovery = false;

        // Let the fluid fixup (Inverter::MeshFixUtoP, keyed on pflag) repair
        // the gas variables via its own neighbor-averaging/backstop.
        // pflag(0, k, j, i) = mhd_inverter_status;

        //Let's try reverting to the initial state and assume that the source term was zero for this step;
        U_new(m_u.UU_RAD, k , j, i) = U_init(m_u.UU_RAD, k , j, i);
        U_new(m_u.U1_RAD, k , j, i) = U_init(m_u.U1_RAD, k , j, i);
        U_new(m_u.U2_RAD, k , j, i) = U_init(m_u.U2_RAD, k , j, i);
        U_new(m_u.U3_RAD, k , j, i) = U_init(m_u.U3_RAD, k , j, i);
        U_new(m_u.UU, k , j, i) = U_init(m_u.UU, k , j, i);
        U_new(m_u.U1, k , j, i) = U_init(m_u.U1, k , j, i);
        U_new(m_u.U2, k , j, i) = U_init(m_u.U2, k , j, i);
        U_new(m_u.U3, k , j, i) = U_init(m_u.U3, k , j, i);

        // Just invert both as if the source term was zero;
        auto mhd_inverter_status = Inverter::u_to_p<Inverter::Type::kastaun>(
        G, U_new, m_u, gam, k, j, i, P_new, m_p, Loci::center, 25, 1e-12);
        pflag(0, k, j, i) = mhd_inverter_status;
        // Now since the u2p for MHD was successful, do it for radiation:
        Real U_rad_final[4] = {U_new(m_u.UU_RAD, k, j, i), U_new(m_u.U1_RAD, k, j, i),
            U_new(m_u.U2_RAD, k, j, i), U_new(m_u.U3_RAD, k, j, i)};
        Real P_rad_final[4];

        auto rad_status = u_to_p_rad(G, U_rad_final, P_rad_final, k, j, i);

        P_new(m_p.UU_RAD, k, j, i) = P_rad_final[0];
        P_new(m_p.U1_RAD, k, j, i) = P_rad_final[1];
        P_new(m_p.U2_RAD, k, j, i) = P_rad_final[2];
        P_new(m_p.U3_RAD, k, j, i) = P_rad_final[3];
    
        return static_cast<int>(StatusImplicitStep::mhdsolve);

    } else {
        successful_prim_recovery = true;

        // Now since the u2p for MHD was successful, do it for radiation:
        Real U_rad_final[4] = {U_new(m_u.UU_RAD, k, j, i), U_new(m_u.U1_RAD, k, j, i),
            U_new(m_u.U2_RAD, k, j, i), U_new(m_u.U3_RAD, k, j, i)};
        Real P_rad_final[4];

        auto rad_status = u_to_p_rad(G, U_rad_final, P_rad_final, k, j, i);
        rinvflag(0, k, j, i) = static_cast<int>(rad_status);
        if (rad_status != StatusRadiationInversion::success) {
            successful_prim_recovery = false;
        } else {
            P_new(m_p.UU_RAD, k, j, i) = P_rad_final[0];
            P_new(m_p.U1_RAD, k, j, i) = P_rad_final[1];
            P_new(m_p.U2_RAD, k, j, i) = P_rad_final[2];
            P_new(m_p.U3_RAD, k, j, i) = P_rad_final[3];
        }
    }

    if (!successful_prim_recovery) {
        // The inverter for MHD worked, but the radiation failed.
        return static_cast<int>(StatusImplicitStep::radsolve);
    }

    if (used_1d_fallback) {
        return static_cast<int>(StatusImplicitStep::onedfallback_success);
    }
    return static_cast<int>(StatusImplicitStep::success);
}

} // namespace RadM1
