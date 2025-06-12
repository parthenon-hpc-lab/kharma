/* 
 *  File: floors_functions.hpp
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

#include "floors.hpp"
#include "onedw.hpp"

/**
 * Device-side functions for applying GRMHD floors
 */
namespace Floors {

/**
 * Apply all ceilings together, currently at most one on velocity and two on internal energy
 * TODO REALLY need to take fflag here and only compute existing values if we know the floor was hit
 * 
 * LOCKSTEP: this function respects P and returns consistent P<->U
 */
KOKKOS_INLINE_FUNCTION void apply_ceilings(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
                                          const Real& gam, const int& k, const int& j, const int& i,
                                          const Floors::Prescription& floors, const Floors::Prescription& floors_inner,
                                          const VariablePack<Real>& U, const VarMap& m_u, const Loci loc=Loci::center)
{
    // Choose our floor scheme
    const Floors::Prescription& myfloors = (floors.radius_dependent_floors
                                            && G.r(k, j, i) < floors.floors_switch_r) ? floors_inner : floors;

    // Compute max values for ceilings
    Real gamma      = GRMHD::lorentz_calc(G, P, m_p, k, j, i, loc);
    Real ktot       = (gam - 1.) * P(m_p.UU, k, j, i) / m::pow(P(m_p.RHO, k, j, i), gam);
    Real u_over_rho = P(m_p.UU, k, j, i) / P(m_p.RHO, k, j, i);

    // 1. Limit gamma with respect to normal observer
    if (gamma > myfloors.gamma_max) {
        Real f = m::sqrt((SQR(myfloors.gamma_max) - 1.) / (SQR(gamma) - 1.));
        VLOOP P(m_p.U1+v, k, j, i) *= f;
    }

    // 2. Limit the entropy by controlling u, to avoid anomalous cooling from funnel wall
    // Note this technically applies the condition *one step sooner* than legacy, since it operates on
    // the entropy as calculated from current conditions, rather than the value kept from the previous
    // step for calculating dissipation.
    if (ktot > myfloors.ktot_max) {
        P(m_p.UU, k, j, i) = myfloors.ktot_max / ktot * P(m_p.UU, k, j, i);
    }

    // 3. Limit the temperature by controlling u.  Can optionally add density instead, implemented in apply_floors
    if (myfloors.temp_adjust_u && P(m_p.UU, k, j, i) / P(m_p.RHO, k, j, i) > myfloors.u_over_rho_max) {
        P(m_p.UU, k, j, i) = myfloors.u_over_rho_max * P(m_p.RHO, k, j, i);
    }
}

KOKKOS_INLINE_FUNCTION int determine_floors(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
                                        const Real& gam, const int& k, const int& j, const int& i, const Floors::Prescription& floors,
                                        const Floors::Prescription& floors_inner, Real& rhoflr_max, Real& uflr_max)
{
    // Choose our floor scheme
    const Floors::Prescription& myfloors = (floors.radius_dependent_floors
                                            && G.r(k, j, i) < floors.floors_switch_r) ? floors_inner : floors;

    // Calculate the different floor values in play:
    // 1. Geometric hard floors, not based on fluid relationships
    // TODO(BSP) can this be cached if it's slow?
    Real rhoflr_geom, uflr_geom;
    if(G.coords.is_spherical()) {
        const GReal r = G.r(k, j, i);
        // r_char sets more aggressive floor close to EH but backs off
        Real rhoscal = (myfloors.use_r_char) ? 1. / ((r*r) * (1 + r / myfloors.r_char)) : 1. / m::sqrt(r*r*r);
        rhoflr_geom = m::max(myfloors.rho_min_geom * rhoscal, myfloors.rho_min_const);
        uflr_geom   = m::max(myfloors.u_min_geom * m::pow(rhoscal, gam), myfloors.u_min_const);
    } else {
        rhoflr_geom = myfloors.rho_min_const;
        uflr_geom   = myfloors.u_min_const;
    }

    // 2. Magnetization ceilings: impose maximum magnetization sigma = bsq/rho, and inverse beta prop. to bsq/U
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
    Real rhoflr_b, uflr_b;
    // Radius-dependent floors.
    // Used with spherical coordinate system, i.e., G.r(k,j,i) exist
    Real bsq = dot(Dtmp.bcon, Dtmp.bcov);
    rhoflr_b = bsq / myfloors.bsq_over_rho_max;
    uflr_b   = bsq / myfloors.bsq_over_u_max;

    // Evaluate max U floor, needed for temp ceiling below
    uflr_max = m::max(uflr_geom, uflr_b);

    const auto& rho = P(m_p.RHO, k, j, i);
    const auto& u = P(m_p.UU, k, j, i);

    int fflag = 0;
    if (!myfloors.temp_adjust_u) {
        // 3. Temperature ceiling: impose maximum temperature u/rho
        // Take floors on U into account
        const double rhoflr_temp = m::max(u, uflr_max) / myfloors.u_over_rho_max;

        // Record hitting temperature ceiling
        fflag |= (rhoflr_temp > rho) * FFlag::TEMP;

        // Evaluate max rho floor
        rhoflr_max = m::max(m::max(rhoflr_geom, rhoflr_b), rhoflr_temp);
    } else {
        // Evaluate max rho floor
        rhoflr_max = m::max(rhoflr_geom, rhoflr_b);
    }

    if (rhoflr_max > rho || uflr_max > u) {
        // Record all the floors that were hit, using bitflags
        // Record Geometric floor hits
        fflag |= (rhoflr_geom > rho) * FFlag::GEOM_RHO;
        fflag |= (uflr_geom > u) * FFlag::GEOM_U;
        // Record Magnetic floor hits
        fflag |= (rhoflr_b > rho) * FFlag::B_RHO;
        fflag |= (uflr_b > u) * FFlag::B_U;
    }

    // Then ceilings, need to record these for FOFC. See real implementation for details
    if (GRMHD::lorentz_calc(G, P, m_p, k, j, i, Loci::center) > myfloors.gamma_max)
        fflag |= FFlag::GAMMA;

    if ((gam - 1.) * P(m_p.UU, k, j, i) / m::pow(P(m_p.RHO, k, j, i), gam) > myfloors.ktot_max)
        fflag |= FFlag::KTOT;

    if (myfloors.temp_adjust_u && (P(m_p.UU, k, j, i) / P(m_p.RHO, k, j, i) > myfloors.u_over_rho_max))
        fflag |= FFlag::TEMP;

    return fflag;
}

#define FLOOR_ONE_ARGS const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p, const Real& gam, \
                        const int& k, const int& j, const int& i, const Real& rhoflr_max, const Real& uflr_max, \
                        const VariablePack<Real>& U, const VarMap& m_u

/**
 * Apply floors of several types in determining how to add mass and internal energy to preserve stability.
 * All floors which might apply are recorded separately, then mass/energy are added *in normal observer frame*
 * 
 * @return pflag: in NOF, a number <32 representing any failure of the U->P solve.  Otherwise 0.
 * 
 * LOCKSTEP: this function respects P and ignores U in order to return consistent P<->U
 */
template<InjectionFrame frame>
KOKKOS_INLINE_FUNCTION int apply_floors(FLOOR_ONE_ARGS);

template<>
KOKKOS_INLINE_FUNCTION int apply_floors<InjectionFrame::fluid>(FLOOR_ONE_ARGS)
{
    P(m_p.RHO, k, j, i) += m::max(0., rhoflr_max - P(m_p.RHO, k, j, i));
    P(m_p.UU, k, j, i)  += m::max(0., uflr_max - P(m_p.UU, k, j, i));
    return 0;
}

template<>
KOKKOS_INLINE_FUNCTION int apply_floors<InjectionFrame::drift>(FLOOR_ONE_ARGS)
{
    // Drift frame floors. Refer to Appendix B3 in https://doi.org/10.1093/mnras/stx364 (hereafter R17)
    const Real lapse2    = 1. / (-G.gcon(Loci::center, j, i, 0, 0));
    double beta[GR_DIM] = {0};
    beta[1] = lapse2 * G.gcon(Loci::center, j, i, 0, 1);
    beta[2] = lapse2 * G.gcon(Loci::center, j, i, 0, 2);
    beta[3] = lapse2 * G.gcon(Loci::center, j, i, 0, 3);

    // Fluid quantities (four velocities have been computed above)
    const Real rho   = P(m_p.RHO, k, j, i);
    const Real uu    = P(m_p.UU, k, j, i);
    const Real pg    = (gam - 1.) * uu;
    const Real w_old = m::max(rho + uu + pg, SMALL_NUM);

    // Normal observer magnetic field
    Real Bcon[GR_DIM] = {0};
    Real Bcov[GR_DIM] = {0};
    if (m_p.B1 >= 0) {
        Bcon[0] = 0;
        Bcon[1] = P(m_p.B1, k, j, i);
        Bcon[2] = P(m_p.B2, k, j, i);
        Bcon[3] = P(m_p.B3, k, j, i);
    }
    DLOOP2 Bcov[mu] += G.gcov(Loci::center, j, i, mu, nu) * Bcon[nu];
    const Real Bsq   = m::max(dot(Bcon, Bcov), SMALL_NUM);
    const Real B_mag = m::sqrt(Bsq);

    // Get four-vectors again
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);

    // Normal observer fluid momentum
    Real Qcov[GR_DIM] = {0};
    Qcov[0] = w_old * Dtmp.ucon[0] * Dtmp.ucov[0] + pg;
    Qcov[1] = w_old * Dtmp.ucon[0] * Dtmp.ucov[1];
    Qcov[2] = w_old * Dtmp.ucon[0] * Dtmp.ucov[2];
    Qcov[3] = w_old * Dtmp.ucon[0] * Dtmp.ucov[3];

    // Momentum along magnetic field lines (must be held constant)
    double QdotB = dot(Bcon, Qcov);

    // Initial parallel velocity (refer R17 Eqn B10)
    Real vpar = QdotB / (B_mag * w_old * Dtmp.ucon[0]*Dtmp.ucon[0]);

    Real ucon_dr[GR_DIM] = {0};
    // t-component of drift velocity (refer R17 Eqn B13)
    ucon_dr[0] = 1. / m::sqrt(1. / (Dtmp.ucon[0]*Dtmp.ucon[0]) + vpar*vpar);
    // spatial components of drift velocity (refer R17 Eqn B11)
    DLOOP1 ucon_dr[mu] = Dtmp.ucon[mu] * (ucon_dr[0] / Dtmp.ucon[0]) - (vpar * Bcon[mu] * ucon_dr[0] / B_mag);

    // Update rho, uu and compute new enthalpy
    P(m_p.RHO, k, j, i) = m::max(rho, rhoflr_max);
    P(m_p.UU, k, j, i)  = m::max(uu, uflr_max);
    const Real pg_new   = (gam - 1.) * P(m_p.UU, k, j, i);
    const Real w_new    = P(m_p.RHO, k, j, i) + P(m_p.UU, k, j, i) + pg_new;

    // New parallel velocity (refer R17 Eqn B14)
    const Real x = (2. * QdotB) / (B_mag * w_new * ucon_dr[0]);
    vpar = x / (1 + m::sqrt(1 + x*x)) * (1. / ucon_dr[0]);

    // New fluid four velocity (refer R17 Eqns B13 and B11)
    Dtmp.ucon[0] = 1. / m::sqrt(1/(ucon_dr[0]*ucon_dr[0]) - vpar*vpar);
    DLOOP1 Dtmp.ucon[mu] = ucon_dr[mu] * (Dtmp.ucon[0] / ucon_dr[0]) + (vpar * Bcon[mu] * Dtmp.ucon[0] / B_mag);
    G.lower(Dtmp.ucon, Dtmp.ucov, k, j, i, Loci::center);

    // New velocity primitives
    P(m_p.U1, k, j, i) = Dtmp.ucon[1] + (beta[1] * Dtmp.ucon[0]);
    P(m_p.U2, k, j, i) = Dtmp.ucon[2] + (beta[2] * Dtmp.ucon[0]);
    P(m_p.U3, k, j, i) = Dtmp.ucon[3] + (beta[3] * Dtmp.ucon[0]);
    return 0;
}

template<>
KOKKOS_INLINE_FUNCTION int apply_floors<InjectionFrame::normal>(FLOOR_ONE_ARGS)
{
    // Add the material in the normal observer frame.
    // 1. Calculate how much material we're adding.
    // This is an estimate, as it's what we'd have to do in fluid frame
    const Real rho_add    = m::max(0., rhoflr_max - P(m_p.RHO, k, j, i));
    const Real u_add      = m::max(0., uflr_max - P(m_p.UU, k, j, i));
    const Real uvec[NVEC] = {0}, B[NVEC] = {0};

    // 2. Calculate the increase in conserved mass/energy corresponding to the new material.
    Real rho_ut, T[GR_DIM];
    GRMHD::p_to_u_mhd(G, rho_add, u_add, uvec, B, gam, k, j, i, rho_ut, T, Loci::center);

    // 3. Add new conserved mass/energy to the current "conserved" state.
    // Also add to the local primitives as a guess
    P(m_p.RHO, k, j, i) += rho_add;
    P(m_p.UU, k, j, i)  += u_add;
    // Add any velocity here
    U(m_u.RHO, k, j, i) += rho_ut;
    U(m_u.UU, k, j, i)  += T[0]; // Note that m_u.U1 != m_u.UU + 1 necessarily
    U(m_u.U1, k, j, i)  += T[1];
    U(m_u.U2, k, j, i)  += T[2];
    U(m_u.U3, k, j, i)  += T[3];
    
    // Recover primitive variables from conserved versions
    // Kastaun would need real vals here...
    const Floors::Prescription floor_tmp = {0}; 
    return Inverter::u_to_p<Inverter::Type::onedw>(G, U, m_u, gam, k, j, i, P, m_p, Loci::center,
                                                     floor_tmp, 8, 1e-8);
}

/**
 * Apply just the geometric floors to a set of local primitives.
 * Specifically called after reconstruction when using non-TVD schemes, e.g. WENO5.
 * Reimplemented to be fast and fit the general prim_to_flux calling convention, including geometry locations
 * 
 * @return fflag: since no inversion is performed, this just returns a flag representing which geometric floors were hit
 * 
 * NOT LOCKSTEP: Operates on and respects primitives *only*
 */
template<typename Local>
KOKKOS_INLINE_FUNCTION int apply_geo_floors(const GRCoordinates& G, Local& P, const VarMap& m,
                                            const Real& gam, const int& j, const int& i,
                                            const Floors::Prescription& floors, const Floors::Prescription& floors_inner,
                                            const Loci loc=Loci::center)
{
    // Choose our floor scheme
    const Floors::Prescription& myfloors = (floors.radius_dependent_floors
                                            && G.r(0, j, i) < floors.floors_switch_r) ? floors_inner : floors;

    // Apply only the geometric floors
    Real rhoflr_geom, uflr_geom;
    if(G.coords.is_spherical()) {
        const GReal r = G.r(0, j, i);
        // r_char sets more aggressive floor close to EH but backs off
        Real rhoscal = (myfloors.use_r_char) ? 1. / ((r*r) * (1 + r / myfloors.r_char)) : 1. / m::sqrt(r*r*r);
        rhoflr_geom = m::max(myfloors.rho_min_geom * rhoscal, myfloors.rho_min_const);
        uflr_geom   = m::max(myfloors.u_min_geom * m::pow(rhoscal, gam), myfloors.u_min_const);
    } else {
        rhoflr_geom = myfloors.rho_min_const;
        uflr_geom   = myfloors.u_min_const;
    }

    int fflag = 0;
    // Record Geometric floor hits
    fflag |= (rhoflr_geom > P(m.RHO)) * FFlag::GEOM_RHO_FLUX;
    fflag |= (uflr_geom > P(m.UU)) * FFlag::GEOM_U_FLUX;

    P(m.RHO) += m::max(0., rhoflr_geom - P(m.RHO));
    P(m.UU)  += m::max(0., uflr_geom - P(m.UU));

    return fflag;
}

template<typename Global>
KOKKOS_INLINE_FUNCTION int apply_geo_floors(const GRCoordinates& G, Global& P, const VarMap& m,
                                            const Real& gam, const int& k, const int& j, const int& i,
                                            const Floors::Prescription& floors, const Floors::Prescription& floors_inner,
                                            const Loci loc=Loci::center)
{
    // Choose our floor scheme
    const Floors::Prescription& myfloors = (floors.radius_dependent_floors
                                            && G.r(0, j, i) < floors.floors_switch_r) ? floors_inner : floors;

    // Apply only the geometric floors
    Real rhoflr_geom, uflr_geom;
    if(G.coords.is_spherical()) {
        const GReal r = G.r(0, j, i);
        // r_char sets more aggressive floor close to EH but backs off
        Real rhoscal = (myfloors.use_r_char) ? 1. / ((r*r) * (1 + r / myfloors.r_char)) : 1. / m::sqrt(r*r*r);
        rhoflr_geom = m::max(myfloors.rho_min_geom * rhoscal, myfloors.rho_min_const);
        uflr_geom   = m::max(myfloors.u_min_geom * m::pow(rhoscal, gam), myfloors.u_min_const);
    } else {
        rhoflr_geom = myfloors.rho_min_const;
        uflr_geom   = myfloors.u_min_const;
    }

    int fflag = 0;
    // Record all the floors that were hit, using bitflags
    // Record Geometric floor hits
    fflag |= (rhoflr_geom > P(m.RHO, k, j, i)) * FFlag::GEOM_RHO_FLUX;
    fflag |= (uflr_geom > P(m.UU, k, j, i)) * FFlag::GEOM_U_FLUX;

    P(m.RHO, k, j, i) += m::max(0., rhoflr_geom - P(m.RHO, k, j, i));
    P(m.UU, k, j, i)  += m::max(0., uflr_geom - P(m.UU, k, j, i));

    return fflag;
}

} // Floors
