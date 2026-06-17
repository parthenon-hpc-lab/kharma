/*
 *  File: flux_functions.hpp
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


#include "emhd.hpp"
#include "gr_coordinates.hpp"
#include "grmhd_functions.hpp"
#include "kharma_utils.hpp"
#include "types.hpp"

/**
 * Device-side functions calc_tensor, prim_to_flux, and vchar, which will depend on
 * the set of enabled packages.
 */

namespace Flux
{

// TODO Q > 0 != emhd_enabled.  Store enablement in emhd_params since we need it anyway
template<typename Local>
KOKKOS_FORCEINLINE_FUNCTION void calc_tensor(const Local& P, const VarMap& m_p,
    const FourVectors D, const EMHD::EMHD_parameters& emhd_params, const Microphysics::EOS::EOS& eos,
    const int& dir, Real T[GR_DIM])
{
    // calc pressure
    Real sie = P(m_p.UU)/P(m_p.RHO) //specific internal energy
    Real pressure = eos.PressureFromDensityInternalEnergy(P(m_p.RHO), sie);
    if ((m_p.Q >= 0 || m_p.DP >= 0) && emhd_params.feedback) {
        // Apply higher-order terms conversion if necessary
        Real qtilde = 0., dPtilde = 0.;
        if (m_p.Q >= 0) qtilde = P(m_p.Q);
        if (m_p.DP >= 0) dPtilde = P(m_p.DP);
        const Real ef = P(m_p.RHO) + P(m_p.UU) + pressure; // \rho * h = rho + u + P.
        const Real cs2 = eos.BulkModulusFromDensityInternalEnergy(P(m_p.RHO),sie)/ef;
        //TODO_EOS: Is this actually what's needed here?
        const Real Theta = pressure/P(m_p.RHO);
        
        // const Real Theta = (gam - 1) * P(m_p.UU) / P(m_p.RHO); //dimensionless temperature ? Check with cora
        // const Real cs2 = gam * (gam - 1) * P(m_p.UU) / (P(m_p.RHO) + gam * P(m_p.UU));
        Real q, dP;
        EMHD::convert_prims_to_q_dP(
            qtilde, dPtilde, P(m_p.RHO), Theta, cs2, emhd_params, q, dP);
        // Then calculate the tensor
        EMHD::calc_tensor(P(m_p.RHO), P(m_p.UU), pressure, q, dP, D, dir, T);
    } else if (m_p.B1 >= 0) {
        // GRMHD stress-energy tensor w/ first index up, second index down
        GRMHD::calc_tensor(P(m_p.RHO), P(m_p.UU), pressure, D, dir, T);
    } else {
        // GRHD stress-energy tensor
        GRHD::calc_tensor(P(m_p.RHO), P(m_p.UU), pressure, D, dir, T);
    }
}
template<typename Global>
KOKKOS_FORCEINLINE_FUNCTION void calc_tensor(const Global& P, const VarMap& m_p,
    const FourVectors D, const EMHD::EMHD_parameters& emhd_params, const Real& gam,
    const int& k, const int& j, const int& i, const int& dir, Real T[GR_DIM])
{
    if ((m_p.Q >= 0 || m_p.DP >= 0) && emhd_params.feedback) {
        // Apply higher-order terms conversion if necessary
        Real qtilde = 0., dPtilde = 0.;
        if (m_p.Q >= 0) qtilde = P(m_p.Q, k, j, i);
        if (m_p.DP >= 0) dPtilde = P(m_p.DP, k, j, i);
        const Real Theta = (gam - 1) * P(m_p.UU, k, j, i) / P(m_p.RHO, k, j, i);
        const Real cs2 = gam * (gam - 1) * P(m_p.UU, k, j, i) /
                         (P(m_p.RHO, k, j, i) + gam * P(m_p.UU, k, j, i));
        Real q, dP;
        EMHD::convert_prims_to_q_dP(
            qtilde, dPtilde, P(m_p.RHO, k, j, i), Theta, cs2, emhd_params, q, dP);

        // Then calculate the tensor
        EMHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i),
            (gam - 1) * P(m_p.UU, k, j, i), q, dP, D, dir, T);
    } else if (m_p.B1 >= 0) {
        // GRMHD stress-energy tensor w/ first index up, second index down
        GRMHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i),
            (gam - 1) * P(m_p.UU, k, j, i), D, dir, T);
    } else {
        // GRHD stress-energy tensor w/ first index up, second index down
        GRHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i),
            (gam - 1) * P(m_p.UU, k, j, i), D, dir, T);
    }
}

/**
 * Turn the primitive variables at a location into:
 * a. conserved variables (dir==0), or
 * b. fluxes in a direction (dir!=0)
 * Keep in mind loc should usually correspond to dir for perpendicuar fluxes
 */
template<typename Local>
KOKKOS_FORCEINLINE_FUNCTION void prim_to_flux(const GRCoordinates& G, const Local& P,
    const VarMap& m_p, const FourVectors D, const EMHD::EMHD_parameters& emhd_params,
    const Microphysics::EOS::EOS& eos, const int& j, const int& i, const int& dir, const Local& flux,
    const VarMap& m_u, const Loci loc = Loci::center)
{
    // Real pressure = eos.PressureFromDensityInternalEnergy(P(m_p.RHO), P(m_p.UU));
    // Real bulk = eos.BulkModulusFromDensityInternalEnergy(P(m_p.RHO), P(m_p.UU));
    // Real gam = bulk/pressure;
    
    Real gdet = G.gdet(loc, j, i);
    // Particle number flux
    flux(m_u.RHO) = P(m_p.RHO) * D.ucon[dir] * gdet;

    // Stress-energy tensor
    Real T[GR_DIM];
    calc_tensor(P, m_p, D, emhd_params, eos, dir, T);
    flux(m_u.UU) = T[0] * gdet + flux(m_u.RHO);
    flux(m_u.U1) = T[1] * gdet;
    flux(m_u.U2) = T[2] * gdet;
    flux(m_u.U3) = T[3] * gdet;

    // Magnetic field
    if (m_u.B1 >= 0) {
        // Magnetic field
        if (dir == 0) {
            VLOOP
                flux(m_u.B1 + v) = P(m_p.B1 + v) * gdet;
        } else {
            // Constraint damping w/Dedner may add also P(m_p.psi) * gdet,
            // but for us this is in the source term
            VLOOP
                flux(m_u.B1 + v) =
                    (D.bcon[v + 1] * D.ucon[dir] - D.bcon[dir] * D.ucon[v + 1]) * gdet;
        }
        // Extra scalar psi for constraint damping, see B_CD
        if (m_u.PSI >= 0) {
            if (dir == 0) {
                flux(m_u.PSI) = P(m_p.PSI) * gdet;
            } else {
                // Psi field update as in Mosta et al (IllinoisGRMHD), alternate
                // explanation Jesse et al (2020)
                // Real alpha = 1. / m::sqrt(-G.gcon(Loci::center, j, i, 0, 0));
                // Real beta_dir = G.gcon(Loci::center, j, i, 0, dir) * alpha * alpha;
                flux(m_u.PSI) =
                    (D.bcon[dir] - G.gcon(Loci::center, j, i, 0, dir) * P(m_p.PSI)) *
                    gdet;
            }
        }
    }

    // EMHD Variables: advect like rho
    if (m_u.Q >= 0) flux(m_u.Q) = P(m_p.Q) * D.ucon[dir] * gdet;
    if (m_u.DP >= 0) flux(m_u.DP) = P(m_p.DP) * D.ucon[dir] * gdet;

    // Electrons: normalized by density
    // TODO_EOS: I think this might need some changes
    if (m_u.KTOT >= 0) {
        flux(m_u.KTOT) = flux(m_u.RHO) * P(m_p.KTOT);
        if (m_u.K_CONSTANT >= 0) flux(m_u.K_CONSTANT) = flux(m_u.RHO) * P(m_p.K_CONSTANT);
        if (m_u.K_HOWES >= 0) flux(m_u.K_HOWES) = flux(m_u.RHO) * P(m_p.K_HOWES);
        if (m_u.K_KAWAZURA >= 0) flux(m_u.K_KAWAZURA) = flux(m_u.RHO) * P(m_p.K_KAWAZURA);
        if (m_u.K_WERNER >= 0) flux(m_u.K_WERNER) = flux(m_u.RHO) * P(m_p.K_WERNER);
        if (m_u.K_ROWAN >= 0) flux(m_u.K_ROWAN) = flux(m_u.RHO) * P(m_p.K_ROWAN);
        if (m_u.K_SHARMA >= 0) flux(m_u.K_SHARMA) = flux(m_u.RHO) * P(m_p.K_SHARMA);
    }
}

template<typename Global>
KOKKOS_FORCEINLINE_FUNCTION void prim_to_flux(const GRCoordinates& G, const Global& P,
    const VarMap& m_p, const FourVectors D, const EMHD::EMHD_parameters& emhd_params,
    const Real& gam, const int& k, const int& j, const int& i, const int& dir,
    Real flux[MAX_VARS], const VarMap& m_u, const Loci loc = Loci::center)
{
    Real gdet = G.gdet(loc, j, i);
    // Particle number flux
    flux[m_u.RHO] = P(m_p.RHO, k, j, i) * D.ucon[dir] * gdet;

    // Stress-energy tensor
    Real T[GR_DIM];
    calc_tensor(P, m_p, D, emhd_params, gam, k, j, i, dir, T);
    flux[m_u.UU] = T[0] * gdet + flux[m_u.RHO];
    flux[m_u.U1] = T[1] * gdet;
    flux[m_u.U2] = T[2] * gdet;
    flux[m_u.U3] = T[3] * gdet;

    // Magnetic field
    if (m_u.B1 >= 0) {
        // Magnetic field
        if (dir == 0) {
            VLOOP
                flux[m_u.B1 + v] = P(m_p.B1 + v, k, j, i) * gdet;
        } else {
            // Constraint damping w/Dedner may add also P(m_p.psi) * gdet,
            // but for us this is in the source term
            VLOOP
                flux[m_u.B1 + v] =
                    (D.bcon[v + 1] * D.ucon[dir] - D.bcon[dir] * D.ucon[v + 1]) * gdet;
        }
        // Extra scalar psi for constraint damping, see B_CD
        if (m_u.PSI >= 0) {
            if (dir == 0) {
                flux[m_u.PSI] = P(m_p.PSI, k, j, i) * gdet;
            } else {
                // Psi field update as in Mosta et al (IllinoisGRMHD), alternate
                // explanation Jesse et al (2020)
                // Real alpha = 1. / m::sqrt(-G.gcon(Loci::center, j, i, 0, 0));
                // Real beta_dir = G.gcon(Loci::center, j, i, 0, dir) * alpha * alpha;
                flux[m_u.PSI] = (D.bcon[dir] - G.gcon(Loci::center, j, i, 0, dir) *
                                                   P(m_p.PSI, k, j, i)) *
                                gdet;
            }
        }
    }

    // EMHD Variables: advect like rho
    if (m_u.Q >= 0) flux[m_u.Q] = P(m_p.Q, k, j, i) * D.ucon[dir] * gdet;
    if (m_u.DP >= 0) flux[m_u.DP] = P(m_p.DP, k, j, i) * D.ucon[dir] * gdet;

    // Electrons: normalized by density
    if (m_u.KTOT >= 0) {
        flux[m_u.KTOT] = flux[m_u.RHO] * P(m_p.KTOT, k, j, i);
        if (m_u.K_CONSTANT >= 0)
            flux[m_u.K_CONSTANT] = flux[m_u.RHO] * P(m_p.K_CONSTANT, k, j, i);
        if (m_u.K_HOWES >= 0) flux[m_u.K_HOWES] = flux[m_u.RHO] * P(m_p.K_HOWES, k, j, i);
        if (m_u.K_KAWAZURA >= 0)
            flux[m_u.K_KAWAZURA] = flux[m_u.RHO] * P(m_p.K_KAWAZURA, k, j, i);
        if (m_u.K_WERNER >= 0)
            flux[m_u.K_WERNER] = flux[m_u.RHO] * P(m_p.K_WERNER, k, j, i);
        if (m_u.K_ROWAN >= 0) flux[m_u.K_ROWAN] = flux[m_u.RHO] * P(m_p.K_ROWAN, k, j, i);
        if (m_u.K_SHARMA >= 0)
            flux[m_u.K_SHARMA] = flux[m_u.RHO] * P(m_p.K_SHARMA, k, j, i);
    }
}

template<typename Global>
KOKKOS_FORCEINLINE_FUNCTION void prim_to_flux(const GRCoordinates& G, const Global& P,
    const VarMap& m_p, const FourVectors D, const EMHD::EMHD_parameters& emhd_params,
    const Real& gam, const int& k, const int& j, const int& i, const int dir,
    const Global& flux, const VarMap& m_u, const Loci loc = Loci::center)
{
    const Real gdet = G.gdet(loc, j, i);
    // Particle number flux
    flux(m_u.RHO, k, j, i) = P(m_p.RHO, k, j, i) * D.ucon[dir] * gdet;

    Real T[GR_DIM];
    calc_tensor(P, m_p, D, emhd_params, gam, k, j, i, dir, T);
    flux(m_u.UU, k, j, i) = T[0] * gdet + flux(m_u.RHO, k, j, i);
    flux(m_u.U1, k, j, i) = T[1] * gdet;
    flux(m_u.U2, k, j, i) = T[2] * gdet;
    flux(m_u.U3, k, j, i) = T[3] * gdet;

    // Magnetic field
    if (m_u.B1 >= 0) {
        // Magnetic field
        if (dir == 0) {
            VLOOP
                flux(m_u.B1 + v, k, j, i) = P(m_p.B1 + v, k, j, i) * gdet;
        } else {
            // Constraint damping w/Dedner may add also P(m_p.psi) * gdet,
            // but for us this is in the source term
            VLOOP
                flux(m_u.B1 + v, k, j, i) =
                    (D.bcon[v + 1] * D.ucon[dir] - D.bcon[dir] * D.ucon[v + 1]) * gdet;
        }
        // Extra scalar psi for constraint damping, see B_CD
        if (m_u.PSI >= 0) {
            if (dir == 0) {
                flux(m_u.PSI, k, j, i) = P(m_p.PSI, k, j, i) * gdet;
            } else {
                // Psi field update as in Mosta et al (IllinoisGRMHD), alternate
                // explanation Jesse et al (2020)
                // Real alpha = 1. / sqrt(-G.gcon(Loci::center, j, i, 0, 0));
                // Real beta_dir = G.gcon(Loci::center, j, i, 0, dir) * alpha * alpha;
                flux(m_u.PSI, k, j, i) =
                    (D.bcon[dir] -
                        G.gcon(Loci::center, j, i, 0, dir) * P(m_p.PSI, k, j, i)) *
                    gdet;
            }
        }
    }

    // EMHD Variables: advect like rho
    if (m_u.Q >= 0) flux(m_u.Q, k, j, i) = P(m_p.Q, k, j, i) * D.ucon[dir] * gdet;
    if (m_u.DP >= 0) flux(m_u.DP, k, j, i) = P(m_p.DP, k, j, i) * D.ucon[dir] * gdet;

    // Electrons: normalized by density
    if (m_u.KTOT >= 0) {
        flux(m_u.KTOT, k, j, i) = flux(m_u.RHO, k, j, i) * P(m_p.KTOT, k, j, i);
        if (m_u.K_CONSTANT >= 0)
            flux(m_u.K_CONSTANT, k, j, i) =
                flux(m_u.RHO, k, j, i) * P(m_p.K_CONSTANT, k, j, i);
        if (m_u.K_HOWES >= 0)
            flux(m_u.K_HOWES, k, j, i) = flux(m_u.RHO, k, j, i) * P(m_p.K_HOWES, k, j, i);
        if (m_u.K_KAWAZURA >= 0)
            flux(m_u.K_KAWAZURA, k, j, i) =
                flux(m_u.RHO, k, j, i) * P(m_p.K_KAWAZURA, k, j, i);
        if (m_u.K_WERNER >= 0)
            flux(m_u.K_WERNER, k, j, i) =
                flux(m_u.RHO, k, j, i) * P(m_p.K_WERNER, k, j, i);
        if (m_u.K_ROWAN >= 0)
            flux(m_u.K_ROWAN, k, j, i) = flux(m_u.RHO, k, j, i) * P(m_p.K_ROWAN, k, j, i);
        if (m_u.K_SHARMA >= 0)
            flux(m_u.K_SHARMA, k, j, i) =
                flux(m_u.RHO, k, j, i) * P(m_p.K_SHARMA, k, j, i);
    }
}

/**
 * P->U for just the GRMHD variables, but using the full tensor.  Needed with floors and
 * in a few places
 */
template<typename Global>
KOKKOS_FORCEINLINE_FUNCTION void prim_to_flux_mhd(const GRCoordinates& G, const Global& P,
    const VarMap& m_p, const FourVectors D, const EMHD::EMHD_parameters& emhd_params,
    const Real& gam, const int& k, const int& j, const int& i, const int dir,
    const Global& flux, const VarMap& m_u, const Loci loc = Loci::center)
{
    const Real& gdet = G.gdet(loc, j, i);
    // Particle number flux
    flux(m_u.RHO, k, j, i) = P(m_p.RHO, k, j, i) * D.ucon[dir] * gdet;

    Real T[GR_DIM];
    calc_tensor(P, m_p, D, emhd_params, gam, k, j, i, dir, T);
    flux(m_u.UU, k, j, i) = T[0] * gdet + flux(m_u.RHO, k, j, i);
    flux(m_u.U1, k, j, i) = T[1] * gdet;
    flux(m_u.U2, k, j, i) = T[2] * gdet;
    flux(m_u.U3, k, j, i) = T[3] * gdet;
}

/**
 * Get the conserved (E)GRMHD variables corresponding to primitives in a zone. Equivalent
 * to prim_to_flux with dir==0
 */
template<typename Local>
KOKKOS_FORCEINLINE_FUNCTION void p_to_u(const GRCoordinates& G, const Local& P,
    const VarMap& m_p, const EMHD::EMHD_parameters& emhd_params, const Real& gam,
    const int& j, const int& i, const Local& U, const VarMap& m_u,
    const Loci& loc = Loci::center)
{
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, P, m_p, j, i, loc, Dtmp);
    prim_to_flux(G, P, m_p, Dtmp, emhd_params, gam, j, i, 0, U, m_u, loc);
}

template<typename Global>
KOKKOS_FORCEINLINE_FUNCTION void p_to_u(const GRCoordinates& G, const Global& P,
    const VarMap& m_p, const EMHD::EMHD_parameters& emhd_params, const Real& gam,
    const int& k, const int& j, const int& i, const Global& U, const VarMap& m_u,
    const Loci& loc = Loci::center)
{
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
    prim_to_flux(G, P, m_p, Dtmp, emhd_params, gam, k, j, i, 0, U, m_u, loc);
}

template<typename Global>
KOKKOS_FORCEINLINE_FUNCTION void p_to_u_mhd(const GRCoordinates& G, const Global& P,
    const VarMap& m_p, const EMHD::EMHD_parameters& emhd_params, const Real& gam,
    const int& k, const int& j, const int& i, const Global& U, const VarMap& m_u,
    const Loci& loc = Loci::center)
{
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
    prim_to_flux_mhd(G, P, m_p, Dtmp, emhd_params, gam, k, j, i, 0, U, m_u, loc);
}

/**
 * Calculate components of magnetosonic velocity from primitive variables
 */
template<typename Local>
KOKKOS_FORCEINLINE_FUNCTION void vchar(const GRCoordinates& G, const Local& P,
    const VarMap& m, const FourVectors& D, const Microphysics::EOS::EOS& eos,
    const EMHD::EMHD_parameters& emhd_params, const int& k, const int& j, const int& i,
    const Loci& loc, const int& dir, Real& cmax, Real& cmin)
{
    // Find sound speed
    //TODO_EOS: Units from singularity-eos are probably wrong here, we need to find a way to convert pressure to code units
  const Real sie = P(m.UU)/P(m.RHO)
    const Real ef1 = P(m.RHO) +  1.666667 * P(m.UU); // \rho * h = rho + u + P.
    const Real pressure = eos.PressureFromDensityInternalEnergy(P(m.RHO),sie);
    fprintf(stderr, "Pressure_sing_eos = %.15e, ideal_gas = %.15e\n", pressure, (1.666667 - 1.) * P(m.UU));
    const Real bulk = eos.BulkModulusFromDensityInternalEnergy(P(m.RHO), sie);
    const Real ef = P(m.RHO) + pressure + P(m.UU);
    //fprintf(stderr, "ef1: %.15e, ef: %.15e, diff: %.15e, gam: %.15e\n", ef1, ef, (ef1 - ef)/ef1, bulk/pressure);
    const Real gam = bulk / pressure;
    // The fluid sound speed should be at most sqrt(gam-1) for a relativistic fluid
    const Real cs2 = clip(bulk / ef, 0., gam - 1.);
    Real cms2;
    if (m.Q >= 0 || m.DP >= 0) {
        // Get the EGRMHD parameters
        Real tau, chi_e, nu_e;
        EMHD::set_parameters(G, P, m, emhd_params, gam, j, i, tau, chi_e, nu_e);

        // Find fast magnetosonic speed
        const Real bsq = dot(D.bcon, D.bcov);
        const Real va2 = bsq / (bsq + ef);
        // TODO_EOS: This might need to be changed for general eos.
        const Real ccond2 =
            (m.Q >= 0) ? (gam - 1.) * emhd_params.conduction_alpha * cs2 : 0.0;
        const Real cvis2 = (m.DP >= 0) ? (4. / 3.) / (P(m.RHO) + (gam * P(m.UU))) *
                                             P(m.RHO) * emhd_params.viscosity_alpha * cs2
                                       : 0.0;

        const Real cs2_emhd =
            0.5 * (cs2 + ccond2 + m::sqrt(cs2 * cs2 + ccond2 * ccond2)) + cvis2;

        cms2 = cs2_emhd + va2 - cs2_emhd * va2;
    } else if (m.B1 >= 0) {
        // Find fast magnetosonic speed
        const Real bsq = dot(D.bcon, D.bcov);
        const Real va2 = bsq / (bsq + ef);

        cms2 = cs2 + va2 - cs2 * va2;
    } else {
        cms2 = cs2;
    }
    // The signal speed should be at most the speed of light
    cms2 = clip(cms2, 0., 1.); // TODO would love to record this...

    // Require that speed of wave measured by observer q.ucon is cms2
    Real A, B, C;
    {
        Real Bcov[GR_DIM] = {1., 0., 0., 0.};
        Real Acov[GR_DIM] = {0};
        Acov[dir] = 1.;

        Real Acon[GR_DIM], Bcon[GR_DIM];
        G.raise(Acov, Acon, k, j, i, loc);
        G.raise(Bcov, Bcon, k, j, i, loc);

        const Real Asq = dot(Acon, Acov);
        const Real Bsq = dot(Bcon, Bcov);
        const Real Au = dot(Acov, D.ucon);
        const Real Bu = dot(Bcov, D.ucon);
        const Real AB = dot(Acon, Bcov);

        A = Bu * Bu - (Bsq + Bu * Bu) * cms2;
        B = 2. * (Au * Bu - (AB + Au * Bu) * cms2);
        C = Au * Au - (Asq + Au * Au) * cms2;
    }

    Real discr = m::sqrt(m::max(B * B - 4. * A * C, 0.));

    Real vp = -(-B + discr) / (2. * A);
    Real vm = -(-B - discr) / (2. * A);

    cmax = m::max(vp, vm);
    cmin = m::min(vp, vm);
}

// This is expressly for updating cmin/max for FOFC zones
// It's named differently because it already took k,j,i so we can't pull the overloading
// w/different signatures trick
template<typename Global>
KOKKOS_FORCEINLINE_FUNCTION void vchar_global(const GRCoordinates& G, const Global& P,
    const VarMap& m, const FourVectors& D, const Real& gam,
    const EMHD::EMHD_parameters& emhd_params, const int& k, const int& j, const int& i,
    const Loci& loc, const int& dir, Real& cmax, Real& cmin)
{
    // Find sound speed
    const Real ef = P(m.RHO, k, j, i) + gam * P(m.UU, k, j, i);
    // The fluid sound speed should be at most sqrt(gam-1) for a relativistic fluid
    const Real cs2 = clip(gam * (gam - 1) * P(m.UU, k, j, i) / ef, 0., gam - 1.);
    Real cms2;
    if (m.Q >= 0 || m.DP >= 0) {
        // Get the EGRMHD parameters
        Real tau, chi_e, nu_e;
        EMHD::set_parameters(G, P, m, emhd_params, gam, k, j, i, tau, chi_e, nu_e);

        // Find fast magnetosonic speed
        const Real bsq = dot(D.bcon, D.bcov);
        const Real va2 = bsq / (bsq + ef);

        const Real ccond2 =
            (m.Q >= 0) ? (gam - 1.) * emhd_params.conduction_alpha * cs2 : 0.0;
        const Real cvis2 =
            (m.DP >= 0) ? (4. / 3.) / (P(m.RHO, k, j, i) + (gam * P(m.UU, k, j, i))) *
                              P(m.RHO, k, j, i) * emhd_params.viscosity_alpha * cs2
                        : 0.0;

        const Real cs2_emhd =
            0.5 * (cs2 + ccond2 + m::sqrt(cs2 * cs2 + ccond2 * ccond2)) + cvis2;

        cms2 = cs2_emhd + va2 - cs2_emhd * va2;
    } else if (m.B1 >= 0) {
        // Find fast magnetosonic speed
        const Real bsq = dot(D.bcon, D.bcov);
        const Real va2 = bsq / (bsq + ef);

        cms2 = cs2 + va2 - cs2 * va2;
    } else {
        cms2 = cs2;
    }
    // The signal speed should be at most the speed of light
    cms2 = clip(cms2, 0., 1.);

    // Require that speed of wave measured by observer q.ucon is cms2
    Real A, B, C;
    {
        Real Bcov[GR_DIM] = {1., 0., 0., 0.};
        Real Acov[GR_DIM] = {0};
        Acov[dir] = 1.;

        Real Acon[GR_DIM], Bcon[GR_DIM];
        G.raise(Acov, Acon, k, j, i, loc);
        G.raise(Bcov, Bcon, k, j, i, loc);

        const Real Asq = dot(Acon, Acov);
        const Real Bsq = dot(Bcon, Bcov);
        const Real Au = dot(Acov, D.ucon);
        const Real Bu = dot(Bcov, D.ucon);
        const Real AB = dot(Acon, Bcov);

        A = Bu * Bu - (Bsq + Bu * Bu) * cms2;
        B = 2. * (Au * Bu - (AB + Au * Bu) * cms2);
        C = Au * Au - (Asq + Au * Au) * cms2;
    }

    Real discr = m::sqrt(m::max(B * B - 4. * A * C, 0.));

    Real vp = -(-B + discr) / (2. * A);
    Real vm = -(-B - discr) / (2. * A);

    cmax = m::max(vp, vm);
    cmin = m::min(vp, vm);
}

} // namespace Flux
