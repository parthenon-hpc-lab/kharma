/* 
 *  File: emhd.hpp
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

#include <parthenon/parthenon.hpp>

#include "grmhd_functions.hpp"

using namespace parthenon;

// Always disabled when implicit solver is disabled
// TODO separate flag, also error properly at runtime
#if DISABLE_IMPLICIT
#define DISABLE_EMHD 1
#endif

/**
 * This physics package implements the Extended GRMHD "EGRMHD" scheme of Chandra et al. 2015,
 * First implemented in GRIM, of Chandra et al. 2017.
 * 
 * It adds variables representing viscosity and heat conduction, with a combination of explicit
 * and implicit source terms; thus it requires a semi-implicit scheme for evolution,
 * implemented in KHARMA as ImexDriver.
 */
namespace EMHD {

enum ClosureType{constant=0, soundspeed, kappa_eta, torus};

class EMHD_parameters {
    public:

        bool higher_order_terms;
        bool feedback;
        ClosureType type;
        Real tau;

        Real conduction_alpha;
        Real viscosity_alpha;

        Real kappa;
        Real eta;

        void print() const
        {
            printf("EMHD Parameters:\n");
            printf("higher order: %d feedback: %d \n",
                    higher_order_terms, feedback);
            printf("kappa: %g eta: %g tau: %g conduction_a: %g viscosity_a: %g \n",
                    kappa, eta, tau, conduction_alpha, viscosity_alpha);
            // TODO closuretype
        }

};

/**
 * Initialization: handle parameters, 
 */
std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

/**
 * Add EGRMHD explicit source terms: anything which can be calculated once
 * and added to the general dU/dt term along with e.g. GRMHD source, wind, etc
 */
TaskStatus AddSource(MeshData<Real> *md, MeshData<Real> *mdudt, IndexDomain domain);

/**
 * Set q and dP to sensible starting values if they are not initialized by the problem.
 * Currently a no-op as sensible values are zeros.
 */
void InitEMHDVariables(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin);

/**
 * Recover primitive qtilde, dPtilde from "conserved" forms {qtilde,dPtilde}*u^0*gdet,
 * and vice versa.
 * These are *not* called in the usual places for explicitly-evolved variables, but instead
 * only on boundaries in order to sync the primitive/conserved variables specifically.
 */
void BlockUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse);
void MeshUtoP(MeshData<Real> *md, IndexDomain domain, bool coarse=false);
void BlockPtoU(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse);

/**
 * Add EGRMHD explicit source terms: anything which can be calculated once
 * and added to the general dU/dt term along with e.g. GRMHD source, wind, etc
 */
TaskStatus AddSource(MeshData<Real> *md, MeshData<Real> *mdudt, IndexDomain domain);

/**
 * Set q and dP to sensible starting values if they are not initialized by the problem.
 * Currently a no-op as sensible values are zeros.
 */
void InitEMHDVariables(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin);

/**
 * Apply limits on the Extended MHD variables q & dP based on instabilities.
 *
 * LOCKSTEP: this function respects P and returns consistent P<->U
 */
void ApplyEMHDLimits(MeshBlockData<Real> *mbd, IndexDomain domain);

/**
 * Get the EMHD parameters needed on the device side.
 * This function exists to be able to easily return a null
 * EMHD_parameters object even if the "EMHD" package is not loaded.
 */
inline EMHD_parameters GetEMHDParameters(Packages_t& packages)
{
    EMHD::EMHD_parameters emhd_params_tmp = {0};
    if (packages.AllPackages().count("EMHD")) {
        emhd_params_tmp = packages.Get("EMHD")->Param<EMHD::EMHD_parameters>("emhd_params");
    }
    return emhd_params_tmp;
}

#if DISABLE_EMHD

template<typename Local>
KOKKOS_INLINE_FUNCTION void set_parameters(const GRCoordinates& G, const Local& P, const VarMap& m_p,
                                           const EMHD_parameters& emhd_params, const Real& gam,
                                           const int& j, const int& i,
                                           Real& tau, Real& chi_e, Real& nu_e) {}

KOKKOS_INLINE_FUNCTION void set_parameters(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
                                           const EMHD_parameters& emhd_params, const Real& gam,
                                           const int& k, const int& j, const int& i,
                                           Real& tau, Real& chi_e, Real& nu_e) {}

KOKKOS_INLINE_FUNCTION void set_parameters_init(const GRCoordinates& G, const Real& rho, const Real& u,
                                           const EMHD_parameters& emhd_params, const Real& gam,
                                           const int& k, const int& j, const int& i,
                                           Real& tau, Real& chi_e, Real& nu_e) {}

KOKKOS_INLINE_FUNCTION void calc_tensor(const Real& rho, const Real& u, const Real& pgas,
                                        const EMHD::EMHD_parameters& emhd_params, 
                                        const Real& q, const Real& dP,
                                        const FourVectors& D, const int& dir,
                                        Real emhd[GR_DIM]) {}

KOKKOS_INLINE_FUNCTION void convert_prims_to_q_dP(const Real& q_tilde, const Real& dP_tilde,
                                        const Real& rho, const Real& Theta, const Real& cs2, 
                                        const EMHD_parameters& emhd_params, Real& q, Real& dP) {}

#else

/**
 * Set chi, nu, tau. Problem dependent
 */
KOKKOS_INLINE_FUNCTION void set_parameters(const GRCoordinates& G, const Real& rho, const Real& u,
                                            const Real& qtilde, const Real& dPtilde, const Real& bsq,
                                            const EMHD_parameters& emhd_params, const Real& gam,
                                            const int& j, const int& i,
                                            Real& tau, Real& chi_e, Real& nu_e)
{
    // Formerly chi_e was only set if conduction was present, nu_e only if viscosity
    // Now assumes these will simply be unused if set when these effects are disabled
    if (emhd_params.type == ClosureType::constant) {
        // Set tau, nu, chi to constants
        tau = emhd_params.tau;
        chi_e = emhd_params.conduction_alpha;
        nu_e = emhd_params.viscosity_alpha;

    } else if (emhd_params.type == ClosureType::soundspeed) {
        // Set tau=const, chi/nu prop. to sound speed squared
        const Real cs2 = (gam * (gam - 1.) * u) / (rho + (gam * u));
        tau = emhd_params.tau;
        chi_e = emhd_params.conduction_alpha * cs2 * tau;
        nu_e = emhd_params.viscosity_alpha * cs2 * tau;

    } else if (emhd_params.type == ClosureType::kappa_eta){
        // Set tau = const, chi = kappa / rho, nu = eta / rho
        tau = emhd_params.tau;
        chi_e = emhd_params.kappa / m::max(rho, SMALL_NUM);
        nu_e = emhd_params.eta / m::max(rho, SMALL_NUM);

    } else if (emhd_params.type == ClosureType::torus) {
        GReal Xembed[GR_DIM];
        G.coord_embed(0, j, i, Loci::center, Xembed);
        const GReal r = Xembed[1];

        // Compute dynamical time scale
        const Real tau_dyn = m::sqrt(r*r*r);
        tau = tau_dyn;

        const Real pg    = (gam - 1.) * u;
        const Real Theta = pg / rho;
        // Compute local sound speed, ensure it is defined and >0
        // Passing NaN disables an upper bound (TODO should we have one?)
        const Real cs2 = clip(gam * pg / (rho + (gam * u)), SMALL_NUM, 0./0.);

        constexpr Real lambda = 0.01;

        // Correction due to heat conduction
        {
            const Real q = (emhd_params.higher_order_terms)
                        ? qtilde * m::sqrt(rho * emhd_params.conduction_alpha * cs2 * Theta * Theta)
                        : qtilde;
            const Real q_max   = emhd_params.conduction_alpha * rho * cs2 * m::sqrt(cs2);
            const Real q_ratio = m::abs(q) / q_max;
            const Real inv_exp_g = m::exp(-(q_ratio - 1.) / lambda);
            const Real f_fmin    = inv_exp_g / (inv_exp_g + 1.) + 1.e-5;

            tau = m::min(tau, f_fmin * tau_dyn);
        }

        // Correction due to pressure anisotropy
        {
            const Real dP = (emhd_params.higher_order_terms)
                        ? dPtilde * sqrt(rho * emhd_params.viscosity_alpha * cs2 * Theta)
                        : dPtilde;
            const Real dP_comp_ratio = m::max(pg - 2./3. * dP, SMALL_NUM) /
                                       m::max(pg + 1./3. * dP, SMALL_NUM);
            const Real dP_max = (dP > 0.)
                              ? m::min(0.5 * bsq * dP_comp_ratio, 1.49 * pg / 1.07)
                              : m::max(-bsq, -2.99 * pg / 1.07);

            const Real dP_ratio = m::abs(dP) / (m::abs(dP_max) + SMALL_NUM);
            const Real inv_exp_g = m::exp((1. - dP_ratio) / lambda);
            const Real f_fmin    = inv_exp_g / (inv_exp_g + 1.) + 1.e-5;

            tau = m::min(tau, f_fmin * tau_dyn);
        }

        // Update thermal diffusivity and kinematic viscosity
        Real max_alpha = (1 - cs2) / (2 * cs2 + 1.e-12);
        chi_e = m::min(max_alpha, emhd_params.conduction_alpha) * cs2 * tau;
        nu_e = m::min(max_alpha, emhd_params.viscosity_alpha) * cs2 * tau;
    }
}
template<typename Local>
KOKKOS_INLINE_FUNCTION void set_parameters(const GRCoordinates& G, const Local& P, const VarMap& m_p,
                                           const EMHD_parameters& emhd_params, const Real& gam,
                                           const int& j, const int& i,
                                           Real& tau, Real& chi_e, Real& nu_e)
{
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, P, m_p, j, i, Loci::center, Dtmp);
    double bsq = m::max(dot(Dtmp.bcon, Dtmp.bcov), SMALL_NUM);
    Real qtilde = (m_p.Q >= 0) ? P(m_p.Q) : 0.;
    Real dPtilde = (m_p.DP >= 0) ? P(m_p.DP) : 0.;
    set_parameters(G, P(m_p.RHO), P(m_p.UU), qtilde, dPtilde,
                    bsq, emhd_params, gam, j, i, tau, chi_e, nu_e);
}

KOKKOS_INLINE_FUNCTION void set_parameters(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
                                           const EMHD_parameters& emhd_params, const Real& gam,
                                           const int& k, const int& j, const int& i,
                                           Real& tau, Real& chi_e, Real& nu_e)
{
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
    double bsq = m::max(dot(Dtmp.bcon, Dtmp.bcov), SMALL_NUM);
    Real qtilde = (m_p.Q >= 0) ? P(m_p.Q, k, j, i) : 0.;
    Real dPtilde = (m_p.DP >= 0) ? P(m_p.DP, k, j, i) : 0.;
    set_parameters(G, P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i), qtilde, dPtilde,
                    bsq, emhd_params, gam, j, i, tau, chi_e, nu_e);
}

/**
 * Get a row of the EMHD stress-energy tensor with first index up, second index down.
 * A factor of m::sqrt(4 pi) is absorbed into the definition of b.
 * Note this must be passed the full q, dP, not the primitive prims.q, usually denote qtilde
 *
 * Entirely local!
 */
KOKKOS_INLINE_FUNCTION void calc_tensor(const Real& rho, const Real& u, const Real& pgas,
                                        const Real& q, const Real& dP,
                                        const FourVectors& D, const int& dir,
                                        Real emhd[GR_DIM])
{
    const Real bsq  = m::max(dot(D.bcon, D.bcov), SMALL_NUM);
    const Real b_mag = m::sqrt(bsq);
    const Real eta  = pgas + rho + u + bsq;
    const Real ptot = pgas + 0.5 * bsq;

    DLOOP1 emhd[mu] = eta * D.ucon[dir] * D.ucov[mu] + ptot * (dir == mu) - D.bcon[dir] * D.bcov[mu]
                    + (q / b_mag) * ((D.ucon[dir] * D.bcov[mu]) + (D.bcon[dir] * D.ucov[mu]))
                    - dP * ((D.bcon[dir] * D.bcov[mu] / bsq) - (1./3.) * ((dir == mu) + D.ucon[dir] * D.ucov[mu]));
}

// Convert q_tilde and dP_tilde (which are primitives) to q and dP
// This is required because the stress-energy tensor depends on q and dP
KOKKOS_INLINE_FUNCTION void convert_prims_to_q_dP(const Real& q_tilde, const Real& dP_tilde,
                                        const Real& rho, const Real& Theta, const Real& cs2, 
                                        const EMHD_parameters& emhd_params, Real& q, Real& dP)
{
    q = q_tilde;
    if (emhd_params.higher_order_terms) {
        if (emhd_params.type == ClosureType::kappa_eta)
            q *= m::sqrt(emhd_params.kappa * Theta * Theta / emhd_params.tau);
        else
            q *= m::sqrt(rho * emhd_params.conduction_alpha * cs2 * Theta * Theta);
    }

    dP = dP_tilde;
    if (emhd_params.higher_order_terms) {
        if (emhd_params.type == ClosureType::kappa_eta)
            dP *= m::sqrt(emhd_params.eta * Theta / emhd_params.tau);
        else
            dP *= m::sqrt(rho * emhd_params.viscosity_alpha * cs2 * Theta);
    }
}
#endif

} // namespace EMHD
