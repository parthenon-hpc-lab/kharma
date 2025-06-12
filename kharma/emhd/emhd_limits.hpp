/* 
 *  File: emhd_limits.hpp
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

#include "emhd.hpp"

#include "flux_functions.hpp"

// Flags for the extended MHD limits
// TODO(BSP) name/list and print like FFlag
#define HIT_Q_LIMIT      1
#define HIT_DP_MAX_LIMIT 2
#define HIT_DP_MIN_LIMIT 4

namespace EMHD {

/**
 * Apply limits on the Extended MHD variables
 * 
 * @return elag, a bitflag indicating whether each particular limit was hit, allowing representation of arbitrary combinations
 * See decs.h for bit names.
 * 
 * The maximum heat flux is limited by the saturated value given by a hot cloud in cold gas.
 * The bounds on the pressure anisotropy as due to the mirror and firehose instability limits.
 * 
 * Although only q, dP are updated here, prim_to_flux updates all conserved. 
 * This shouldn't be an issue though since PtoU in analytic and will result in the same value for the ideal MHD variables.
 */
KOKKOS_INLINE_FUNCTION int apply_instability_limits(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
                                          const Real& gam, const EMHD::EMHD_parameters& emhd_params, 
                                          const int& k, const int& j, const int& i,
                                          const VariablePack<Real>& U, const VarMap& m_u, const Loci loc=Loci::center)
{
    int eflag = 0;

    Real rho      = P(m_p.RHO, k, j, i);
    Real uu       = P(m_p.UU, k, j, i);
    Real qtilde  = (m_p.Q >= 0) ? P(m_p.Q, k, j, i) : 0.;
    Real dPtilde = (m_p.DP >= 0) ? P(m_p.DP, k, j, i) : 0.;

    Real pg    = (gam - 1.) * uu;
    Real Theta = pg / rho;
    Real cs    = m::sqrt(gam * pg / (rho + (gam * uu)));

    FourVectors D;
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, D);
    Real bsq = m::max(dot(D.bcon, D.bcov), SMALL_NUM);

    Real tau, chi_e, nu_e;
    EMHD::set_parameters(G, P, m_p, emhd_params, gam, k, j, i, tau, chi_e, nu_e);

    Real q, dP;
    EMHD::convert_prims_to_q_dP(qtilde, dPtilde, rho, Theta, cs*cs, emhd_params, q, dP);

    // if (i==128 && j==128) {
    //     printf("\n---INSTABILITY LIMITS---\n");
    //     printf("---Before applying limits---\n");
    //     printf("Bsq:      %g\n", bsq);
    //     printf("dP_tilde: %g\n", P(m_p.DP, k, j, i));
    //     printf("dP:       %g\n", dP);
    // }

    if (m_p.Q >= 0) {
        Real qmax         = 1.07 * rho * cs*cs*cs;
        Real max_frac     = m::max(m::abs(q) / qmax, 1.);
        if (m::abs(q) / qmax > 1.)
            eflag |= HIT_Q_LIMIT;

        P(m_p.Q, k, j, i) = P(m_p.Q, k, j, i) / max_frac;
    }

    if (m_p.DP >= 0) {

        Real dP_comp_ratio = m::max(pg - 2./3. * dP, SMALL_NUM) / m::max(pg + 1./3. * dP, SMALL_NUM);
        Real dP_plus       = m::min(1.07 * 0.5 * bsq * dP_comp_ratio, 1.49 * pg);
        Real dP_minus      = m::max(-1.07 * bsq, -2.99 * pg);

        if (dP > 0. && (dP / dP_plus > 1.))
            eflag |= HIT_DP_MAX_LIMIT;
        else if (dP < 0. && (dP / dP_minus > 1.))
            eflag |= HIT_DP_MIN_LIMIT;
        
        if (dP > 0.)
            P(m_p.DP, k, j, i) = P(m_p.DP, k, j, i) * (1. / m::max(dP / dP_plus, 1.));
        else
            P(m_p.DP, k, j, i) = P(m_p.DP, k, j, i) * (1. / m::max(dP / dP_minus, 1.));

    }

    Flux::p_to_u(G, P, m_p, emhd_params, gam, k, j, i, U, m_u);

    return eflag;
}    

} // EMHD
