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

#include "decs.hpp"

#include "reconstruction.hpp"

using KReconstruction::slope_calc;

/**
 * Utilities for the EMHD source terms, things we might conceivably use somewhere else,
 * or use *from* somewhere else instead of here.
 *
 * 1. Slopes at faces using various linear reconstructions.  Since this is unrelated to
 *    reconstructing all prims, and only called zone-wise, the "same" recon algos are
 * reimplemented here
 * 2. Calculate gradient of each component of ucov &
 */

namespace EMHD
{

// Compute gradient of four velocities and temperature
// Called by emhd_explicit_sources
template<KReconstruction::Type recon>
KOKKOS_INLINE_FUNCTION void gradient_calc(const GRCoordinates& G,
    const VariablePack<Real>& Temps, const int& uvec_index, const int& theta_index,
    const int& b, const int& k, const int& j, const int& i, const bool& do_3d,
    const bool& do_2d, Real grad_ucov[GR_DIM][GR_DIM], Real grad_Theta[GR_DIM])
{
    // Compute gradient of ucov
    DLOOP1 {
        grad_ucov[0][mu] = 0;
        // slope in direction nu of component mu
        grad_ucov[1][mu] = slope_calc<recon, X1DIR>(G, Temps, uvec_index + mu, k, j, i);
        grad_ucov[2][mu] =
            (do_2d) ? slope_calc<recon, X2DIR>(G, Temps, uvec_index + mu, k, j, i) : 0.;
        grad_ucov[3][mu] =
            (do_3d) ? slope_calc<recon, X3DIR>(G, Temps, uvec_index + mu, k, j, i) : 0.;
    }
    // TODO skip this if flat space?
    DLOOP3
        grad_ucov[mu][nu] -= G.conn(j, i, lam, mu, nu) * Temps(uvec_index + lam, k, j, i);

    // Compute temperature gradient
    // Time derivative component is computed in time_derivative_sources
    grad_Theta[0] = 0;
    grad_Theta[1] = slope_calc<recon, X1DIR>(G, Temps, theta_index, k, j, i);
    grad_Theta[2] =
        (do_2d) ? slope_calc<recon, X2DIR>(G, Temps, theta_index, k, j, i) : 0.;
    grad_Theta[3] =
        (do_3d) ? slope_calc<recon, X3DIR>(G, Temps, theta_index, k, j, i) : 0.;
}

} // namespace EMHD
