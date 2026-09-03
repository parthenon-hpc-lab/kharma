/* 
 *  File: force_free_functions.hpp
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

// TODO remove and make runtime
// It needs to modify behavior *inside* prim_to_flux tho
#define FORCE_FREE_COLD

namespace Force_Free {

/**
 * Get a row of the E&M-only stress-energy tensor with first index up, second index down.
 * A factor of m::sqrt(4 pi) is absorbed into the definition of b.
 * See Chael '24.
 *
 * Entirely local!
 */
KOKKOS_INLINE_FUNCTION void calc_tensor(const FourVectors& D, const int dir,
                                        Real em[GR_DIM])
{
    const Real bsq = dot(D.bcon, D.bcov);
    DLOOP1 {
        em[mu] = bsq * D.ucon[dir] * D.ucov[mu] +
                  0.5 * bsq * (dir == mu) -
                  D.bcon[dir] * D.bcov[mu];
    }
}

KOKKOS_INLINE_FUNCTION Real calc_ufromS(Real S, Real rho, Real gam)
{
    const Real gm1 = gam - 1.;
#ifdef NOLOGINS
    return m::pow(rho,gamma)*(S/rho);
#else
    return m::pow(m::pow(rho, 1.0 / gm1 + 1.) * m::exp(S / rho), gm1) / gm1;
#endif
}

}
