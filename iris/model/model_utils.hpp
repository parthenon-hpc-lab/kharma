/*
 *  File: model_utils.hpp
 *
 *  BSD 3-Clause License
 *
 *  Copyright (c) 2025, Iris contributors
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

// Iris headers
#include "units.hpp"

// KHARMA headers
#include "decs.hpp"
#include "matrix.hpp"

/* return angle between magnetic field and wavevector */
KOKKOS_INLINE_FUNCTION double get_bk_mu(const double Kcon[GR_DIM],
    const double Ucov[GR_DIM], const double Bcov[GR_DIM], const double& B)
{
    if (B == 0.) return 0.; // cosine of pi/2

    double k = m::abs(dot(Kcon, Ucov));

    double mu = dot(Kcon, Bcov) / (k * B);

    if (m::abs(mu) > 1.) mu /= m::abs(mu);

    return mu;
}

/* get frequency in fluid frame, in Hz */
KOKKOS_INLINE_FUNCTION double get_fluid_nu(
    const double Kcon[GR_DIM], const double Ucov[GR_DIM])
{
    // Get the energy in electron rest-mass units and divide
    return -(Kcon[0] * Ucov[0] + Kcon[1] * Ucov[1] + Kcon[2] * Ucov[2] +
               Kcon[3] * Ucov[3]) *
           ME * CL * CL / HPL;
}
