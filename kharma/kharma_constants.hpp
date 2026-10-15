/*
 *  File: kharma_constants.hpp
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

// Compile-time hardcoded constants, one section per namespace.
// When passing them in functions that accept const &, pass them
// as e.g. (Real)LFAC_CAP to create a temporary copy,
// because nvcc does not bind constexpr variables with const &.

#include "decs.hpp"

namespace Floors
{
// Hard cap on the Lorentz factor
static constexpr Real LFAC_CAP = 50.;
} // namespace Floors

namespace Inverter
{
// Default iterations & tolerance of each inverter (inverter/iter_max, inverter/err_tol)
static constexpr int KASTAUN_ITER_MAX = 25;
static constexpr Real KASTAUN_ERR_TOL = 1.e-14;
static constexpr int ONEDW_ITER_MAX = 8;
static constexpr Real ONEDW_ERR_TOL = 1.e-8;
// Default iterations of the backstop bisections (inverter/backstop_iter_max)
static constexpr int BACKSTOP_ITER_MAX = 100;
// Fixups average a bad zone over its good neighbors, or fall back to floors if the
// neighbors' total weight is below this
static constexpr Real NEIGHBOR_WSUM_MIN = 1.e-10;
} // namespace Inverter

namespace EMHD
{
// The pressure anisotropy dP = P_perp - P_par must keep P_par >= 0 (dP <= 3/2 P) and
// P_perp >= 0 (dP >= -3 P): stay just inside those bounds
static constexpr Real DP_MAX_OVER_P = 1.49;
static constexpr Real DP_MIN_OVER_P = -2.99;
// Ratio of the hard limits (emhd_limits.hpp) to the soft limits that shorten the
// relaxation time (emhd.hpp)
static constexpr Real LIMIT_MARGIN = 1.07;
// Minimum relaxation time, as a fraction of the dynamical time
static constexpr Real TAU_MIN_FRAC = 1.e-5;
} // namespace EMHD

namespace RadM1
{
// Radiation energy density set when the radiation inversion fails
static constexpr Real ERAD_FALLBACK = 1.e-300;
// Tolerance of the Kastaun inversions inside the implicit radiation solve, looser than
// Inverter::KASTAUN_ERR_TOL
static constexpr Real IMPLICIT_KASTAUN_ERR_TOL = 1.e-12;
} // namespace RadM1

namespace Units
{
// Physical constants in CGS
static constexpr Real SIGMA_SB_CGS = 5.670374419e-5;
static constexpr Real SIGMA_THOMSON_CGS = 6.6524587158e-25;
// Electron and proton masses
static constexpr Real ME_CGS = 9.1093826e-28;
static constexpr Real MP_CGS = 1.67262171e-24;
static constexpr Real MSUN_CGS = 1.989e33;
// Electron scattering opacity of fully ionized hydrogen, cm^2/g
static constexpr Real KAPPA_ES_CGS = 0.4;
} // namespace Units
