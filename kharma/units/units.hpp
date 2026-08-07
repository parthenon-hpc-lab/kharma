/*
 *  File: units.hpp
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

namespace Units
{

inline Real M_BH = 1.0;
inline Real rho_scale = 1.0;
inline Real L_unit = 1.0;
inline Real T_unit = 1.0;
inline Real U_unit = 1.0;
// Constants in CGS units.
constexpr Real Gnewt_cgs = 6.67430e-8;
constexpr Real clight_cgs = 2.99792458e10;
constexpr Real msun_cgs = 1.989e33;

/**
 * Initialize the Units package.
 */
inline std::shared_ptr<KHARMAPackage> Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages)
{
    // Create the package object
    auto pkg = std::make_shared<KHARMAPackage>("Units");
    Params& params = pkg->AllParams();

    // Read parameters from par file
    // It defaults to 10 solar masses, but it should be always set in the par file.
    M_BH = pin->GetOrAddReal("units", "M_BH", 10.0);
    rho_scale = pin->GetOrAddReal("units", "rho_scale", 1e-6);
    // TODO: instead of having a rho_scale parameter
    // We could have a mdot parameter, if we knew the standard accretion rate a normal
    // SANE and MAD initialization would produce. Then we could calculate the rho_scale
    // from the desired mdot.

    Real M_BH_CGS = M_BH * msun_cgs;
    L_unit = (Gnewt_cgs * M_BH_CGS) / (clight_cgs * clight_cgs); // GM/c^2
    T_unit = L_unit / clight_cgs;                                // GM/c^3
    U_unit = rho_scale * clight_cgs * clight_cgs; // rho * c^2 (Energy Density)

    // 4. Store them in the package parameters so other packages (like RadM1) can find
    // them
    params.Add("M_BH", M_BH); // Store in Solar Masses
    params.Add("rho_scale", rho_scale);

    // Store Derived CGS units
    params.Add("length_unit_cgs", L_unit);
    params.Add("time_unit_cgs", T_unit);
    params.Add("energy_unit_cgs", U_unit);

    // Print an output of all the scales
    printf("Units initialized with M_BH = %e Msun, rho_scale = %e g/cm^3\n", M_BH,
        rho_scale);
    printf("Derived units: length_unit = %e cm, time_unit = %e s, energy_unit = %e "
           "erg/cm^3\n",
        L_unit, T_unit, U_unit);

    return pkg;
}
}
