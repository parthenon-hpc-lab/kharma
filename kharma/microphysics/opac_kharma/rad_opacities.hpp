/*
 *  File: rad_opacities.hpp
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

class RadOpac
{
  public:
    int opacity_type;
    Real const_sigma;
    Real const_kappa_a;
    Real const_kappa_sc;
    Units::UnitConversions units_cgs;
    Microphysics::Opacities sing_opac;

    KOKKOS_INLINE_FUNCTION
    Real kappa_a(Real rho, Real Tg) const
    {
        switch (static_cast<OpacityType>(opacity_type)) {
            case OpacityType::Constant:
                return const_kappa_a;
            case OpacityType::ShocktubeConstant:
                return rho * const_kappa_a;
            case OpacityType::Bondi: {
                const Real T_cgs = m::abs(Tg) * units_cgs.GetTemperatureCodeToCGS();
                const Real rho_cgs = rho * units_cgs.GetMassDensityCodeToCGS();
                // 1.0e23 to match harmrad
                const Real kappa_a_cgs = 1.0e23 * m::pow(T_cgs, -3.5) * rho_cgs * rho_cgs;
                // make it scale free
                return kappa_a_cgs * units_cgs.GetLengthCodeToCGS();
            }
            case OpacityType::Transparent:
                return 0.0;
            case OpacityType::ThermalEquilibrium: {
                const Real rho_cgs = rho * units_cgs.GetMassDensityCodeToCGS();
                return 0.4 * rho_cgs * units_cgs.GetLengthCodeToCGS();
            }
            default: {
                return sing_opac.PlanckMeanAbsorptionCoefficient(rho, Tg);
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    Real kappa_sc(Real rho, Real Tg) const
    {
        switch (static_cast<OpacityType>(opacity_type)) {
            case OpacityType::Constant:
                return const_kappa_sc;
            case OpacityType::ShocktubeConstant:
                return const_kappa_sc;
            case OpacityType::Transparent:
                return 0.0;
            case OpacityType::Bondi: {
                const Real rho_cgs = rho * units_cgs.GetMassDensityCodeToCGS();
                const Real kappa_sc_cgs = 0.4 * rho_cgs;
                return kappa_sc_cgs * units_cgs.GetLengthCodeToCGS();
            }
            case OpacityType::ThermalEquilibrium:
                return 0.0;
            default: {
                return sing_opac.RosselandMeanScatteringCoefficient(rho, Tg);
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    Real JBB(Real Tg) const
    {
        switch (static_cast<OpacityType>(opacity_type)) {
            case OpacityType::Constant:
                return 4.0 * const_sigma * (Tg * Tg * Tg * Tg);
            case OpacityType::ShocktubeConstant:
                return 4.0 * const_sigma * (Tg * Tg * Tg * Tg);
            case OpacityType::Bondi: {
                const Real energy_density_cgs =
                    units_cgs.GetEnergyCodeToCGS() /
                    m::pow(units_cgs.GetLengthCodeToCGS(), 3.0);
                const Real sigma_rad =
                    5.670374419e-5 / pc::c /
                    (energy_density_cgs /
                        m::pow(units_cgs.GetTemperatureCodeToCGS(), 4.0));
                return 4.0 * sigma_rad * (Tg * Tg * Tg * Tg);
            }
            case OpacityType::ThermalEquilibrium:
                return 4.0 * const_sigma * (Tg * Tg * Tg * Tg);
            case OpacityType::Transparent:
                return 0.0;
            default: {
                return sing_opac.EnergyDensityFromTemperature(Tg);
            }
        }
    }
};
