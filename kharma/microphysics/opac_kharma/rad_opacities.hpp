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
    int fit_type;
    Real const_sigma;
    Real const_kappa_a;
    Real const_kappa_sc;
    Real mean_molecular_weight;
    Units::UnitConversions units_cgs;
    Microphysics::Opacities sing_opac;
    enum class FitType { AGN, XRB };

    FitType regime;

    KOKKOS_INLINE_FUNCTION
    Real kappa_a(Real rho, Real Tg, Real bsq, Real Trad) const
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
                return sing_opac.PlanckMeanAbsorptionCoefficient(rho, Tg) +
                       synchrotron(rho, Tg, Trad, bsq);
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    Real kappa_sc(Real rho, Real Tg, Real bsq) const
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
    Real JBB(Real Tg, Real rho, Real bsq) const
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

    KOKKOS_INLINE_FUNCTION
    Real Trad(Real E_hat, Real rho, Real bsq) const
    {
        switch (static_cast<OpacityType>(opacity_type)) {
            case OpacityType::Constant:
            case OpacityType::ShocktubeConstant:
            case OpacityType::ThermalEquilibrium:
                return m::pow(m::max(E_hat, 0.0) / (4.0 * const_sigma), 0.25);
            case OpacityType::Bondi: {
                const Real energy_density_cgs =
                    units_cgs.GetEnergyCodeToCGS() /
                    m::pow(units_cgs.GetLengthCodeToCGS(), 3.0);
                const Real sigma_rad =
                    5.670374419e-5 / pc::c /
                    (energy_density_cgs /
                        m::pow(units_cgs.GetTemperatureCodeToCGS(), 4.0));
                return m::pow(m::max(E_hat, 0.0) / (4.0 * sigma_rad), 0.25);
            }
            case OpacityType::Transparent:
                return 0.0;
            default:
                return sing_opac.TemperatureFromEnergyDensity(E_hat);
        }
    }

  private:
    KOKKOS_INLINE_FUNCTION
    Real synchrotron(const Real rho, const Real Tg, const Real Trad, const Real bsq) const
    {
        if (bsq <= 0.0) return 0.0;
        const Real rho_cgs = rho * units_cgs.GetMassDensityCodeToCGS();
        const Real Tg_cgs = m::abs(Tg) * units_cgs.GetTemperatureCodeToCGS();
        const Real Trad_cgs = m::abs(Trad) * units_cgs.GetTemperatureCodeToCGS();
        const Real energy_density_scale =
            units_cgs.GetEnergyCodeToCGS() / m::pow(units_cgs.GetLengthCodeToCGS(), 3.0);
        const Real B_cgs = m::sqrt(4.0 * M_PI * bsq * energy_density_scale); // Gauss

        const Real n_e = rho_cgs / (mean_molecular_weight *
                                       pc::mp); // mu: mean molecular weight per electron
        const Real Theta_e = pc::kb * Tg_cgs / (pc::me * pc::c * pc::c);
        const Real nu_B =
            pc::qe * B_cgs / (2.0 * M_PI * pc::me * pc::c); // cyclotron frequency
        const Real nu_mu = 1.5 * nu_B * Theta_e * Theta_e;  // thermal synchrotron peak

        // Eventually, we will want to evolve photon number, therefore chi will be used,
        // for now, chi = 0 returns the fit for no photon number. const Real chi =
        // mu/(pc::kb * T_cgs);
        const Real chi = 0;

        const Real phi = m::max(pc::kb * Trad_cgs / (pc::h * nu_mu), 1.0e-5);
        const Real exp_chi = m::exp(-chi);

        // AGN coefficients
        Real a, b, c, d, e;
        // (the 0.99 cap is the one McKinney et al. 2017, Sec. 3.3, use for synchrotron)
        const Real xc = m::min(m::max(exp_chi, 0.), 0.99);
        const Real omx = 1. - xc;
        if (regime == FitType::AGN) {
            // McKinney et al. 2017, eq. D8
            a = -0.0295 * m::pow(xc, 2.29) - 0.143 * m::pow(omx, 0.251) + 0.236;
            b = 0.00977 * m::pow(xc, 730.) + 0.0291 * m::pow(omx, 0.48) + 2.58;
            c = 1.29 * m::pow(xc, 1.59) + 3.46 * m::pow(omx, 0.234) + 2.15;
            d = -78.1 * m::pow(xc, 66.) - 40.3 * m::pow(omx, 0.899) + 87.4;
            e = 0.415 * m::pow(xc, 0.399) + 1.04 * m::pow(omx, 0.252) + 2.68;
        } else if (regime == FitType::XRB) {
            // McKinney et al. 2017, eq. D10
            a = -2.31e-8 * m::pow(xc, 34.) - 8.24e-9 * m::pow(omx, 2.42) + 1.27;
            b = -0.0261 * m::pow(xc, 738.) - 0.00475 * m::pow(omx, 1.55) + 1.06;
            c = 0.000179 * m::pow(xc, 432.) + 0.0000411 * m::pow(omx, 0.372) + 0.000584;
            d = -17.7 * m::pow(xc, 49.4) - 3.33 * m::pow(omx, 2.76) + 18.3;
            e = 0.427 * m::pow(xc, 0.654) + 1.23 * m::pow(omx, 0.214) + 2.49;
        }

        Real kappa_cgs =
            5.85374e-14 * n_e * phi / (Theta_e * Theta_e * Theta_e * Trad_cgs) / rho_cgs;
        kappa_cgs *= 1.0 / (1.0 / (a * m::pow(phi, -b) * m::log(1.0 + c * phi)) +
                               1.0 / (d * m::pow(phi, -e)));

        Real kappa = kappa_cgs * rho_cgs * units_cgs.GetLengthCodeToCGS();
        if (!isfinite(kappa)) kappa = 0.0;
        return kappa;
    }
};
