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
    bool use_bremsstrahlung, use_synchrotron, use_tops_table;
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
                Real kappa = 0.0;
                if (use_tops_table) {
                    kappa += tops_absorption(rho, Tg, Trad);
                } else if (use_bremsstrahlung) {
                    kappa += sing_opac.PlanckMeanAbsorptionCoefficient(rho, Tg);
                }
                if (use_synchrotron) kappa += synchrotron(rho, Tg, Trad, bsq);
                return kappa;
            }
        }
    }

    // Absorption of the radiation at Trad (k_rad) and Kirchhoff emission at Tg (k_gas).
    // Trad enters through synchrotron and, above the TOPS table, the bremsstrahlung fit;
    // within the table the lookup is evaluated once
    KOKKOS_INLINE_FUNCTION
    void kappa_a_rad_gas(
        Real rho, Real Tg, Real bsq, Real Trad, Real& k_rad, Real& k_gas) const
    {
        const bool sync = use_synchrotron &&
                          static_cast<OpacityType>(opacity_type) == OpacityType::Default;
        k_rad = kappa_a(rho, Tg, sync ? 0.0 : bsq, Trad); // synchrotron is 0 at bsq = 0
        k_gas = above_tops_table(Tg) ? kappa_a(rho, Tg, sync ? 0.0 : bsq, Tg) : k_rad;
        if (sync) {
            k_rad += synchrotron(rho, Tg, Trad, bsq);
            k_gas += synchrotron(rho, Tg, Tg, bsq);
        }
    }

    KOKKOS_INLINE_FUNCTION
    Real kappa_sc(Real rho, Real Tg, Real bsq, Real Trad) const
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
                if (sing_opac.UseTopsScattering()) return tops_scattering(rho, Tg, Trad);
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
    Real klein_nishina(const Real T_cgs) const
    {
        return 1.0 / (1.0 + m::pow(T_cgs / 4.5e8, 0.86));
    }

    // Spiner extrapolates log(kappa) linearly from the edge cells of a table, which
    // can give unphysical opacities elsewhere. So the TOPS lookups are clamped to the
    // table, and outside it:
    //  - T > T_max: absorption from the McKinney et al. 2017 free-free fit, at
    //    zeta = Trad/Tg (zeta = 1 for the emission).
    //  - T < T_min: hold the T_min value.
    //  - rho > rho_max: hold kappa = alpha/rho at the edge,
    //  - rho < rho_min: absorption kappa ~ rho (free-free, as gas only gets more
    //    ionized), scattering kappa held at the edge (stays below Thomson).
    // The scattering's Klein-Nishina reduction is taken at Trad: the table holds the LTE
    // value at the gas temperature, a reduction that cold radiation in hot gas does not
    // have.

    // Gas hotter than the TOPS absorption table (bremsstrahlung fit)
    KOKKOS_INLINE_FUNCTION
    bool above_tops_table(const Real Tg) const
    {
        return static_cast<OpacityType>(opacity_type) == OpacityType::Default &&
               use_tops_table &&
               m::abs(Tg) > sing_opac.TopsAbsorptionBounds().T_max /
                                units_cgs.GetTemperatureCodeToCGS();
    }

    // TOPS Planck-mean absorption coefficient (code units) of radiation at Trad
    KOKKOS_INLINE_FUNCTION
    Real tops_absorption(const Real rho, const Real Tg, const Real Trad) const
    {
        const Microphysics::TableBounds bounds = sing_opac.TopsAbsorptionBounds();
        const Real rho_unit = units_cgs.GetMassDensityCodeToCGS();
        const Real T_unit = units_cgs.GetTemperatureCodeToCGS();
        const Real T_max = bounds.T_max / T_unit;
        const Real T = m::abs(Tg);
        if (T > T_max) return bremsstrahlung(rho, T, isfinite(Trad) ? Trad : T);

        const Real rho_min = bounds.rho_min / rho_unit;
        const Real rho_q = m::min(m::max(rho, rho_min), bounds.rho_max / rho_unit);
        const Real T_q = m::max(T, bounds.T_min / T_unit);

        Real fac = rho / rho_q;
        if (rho < rho_min) fac *= rho / rho_min;
        return fac * sing_opac.TopsTableAbsorptionCoefficient(rho_q, T_q);
    }

    // TOPS Rosseland-mean scattering coefficient (code units) of radiation at Trad
    KOKKOS_INLINE_FUNCTION
    Real tops_scattering(const Real rho, const Real Tg, const Real Trad) const
    {
        const Microphysics::TableBounds bounds = sing_opac.TopsScatteringBounds();
        const Real rho_unit = units_cgs.GetMassDensityCodeToCGS();
        const Real T_unit = units_cgs.GetTemperatureCodeToCGS();
        const Real T_max = bounds.T_max / T_unit;
        const Real T = m::abs(Tg);
        const Real rho_q =
            m::min(m::max(rho, bounds.rho_min / rho_unit), bounds.rho_max / rho_unit);
        const Real T_q = m::min(m::max(T, bounds.T_min / T_unit), T_max);

        // Klein-Nishina at the photon temperature in place of the table's at T_q
        const Real T_ph = isfinite(Trad) ? m::abs(Trad) : T;
        const Real fac = rho / rho_q * klein_nishina(T_ph * T_unit) /
                         klein_nishina(T_q * T_unit);
        return fac * sing_opac.RosselandMeanScatteringCoefficient(rho_q, T_q);
    }

    KOKKOS_INLINE_FUNCTION
    Real bremsstrahlung(const Real rho, const Real Tg, const Real Trad) const
    {
        const Real rho_cgs = rho * units_cgs.GetMassDensityCodeToCGS();
        const Real Tg_cgs = m::abs(Tg) * units_cgs.GetTemperatureCodeToCGS();
        const Real Trad_cgs = m::abs(Trad) * units_cgs.GetTemperatureCodeToCGS();
        if (Tg_cgs <= 0.0) return 0.0;
        const Real X = sing_opac.HydrogenMassFraction();
        const Real Z = sing_opac.MetalMassFraction();
        const Real Theta_e = pc::kb * Tg_cgs / (pc::me * pc::c * pc::c);

        Real R_ei, R_ee;
        if (Theta_e <= 1.0) {
            R_ei = 1.0 + 1.76 * m::pow(Theta_e, 1.34);
            R_ee =
                1.7 * Theta_e *
                (1.0 + 1.1 * Theta_e + Theta_e * Theta_e - 1.06 * m::pow(Theta_e, 2.5));
        } else {
            R_ei = 1.4 * m::sqrt(Theta_e) * (m::log(1.12 * Theta_e + 0.48) + 1.5);
            R_ee = 1.7 * m::sqrt(Theta_e) * 1.46 * (1.28 + m::log(1.12 * Theta_e));
        }

        const Real zeta = m::min(m::max(Trad_cgs / Tg_cgs, 1.0e-5), 1.0e5);
        const Real a = 0.532, b = 3.14, c = 4.52;
        const Real kappa_ei_cgs = 1.2e24 * m::pow(Tg_cgs, -3.5) * rho_cgs * rho_cgs *
                                  (1.0 + X) * (1.0 - Z) * R_ei * a * m::pow(zeta, -b) *
                                  m::log(1.0 + c * zeta);
        const Real kappa_ee_cgs =
            kappa_ei_cgs * (1.0 + X) / (2.0 * (1.0 - Z)) * R_ee / R_ei;

        Real kappa = (kappa_ei_cgs + kappa_ee_cgs) * units_cgs.GetLengthCodeToCGS();
        if (!isfinite(kappa)) kappa = 0.0;
        return kappa;
    }

    KOKKOS_INLINE_FUNCTION
    Real synchrotron(const Real rho, const Real Tg, const Real Trad, const Real bsq) const
    {
        if (bsq <= 0.0) return 0.0;
        const Real rho_cgs = rho * units_cgs.GetMassDensityCodeToCGS();
        const Real Tg_cgs = m::abs(Tg) * units_cgs.GetTemperatureCodeToCGS();
        const Real Trad_cgs = m::abs(Trad) * units_cgs.GetTemperatureCodeToCGS();
        const Real L = units_cgs.GetLengthCodeToCGS();
        const Real energy_density_scale = units_cgs.GetEnergyCodeToCGS() / (L * L * L);
        const Real B_cgs = m::sqrt(4.0 * M_PI * bsq * energy_density_scale); // Gauss

        const Real n_e =
            rho_cgs * (1.0 + sing_opac.HydrogenMassFraction()) / (2.0 * pc::mp);
        const Real Theta_e = pc::kb * Tg_cgs / (pc::me * pc::c * pc::c);
        const Real nu_B =
            pc::qe * B_cgs / (2.0 * M_PI * pc::me * pc::c); // cyclotron frequency
        const Real nu_mu = 1.5 * nu_B * Theta_e * Theta_e;  // thermal synchrotron peak
        const Real phi = m::max(pc::kb * Trad_cgs / (pc::h * nu_mu), 1.0e-5);

        // McKinney et al. 2017 eq. D8 (AGN) / D10 (XRB) at e^{-xi} = 1: no photon number
        // is evolved, so the radiation is Planck
        const bool agn = (regime == FitType::AGN);
        const Real a = agn ? 0.2065 : 1.2699999769, b = agn ? 2.58977 : 1.0339;
        const Real c = agn ? 3.44 : 0.000763, d = agn ? 9.3 : 0.616;
        const Real e = agn ? 3.095 : 2.917;

        Real kappa_cgs =
            5.85374e-14 * n_e * phi / (Theta_e * Theta_e * Theta_e * Trad_cgs) / rho_cgs;
        kappa_cgs *= 1.0 / (1.0 / (a * m::pow(phi, -b) * m::log(1.0 + c * phi)) +
                               1.0 / (d * m::pow(phi, -e)));

        Real kappa = kappa_cgs * rho_cgs * units_cgs.GetLengthCodeToCGS();
        if (!isfinite(kappa)) kappa = 0.0;
        return kappa;
    }
};
