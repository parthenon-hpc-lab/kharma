// © 2021-2022. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.

#ifndef MICROPHYSICS_OPAC_OPAC_HPP_
#define MICROPHYSICS_OPAC_OPAC_HPP_

#include <memory>
#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

#include "kharma_package.hpp"
// Include neutrinos opacities
#include <singularity-opac/neutrinos/mean_opacity_neutrinos.hpp>
#include <singularity-opac/neutrinos/mean_s_opacity_neutrinos.hpp>
#include <singularity-opac/neutrinos/opac_neutrinos.hpp>
#include <singularity-opac/neutrinos/s_opac_neutrinos.hpp>

#include <singularity-opac/photons/mean_opacity_photons.hpp>
#include <singularity-opac/photons/mean_photon_s_variant.hpp>
#include <singularity-opac/photons/mean_s_opacity_photons.hpp>
#include <singularity-opac/photons/opac_photons.hpp>
#include <singularity-opac/photons/s_opac_photons.hpp>

namespace Microphysics
{

// Density and temperature range (CGS) covered by a tabulated opacity
struct TableBounds
{
    Real rho_min = 0.0;
    Real rho_max = 0.0;
    Real T_min = 0.0;
    Real T_max = 0.0;
};

class Opacities
{
    using Opacity = singularity::photons::Opacity;
    using MeanOpacity = singularity::photons::MeanOpacity;
    using MeanNonCGSUnits = singularity::photons::MeanNonCGSUnits<MeanOpacity>;
    using SOpacity = singularity::photons::SOpacity;
    using MeanSOpacityBase = singularity::photons::MeanSOpacityBase;
    using MeanSOpacity = singularity::photons::impl::MeanSVariant<MeanSOpacityBase,
        singularity::photons::MeanNonCGSUnitsS<MeanSOpacityBase>>;
    using MeanNonCGSUnitsS = singularity::photons::MeanNonCGSUnitsS<MeanSOpacity>;

  public:
    Opacities() = default;
    KOKKOS_FUNCTION
    Opacities(const Opacity& opac, const MeanOpacity& m_opac, const SOpacity& s_opac,
        const MeanSOpacity& m_s_opac, const MeanOpacity& m_opac_tops,
        const bool tops_scattering, const TableBounds& tops_abs_bounds,
        const TableBounds& tops_sca_bounds, const Real X, const Real Z)
        : opac_(opac)
        , m_opac_(m_opac)
        , s_opac_(s_opac)
        , m_s_opac_(m_s_opac)
        , m_opac_tops_(m_opac_tops)
        , tops_scattering_(tops_scattering)
        , tops_abs_bounds_(tops_abs_bounds)
        , tops_sca_bounds_(tops_sca_bounds)
        , X_(X)
        , Z_(Z)
    {}

    /// Radiation equation of state calls
    KOKKOS_INLINE_FUNCTION
    Real EnergyDensityFromTemperature(const Real& T) const
    {
        return opac_.EnergyDensityFromTemperature(T);
    }

    KOKKOS_INLINE_FUNCTION
    Real TemperatureFromEnergyDensity(const Real& E) const
    {
        return opac_.TemperatureFromEnergyDensity(E);
    }

    KOKKOS_INLINE_FUNCTION
    Real ThermalDistributionOfTNu(const Real& T, const Real& nu) const
    {
        return opac_.ThermalDistributionOfTNu(T, nu);
    }

    /// Absorption/emission quantities
    KOKKOS_INLINE_FUNCTION
    Real Emissivity(const Real& rho, const Real& T, Real* lambda = nullptr) const
    {
        return opac_.Emissivity(rho, T, lambda);
    }

    KOKKOS_INLINE_FUNCTION
    Real NumberEmissivity(const Real& rho, const Real& T, Real* lambda = nullptr) const
    {
        return opac_.NumberEmissivity(rho, T, lambda);
    }

    KOKKOS_INLINE_FUNCTION
    Real EmissivityPerNu(
        const Real& rho, const Real& T, const Real nu, Real* lambda = nullptr) const
    {
        return opac_.EmissivityPerNu(rho, T, nu, lambda);
    }

    KOKKOS_INLINE_FUNCTION
    Real AbsorptionCoefficient(
        const Real& rho, const Real& T, const Real nu, Real* lambda = nullptr) const
    {
        return opac_.AbsorptionCoefficient(rho, T, nu, lambda);
    }

    KOKKOS_INLINE_FUNCTION
    Real AngleAveragedAbsorptionCoefficient(
        const Real& rho, const Real& T, const Real nu, Real* lambda = nullptr) const
    {
        return opac_.AngleAveragedAbsorptionCoefficient(rho, T, nu, lambda);
    }

    // Scattering quantities
    KOKKOS_INLINE_FUNCTION
    Real TotalScatteringCoefficient(
        const Real& rho, const Real& T, const Real nu, Real* lambda = nullptr) const
    {
        return s_opac_.TotalScatteringCoefficient(rho, T, nu, lambda);
    }

    KOKKOS_INLINE_FUNCTION
    Real PlanckMeanAbsorptionCoefficient(const Real& rho, const Real& T) const
    {
        return m_opac_.PlanckGroupAbsorptionCoefficient(rho, T, 0);
    }

    KOKKOS_INLINE_FUNCTION
    Real RosselandMeanAbsorptionCoefficient(const Real& rho, const Real& T) const
    {
        return m_opac_.RosselandGroupAbsorptionCoefficient(rho, T, 0);
    }

    KOKKOS_INLINE_FUNCTION
    Real TopsTableAbsorptionCoefficient(const Real& rho, const Real& T) const
    {
        return m_opac_tops_.PlanckGroupAbsorptionCoefficient(rho, T, 0);
    }

    /// Mean scattering opacities
    KOKKOS_INLINE_FUNCTION
    Real RosselandMeanScatteringCoefficient(const Real& rho, const Real& T) const
    {
        return m_s_opac_.RosselandGroupScatteringCoefficient(rho, T, 0);
    }

    KOKKOS_INLINE_FUNCTION bool UseTopsScattering() const { return tops_scattering_; }
    KOKKOS_INLINE_FUNCTION TableBounds TopsAbsorptionBounds() const
    {
        return tops_abs_bounds_;
    }
    KOKKOS_INLINE_FUNCTION TableBounds TopsScatteringBounds() const
    {
        return tops_sca_bounds_;
    }
    KOKKOS_INLINE_FUNCTION Real HydrogenMassFraction() const { return X_; }
    KOKKOS_INLINE_FUNCTION Real MetalMassFraction() const { return Z_; }

  private:
    Opacity opac_;
    MeanOpacity m_opac_;
    SOpacity s_opac_;
    MeanSOpacity m_s_opac_;
    MeanOpacity m_opac_tops_;
    bool tops_scattering_ = false;
    TableBounds tops_abs_bounds_;
    TableBounds tops_sca_bounds_;
    Real X_ = 0.7;
    Real Z_ = 0.02;
};

namespace Opacity
{
std::shared_ptr<KHARMAPackage> Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages);
} // namespace Opacity

} // namespace Microphysics

#endif // MICROPHYSICS_OPAC_OPAC_HPP_
