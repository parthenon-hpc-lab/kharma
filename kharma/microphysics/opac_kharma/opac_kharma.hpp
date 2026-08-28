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

#include <singularity-opac/photons/mean_opacity_photons.hpp>
#include <singularity-opac/photons/mean_s_opacity_photons.hpp>
#include <singularity-opac/photons/opac_photons.hpp>
#include <singularity-opac/photons/s_opac_photons.hpp>

namespace Microphysics {


class Opacities {
  using Opacity = singularity::photons::Opacity;
  using MeanOpacity = singularity::photons::MeanOpacity;
  using MeanNonCGSUnits = singularity::photons::MeanNonCGSUnits<MeanOpacity>;
  using SOpacity = singularity::photons::SOpacity;
  using MeanSOpacity = singularity::photons::MeanSOpacity;
  using MeanNonCGSUnitsS = singularity::photons::MeanNonCGSUnitsS<MeanSOpacity>;

 public:
  Opacities() = default;
  KOKKOS_FUNCTION
  Opacities(const Opacity &opac, const MeanOpacity &m_opac, const SOpacity &s_opac,
            const MeanSOpacity &m_s_opac)
      : opac_(opac), m_opac_(m_opac), s_opac_(s_opac), m_s_opac_(m_s_opac) {}

  /// Radiation equation of state calls
  KOKKOS_INLINE_FUNCTION
  Real EnergyDensityFromTemperature(const Real &T) const {
    return opac_.EnergyDensityFromTemperature(T);
  }

  KOKKOS_INLINE_FUNCTION
  Real TemperatureFromEnergyDensity(const Real &E) const {
    return opac_.TemperatureFromEnergyDensity(E);
  }

  KOKKOS_INLINE_FUNCTION
  Real ThermalDistributionOfTNu(const Real &T, const Real &nu) const {
    return opac_.ThermalDistributionOfTNu(T, nu);
  }

  /// Absorption/emission quantities
  KOKKOS_INLINE_FUNCTION
  Real Emissivity(const Real &rho, const Real &T, Real *lambda = nullptr) const {
    return opac_.Emissivity(rho, T, lambda);
  }

  KOKKOS_INLINE_FUNCTION
  Real NumberEmissivity(const Real &rho, const Real &T, Real *lambda = nullptr) const {
    return opac_.NumberEmissivity(rho, T, lambda);
  }

  KOKKOS_INLINE_FUNCTION
  Real EmissivityPerNu(const Real &rho, const Real &T, const Real nu,
                       Real *lambda = nullptr) const {
    return opac_.EmissivityPerNu(rho, T, nu, lambda);
  }

  KOKKOS_INLINE_FUNCTION
  Real AbsorptionCoefficient(const Real &rho, const Real &T, const Real nu,
                             Real *lambda = nullptr) const {
    return opac_.AbsorptionCoefficient(rho, T, nu, lambda);
  }

  KOKKOS_INLINE_FUNCTION
  Real AngleAveragedAbsorptionCoefficient(const Real &rho, const Real &T, const Real nu,
                                          Real *lambda = nullptr) const {
    return opac_.AngleAveragedAbsorptionCoefficient(rho, T, nu, lambda);
  }

  // Scattering quantities
  KOKKOS_INLINE_FUNCTION
  Real TotalScatteringCoefficient(const Real &rho, const Real &T, const Real nu,
                                  Real *lambda = nullptr) const {
    return s_opac_.TotalScatteringCoefficient(rho, T, nu, lambda);
  }

  
  KOKKOS_INLINE_FUNCTION
  Real PlanckMeanAbsorptionCoefficient(const Real &rho, const Real &T) const {
    return m_opac_.PlanckMeanAbsorptionCoefficient(rho, T);
  }

  KOKKOS_INLINE_FUNCTION
  Real RosselandMeanAbsorptionCoefficient(const Real &rho, const Real &T) const {
    return m_opac_.RosselandMeanAbsorptionCoefficient(rho, T);
  }

  /// Mean scattering opacities
  KOKKOS_INLINE_FUNCTION
  Real RosselandMeanScatteringCoefficient(const Real &rho, const Real &T) const {
    return m_s_opac_.RosselandMeanTotalScatteringCoefficient(rho, T);
  }

 private:
  Opacity opac_;
  MeanOpacity m_opac_;
  SOpacity s_opac_;
  MeanSOpacity m_s_opac_;
};

namespace Opacity {
std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);
} // namespace Opacity

} // namespace Microphysics

#endif // MICROPHYSICS_OPAC_OPAC_HPP_
