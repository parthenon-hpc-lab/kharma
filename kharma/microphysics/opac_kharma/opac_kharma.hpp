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

// singularity includes
#include <singularity-opac/base/radiation_types.hpp>
#include <singularity-opac/neutrinos/mean_opacity_neutrinos.hpp>
#include <singularity-opac/neutrinos/mean_s_opacity_neutrinos.hpp>
#include <singularity-opac/neutrinos/opac_neutrinos.hpp>
#include <singularity-opac/neutrinos/s_opac_neutrinos.hpp>

namespace Microphysics {

using RadiationType = singularity::RadiationType;

/// One class to contain all opacity objects and wrap the subset of calls we use in
/// phoebus for convenience.
class Opacities {
  using Opacity = singularity::neutrinos::Opacity;
  using MeanOpacity = singularity::neutrinos::MeanOpacity;
  using MeanNonCGSUnits = singularity::neutrinos::MeanNonCGSUnits<MeanOpacity>;
  using SOpacity = singularity::neutrinos::SOpacity;
  using MeanSOpacity = singularity::neutrinos::MeanSOpacity;
  using MeanNonCGSUnitsS = singularity::neutrinos::MeanNonCGSUnitsS<MeanSOpacity>;

 public:
  Opacities() = default;
  KOKKOS_FUNCTION
  Opacities(const Opacity &opac, const MeanOpacity &m_opac, const SOpacity &s_opac,
            const MeanSOpacity &m_s_opac)
      : opac_(opac), m_opac_(m_opac), s_opac_(s_opac), m_s_opac_(m_s_opac) {}

  /// Radiation equation of state calls
  KOKKOS_INLINE_FUNCTION
  Real EnergyDensityFromTemperature(const Real &T, const RadiationType &type) const {
    return opac_.EnergyDensityFromTemperature(T, type);
  }

  KOKKOS_INLINE_FUNCTION
  Real TemperatureFromEnergyDensity(const Real &E, const RadiationType &type) const {
    return opac_.TemperatureFromEnergyDensity(E, type);
  }

  KOKKOS_INLINE_FUNCTION
  Real ThermalDistributionOfTNu(const Real &T, const RadiationType &type,
                                const Real &nu) const {
    return opac_.ThermalDistributionOfTNu(T, type, nu);
  }

  /// Absoprtion/emission quantities
  KOKKOS_INLINE_FUNCTION
  Real Emissivity(const Real &rho, const Real &T, const Real &Ye,
                  const RadiationType &type, Real *lambda = nullptr) const {
    return opac_.Emissivity(rho, T, Ye, type, lambda);
  }

  KOKKOS_INLINE_FUNCTION
  Real NumberEmissivity(const Real &rho, const Real &T, const Real &Ye,
                        const RadiationType &type, Real *lambda = nullptr) const {
    return opac_.NumberEmissivity(rho, T, Ye, type, lambda);
  }

  KOKKOS_INLINE_FUNCTION
  Real EmissivityPerNu(const Real &rho, const Real &T, const Real &Ye,
                       const RadiationType &type, const Real nu,
                       Real *lambda = nullptr) const {
    return opac_.EmissivityPerNu(rho, T, Ye, type, nu, lambda);
  }

  KOKKOS_INLINE_FUNCTION
  Real AbsorptionCoefficient(const Real &rho, const Real &T, const Real &Ye,
                             const RadiationType &type, const Real nu,
                             Real *lambda = nullptr) const {
    return opac_.AbsorptionCoefficient(rho, T, Ye, type, nu, lambda);
  }

  KOKKOS_INLINE_FUNCTION
  Real AngleAveragedAbsorptionCoefficient(const Real &rho, const Real &T, const Real &Ye,
                                          const RadiationType &type, const Real nu,
                                          Real *lambda = nullptr) const {
    return opac_.AngleAveragedAbsorptionCoefficient(rho, T, Ye, type, nu, lambda);
  }

  // Scattering quantities
  KOKKOS_INLINE_FUNCTION
  Real TotalScatteringCoefficient(const Real &rho, const Real &T, const Real &Ye,
                                  const RadiationType &type, const Real nu,
                                  Real *lambda = nullptr) const {
    return s_opac_.TotalScatteringCoefficient(rho, T, Ye, type, nu, lambda);
  }

  /// Mean absorption opacities
  KOKKOS_INLINE_FUNCTION
  Real RosselandMeanAbsorptionCoefficient(const Real &rho, const Real &T, const Real &Ye,
                                          const RadiationType &type) const {
    return m_opac_.RosselandMeanAbsorptionCoefficient(rho, T, Ye, type);
  }

  /// Mean scattering opacities
  KOKKOS_INLINE_FUNCTION
  Real RosselandMeanScatteringCoefficient(const Real &rho, const Real &T, const Real &Ye,
                                          const RadiationType &type) const {
    return m_s_opac_.RosselandMeanTotalScatteringCoefficient(rho, T, Ye, type);
  }

 private:
  Opacity opac_;
  MeanOpacity m_opac_;
  SOpacity s_opac_;
  MeanSOpacity m_s_opac_;
};

namespace Opacity {
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
} // namespace Opacity

} // namespace Microphysics

#endif // MICROPHYSICS_OPAC_OPAC_HPP_
