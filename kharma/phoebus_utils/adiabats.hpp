// © 2021. Triad National Security, LLC. All rights reserved.  This
// program was produced under U.S. Government contract
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

#ifndef PHOEBUS_UTILS_ADIABATS_HPP_
#define PHOEBUS_UTILS_ADIABATS_HPP_

#include <limits>

#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

#include <spiner/databox.hpp>

#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "phoebus_utils/root_find.hpp"

namespace Adiabats {

using Microphysics::EOS::EOS;
using DataBox = Spiner::DataBox<Real>;

template <typename EOS>
inline void GetRhoBounds(const EOS &eos, const Real rho_min, const Real rho_max,
                         const Real T_min, const Real T_max, const Real Ye, const Real S0,
                         Real &lrho_min_new, Real &lrho_max_new) {

  // adjust bounds slightly to avoid edges
  constexpr Real ADJUST = 0.01;
  root_find::RootFind root_find;
  root_find::RootFindStatus status;
  Real lambda[2];
  lambda[0] = Ye;

  const Real epsilon = std::numeric_limits<Real>::epsilon();

  // lower density bound
  Real guess = 1.1 * rho_min;
  Real T = T_min;
  auto target = [&](const Real Rho) {
    return eos.EntropyFromDensityTemperature(Rho, T, lambda) - S0;
  };
  lrho_min_new =
      root_find.regula_falsi(target, rho_min, rho_max, epsilon * guess, guess, &status);
  if (status == root_find::RootFindStatus::failure) {
    lrho_min_new = rho_min;
  };
  lrho_min_new = std::log10(lrho_min_new);
  lrho_min_new += ADJUST * std::abs(lrho_min_new);

  // upper density bound
  guess = 0.1 * rho_max;
  T = T_max; // go down a bit?
  lrho_max_new =
      root_find.regula_falsi(target, rho_min, rho_max, epsilon * guess, guess, &status);
  if (status == root_find::RootFindStatus::failure) {
    lrho_max_new = rho_max;
  };
  lrho_max_new = std::log10(lrho_max_new);

  lrho_max_new -= ADJUST * (lrho_max_new - lrho_min_new);
}

// sample log rho
inline void SampleRho(DataBox lrho, const Real lrho_min, const Real lrho_max,
                      const int n_samps) {
  const Real dlrho = (lrho_max - lrho_min) / n_samps;

  for (int i = 0; i < n_samps; i++) {
    lrho(i) = lrho_min + i * dlrho;
  }
}

/**
 * Given Ye and a target entropy, compute density and temperature of constant entropy
 **/
template <typename EOS>
inline void ComputeAdiabats(DataBox lrho, DataBox temp, const EOS &eos, const Real Ye,
                            const Real S0, const Real T_min, const Real T_max,
                            const int n_samps) {

  const Real guess0 = (T_max - T_min) / 2.0;
  const Real epsilon = std::numeric_limits<Real>::epsilon();

  for (int i = 0; i < n_samps; i++) {
    const Real Rho = std::pow(10.0, lrho(i));
    Real lambda[2];
    lambda[0] = Ye;

    auto target = [&](const Real T) {
      return eos.EntropyFromDensityTemperature(Rho, T, lambda) - S0;
    };

    const Real guess = guess0;
    root_find::RootFind root_find;
    temp(i) = root_find.regula_falsi(target, T_min, T_max, epsilon * guess, guess);
  }
}

/**
 * Find the minimum enthalpy along an adiabat as computed above
 **/
template <typename EOS>
inline Real MinEnthalpy(DataBox lrho, DataBox temp, const Real Ye, const EOS &eos,
                        const int n_samps) {
  Real min_enthalpy = 1e30;
  for (int i = 0; i < n_samps; i++) {
    const Real Rho = std::pow(10.0, lrho(i));
    const Real T = temp.interpToReal(lrho(i));
    Real lambda[2];
    lambda[0] = Ye;

    auto enthalpy_sc = [&]() {
      const Real P = eos.PressureFromDensityTemperature(Rho, T, lambda);
      const Real e = eos.InternalEnergyFromDensityTemperature(Rho, T, lambda);
      return 1.0 + e + P / Rho;
    };

    const Real h = enthalpy_sc();
    min_enthalpy = (h < min_enthalpy) ? h : min_enthalpy;
  }
  return min_enthalpy;
}

} // namespace Adiabats
#endif // PHOEBUS_UTILS_ADIABATS_HPP_
