// © 2022. Triad National Security, LLC. All rights reserved.  This
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

#ifndef GEOMETRY_FLRW_HPP_
#define GEOMETRY_FLRW_HPP_

// Parthenon includes
#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

// Phoebus includes
#include "geometry/analytic_system.hpp"
#include "geometry/cached_system.hpp"
#include "geometry/geometry_defaults.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/linear_algebra.hpp"
#include "phoebus_utils/robust.hpp"

namespace Geometry {

// A simplified toy cosmology for testing time-dependent metrics
// Assumes ds^2 = -dt^2 + a^2 delta_{ij} dx^i dx^j for delta = Chronicker Delta
// We also assume da/dt = const = C, i.e., d^2a/dt^2 = 0
// and hubble param H = (1/a)(da/dt) = C/a
// This also implies vanishing spatial curvature... i.e., flat spacetime.
// This implies d rho/dt = -3 (rho + P) H
// where here rho is rho_0 + u, i.e., the total energy density.
// Beware this spacetime has cosmological horizons and a singularity.
// For example, t = -a0/dadt produces a = 0, i.e., the Big Bang.
// Treatment taken from Chapter 8 of Carroll
class FLRW {
 public:
  FLRW() : a0_(1.), dadt_(1.) {}
  FLRW(const Real &a0, const Real &dadt) : a0_(a0), dadt_(dadt) {}
  KOKKOS_INLINE_FUNCTION
  Real Lapse(Real X0, Real X1, Real X2, Real X3) const { return 1.; }
  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(Real X0, Real X1, Real X2, Real X3, Real beta[NDSPACE]) const {
    beta[0] = beta[1] = beta[2] = 0;
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(Real X0, Real X1, Real X2, Real X3, Real g[NDFULL][NDFULL]) const {
    LinearAlgebra::SetZero(g, NDFULL, NDFULL);
    const Real a = a_(X0);
    g[0][0] = -1;
    g[1][1] = g[2][2] = g[3][3] = a * a;
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(Real X0, Real X1, Real X2, Real X3,
                              Real g[NDFULL][NDFULL]) const {
    LinearAlgebra::SetZero(g, NDFULL, NDFULL);
    const Real a = a_(X0);
    g[0][0] = -1;
    g[1][1] = g[2][2] = g[3][3] = robust::ratio(1, a * a);
  }
  KOKKOS_INLINE_FUNCTION
  void Metric(Real X0, Real X1, Real X2, Real X3, Real gamma[NDSPACE][NDSPACE]) const {
    LinearAlgebra::SetZero(gamma, NDSPACE, NDSPACE);
    const Real a = a_(X0);
    gamma[0][0] = gamma[1][1] = gamma[2][2] = a * a;
  }
  KOKKOS_INLINE_FUNCTION
  void MetricInverse(Real X0, Real X1, Real X2, Real X3,
                     Real gamma[NDSPACE][NDSPACE]) const {
    LinearAlgebra::SetZero(gamma, NDSPACE, NDSPACE);
    const Real a = a_(X0);
    gamma[0][0] = gamma[1][1] = gamma[2][2] = robust::ratio(1, a * a);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetGamma(Real X0, Real X1, Real X2, Real X3) const {
    // ((a^2)^3)^(1/2) = a^3
    const Real a = a_(X0);
    return a * a * a;
  }
  KOKKOS_INLINE_FUNCTION
  Real DetG(Real X0, Real X1, Real X2, Real X3) const {
    // ((a^2)^3)^(1/2) = a^3
    const Real a = a_(X0);
    return a * a * a;
  }
  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(Real X0, Real X1, Real X2, Real X3,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    const Real a = a_(X0);
    const Real aadot = a * dadt_;
    LinearAlgebra::SetZero(Gamma, NDFULL, NDFULL, NDFULL);
    Gamma[0][1][1] = Gamma[0][2][2] = Gamma[0][3][3] = -aadot;
    Gamma[1][0][1] = Gamma[1][1][0] = aadot;
    Gamma[2][0][2] = Gamma[2][2][0] = aadot;
    Gamma[3][0][3] = Gamma[3][3][0] = aadot;
  }
  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(Real X0, Real X1, Real X2, Real X3,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    const Real a = a_(X0);
    LinearAlgebra::SetZero(dg, NDFULL, NDFULL, NDFULL);
    dg[1][1][0] = dg[2][2][0] = dg[3][3][0] = 2 * a * dadt_;
  }
  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const {
    LinearAlgebra::SetZero(da, NDFULL);
  }
  KOKKOS_INLINE_FUNCTION
  void Coords(Real X0, Real X1, Real X2, Real X3, Real C[NDFULL]) const {
    C[0] = X0;
    C[1] = X1;
    C[2] = X2;
    C[3] = X3;
  }

 private:
  KOKKOS_INLINE_FUNCTION
  Real a_(const Real t) const { return a0_ + dadt_ * t; }
  Real a0_; // These cannot be const, or the compiler deletes the copy assignment operator
  Real dadt_;
};

using FLRWMeshBlock = Analytic<FLRW, IndexerMeshBlock>;
using FLRWMesh = Analytic<FLRW, IndexerMesh>;
using CFLRWMeshBlock = CachedOverMeshBlock<FLRWMeshBlock>;
using CFLRWMesh = CachedOverMesh<FLRWMesh>;

template <>
void Initialize<FLRWMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);
template <>
void Initialize<CFLRWMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);

} // namespace Geometry

#endif // GEOMETRY_FLRW_HPP_
