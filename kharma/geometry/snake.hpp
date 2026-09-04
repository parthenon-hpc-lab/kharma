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

#ifndef GEOMETRY_SNAKE_HPP_
#define GEOMETRY_SNAKE_HPP_

#include <array>
#include <cmath>

// Parthenon includes
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

// phoebus includes
#include "geometry/analytic_system.hpp"
#include "geometry/cached_system.hpp"
#include "geometry/geometry_defaults.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/linear_algebra.hpp"

namespace Geometry {

class Snake {
 public:
  Snake() = default;
  KOKKOS_INLINE_FUNCTION
  Snake(const Real a, const Real k) : a_(a), k_(k), alpha_(1), vy_(0) {}
  KOKKOS_INLINE_FUNCTION
  Snake(const Real a, const Real k, const Real alpha, const Real vy)
      : a_(a), k_(k), alpha_(alpha), vy_(vy) {
    PARTHENON_REQUIRE(0 < alpha_, "Lapse must positive");
  }

  KOKKOS_INLINE_FUNCTION
  Real Lapse(Real X0, Real X1, Real X2, Real X3) const { return alpha_; }
  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(Real X0, Real X1, Real X2, Real X3, Real beta[NDSPACE]) const {
    beta[0] = 0;
    beta[1] = vy_;
    beta[2] = 0;
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(Real X0, Real X1, Real X2, Real X3, Real g[NDFULL][NDFULL]) const {
    const Real d = GetDelta(X1);
    g[0][0] = (vy_ - alpha_) * (vy_ + alpha_);
    g[0][1] = g[1][0] = -d * vy_;
    g[0][2] = g[2][0] = vy_;
    g[0][3] = g[3][0] = 0;
    g[1][1] = 1 + d * d;
    g[1][2] = g[2][1] = -d;
    g[1][3] = g[3][1] = 0;
    g[2][2] = 1;
    g[2][3] = g[3][2] = 0;
    g[3][3] = 1;
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(Real X0, Real X1, Real X2, Real X3,
                              Real g[NDFULL][NDFULL]) const {
    const Real d = GetDelta(X1);
    Real alpha2 = (alpha_ * alpha_);
    Real ialpha2 = 1. / alpha2;
    g[0][0] = -ialpha2;
    g[0][1] = g[1][0] = 0;
    g[0][2] = g[2][0] = vy_ * ialpha2;
    g[0][3] = g[3][0] = 0;
    g[1][1] = 1;
    g[1][2] = g[2][1] = d;
    g[1][3] = g[3][1] = 0;
    g[2][2] = 1 - vy_ * vy_ * ialpha2 + d * d;
    g[2][3] = g[3][2] = 0;
    g[3][3] = 1;
  }
  KOKKOS_INLINE_FUNCTION
  void Metric(Real X0, Real X1, Real X2, Real X3, Real gamma[NDSPACE][NDSPACE]) const {
    LinearAlgebra::SetZero(gamma, NDSPACE, NDSPACE);
    const Real d = GetDelta(X1);
    gamma[0][0] = d * d + 1.;
    gamma[1][0] = gamma[0][1] = -d;
    gamma[1][1] = gamma[2][2] = 1;
  }
  KOKKOS_INLINE_FUNCTION
  void MetricInverse(Real X0, Real X1, Real X2, Real X3,
                     Real gamma[NDSPACE][NDSPACE]) const {
    const Real d = GetDelta(X1);
    gamma[0][0] = 1;
    gamma[1][0] = gamma[0][1] = d;
    gamma[2][0] = gamma[0][2] = 0;
    gamma[1][1] = d * d + 1;
    gamma[1][2] = gamma[2][1] = 0;
    gamma[2][2] = 1;
  }
  KOKKOS_INLINE_FUNCTION
  Real DetGamma(Real X0, Real X1, Real X2, Real X3) const { return 1.; }
  KOKKOS_INLINE_FUNCTION
  Real DetG(Real X0, Real X1, Real X2, Real X3) const { return alpha_; }

  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(Real X0, Real X1, Real X2, Real X3,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    const Real a2 = a_ * a_;
    const Real k2 = k_ * k_;
    const Real k3 = k_ * k_ * k_;
    LinearAlgebra::SetZero(Gamma, NDFULL, NDFULL, NDFULL);
    Gamma[0][1][1] = a_ * k2 * vy_ * sin(k_ * X1);
    Gamma[1][1][1] = -a2 * k3 * sin(2. * k_ * X1) / 2.;
    Gamma[2][1][1] = a_ * k2 * sin(k_ * X1);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(Real X0, Real X1, Real X2, Real X3,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    const Real a2 = a_ * a_;
    const Real k2 = k_ * k_;
    const Real k3 = k_ * k_ * k_;
    LinearAlgebra::SetZero(dg, NDFULL, NDFULL, NDFULL);
    dg[0][1][1] = dg[1][0][1] = a_ * k2 * vy_ * sin(k_ * X1);
    dg[1][1][1] = -a2 * k3 * sin(2. * k_ * X1);
    dg[2][1][1] = dg[1][2][1] = a_ * k2 * sin(k_ * X1);
  }
  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const {
    for (int mu = 0; mu < NDFULL; ++mu) {
      da[mu] = 0;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void Coords(Real X0, Real X1, Real X2, Real X3, Real C[NDFULL]) const {
    C[0] = X0;
    C[1] = X1;
    C[2] = X2;
    C[3] = X3;
  }

 private:
  Real a_ = 0;
  Real k_ = 0;
  Real alpha_ = 1;
  Real vy_ = 0;

  KOKKOS_INLINE_FUNCTION
  Real GetDelta(Real X1) const { return a_ * k_ * cos(k_ * X1); }
};

using SnakeMeshBlock = Analytic<Snake, IndexerMeshBlock>;
using SnakeMesh = Analytic<Snake, IndexerMesh>;

using CSnakeMeshBlock = CachedOverMeshBlock<SnakeMeshBlock>;
using CSnakeMesh = CachedOverMesh<SnakeMesh>;

template <>
void Initialize<SnakeMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);
template <>
void Initialize<CSnakeMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);

} // namespace Geometry

#endif // GEOMETRY_SNAKE_HPP_
