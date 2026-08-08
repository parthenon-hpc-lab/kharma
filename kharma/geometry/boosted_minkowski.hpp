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

#ifndef GEOMETRY_BOOSTED_MINKOWSKI_HPP_
#define GEOMETRY_BOOSTED_MINKOWSKI_HPP_

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

/*
 * Minkowski coordinates under the transformation
 * x -> x - vx t
 * y -> y - vy t
 * z -> z - vz t
 * where -1 < vx, vy, vz < 1
 */
class BoostedMinkowski {
 public:
  BoostedMinkowski() : vx_(0), vy_(0), vz_(0) {}
  BoostedMinkowski(const Real &vx, const Real &vy, const Real &vz)
      : vx_(vx), vy_(vy), vz_(vz) {}
  KOKKOS_INLINE_FUNCTION
  Real Lapse(Real X0, Real X1, Real X2, Real X3) const { return 1.; }
  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(Real X0, Real X1, Real X2, Real X3, Real beta[NDSPACE]) const {
    beta[0] = vx_;
    beta[1] = vy_;
    beta[2] = vz_;
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(Real X0, Real X1, Real X2, Real X3, Real g[NDFULL][NDFULL]) const {
    LinearAlgebra::SetZero(g, NDFULL, NDFULL);
    g[0][0] = -1 + vx_ * vx_ + vy_ * vy_ + vz_ * vz_;
    g[1][1] = g[2][2] = g[3][3] = 1;
    g[0][1] = g[1][0] = vx_;
    g[0][2] = g[2][0] = vy_;
    g[0][3] = g[3][0] = vz_;
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(Real X0, Real X1, Real X2, Real X3,
                              Real g[NDFULL][NDFULL]) const {
    g[0][0] = -1;
    g[0][1] = g[1][0] = vx_;
    g[0][2] = g[2][0] = vy_;
    g[0][3] = g[3][0] = vz_;
    g[1][1] = 1 - vx_ * vx_;
    g[2][2] = 1 - vy_ * vy_;
    g[3][3] = 1 - vz_ * vz_;
    g[1][2] = g[2][1] = -vx_ * vy_;
    g[1][3] = g[3][1] = -vx_ * vz_;
    g[2][3] = g[3][2] = -vy_ * vz_;
  }
  KOKKOS_INLINE_FUNCTION
  void Metric(Real X0, Real X1, Real X2, Real X3, Real gamma[NDSPACE][NDSPACE]) const {
    LinearAlgebra::SetZero(gamma, NDSPACE, NDSPACE);
    gamma[0][0] = gamma[1][1] = gamma[2][2] = 1;
  }
  KOKKOS_INLINE_FUNCTION
  void MetricInverse(Real X0, Real X1, Real X2, Real X3,
                     Real gamma[NDSPACE][NDSPACE]) const {
    LinearAlgebra::SetZero(gamma, NDSPACE, NDSPACE);
    gamma[0][0] = gamma[1][1] = gamma[2][2] = 1;
  }
  KOKKOS_INLINE_FUNCTION
  Real DetGamma(Real X0, Real X1, Real X2, Real X3) const { return 1.; }
  KOKKOS_INLINE_FUNCTION
  Real DetG(Real X0, Real X1, Real X2, Real X3) const { return 1.; }

  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(Real X0, Real X1, Real X2, Real X3,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    LinearAlgebra::SetZero(Gamma, NDFULL, NDFULL, NDFULL);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(Real X0, Real X1, Real X2, Real X3,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    LinearAlgebra::SetZero(dg, NDFULL, NDFULL, NDFULL);
  }
  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const {
    LinearAlgebra::SetZero(da, NDFULL);
  }
  KOKKOS_INLINE_FUNCTION
  void Coords(Real X0, Real X1, Real X2, Real X3, Real C[NDFULL]) const {
    C[0] = X0;
    C[1] = X1 + vx_ * X0;
    C[2] = X2 + vy_ * X0;
    C[3] = X3 + vz_ * X0;
  }

 private:
  Real vx_, vy_, vz_;
};

using BoostedMinkowskiMeshBlock = Analytic<BoostedMinkowski, IndexerMeshBlock>;
using BoostedMinkowskiMesh = Analytic<BoostedMinkowski, IndexerMesh>;
using CBoostedMinkowskiMeshBlock = CachedOverMeshBlock<BoostedMinkowskiMeshBlock>;
using CBoostedMinkowskiMesh = CachedOverMesh<BoostedMinkowskiMesh>;

template <>
void Initialize<BoostedMinkowskiMeshBlock>(ParameterInput *pin,
                                           StateDescriptor *geometry);
template <>
void Initialize<CBoostedMinkowskiMeshBlock>(ParameterInput *pin,
                                            StateDescriptor *geometry);
} // namespace Geometry

#endif // GEOMETRY_BOOSTED_MINKOWSKI_HPP_
