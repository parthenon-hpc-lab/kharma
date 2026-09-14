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

#ifndef GEOMETRY_SPHERICAL_MINKOWSKI_HPP_
#define GEOMETRY_SPHERICAL_MINKOWSKI_HPP_

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
#include "phoebus_utils/robust.hpp"

namespace Geometry {

class SphericalMinkowski {
 public:
  KOKKOS_INLINE_FUNCTION
  Real Lapse(Real X0, Real X1, Real X2, Real X3) const { return 1.; }
  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(Real X0, Real X1, Real X2, Real X3, Real beta[NDSPACE]) const {
    LinearAlgebra::SetZero(beta, NDSPACE);
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(Real X0, Real X1, Real X2, Real X3, Real g[NDFULL][NDFULL]) const {
    LinearAlgebra::SetZero(g, NDFULL, NDFULL);
    Real r2 = X1 * X1;
    Real sth = std::sin(X2);
    Real sth2 = sth * sth;
    g[0][0] = -1;
    g[1][1] = 1;
    g[2][2] = r2;
    g[3][3] = r2 * sth2;
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(Real X0, Real X1, Real X2, Real X3,
                              Real g[NDFULL][NDFULL]) const {
    LinearAlgebra::SetZero(g, NDFULL, NDFULL);
    const Real sth = std::sin(X2);
    g[0][0] = -1;
    g[1][1] = 1;
    g[2][2] = robust::ratio(1, X1 * X1);
    g[3][3] = robust::ratio(1, X1 * X1 * sth * sth);
  }
  KOKKOS_INLINE_FUNCTION
  void Metric(Real X0, Real X1, Real X2, Real X3, Real g[NDSPACE][NDSPACE]) const {
    LinearAlgebra::SetZero(g, NDSPACE, NDSPACE);
    g[0][0] = 1;
    g[1][1] = X1 * X1;
    g[2][2] = X1 * std::sin(X2);
    g[2][2] *= g[2][2];
  }
  KOKKOS_INLINE_FUNCTION
  void MetricInverse(Real X0, Real X1, Real X2, Real X3, Real g[NDSPACE][NDSPACE]) const {
    LinearAlgebra::SetZero(g, NDSPACE, NDSPACE);
    const Real sth = std::sin(X2);
    g[0][0] = 1;
    g[1][1] = robust::ratio(1, X1 * X1);
    g[2][2] = robust::ratio(1, X1 * X1 * sth * sth);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetGamma(Real X0, Real X1, Real X2, Real X3) const {
    return X1 * X1 * std::abs(std::sin(X2));
  }
  KOKKOS_INLINE_FUNCTION
  Real DetG(Real X0, Real X1, Real X2, Real X3) const {
    return X1 * X1 * std::abs(std::sin(X2));
  }
  // Gamma^mu_{nu sigma}
  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(Real X0, Real X1, Real X2, Real X3,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    Utils::SetConnectionCoeffByFD(*this, Gamma, X0, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(Real X0, Real X1, Real X2, Real X3,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    using namespace Utils;
    LinearAlgebra::SetZero(dg, NDFULL, NDFULL, NDFULL);
    const Real r = std::abs(X1);
    const Real sth = std::sin(X2);
    const Real s2th = std::sin(2 * X2);
    dg[2][2][1] = 2 * r;
    dg[3][3][1] = 2 * r * sth * sth;
    dg[3][3][2] = r * r * s2th;
  }
  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const {
    LinearAlgebra::SetZero(da, NDFULL);
  }

  KOKKOS_INLINE_FUNCTION
  void Coords(Real X0, Real X1, Real X2, Real X3, Real C[NDFULL]) const {
    const Real r = std::abs(X1);
    const Real sth = std::sin(X2);
    const Real cth = std::cos(X2);
    const Real sph = std::sin(X3);
    const Real cph = std::cos(X3);
    C[0] = X0;
    C[1] = r * sth * cph;
    C[2] = r * sth * sph;
    C[3] = r * cth;
  }
};

using SphMinkowskiMeshBlock = Analytic<SphericalMinkowski, IndexerMeshBlock>;
using SphMinkowskiMesh = Analytic<SphericalMinkowski, IndexerMesh>;

using CSphMinkowskiMeshBlock = CachedOverMeshBlock<SphMinkowskiMeshBlock>;
using CSphMinkowskiMesh = CachedOverMesh<SphMinkowskiMesh>;

template <>
void Initialize<CSphMinkowskiMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);

} // namespace Geometry

#endif // GEOMETRY_SPHERICAL_MINKOWSKI_HPP_
