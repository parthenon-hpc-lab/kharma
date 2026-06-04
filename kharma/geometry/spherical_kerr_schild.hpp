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

#ifndef GEOMETRY_SPHERICAL_KERR_SCHILD_HPP_
#define GEOMETRY_SPHERICAL_KERR_SCHILD_HPP_

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

// Spherical Kerr-Schild
// No singularity at the horizon
// Formulae computed with Mathematica
// Mathematica notebooks stored
// Assumes WLOG that G = c = M = 1.
// TODO(JMM): Should we modify to accept arbitrary M?
class SphericalKerrSchild {
 public:
  SphericalKerrSchild() = default;
  KOKKOS_INLINE_FUNCTION
  SphericalKerrSchild(Real a) : a_(a), a2_(a * a) {}

  KOKKOS_INLINE_FUNCTION
  Real Lapse(Real X0, Real X1, Real X2, Real X3) const {
    const Real r = std::abs(X1);
    const Real cth = std::cos(X2);
    const Real rho2 = rho2_(r, cth);
    return std::sqrt(robust::ratio(rho2, 2 * r + rho2));
  }

  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(Real X0, Real X1, Real X2, Real X3, Real beta[NDSPACE]) const {
    using namespace Utils;
    using namespace robust;
    Real r, th, cth, sth;
    rth_(X1, X2, r, th, cth, sth);
    const Real rho2 = rho2_(r, cth);
    LinearAlgebra::SetZero(beta, NDSPACE);
    beta[0] = ratio(2 * r, 2 * r + rho2);
  }

  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(Real X0, Real X1, Real X2, Real X3, Real g[NDFULL][NDFULL]) const {
    using namespace Utils;
    using namespace robust;
    Real r, th, cth, sth, sth2, rho2;
    vars_(X1, X2, r, th, cth, sth, sth2, rho2);
    LinearAlgebra::SetZero(g, NDFULL, NDFULL);
    // do this first b/c it shows up lots of places
    g[0][1] = g[1][0] = ratio(2 * r, rho2);

    g[0][0] = -1 + g[0][1];
    g[0][3] = g[3][0] = -ratio(2 * a_ * r * sth2, rho2);
    g[1][1] = 1 + g[0][1];
    g[1][3] = g[3][1] = -a_ * (1 + g[0][1]) * sth2;
    g[2][2] = rho2;
    g[3][3] = (rho2 + a2_ * (1 + g[0][1]) * sth2) * sth2;
  }

  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(Real X0, Real X1, Real X2, Real X3,
                              Real g[NDFULL][NDFULL]) const {
    using namespace Utils;
    using namespace robust;
    Real r, th, cth, sth, sth2, rho2;
    vars_(X1, X2, r, th, cth, sth, sth2, rho2);
    LinearAlgebra::SetZero(g, NDFULL, NDFULL);
    // do this first b/c it shows up lots of places
    g[0][1] = g[1][0] = ratio(2 * r, rho2);

    g[0][0] = -1 - g[0][1];
    g[1][1] = ratio(-2 * r + rho2 + a2_ * sth2, rho2);
    g[1][3] = g[3][1] = ratio(a_, rho2);
    g[2][2] = ratio(1, rho2);
    g[3][3] = ratio(1, rho2 * sth2);
  }

  KOKKOS_INLINE_FUNCTION
  void Metric(Real X0, Real X1, Real X2, Real X3, Real g[NDSPACE][NDSPACE]) const {
    using namespace Utils;
    using namespace robust;
    Real r, th, cth, sth, sth2, rho2;
    vars_(X1, X2, r, th, cth, sth, sth2, rho2);
    const Real br = ratio(2 * r, rho2);
    LinearAlgebra::SetZero(g, NDSPACE, NDSPACE);
    g[0][0] = 1 + br;
    g[0][2] = g[2][0] = -a_ * (1 + br) * sth2;
    g[1][1] = rho2;
    g[2][2] = (rho2 + a2_ * (1 + br) * sth2) * sth2;
  }

  KOKKOS_INLINE_FUNCTION
  void MetricInverse(Real X0, Real X1, Real X2, Real X3, Real g[NDSPACE][NDSPACE]) const {
    using namespace Utils;
    using namespace robust;
    Real r, th, cth, sth, sth2, rho2;
    vars_(X1, X2, r, th, cth, sth, sth2, rho2);
    LinearAlgebra::SetZero(g, NDSPACE, NDSPACE);
    g[0][0] = ratio(rho2, 2 * r + rho2) + ratio(a2_ * sth2, rho2);
    g[0][2] = g[2][0] = ratio(a_, rho2);
    g[1][1] = ratio(1, rho2);
    g[2][2] = ratio(1, rho2 * sth2);
  }

  KOKKOS_INLINE_FUNCTION
  Real DetGamma(Real X0, Real X1, Real X2, Real X3) const {
    Real r, th, cth, sth, sth2, rho2;
    vars_(X1, X2, r, th, cth, sth, sth2, rho2);
    return std::sqrt(std::abs(rho2 * (2 * r + rho2) * sth2));
  }
  KOKKOS_INLINE_FUNCTION
  Real DetG(Real X0, Real X1, Real X2, Real X3) const {
    Real r, th, cth, sth, sth2, rho2;
    vars_(X1, X2, r, th, cth, sth, sth2, rho2);
    return std::sqrt(std::abs(rho2 * rho2 * sth2));
  }

  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(Real X0, Real X1, Real X2, Real X3,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    // Utils::SetConnectionCoeffByFD(*this, Gamma, X0, X1, X2, X3);
    Real dg[NDFULL][NDFULL][NDFULL];
    MetricDerivative(X0, X1, X2, X3, dg);
    SPACETIMELOOP3(a, b, c) {
      Gamma[c][b][a] = 0.5 * (dg[c][a][b] + dg[c][b][a] - dg[a][b][c]);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(Real X0, Real X1, Real X2, Real X3,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    using namespace Utils;
    using namespace robust;
    Real r, th, cth, sth, sth2, rho2;
    vars_(X1, X2, r, th, cth, sth, sth2, rho2);
    const Real r2 = r * r;
    const Real r3 = r2 * r;
    const Real r4 = r2 * r2;
    const Real r5 = r4 * r;
    const Real a4 = a2_ * a2_;
    const Real rho4 = rho2 * rho2;
    const Real cth2 = cth * cth;
    const Real s2th = std::sin(2 * th);
    const Real c2th = std::cos(2 * th);
    LinearAlgebra::SetZero(dg, NDFULL, NDFULL, NDFULL);

    dg[0][0][1] = 2 * ratio(rho2 - 2 * r2, rho4);
    dg[0][1][1] = dg[1][0][1] = dg[0][0][1];
    dg[0][3][1] = dg[3][0][1] = ratio(2 * a_ * (r2 - a2_ * cth2) * sth2, rho4);
    dg[1][1][1] = dg[0][0][1];
    dg[1][3][1] = dg[3][1][1] = dg[0][3][1];
    dg[2][2][1] = 2 * r;
    dg[3][3][1] = ratio(2 * sth2 *
                            (r5 + a4 * r * cth2 * cth2 - a2_ * r2 * sth2 +
                             cth2 * (2 * a2_ * r3 + a4 * sth2)),
                        rho4);

    dg[0][0][2] = ratio(4 * r * (rho2 - r2) * sth, rho4 * cth);
    dg[0][1][2] = dg[1][0][2] = dg[0][0][2];
    dg[0][3][2] = dg[3][0][2] = -ratio(2 * a_ * r * (a2_ + r2) * s2th, rho4);
    dg[1][1][2] = dg[0][0][2];
    dg[1][3][2] = dg[3][1][2] =
        -ratio(2 * a_ * cth *
                   (r3 * (2 + r) + a4 * cth2 * cth2 + a2_ * r * (2 + r + r * c2th)) * sth,
               rho4);
    dg[2][2][2] = -a2_ * s2th;
    dg[3][3][2] =
        (a2_ + r * (r - 2) + ratio(2 * r * (a2_ + r2) * (a2_ + r2), rho4)) * s2th;
  }

  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const {
    using namespace Utils;
    using namespace robust;
    Real r, th, cth, sth, sth2, rho2;
    vars_(X1, X2, r, th, cth, sth, sth2, rho2);
    const Real rho4 = rho2 * rho2;
    LinearAlgebra::SetZero(da, NDFULL);
    da[1] = ratio(r, rho2) - ratio(1 + r, 2 * r + rho2);
    da[2] = -ratio(2 * a2_ * r * cth * sth, 2 * r * rho2 + rho4);
  }

  KOKKOS_INLINE_FUNCTION
  void Coords(Real X0, Real X1, Real X2, Real X3, Real C[NDFULL]) const {
    const Real r = std::abs(X1);
    const Real sth = std::sin(X2);
    const Real cth = std::cos(X2);
    const Real sph = std::sin(X3);
    const Real cph = std::cos(X3);
    const Real r2a2 = std::sqrt(r * r + a2_);
    C[0] = X0; // oblate spheroidal coordinates
    C[1] = r2a2 * sth * cph;
    C[2] = r2a2 * sth * sph;
    C[3] = r * cth;
  }

 private:
  KOKKOS_INLINE_FUNCTION
  void vars_(Real X1, Real X2, Real &r, Real &th, Real &cth, Real &sth, Real &sth2,
             Real &rho2) const {
    rth_(X1, X2, r, th, cth, sth);
    sth2 = sth * sth;
    rho2 = rho2_(r, cth);
  }
  KOKKOS_INLINE_FUNCTION
  void rth_(Real X1, Real X2, Real &r, Real &th, Real &cth, Real &sth) const {
    r = std::abs(X1);
    th = X2;
    cth = std::cos(th);
    sth = std::sin(th);
  }
  KOKKOS_INLINE_FUNCTION
  Real rho2_(Real r, Real cth) const { // cth = cos(th)
    return r * r + a2_ * cth * cth;
  }
  Real a_ = 0;
  Real a2_ = 0;
};

using SphericalKSMeshBlock = Analytic<SphericalKerrSchild, IndexerMeshBlock>;
using SphericalKSMesh = Analytic<SphericalKerrSchild, IndexerMesh>;

using CSphericalKSMeshBlock = CachedOverMeshBlock<SphericalKSMeshBlock>;
using CSphericalKSMesh = CachedOverMesh<SphericalKSMesh>;

template <>
void Initialize<SphericalKSMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);
template <>
void Initialize<CSphericalKSMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);

} // namespace Geometry

#endif // GEOMETRY_SPHERICAL_KERR_SCHILD_HPP_
