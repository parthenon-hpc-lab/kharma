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
#include "geometry/modified_system.hpp"
#include "geometry/sph2cart.hpp"
#include "monopole_gr/monopole_gr_base.hpp"
#include "phoebus_utils/linear_algebra.hpp"
#include "phoebus_utils/robust.hpp"

namespace Geometry {

/*
 * Metric from Monopole GR module.
 * Assumes spherical symmetry for the spacetime.
 * This class also assumes spherical coordinates for the fluid sector.

 * Since the metric and fluid radial grids don't necessarily align, we
 * interpolate the metric quantities and use the AnalyticSystem
 * modifier to fill in the missing functionality.
 */
class MonopoleSph {
 public:
  MonopoleSph() = default;
  KOKKOS_INLINE_FUNCTION
  MonopoleSph(const MonopoleGR::Hypersurface_t &hypersurface,
              const MonopoleGR::Alpha_t &alpha, const MonopoleGR::Beta_t &beta,
              const MonopoleGR::Gradients_t &gradients, const MonopoleGR::Radius &rgrid)
      : hypersurface_(hypersurface), alpha_(alpha), beta_(beta), gradients_(gradients),
        rgrid_(rgrid) {}

  KOKKOS_INLINE_FUNCTION
  Real Lapse(Real X0, Real X1, Real X2, Real X3) const {
    const Real r = std::abs(X1);
    return MonopoleGR::Interpolate(r, alpha_, rgrid_);
  }

  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(Real X0, Real X1, Real X2, Real X3, Real beta[NDSPACE]) const {
    const Real r = std::abs(X1);
    LinearAlgebra::SetZero(beta, NDSPACE);
    beta[0] = MonopoleGR::Interpolate(r, beta_, rgrid_);
  }

  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(Real X0, Real X1, Real X2, Real X3, Real g[NDFULL][NDFULL]) const {
    const Real r = std::abs(X1);
    const Real r2 = r * r;
    const Real sth = std::sin(X2);
    const Real alpha = MonopoleGR::Interpolate(r, alpha_, rgrid_);
    const Real beta = MonopoleGR::Interpolate(r, beta_, rgrid_);
    const Real a =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::A);
    const Real a2 = a * a;

    LinearAlgebra::SetZero(g, NDFULL, NDFULL);
    g[0][0] = -alpha * alpha + a2 * beta * beta;
    g[0][1] = g[1][0] = a2 * beta;
    g[1][1] = a2;
    g[2][2] = r2;
    g[3][3] = r2 * sth * sth;
  }

  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(Real X0, Real X1, Real X2, Real X3,
                              Real g[NDFULL][NDFULL]) const {
    using robust::ratio;
    const Real r = std::abs(X1);
    const Real ir2 = ratio(1., r * r);
    const Real sth = std::sin(X2);
    const Real alpha = MonopoleGR::Interpolate(r, alpha_, rgrid_);
    const Real ialpha2 = ratio(1., alpha * alpha);
    const Real beta = MonopoleGR::Interpolate(r, beta_, rgrid_);
    const Real a =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::A);

    LinearAlgebra::SetZero(g, NDFULL, NDFULL);
    g[0][0] = -ialpha2;
    g[0][1] = g[1][0] = beta * ialpha2;
    g[1][1] = ratio(1., a * a) - beta * beta * ialpha2;
    g[2][2] = ir2;
    g[3][3] = ir2 * ratio(1., sth * sth);
  }

  KOKKOS_INLINE_FUNCTION
  void Metric(Real X0, Real X1, Real X2, Real X3, Real g[NDSPACE][NDSPACE]) const {
    const Real r = std::abs(X1);
    const Real r2 = r * r;
    const Real sth = std::sin(X2);
    const Real a =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::A);
    LinearAlgebra::SetZero(g, NDSPACE, NDSPACE);
    g[0][0] = a * a;
    g[1][1] = r2;
    g[2][2] = r2 * sth * sth;
  }

  KOKKOS_INLINE_FUNCTION
  void MetricInverse(Real X0, Real X1, Real X2, Real X3, Real g[NDSPACE][NDSPACE]) const {
    using robust::ratio;
    const Real r = std::abs(X1);
    const Real ir2 = ratio(1., r * r);
    const Real sth = std::sin(X2);
    const Real a =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::A);
    LinearAlgebra::SetZero(g, NDSPACE, NDSPACE);
    g[0][0] = ratio(1., a * a);
    g[1][1] = ir2;
    g[2][2] = ir2 * ratio(1., sth * sth);
  }

  KOKKOS_INLINE_FUNCTION
  Real DetGamma(Real X0, Real X1, Real X2, Real X3) const {
    const Real r = std::abs(X1);
    const Real sth = std::sin(X2);
    const Real a =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::A);
    return a * r * r * std::abs(sth);
  }

  KOKKOS_INLINE_FUNCTION
  Real DetG(Real X0, Real X1, Real X2, Real X3) const {
    const Real r = std::abs(X1);
    const Real alpha = MonopoleGR::Interpolate(r, alpha_, rgrid_);
    return alpha * DetGamma(X0, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(Real X0, Real X1, Real X2, Real X3,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    const Real r = std::abs(X1);
    const Real th = X2;
    const Real sth = std::sin(th);
    const Real alpha = MonopoleGR::Interpolate(r, alpha_, rgrid_);
    const Real beta = MonopoleGR::Interpolate(r, beta_, rgrid_);
    const Real a =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::A);
    const Real dadr =
        MonopoleGR::Interpolate(r, gradients_, rgrid_, MonopoleGR::Gradients::DADR);
    const Real dalphadr =
        MonopoleGR::Interpolate(r, gradients_, rgrid_, MonopoleGR::Gradients::DALPHADR);
    const Real dbetadr =
        MonopoleGR::Interpolate(r, gradients_, rgrid_, MonopoleGR::Gradients::DBETADR);
    const Real dadt =
        MonopoleGR::Interpolate(r, gradients_, rgrid_, MonopoleGR::Gradients::DADT);
    const Real dalphadt =
        MonopoleGR::Interpolate(r, gradients_, rgrid_, MonopoleGR::Gradients::DALPHADT);
    const Real dbetadt =
        MonopoleGR::Interpolate(r, gradients_, rgrid_, MonopoleGR::Gradients::DBETADT);

    LinearAlgebra::SetZero(dg, NDFULL, NDFULL, NDFULL);
    // d/dt
    dg[0][0][0] =
        2 * a * beta * beta * dadt - 2 * alpha * dalphadt + 2 * a * a * beta * dbetadt;
    dg[1][0][0] = dg[0][1][0] = 2 * a * beta * dadt + a * a * dbetadt;
    dg[1][1][0] = 2 * a * dadt;
    // d/dr
    dg[0][0][1] = -2 * alpha * dalphadr + 2 * a * beta * (beta * dadr + a * dbetadr);
    dg[1][0][1] = dg[0][1][1] = a * (2 * beta * dadr + a * dbetadr);
    dg[1][1][1] = 2 * a * dadr;
    dg[2][2][1] = 2 * r;
    dg[3][3][1] = 2 * r * sth * sth;
    // d/dth
    dg[3][3][2] = r * r * std::sin(2 * th);
    // d/dph = 0
  }

  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(Real X0, Real X1, Real X2, Real X3,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    // This is less error prone than a hardcoded version.
    Utils::SetConnectionCoeffByFD(*this, Gamma, X0, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const {
    const Real r = std::abs(X1);
    const Real alpha = MonopoleGR::Interpolate(r, alpha_, rgrid_);
    const Real dalphadr =
        MonopoleGR::Interpolate(r, gradients_, rgrid_, MonopoleGR::Gradients::DALPHADR);
    const Real dalphadt =
        MonopoleGR::Interpolate(r, gradients_, rgrid_, MonopoleGR::Gradients::DALPHADT);

    LinearAlgebra::SetZero(da, NDFULL);
    da[0] = dalphadt / alpha;
    da[1] = dalphadr / alpha;
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

 private:
  MonopoleGR::Hypersurface_t hypersurface_;
  MonopoleGR::Alpha_t alpha_;
  MonopoleGR::Beta_t beta_;
  MonopoleGR::Gradients_t gradients_;
  MonopoleGR::Radius rgrid_;
};

using MplSphMeshBlock = Analytic<MonopoleSph, IndexerMeshBlock>;
using MplSphMesh = Analytic<MonopoleSph, IndexerMesh>;

template <>
void Initialize<MplSphMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);

using CMplSphMeshBlock = CachedOverMeshBlock<Analytic<MonopoleSph, IndexerMeshBlock>>;
using CMplSphMesh = CachedOverMesh<Analytic<MonopoleSph, IndexerMesh>>;

template <>
void Initialize<CMplSphMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);

// --------------------------------------------------------------------------------

/*
 * Cartesian Monopole class.
 * Built explicitly on modifiers.
 */
using MonopoleCart = Modified<MonopoleSph, SphericalToCartesian, true>;

using MplCartMeshBlock = Analytic<MonopoleCart, IndexerMeshBlock>;
using CMplCartMeshBlock = CachedOverMeshBlock<Analytic<MonopoleCart, IndexerMeshBlock>>;

using MplCartMesh = Analytic<MonopoleCart, IndexerMesh>;
using CMplCartMesh = CachedOverMesh<Analytic<MonopoleCart, IndexerMesh>>;

template <>
void Initialize<CMplCartMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);

} // namespace Geometry
