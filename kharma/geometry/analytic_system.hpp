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

#ifndef GEOMETRY_ANALYTIC_SYSTEM_
#define GEOMETRY_ANALYTIC_SYSTEM_

#include <array>
#include <cmath>
#include <limits>
#include <utility>

// Parthenon includes
#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

// phoebus includes
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/cell_locations.hpp"

using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;
using parthenon::ParArray1D;

namespace Geometry {

// Boiler plate for analytic coordinate systems so you only need to
// define the functions of X0,X1,X2,X3, not all the functions of indices.
/*
  To utilize this template, the fullowing methods are expected:
  - A constructor/destructor
  - Lapse(X0, X1, X2, X3)
  - ContravariantShift(X0, X1, X2, X3, beta)
  - Metric(X0, X1, X2, X3, gamma)
  - MetricInverse(X0, X1, X2, X3, gamma)
  - DetGamma(X0, X1, X2, X3)
  - DetG(X0, X1, X2, X3)
  - ConnectionCoefficient(X0, X1, X2, X3, Gamma)
  - MetricDerivative(X0, X1, X2, X3, dg)
  - GradLnAlpha(X0, X1, X2, X3, )
  - C0(X0, X1, X2, X3)
  - C1(X0, X1, X2, X3)
  - C2(X0, X1, X2, X3)
  - C3(X0, X1, X2, X3)

  The user must also specify a MeshBlock or Mesh indexer.
 */
class IndexerMeshBlock {
 public:
  IndexerMeshBlock() = default;
  IndexerMeshBlock(const Coordinates_t &coordinates) : coordinates_(coordinates) {}
  KOKKOS_INLINE_FUNCTION
  void GetX(CellLocation loc, int b, int k, int j, int i, Real &X1, Real &X2,
            Real &X3) const {
    return GetX(loc, k, j, i, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  void GetX(CellLocation loc, int k, int j, int i, Real &X1, Real &X2, Real &X3) const {
    // defaults
    X1 = coordinates_.Xc<1>(i);
    X2 = coordinates_.Xc<2>(j);
    X3 = coordinates_.Xc<3>(k);
    // overwrite
    switch (loc) {
    case CellLocation::Face1:
      X1 = coordinates_.Xf<1>(i);
      break;
    case CellLocation::Face2:
      X2 = coordinates_.Xf<2>(j);
      break;
    case CellLocation::Face3:
      X3 = coordinates_.Xf<3>(k);
      break;
    case CellLocation::Corn:
      X1 = coordinates_.Xf<1>(i);
      X2 = coordinates_.Xf<2>(j);
      X3 = coordinates_.Xf<3>(k);
      break;
    default: // CellLocation::Cent:
      break;
    }
    return;
  }

 private:
  Coordinates_t coordinates_;
};
inline IndexerMeshBlock GetIndexer(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetParentPointer();
  auto &coords = pmb->coords;
  return IndexerMeshBlock(coords);
}

class IndexerMesh {
 public:
  IndexerMesh() = default;
  IndexerMesh(const ParArray1D<Coordinates_t> &coordinates) : coordinates_(coordinates) {}
  KOKKOS_INLINE_FUNCTION
  void GetX(CellLocation loc, int k, int j, int i, Real &X1, Real &X2, Real &X3) const {
    PARTHENON_FAIL("A MeshBlock index is required for indexing into the mesh");
  }
  KOKKOS_INLINE_FUNCTION
  void GetX(CellLocation loc, int b, int k, int j, int i, Real &X1, Real &X2,
            Real &X3) const {
    // defaults
    X1 = coordinates_(b).Xc<1>(i);
    X2 = coordinates_(b).Xc<2>(j);
    X3 = coordinates_(b).Xc<3>(k);
    // overwrite
    switch (loc) {
    case CellLocation::Face1:
      X1 = coordinates_(b).Xf<1>(i);
      break;
    case CellLocation::Face2:
      X2 = coordinates_(b).Xf<2>(j);
      break;
    case CellLocation::Face3:
      X3 = coordinates_(b).Xf<3>(k);
      break;
    case CellLocation::Corn:
      X1 = coordinates_(b).Xf<1>(i);
      X2 = coordinates_(b).Xf<2>(j);
      X3 = coordinates_(b).Xf<3>(k);
      break;
    default: // CellLocation::Cent:
      break;
    }
    return;
  }

 private:
  ParArray1D<Coordinates_t> coordinates_;
};
inline IndexerMesh GetIndexer(MeshData<Real> *rc) {
  auto pmesh = rc->GetParentPointer();
  int nblocks = rc->NumBlocks();
  ParArray1D<Coordinates_t> coords("GetCoordinateSystem::coords", nblocks);
  auto coords_h = Kokkos::create_mirror_view(coords);
  for (int i = 0; i < nblocks; ++i) {
    coords_h(i) = rc->GetBlockData(i)->GetBlockPointer()->coords;
  }
  Kokkos::deep_copy(coords, coords_h);
  return IndexerMesh(coords);
}

template <typename System, typename Indexer>
class Analytic {
 public:
  Analytic() = default;
  template <typename... Args>
  Analytic(const Indexer &indexer, Args... args)
      : X0_(0), indexer_(indexer), system_(std::forward<Args>(args)...) {}
  template <typename... Args>
  Analytic(Real X0, const Indexer &indexer, Args... args)
      : X0_(X0), indexer_(indexer), system_(std::forward<Args>(args)...) {}
  KOKKOS_INLINE_FUNCTION
  Real Lapse(Real X0, Real X1, Real X2, Real X3) const {
    return system_.Lapse(X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real Lapse(int b, Real X0, Real X1, Real X2, Real X3) const {
    return system_.Lapse(X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real Lapse(CellLocation loc, int k, int j, int i) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, k, j, i, X1, X2, X3);
    return Lapse(X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real Lapse(CellLocation loc, int b, int k, int j, int i) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, b, k, j, i, X1, X2, X3);
    return Lapse(X0_, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(Real X0, Real X1, Real X2, Real X3, Real beta[NDSPACE]) const {
    return system_.ContravariantShift(X0, X1, X2, X3, beta);
  }
  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(int b, Real X0, Real X1, Real X2, Real X3,
                          Real beta[NDSPACE]) const {
    return system_.ContravariantShift(X0, X1, X2, X3, beta);
  }
  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(CellLocation loc, int k, int j, int i,
                          Real beta[NDSPACE]) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, k, j, i, X1, X2, X3);
    return ContravariantShift(X0_, X1, X2, X3, beta);
  }
  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(CellLocation loc, int b, int k, int j, int i,
                          Real beta[NDSPACE]) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, b, k, j, i, X1, X2, X3);
    return ContravariantShift(X0_, X1, X2, X3, beta);
  }

  KOKKOS_INLINE_FUNCTION
  void Metric(Real X0, Real X1, Real X2, Real X3, Real gamma[NDSPACE][NDSPACE]) const {
    return system_.Metric(X0, X1, X2, X3, gamma);
  }
  KOKKOS_INLINE_FUNCTION
  void Metric(int b, Real X0, Real X1, Real X2, Real X3,
              Real gamma[NDSPACE][NDSPACE]) const {
    return system_.Metric(X0, X1, X2, X3, gamma);
  }
  KOKKOS_INLINE_FUNCTION
  void Metric(CellLocation loc, int k, int j, int i, Real gamma[NDSPACE][NDSPACE]) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, k, j, i, X1, X2, X3);
    return Metric(X0_, X1, X2, X3, gamma);
  }
  KOKKOS_INLINE_FUNCTION
  void Metric(CellLocation loc, int b, int k, int j, int i,
              Real gamma[NDSPACE][NDSPACE]) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, b, k, j, i, X1, X2, X3);
    return Metric(X0_, X1, X2, X3, gamma);
  }

  KOKKOS_INLINE_FUNCTION
  void MetricInverse(Real X0, Real X1, Real X2, Real X3,
                     Real gamma[NDSPACE][NDSPACE]) const {
    return system_.MetricInverse(X0, X1, X2, X3, gamma);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricInverse(int b, Real X0, Real X1, Real X2, Real X3,
                     Real gamma[NDSPACE][NDSPACE]) const {
    return system_.MetricInverse(X0, X1, X2, X3, gamma);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricInverse(CellLocation loc, int k, int j, int i,
                     Real gamma[NDSPACE][NDSPACE]) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, k, j, i, X1, X2, X3);
    return MetricInverse(X0_, X1, X2, X3, gamma);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricInverse(CellLocation loc, int b, int k, int j, int i,
                     Real gamma[NDSPACE][NDSPACE]) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, b, k, j, i, X1, X2, X3);
    return MetricInverse(X0_, X1, X2, X3, gamma);
  }

  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(Real X0, Real X1, Real X2, Real X3, Real g[NDFULL][NDFULL]) const {
    return system_.SpacetimeMetric(X0, X1, X2, X3, g);
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(int b, Real X0, Real X1, Real X2, Real X3,
                       Real g[NDFULL][NDFULL]) const {
    return system_.SpacetimeMetric(X0, X1, X2, X3, g);
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(CellLocation loc, int k, int j, int i,
                       Real g[NDFULL][NDFULL]) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, k, j, i, X1, X2, X3);
    return SpacetimeMetric(X0_, X1, X2, X3, g);
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(CellLocation loc, int b, int k, int j, int i,
                       Real g[NDFULL][NDFULL]) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, b, k, j, i, X1, X2, X3);
    return SpacetimeMetric(X0_, X1, X2, X3, g);
  }

  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(Real X0, Real X1, Real X2, Real X3,
                              Real g[NDFULL][NDFULL]) const {
    return system_.SpacetimeMetricInverse(X0, X1, X2, X3, g);
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(int b, Real X0, Real X1, Real X2, Real X3,
                              Real g[NDFULL][NDFULL]) const {
    return system_.SpacetimeMetricInverse(X0, X1, X2, X3, g);
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(CellLocation loc, int k, int j, int i,
                              Real g[NDFULL][NDFULL]) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, k, j, i, X1, X2, X3);
    return SpacetimeMetricInverse(X0_, X1, X2, X3, g);
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(CellLocation loc, int b, int k, int j, int i,
                              Real g[NDFULL][NDFULL]) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, b, k, j, i, X1, X2, X3);
    return SpacetimeMetricInverse(X0_, X1, X2, X3, g);
  }

  KOKKOS_INLINE_FUNCTION
  Real DetGamma(Real X0, Real X1, Real X2, Real X3) const {
    return system_.DetGamma(X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetGamma(int b, Real X0, Real X1, Real X2, Real X3) const {
    return system_.DetGamma(X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetGamma(CellLocation loc, int k, int j, int i) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, k, j, i, X1, X2, X3);
    return DetGamma(X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetGamma(CellLocation loc, int b, int k, int j, int i) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, b, k, j, i, X1, X2, X3);
    return DetGamma(X0_, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  Real DetG(Real X0, Real X1, Real X2, Real X3) const {
    return system_.DetG(X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetG(int b, Real X0, Real X1, Real X2, Real X3) const {
    return system_.DetG(X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetG(CellLocation loc, int k, int j, int i) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, k, j, i, X1, X2, X3);
    return DetG(X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetG(CellLocation loc, int b, int k, int j, int i) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, b, k, j, i, X1, X2, X3);
    return DetG(X0_, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(Real X0, Real X1, Real X2, Real X3,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    return system_.ConnectionCoefficient(X0, X1, X2, X3, Gamma);
  }
  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(int b, Real X0, Real X1, Real X2, Real X3,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    return system_.ConnectionCoefficient(X0, X1, X2, X3, Gamma);
  }
  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(CellLocation loc, int k, int j, int i,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, k, j, i, X1, X2, X3);
    return system_.ConnectionCoefficient(X0_, X1, X2, X3, Gamma);
  }
  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(CellLocation loc, int b, int k, int j, int i,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, b, k, j, i, X1, X2, X3);
    return system_.ConnectionCoefficient(X0_, X1, X2, X3, Gamma);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(Real X0, Real X1, Real X2, Real X3,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    return system_.MetricDerivative(X0, X1, X2, X3, dg);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(int b, Real X0, Real X1, Real X2, Real X3,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    return system_.MetricDerivative(X0, X1, X2, X3, dg);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(CellLocation loc, int k, int j, int i,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, k, j, i, X1, X2, X3);
    return system_.MetricDerivative(X0_, X1, X2, X3, dg);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(CellLocation loc, int b, int k, int j, int i,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, b, k, j, i, X1, X2, X3);
    return system_.MetricDerivative(X0_, X1, X2, X3, dg);
  }

  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const {
    return system_.GradLnAlpha(X0, X1, X2, X3, da);
  }
  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(int b, Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const {
    return system_.GradLnAlpha(X0, X1, X2, X3, da);
  }
  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(CellLocation loc, int k, int j, int i, Real da[NDFULL]) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, k, j, i, X1, X2, X3);
    return system_.GradLnAlpha(X0_, X1, X2, X3, da);
  }
  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(CellLocation loc, int b, int k, int j, int i, Real da[NDFULL]) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, b, k, j, i, X1, X2, X3);
    return system_.GradLnAlpha(X0_, X1, X2, X3, da);
  }

  KOKKOS_INLINE_FUNCTION
  void Coords(Real X0, Real X1, Real X2, Real X3, Real C[NDFULL]) const {
    return system_.Coords(X0, X1, X2, X3, C);
  }
  KOKKOS_INLINE_FUNCTION
  void Coords(int b, Real X0, Real X1, Real X2, Real X3, Real C[NDFULL]) const {
    return system_.Coords(X0, X1, X2, X3, C);
  }
  KOKKOS_INLINE_FUNCTION
  void Coords(CellLocation loc, int k, int j, int i, Real C[NDFULL]) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, k, j, i, X1, X2, X3);
    return system_.Coords(X0_, X1, X2, X3, C);
  }
  KOKKOS_INLINE_FUNCTION
  void Coords(CellLocation loc, int b, int k, int j, int i, Real C[NDFULL]) const {
    Real X1, X2, X3;
    indexer_.GetX(loc, b, k, j, i, X1, X2, X3);
    return system_.Coords(X0_, X1, X2, X3, C);
  }

 private:
  Real X0_ = 0;
  Indexer indexer_;
  System system_;
};
} // namespace Geometry

#endif // GEOMETRY_ANALYTIC_SYSTEM_
