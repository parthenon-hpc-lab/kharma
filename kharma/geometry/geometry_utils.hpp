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

#ifndef GEOMETRY_GEOMETRY_UTILS_HPP_
#define GEOMETRY_GEOMETRY_UTILS_HPP_

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// Parthenon includes
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

// Phoebus includes
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/linear_algebra.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/variables.hpp"

namespace Geometry {
constexpr int NDSPACE = 3;
constexpr int NDFULL = NDSPACE + 1;
} // namespace Geometry

#define SPACELOOP(i) for (int i = 0; i < Geometry::NDSPACE; i++)
#define SPACETIMELOOP(mu) for (int mu = 0; mu < Geometry::NDFULL; mu++)

#define SPACELOOP2(i, j) SPACELOOP(i) SPACELOOP(j)
#define SPACETIMELOOP2(mu, nu) SPACETIMELOOP(mu) SPACETIMELOOP(nu)
#define SPACELOOP3(i, j, k) SPACELOOP(i) SPACELOOP(j) SPACELOOP(k)
#define SPACETIMELOOP3(mu, nu, sigma)                                                    \
  SPACETIMELOOP(mu) SPACETIMELOOP(nu) SPACETIMELOOP(sigma)

namespace Geometry {
namespace Utils {

struct MeshBlockShape {
  MeshBlockShape(ParameterInput *pin) {
    ng = pin->GetOrAddInteger("parthenon/mesh", "nghost", 2);
    int mesh_nx1 = pin->GetInteger("parthenon/mesh", "nx1");
    int mesh_nx2 = pin->GetInteger("parthenon/mesh", "nx2");
    int mesh_nx3 = pin->GetInteger("parthenon/mesh", "nx3");
    ndim = (mesh_nx3 > 1 ? 3 : (mesh_nx2 > 1 ? 2 : 1));

    nx1 = pin->GetOrAddInteger("parthenon/meshblock", "nx1", mesh_nx1) + 2 * ng;
    nx2 = (ndim >= 2)
              ? pin->GetOrAddInteger("parthenon/meshblock", "nx2", mesh_nx2) + 2 * ng
              : 1;
    nx3 = (ndim >= 3)
              ? pin->GetOrAddInteger("parthenon/meshblock", "nx3", mesh_nx3) + 2 * ng
              : 1;
  }
  int nx1, nx2, nx3, ndim, ng;
};

KOKKOS_INLINE_FUNCTION void
SetConnectionCoeffFromMetricDerivs(const Real dg[NDFULL][NDFULL][NDFULL],
                                   Real Gamma[NDFULL][NDFULL][NDFULL]) {
  SPACETIMELOOP3(a, b, c) {
    Gamma[a][b][c] = 0.5 * (dg[b][a][c] + dg[c][a][b] - dg[b][c][a]);
  }
}

template <typename System, typename... Args>
KOKKOS_INLINE_FUNCTION void SetConnectionCoeffByFD(const System &s,
                                                   Real Gamma[NDFULL][NDFULL][NDFULL],
                                                   Args... args) {
  Real dg[NDFULL][NDFULL][NDFULL];
  s.MetricDerivative(std::forward<Args>(args)..., dg);
  SetConnectionCoeffFromMetricDerivs(dg, Gamma);
}

// TODO(JMM) Currently assumes static metric
template <typename System>
KOKKOS_INLINE_FUNCTION void SetGradLnAlphaByFD(const System &s, Real dx, Real X0, Real X1,
                                               Real X2, Real X3, Real da[NDFULL]) {
  Real EPS = std::pow(std::numeric_limits<Real>::epsilon(), 1.0 / 3.0);
  LinearAlgebra::SetZero(da, NDFULL);
  for (int d = 1; d < NDFULL; ++d) {
    Real X1p = X1;
    Real X1m = X1;
    Real X2p = X2;
    Real X2m = X2;
    Real X3p = X3;
    Real X3m = X3;
    if (d == 1) {
      dx = std::max(std::abs(X1), 1.0) * EPS;
      X1p += dx;
      X1m -= dx;
    } else if (d == 2) {
      dx = std::max(std::abs(X2), 1.0) * EPS;
      X2p += dx;
      X2m -= dx;
    } else if (d == 3) {
      dx = std::max(std::abs(X3), 1.0) * EPS;
      X3p += dx;
      X3m -= dx;
    }
    Real alpham = s.Lapse(X0, X1m, X2m, X3m);
    Real alphap = s.Lapse(X0, X1p, X2p, X3p);
    da[d] = robust::ratio(alphap - alpham, dx * (alpham + alphap));
  }
}

KOKKOS_INLINE_FUNCTION
void Lower(const double Vcon[NDFULL], const double Gcov[NDFULL][NDFULL],
           double Vcov[NDFULL]) {
  SPACETIMELOOP(mu) {
    Vcov[mu] = 0.;
    SPACETIMELOOP(nu) { Vcov[mu] += Gcov[mu][nu] * Vcon[nu]; }
  }
}

KOKKOS_INLINE_FUNCTION
void Raise(const double Vcov[NDFULL], const double Gcon[NDFULL][NDFULL],
           double Vcon[NDFULL]) {
  SPACETIMELOOP(mu) {
    Vcon[mu] = 0.;
    SPACETIMELOOP(nu) { Vcon[mu] += Gcon[mu][nu] * Vcov[nu]; }
  }
}

KOKKOS_INLINE_FUNCTION
int KroneckerDelta(const int a, const int b) {
  if (a == b) {
    return 1;
  } else {
    return 0;
  }
}

// TODO(JMM): Currently assumes static metric
template <typename System>
KOKKOS_INLINE_FUNCTION void SetMetricGradientByFD(const System &s, Real dx, Real X0,
                                                  Real X1, Real X2, Real X3,
                                                  Real dg[NDFULL][NDFULL][NDFULL]) {
  Real EPS = std::pow(std::numeric_limits<Real>::epsilon(), 1.0 / 3.0);
  LinearAlgebra::SetZero(dg, NDFULL, NDFULL, NDFULL);
  Real gl[NDFULL][NDFULL];
  Real gr[NDFULL][NDFULL];
  for (int d = 1; d < NDFULL; ++d) {
    Real X1L = X1;
    Real X2L = X2;
    Real X3L = X3;
    Real X1R = X1;
    Real X2R = X2;
    Real X3R = X3;
    if (d == 1) {
      dx = std::max(std::abs(X1), 1.0) * EPS;
      X1L -= dx;
      X1R += dx;
    }
    if (d == 2) {
      dx = std::max(std::abs(X2), 1.0) * EPS;
      X2L -= dx;
      X2R += dx;
    }
    if (d == 3) {
      dx = std::max(std::abs(X3), 1.0) * EPS;
      X3L -= dx;
      X3R += dx;
    }
    s.SpacetimeMetric(X0, X1R, X2R, X3R, gr);
    s.SpacetimeMetric(X0, X1L, X2L, X3L, gl);
    SPACETIMELOOP(mu) {
      SPACETIMELOOP(nu) {
        dg[mu][nu][d] = robust::ratio(gr[mu][nu] - gl[mu][nu], 2 * dx);
      }
    }
  }
}
KOKKOS_FORCEINLINE_FUNCTION
int Flatten2(int m, int n, int size) {
  static constexpr int ind[4][4] = {
      {0, 1, 3, 6}, {1, 2, 4, 7}, {3, 4, 5, 8}, {6, 7, 8, 9}};
  PARTHENON_DEBUG_REQUIRE(0 <= m && 0 <= n && m < size && n < size, "bounds");
  return ind[m][n];
}
template <int size>
KOKKOS_INLINE_FUNCTION int SymSize() {
  return size * size - (size * (size - 1)) / 2;
}

} // namespace Utils

namespace impl {
template <typename Data, typename System>
void SetGeometryDefault(Data *rc, const System &system) {
  std::vector<std::string> coord_names = {geometric_variables::cell_coords::name(),
                                          geometric_variables::node_coords::name()};
  PackIndexMap imap;
  auto pack = rc->PackVariables(coord_names, imap);
  PARTHENON_REQUIRE(imap["g.c.coord"].second >= 0, "g.c.coord exists");
  PARTHENON_REQUIRE(imap["g.n.coord"].second >= 0, "g.n.coord exists");
  PARTHENON_REQUIRE(imap["g.c.coord"].second - imap["g.c.coord"].first + 1 == 4,
                    "g.c.coord has correct shape");
  PARTHENON_REQUIRE(imap["g.n.coord"].second - imap["g.n.coord"].first + 1 == 4,
                    "g.n.coord has correct shape");
  int icoord_c = imap["g.c.coord"].first;
  int icoord_n = imap["g.n.coord"].first;

  auto lamb = KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                            CellLocation loc) {
    Real C[NDFULL];
    int icoord = (loc == CellLocation::Cent) ? icoord_c : icoord_n;
    system.Coords(loc, b, k, j, i, C);
    SPACETIMELOOP(mu) pack(b, icoord + mu, k, j, i) = C[mu];
  };
  IndexRange ib = rc->GetBoundsI(IndexDomain::entire);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = rc->GetBoundsK(IndexDomain::entire);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetGeometry::Set Cached data, Cent", DevExecSpace(), 0,
      pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        lamb(b, k, j, i, CellLocation::Cent);
      });
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetGeometry::Set Cached data, Corn", DevExecSpace(), 0,
      pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e + 1, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        lamb(b, k, j, i, CellLocation::Corn);
      });
}
} // namespace impl

} // namespace Geometry

#endif // GEOMETRY_GEOMETRY_UTILS_HPP_
