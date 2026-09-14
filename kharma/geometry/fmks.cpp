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
#include "geometry/mckinney_gammie_ryan.hpp"
#include "geometry/modified_system.hpp"
#include "geometry/spherical_kerr_schild.hpp"
#include "phoebus_utils/linear_algebra.hpp"

#include "geometry/fmks.hpp"

namespace Geometry {

template <>
void Initialize<FMKSMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  Initialize<SphericalKSMeshBlock>(pin, geometry);
  Params &params = geometry->AllParams();
  Real dxfd_geom = pin->GetOrAddReal("geometry", "finite_differences_dx", 1e-8);
  Real dxfd = pin->GetOrAddReal("coordinates", "finite_differences_dx", dxfd_geom);
  bool derefine_poles = pin->GetOrAddBoolean("coordinates", "derefine_poles", true);
  Real h = pin->GetOrAddReal("coordinates", "hslope", 0.3);
  Real xt = pin->GetOrAddReal("coordinates", "poly_xt", 0.82);
  Real alpha = pin->GetOrAddReal("coordinates", "poly_alpha", 14);
  Real x0 = pin->GetReal("parthenon/mesh", "x1min");
  Real smooth = pin->GetOrAddReal("coordinates", "smooth", 0.5);
  const Real hexp_br_large = 1.e30;
  PARTHENON_REQUIRE(hexp_br_large < std::numeric_limits<Real>::max(),
                    "Default hyperexponential break radius must be representable!");
  Real hexp_br = pin->GetOrAddReal("coordinates", "hexp_br", hexp_br_large);
  Real hexp_nsq = pin->GetOrAddReal("coordinates", "hexp_nsq", 4.0);
  Real hexp_csq = pin->GetOrAddReal("coordinates", "hexp_csq", 4.0);
  Real a = pin->GetReal("geometry", "a");
  Real Rh = 1.0 + sqrt(1.0 - a * a);
  Real xh = log(Rh);
  Real Z1 = 1.0 + std::cbrt(1.0 - a * a) * (std::cbrt(1.0 + a) + std::cbrt(1.0 - a));
  Real Z2 = sqrt(3.0 * a * a + Z1);
  Real r_isco_p = 3.0 + Z2 - sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)); // prograde
  Real r_isco_r = 3.0 + Z2 + sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)); // retrograde
  if (!params.hasKey("dxfd")) {
    params.Add("dxfd", dxfd);
  }
  params.Add("derefine_poles", derefine_poles);
  params.Add("h", h);
  params.Add("xt", xt);
  params.Add("alpha", alpha);
  // Require x0 to be <= x1 of the innermost ghost cell face to avoid issues with mapping
  // theta outside of [0, pi] This calculation assumes no mesh refinement, but works for
  // any refinement because ghost cell layers of refined blocks will always have smaller
  // extents
  const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
  const Real x1max = pin->GetReal("parthenon/mesh", "x1max");
  const int nx1 = pin->GetInteger("parthenon/mesh", "nx1");
  const Real dx1 = (x1max - x1min) / nx1;
  const int nghost = pin->GetInteger("parthenon/mesh", "nghost");
  x0 = std::min<Real>(x0, std::max<Real>(0., x1min - nghost * dx1));
  params.Add("x0", x0);
  params.Add("smooth", smooth);
  params.Add("hexp_br", hexp_br);
  params.Add("hexp_nsq", hexp_nsq);
  params.Add("hexp_csq", hexp_csq);
  params.Add("Rh", Rh);
  params.Add("xh", xh);
  params.Add("r_isco_p", r_isco_p);
  params.Add("r_isco_r", r_isco_r);
}
template <>
void SetGeometry<FMKSMeshBlock>(MeshBlockData<Real> *rc) {}
template <>
void SetGeometry<FMKSMesh>(MeshData<Real> *rc) {}

template <>
FMKSMeshBlock GetCoordinateSystem<FMKSMeshBlock>(MeshBlockData<Real> *rc) {
  auto &pkg = rc->GetMeshPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a = pkg->Param<Real>("a");
  Real dxfd = pkg->Param<Real>("dxfd");
  auto transformation = GetTransformation<McKinneyGammieRyan>(pkg.get());
  return FMKSMeshBlock(indexer, dxfd, transformation, a);
}
template <>
FMKSMesh GetCoordinateSystem<FMKSMesh>(MeshData<Real> *rc) {
  auto &pkg = rc->GetMeshPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a = pkg->Param<Real>("a");
  Real dxfd = pkg->Param<Real>("dxfd");
  auto transformation = GetTransformation<McKinneyGammieRyan>(pkg.get());
  return FMKSMesh(indexer, dxfd, transformation, a);
}

template <>
void Initialize<CFMKSMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  InitializeCachedCoordinateSystem<FMKSMeshBlock>(pin, geometry);
}
template <>
CFMKSMeshBlock GetCoordinateSystem<CFMKSMeshBlock>(MeshBlockData<Real> *rc) {
  return GetCachedCoordinateSystem<FMKSMeshBlock>(rc);
}
template <>
CFMKSMesh GetCoordinateSystem<CFMKSMesh>(MeshData<Real> *rc) {
  return GetCachedCoordinateSystem<FMKSMesh>(rc);
}
template <>
void SetGeometry<CFMKSMeshBlock>(MeshBlockData<Real> *rc) {
  SetCachedCoordinateSystem<FMKSMeshBlock>(rc);
}
template <>
void SetGeometry<CFMKSMesh>(MeshData<Real> *rc) {
  SetCachedCoordinateSystem<FMKSMesh>(rc);
}
} // namespace Geometry
