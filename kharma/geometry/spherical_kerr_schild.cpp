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
#include "phoebus_utils/linear_algebra.hpp"

#include "geometry/spherical_kerr_schild.hpp"

namespace Geometry {

template <>
void Initialize<SphericalKSMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  Params &params = geometry->AllParams();
  Real a = pin->GetOrAddReal("geometry", "a", 0);
  params.Add("a", a);
}

template <>
void SetGeometry<SphericalKSMeshBlock>(MeshBlockData<Real> *rc) {}
template <>
void SetGeometry<SphericalKSMesh>(MeshData<Real> *rc) {}

template <>
SphericalKSMeshBlock GetCoordinateSystem<SphericalKSMeshBlock>(MeshBlockData<Real> *rc) {
  auto &pkg = rc->GetMeshPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a = pkg->Param<Real>("a");
  return SphericalKSMeshBlock(indexer, a);
}
template <>
SphericalKSMesh GetCoordinateSystem<SphericalKSMesh>(MeshData<Real> *rc) {
  auto &pkg = rc->GetMeshPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a = pkg->Param<Real>("a");
  return SphericalKSMesh(indexer, a);
}

template <>
void Initialize<CSphericalKSMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  InitializeCachedCoordinateSystem<SphericalKSMeshBlock>(pin, geometry);
}
template <>
CSphericalKSMeshBlock
GetCoordinateSystem<CSphericalKSMeshBlock>(MeshBlockData<Real> *rc) {
  return GetCachedCoordinateSystem<SphericalKSMeshBlock>(rc);
}
template <>
CSphericalKSMesh GetCoordinateSystem<CSphericalKSMesh>(MeshData<Real> *rc) {
  return GetCachedCoordinateSystem<SphericalKSMesh>(rc);
}
template <>
void SetGeometry<CSphericalKSMeshBlock>(MeshBlockData<Real> *rc) {
  SetCachedCoordinateSystem<SphericalKSMeshBlock>(rc);
}
template <>
void SetGeometry<CSphericalKSMesh>(MeshData<Real> *rc) {
  SetCachedCoordinateSystem<SphericalKSMesh>(rc);
}

} // namespace Geometry
