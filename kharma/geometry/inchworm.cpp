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

#include "geometry/inchworm.hpp"

namespace Geometry {

template <>
void Initialize<InchwormMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  Params &params = geometry->AllParams();
  Real a = pin->GetOrAddReal("geometry", "a", 0.3);
  Real k = pin->GetOrAddReal("geometry", "k", M_PI / 2.);
  params.Add("a", a);
  params.Add("k", k);
}

template <>
void SetGeometry<InchwormMeshBlock>(MeshBlockData<Real> *rc) {}

template <>
void SetGeometry<InchwormMesh>(MeshData<Real> *rc) {}

template <>
InchwormMeshBlock GetCoordinateSystem<InchwormMeshBlock>(MeshBlockData<Real> *rc) {
  auto &pkg = rc->GetMeshPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a = pkg->Param<Real>("a");
  Real k = pkg->Param<Real>("k");
  return InchwormMeshBlock(indexer, a, k);
}
template <>
InchwormMesh GetCoordinateSystem<InchwormMesh>(MeshData<Real> *rc) {
  auto &pkg = rc->GetMeshPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a = pkg->Param<Real>("a");
  Real k = pkg->Param<Real>("k");
  return InchwormMesh(indexer, a, k);
}

template <>
void Initialize<CInchwormMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  InitializeCachedCoordinateSystem<InchwormMeshBlock>(pin, geometry);
}
template <>
CInchwormMeshBlock GetCoordinateSystem<CInchwormMeshBlock>(MeshBlockData<Real> *rc) {
  return GetCachedCoordinateSystem<InchwormMeshBlock>(rc);
}
template <>
CInchwormMesh GetCoordinateSystem<CInchwormMesh>(MeshData<Real> *rc) {
  return GetCachedCoordinateSystem<InchwormMesh>(rc);
}
template <>
void SetGeometry<CInchwormMeshBlock>(MeshBlockData<Real> *rc) {
  SetCachedCoordinateSystem<InchwormMeshBlock>(rc);
}
template <>
void SetGeometry<CInchwormMesh>(MeshData<Real> *rc) {
  SetCachedCoordinateSystem<InchwormMesh>(rc);
}
} // namespace Geometry
