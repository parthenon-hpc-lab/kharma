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

#include "geometry/snake.hpp"

namespace Geometry {

template <>
void Initialize<SnakeMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  Params &params = geometry->AllParams();
  Real a = pin->GetOrAddReal("geometry", "a", 0.3);
  Real alpha = pin->GetOrAddReal("geometry", "alpha", 1.0);
  Real vy = pin->GetOrAddReal("geometry", "vy", 0.0);
  Real k = pin->GetOrAddReal("geometry", "k", 2 * M_PI);
  Real kmult = pin->GetOrAddReal("geometry", "kmult", 1);
  k *= kmult;
  params.Add("a", a);
  params.Add("k", k);
  params.Add("alpha", alpha);
  params.Add("vy", vy);
}

template <>
void SetGeometry<SnakeMeshBlock>(MeshBlockData<Real> *rc) {}

template <>
void SetGeometry<SnakeMesh>(MeshData<Real> *rc) {}

template <>
SnakeMeshBlock GetCoordinateSystem<SnakeMeshBlock>(MeshBlockData<Real> *rc) {
  auto &pkg = rc->GetMeshPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a = pkg->Param<Real>("a");
  Real k = pkg->Param<Real>("k");
  Real alpha = pkg->Param<Real>("alpha");
  Real vy = pkg->Param<Real>("vy");
  return SnakeMeshBlock(indexer, a, k, alpha, vy);
}
template <>
SnakeMesh GetCoordinateSystem<SnakeMesh>(MeshData<Real> *rc) {
  auto &pkg = rc->GetMeshPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a = pkg->Param<Real>("a");
  Real k = pkg->Param<Real>("k");
  Real alpha = pkg->Param<Real>("alpha");
  Real vy = pkg->Param<Real>("vy");
  return SnakeMesh(indexer, a, k, alpha, vy);
}

template <>
void Initialize<CSnakeMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  InitializeCachedCoordinateSystem<SnakeMeshBlock>(pin, geometry);
}
template <>
CSnakeMeshBlock GetCoordinateSystem<CSnakeMeshBlock>(MeshBlockData<Real> *rc) {
  return GetCachedCoordinateSystem<SnakeMeshBlock>(rc);
}
template <>
CSnakeMesh GetCoordinateSystem<CSnakeMesh>(MeshData<Real> *rc) {
  return GetCachedCoordinateSystem<SnakeMesh>(rc);
}
template <>
void SetGeometry<CSnakeMeshBlock>(MeshBlockData<Real> *rc) {
  SetCachedCoordinateSystem<SnakeMeshBlock>(rc);
}
template <>
void SetGeometry<CSnakeMesh>(MeshData<Real> *rc) {
  SetCachedCoordinateSystem<SnakeMesh>(rc);
}

} // namespace Geometry
