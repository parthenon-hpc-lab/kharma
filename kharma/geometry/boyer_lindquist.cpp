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

// phoebus includes
#include "geometry/analytic_system.hpp"
#include "geometry/cached_system.hpp"
#include "geometry/geometry_defaults.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/linear_algebra.hpp"

#include "geometry/boyer_lindquist.hpp"

namespace Geometry {

// Don't need to overwrite SetGeometry
template <>
void Initialize<BoyerLindquistMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  Params &params = geometry->AllParams();
  Real a = pin->GetOrAddReal("geometry", "a", 0);
  Real dxfd = pin->GetOrAddReal("geometry", "finite_differences_dx", 1e-8);
  params.Add("a", a);
  params.Add("dxfd", dxfd);
}

template <>
void SetGeometry<BoyerLindquistMeshBlock>(MeshBlockData<Real> *rc) {}

template <>
void SetGeometry<BoyerLindquistMesh>(MeshData<Real> *rc) {}

template <>
BoyerLindquistMeshBlock
GetCoordinateSystem<BoyerLindquistMeshBlock>(MeshBlockData<Real> *rc) {
  auto &pkg = rc->GetMeshPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a = pkg->Param<Real>("a");
  Real dxfd = pkg->Param<Real>("dxfd");
  return BoyerLindquistMeshBlock(indexer, a, dxfd);
}
template <>
BoyerLindquistMesh GetCoordinateSystem<BoyerLindquistMesh>(MeshData<Real> *rc) {
  auto &pkg = rc->GetMeshPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a = pkg->Param<Real>("a");
  Real dxfd = pkg->Param<Real>("dxfd");
  return BoyerLindquistMesh(indexer, a, dxfd);
}

template <>
void Initialize<CBoyerLindquistMeshBlock>(ParameterInput *pin,
                                          StateDescriptor *geometry) {
  InitializeCachedCoordinateSystem<BoyerLindquistMeshBlock>(pin, geometry);
}
template <>
CBoyerLindquistMeshBlock
GetCoordinateSystem<CBoyerLindquistMeshBlock>(MeshBlockData<Real> *rc) {
  return GetCachedCoordinateSystem<BoyerLindquistMeshBlock>(rc);
}
template <>
CBoyerLindquistMesh GetCoordinateSystem<CBoyerLindquistMesh>(MeshData<Real> *rc) {
  return GetCachedCoordinateSystem<BoyerLindquistMesh>(rc);
}
template <>
void SetGeometry<CBoyerLindquistMeshBlock>(MeshBlockData<Real> *rc) {
  SetCachedCoordinateSystem<BoyerLindquistMeshBlock>(rc);
}
template <>
void SetGeometry<CBoyerLindquistMesh>(MeshData<Real> *rc) {
  SetCachedCoordinateSystem<BoyerLindquistMesh>(rc);
}
} // namespace Geometry
