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

#include "geometry/minkowski.hpp"

namespace Geometry {

template <>
void SetGeometry<MinkowskiMeshBlock>(MeshBlockData<Real> *rc) {}
template <>
void SetGeometry<MinkowskiMesh>(MeshData<Real> *rc) {}

template <>
MinkowskiMeshBlock GetCoordinateSystem<MinkowskiMeshBlock>(MeshBlockData<Real> *rc) {
  auto indexer = GetIndexer(rc);
  return MinkowskiMeshBlock(indexer);
}
template <>
MinkowskiMesh GetCoordinateSystem<MinkowskiMesh>(MeshData<Real> *rc) {
  auto indexer = GetIndexer(rc);
  return MinkowskiMesh(indexer);
}

template <>
void Initialize<CMinkowskiMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  InitializeCachedCoordinateSystem<MinkowskiMeshBlock>(pin, geometry);
}
template <>
CMinkowskiMeshBlock GetCoordinateSystem<CMinkowskiMeshBlock>(MeshBlockData<Real> *rc) {
  return GetCachedCoordinateSystem<MinkowskiMeshBlock>(rc);
}
template <>
CMinkowskiMesh GetCoordinateSystem<CMinkowskiMesh>(MeshData<Real> *rc) {
  return GetCachedCoordinateSystem<MinkowskiMesh>(rc);
}
template <>
void SetGeometry<CMinkowskiMeshBlock>(MeshBlockData<Real> *rc) {
  SetCachedCoordinateSystem<MinkowskiMeshBlock>(rc);
}
template <>
void SetGeometry<CMinkowskiMesh>(MeshData<Real> *rc) {
  SetCachedCoordinateSystem<MinkowskiMesh>(rc);
}
} // namespace Geometry
