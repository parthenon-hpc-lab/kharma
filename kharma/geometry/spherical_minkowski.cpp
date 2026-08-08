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

#include "geometry/spherical_minkowski.hpp"

namespace Geometry {

template <>
SphMinkowskiMeshBlock
GetCoordinateSystem<SphMinkowskiMeshBlock>(MeshBlockData<Real> *rc) {
  auto indexer = GetIndexer(rc);
  return SphMinkowskiMeshBlock(indexer);
}

template <>
void SetGeometry<SphMinkowskiMeshBlock>(MeshBlockData<Real> *rc) {}

template <>
SphMinkowskiMesh GetCoordinateSystem<SphMinkowskiMesh>(MeshData<Real> *rc) {
  auto indexer = GetIndexer(rc);
  return SphMinkowskiMesh(indexer);
}

template <>
void SetGeometry<SphMinkowskiMesh>(MeshData<Real> *rc) {}

template <>
void Initialize<CSphMinkowskiMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  InitializeCachedCoordinateSystem<SphMinkowskiMeshBlock>(pin, geometry);
}
template <>
CSphMinkowskiMeshBlock
GetCoordinateSystem<CSphMinkowskiMeshBlock>(MeshBlockData<Real> *rc) {
  return GetCachedCoordinateSystem<SphMinkowskiMeshBlock>(rc);
}
template <>
CSphMinkowskiMesh GetCoordinateSystem<CSphMinkowskiMesh>(MeshData<Real> *rc) {
  return GetCachedCoordinateSystem<SphMinkowskiMesh>(rc);
}
template <>
void SetGeometry<CSphMinkowskiMeshBlock>(MeshBlockData<Real> *rc) {
  SetCachedCoordinateSystem<SphMinkowskiMeshBlock>(rc);
}
template <>
void SetGeometry<CSphMinkowskiMesh>(MeshData<Real> *rc) {
  SetCachedCoordinateSystem<SphMinkowskiMesh>(rc);
}

} // namespace Geometry
