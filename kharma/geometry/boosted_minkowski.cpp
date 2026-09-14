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

// Phoebus includes
#include "geometry/analytic_system.hpp"
#include "geometry/cached_system.hpp"
#include "geometry/geometry_defaults.hpp"
#include "geometry/geometry_utils.hpp"

#include "geometry/boosted_minkowski.hpp"

namespace Geometry {

template <>
void Initialize<BoostedMinkowskiMeshBlock>(ParameterInput *pin,
                                           StateDescriptor *geometry) {
  Params &params = geometry->AllParams();
  Real vx = pin->GetOrAddReal("coordinates", "vx", 0);
  Real vy = pin->GetOrAddReal("coordinates", "vy", 0);
  Real vz = pin->GetOrAddReal("coordinates", "vz", 0);
  PARTHENON_REQUIRE_THROWS(-1 < vx && vx < 1, "vx must be subluminal");
  PARTHENON_REQUIRE_THROWS(-1 < vy && vy < 1, "vy must be subluminal");
  PARTHENON_REQUIRE_THROWS(-1 < vz && vz < 1, "vz must be subluminal");
  params.Add("vx", vx);
  params.Add("vy", vy);
  params.Add("vz", vz);
}
template <>
void Initialize<CBoostedMinkowskiMeshBlock>(ParameterInput *pin,
                                            StateDescriptor *geometry) {
  InitializeCachedCoordinateSystem<BoostedMinkowskiMeshBlock>(pin, geometry);
}

template <>
BoostedMinkowskiMeshBlock
GetCoordinateSystem<BoostedMinkowskiMeshBlock>(MeshBlockData<Real> *rc) {
  auto &pkg = rc->GetMeshPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real vx = pkg->Param<Real>("vx");
  Real vy = pkg->Param<Real>("vy");
  Real vz = pkg->Param<Real>("vz");
  return BoostedMinkowskiMeshBlock(indexer, vx, vy, vz);
}

template <>
BoostedMinkowskiMesh GetCoordinateSystem<BoostedMinkowskiMesh>(MeshData<Real> *rc) {
  auto &pkg = rc->GetMeshPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real vx = pkg->Param<Real>("vx");
  Real vy = pkg->Param<Real>("vy");
  Real vz = pkg->Param<Real>("vz");
  return BoostedMinkowskiMesh(indexer, vx, vy, vz);
}

template <>
CBoostedMinkowskiMeshBlock
GetCoordinateSystem<CBoostedMinkowskiMeshBlock>(MeshBlockData<Real> *rc) {
  return GetCachedCoordinateSystem<BoostedMinkowskiMeshBlock>(rc);
}

template <>
CBoostedMinkowskiMesh GetCoordinateSystem<CBoostedMinkowskiMesh>(MeshData<Real> *rc) {
  return GetCachedCoordinateSystem<BoostedMinkowskiMesh>(rc);
}

template <>
void SetGeometry<BoostedMinkowskiMeshBlock>(MeshBlockData<Real> *rc) {}

template <>
void SetGeometry<BoostedMinkowskiMesh>(MeshData<Real> *rc) {}

template <>
void SetGeometry<CBoostedMinkowskiMeshBlock>(MeshBlockData<Real> *rc) {
  SetCachedCoordinateSystem<BoostedMinkowskiMeshBlock>(rc);
}

template <>
void SetGeometry<CBoostedMinkowskiMesh>(MeshData<Real> *rc) {
  SetCachedCoordinateSystem<BoostedMinkowskiMesh>(rc);
}

} // namespace Geometry
