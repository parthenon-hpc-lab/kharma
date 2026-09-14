// © 2022. Triad National Security, LLC. All rights reserved.  This
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
// publicly, and to permit others to do so

// Parthenon includes
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

// Phoebus Includes
#include "geometry/analytic_system.hpp"
#include "geometry/cached_system.hpp"
#include "geometry/geometry_defaults.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/linear_algebra.hpp"
#include "phoebus_utils/robust.hpp"

// FLRW
#include "geometry/flrw.hpp"

namespace Geometry {

template <>
void Initialize<FLRWMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  Params &params = geometry->AllParams();

  Real a0 = pin->GetOrAddReal("coordinates", "a0", 1);
  PARTHENON_REQUIRE_THROWS(a0 > 0, "Scale factor must not be singular");
  params.Add("a0", a0);

  Real dadt = pin->GetOrAddReal("coordinates", "dadt", 1);
  params.Add("dadt", dadt);

  Real t = 0;
  params.Add("time", t, true); // Will need updates from simulation

  // Registers with the caching machinery that the metric evolves in time
  bool time_dependent = true;
  params.Add("time_dependent", time_dependent);
}

template <>
void Initialize<CFLRWMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  InitializeCachedCoordinateSystem<FLRWMeshBlock>(pin, geometry);
}

template <>
FLRWMeshBlock GetCoordinateSystem<FLRWMeshBlock>(MeshBlockData<Real> *rc) {
  auto &pkg = rc->GetMeshPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a0 = pkg->Param<Real>("a0");
  Real dadt = pkg->Param<Real>("dadt");
  Real t = pkg->Param<Real>("time");
  return FLRWMeshBlock(t, indexer, a0, dadt);
}

template <>
FLRWMesh GetCoordinateSystem<FLRWMesh>(MeshData<Real> *rc) {
  auto &pkg = rc->GetMeshPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a0 = pkg->Param<Real>("a0");
  Real dadt = pkg->Param<Real>("dadt");
  Real t = pkg->Param<Real>("time");
  return FLRWMesh(t, indexer, a0, dadt);
}

template <>
CFLRWMeshBlock GetCoordinateSystem<CFLRWMeshBlock>(MeshBlockData<Real> *rc) {
  return GetCachedCoordinateSystem<FLRWMeshBlock>(rc);
}

template <>
CFLRWMesh GetCoordinateSystem<CFLRWMesh>(MeshData<Real> *rc) {
  return GetCachedCoordinateSystem<FLRWMesh>(rc);
}

template <>
void SetGeometry<FLRWMeshBlock>(MeshBlockData<Real> *rc) {}

template <>
void SetGeometry<FLRWMesh>(MeshData<Real> *rc) {}

template <>
void SetGeometry<CFLRWMeshBlock>(MeshBlockData<Real> *rc) {
  SetCachedCoordinateSystem<FLRWMeshBlock>(rc);
}

template <>
void SetGeometry<CFLRWMesh>(MeshData<Real> *rc) {
  SetCachedCoordinateSystem<FLRWMesh>(rc);
}

} // namespace Geometry
