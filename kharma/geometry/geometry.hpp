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

#ifndef GEOMETRY_GEOMETRY_HPP_
#define GEOMETRY_GEOMETRY_HPP_

#include <memory>

#include <parthenon/package.hpp>

#include "geometry/coordinate_systems.hpp"
#include "geometry/tetrads.hpp"

using namespace parthenon::package::prelude;

namespace Geometry {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

// Set geometry data on grid, if needed.
// Potentially a very expensive operation. You only want to do this
// once, but it is deally done per meshblock, right at the beginning of
// a problem generator.
void SetGeometryBlock(MeshBlock *pmb, ParameterInput *pin);

// Same as SetGeometryBlock, but a task for the task list.
// Supports MeshBlockData or MeshData.
// Template specializations are in geometry.cpp
template <typename Data>
TaskStatus UpdateGeometry(Data *rc);

CoordSysMeshBlock GetCoordinateSystem(MeshBlockData<Real> *rc);
CoordSysMesh GetCoordinateSystem(MeshData<Real> *rc);

} // namespace Geometry

#endif // GEOMETRY_GEOMETRY_HPP_
