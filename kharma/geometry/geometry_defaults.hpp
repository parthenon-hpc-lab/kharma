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
// publicly, and to permit others to do so.

#ifndef GEOMETRY_GEOMETRY_DEFAULTS_HPP_
#define GEOMETRY_GEOMETRY_DEFAULTS_HPP_

#include <string>
#include <vector>

// Parthenon includes
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

// Phoebus includes
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/cell_locations.hpp"

namespace Geometry {

template <typename System>
void Initialize(ParameterInput *pin, StateDescriptor *geometry) {}

template <typename System>
System GetCoordinateSystem(MeshBlockData<Real> *rc);

template <typename System>
System GetCoordinateSystem(MeshData<Real> *rc);

template <typename System>
void SetGeometry(MeshBlockData<Real> *rc);

template <typename System>
void SetGeometry(MeshData<Real> *rc);

template <typename Transformation>
Transformation GetTransformation(StateDescriptor *pkg);

} // namespace Geometry

#endif // GEOMETRY_GEOMETRY_DEFAULTS_HPP_
