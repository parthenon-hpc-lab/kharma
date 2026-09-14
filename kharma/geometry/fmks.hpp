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

#ifndef GEOMETRY_FMKS_HPP_
#define GEOMETRY_FMKS_HPP_

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
#include "geometry/mckinney_gammie_ryan.hpp"
#include "geometry/modified_system.hpp"
#include "geometry/spherical_kerr_schild.hpp"
#include "phoebus_utils/linear_algebra.hpp"

namespace Geometry {

// Further Modified Kerr-Schild
// first presented in McKinney and Gammie, ApJ 611:977-995, 2004,
// where it was called MKS or Modified Kerr-Schild.
// Later modified by Ben Ryan for the bhlight code.
// https://github.com/AFD-Illinois/ebhlight
// https://github.com/lanl/nubhlight
using FMKS = Modified<SphericalKerrSchild, McKinneyGammieRyan>;
using FMKSMesh = Analytic<FMKS, IndexerMesh>;
using FMKSMeshBlock = Analytic<FMKS, IndexerMeshBlock>;
using CFMKSMesh = CachedOverMesh<Analytic<FMKS, IndexerMesh>>;
using CFMKSMeshBlock = CachedOverMeshBlock<Analytic<FMKS, IndexerMeshBlock>>;

template <>
void Initialize<FMKSMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);
template <>
void Initialize<CFMKSMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);

} // namespace Geometry

#endif // GEOMETRY_FMKS_HPP_
