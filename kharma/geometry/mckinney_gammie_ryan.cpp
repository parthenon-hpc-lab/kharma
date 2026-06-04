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
#include "geometry/geometry_defaults.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/linear_algebra.hpp"

#include "geometry/mckinney_gammie_ryan.hpp"

namespace Geometry {

template <>
McKinneyGammieRyan GetTransformation<McKinneyGammieRyan>(StateDescriptor *pkg) {
  bool derefine_poles = pkg->Param<bool>("derefine_poles");
  Real h = pkg->Param<Real>("h");
  Real xt = pkg->Param<Real>("xt");
  Real alpha = pkg->Param<Real>("alpha");
  Real x0 = pkg->Param<Real>("x0");
  Real smooth = pkg->Param<Real>("smooth");
  Real hexp_br = pkg->Param<Real>("hexp_br");
  Real hexp_nsq = pkg->Param<Real>("hexp_nsq");
  Real hexp_csq = pkg->Param<Real>("hexp_csq");
  return McKinneyGammieRyan(derefine_poles, h, xt, alpha, x0, smooth, hexp_br, hexp_nsq,
                            hexp_csq);
}

} // namespace Geometry
