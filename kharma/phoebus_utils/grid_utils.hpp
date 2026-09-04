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

#ifndef PHOEBUS_UTILS_GRID_UTILS_HPP_
#define PHOEBUS_UTILS_GRID_UTILS_HPP_

// Parthenon includes
#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

namespace Coordinates {

using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;

template <int dir>
KOKKOS_FORCEINLINE_FUNCTION Real GetXv(const int i, const Coordinates_t &coord);
template <>
KOKKOS_FORCEINLINE_FUNCTION Real GetXv<X1DIR>(const int i, const Coordinates_t &coord) {
  return coord.Xc<1>(i);
}
template <>
KOKKOS_FORCEINLINE_FUNCTION Real GetXv<X2DIR>(const int i, const Coordinates_t &coord) {
  return coord.Xc<2>(i);
}
template <>
KOKKOS_FORCEINLINE_FUNCTION Real GetXv<X3DIR>(const int i, const Coordinates_t &coord) {
  return coord.Xc<3>(i);
}
KOKKOS_FORCEINLINE_FUNCTION Real GetXv(const int i, const int dir,
                                       const Coordinates_t &coord) {
  switch (dir) {
  case X1DIR:
    return GetXv<X1DIR>(i, coord);
  case X2DIR:
    return GetXv<X2DIR>(i, coord);
  case X3DIR:
    return GetXv<X3DIR>(i, coord);
  default:
    PARTHENON_FAIL("Invalid coordinate direction");
  }
}

} // namespace Coordinates

namespace PhoebusUtils {
inline Real GetRegionVolume(const parthenon::RegionSize &reg) {
  using parthenon::CoordinateDirection;
  Real vol = 1;
  for (int d = X1DIR; d <= X3DIR; ++d) {
    CoordinateDirection dir = static_cast<CoordinateDirection>(d);
    vol *= (reg.xmax(dir) - reg.xmin(dir));
  }
  return vol;
}
} // namespace PhoebusUtils

#endif // PHOEBUS_UTILS_GRID_UTILS_HPP_
