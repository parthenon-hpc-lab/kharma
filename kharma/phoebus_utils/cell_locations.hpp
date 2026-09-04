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

#ifndef PHOEBUS_UTILS_CELL_LOCATIONS_HPP_
#define PHOEBUS_UTILS_CELL_LOCATIONS_HPP_

#include <defs.hpp>
#include <kokkos_abstraction.hpp>

constexpr int NUM_CELL_LOCATIONS = 5;
enum class CellLocation { Cent, Face1, Face2, Face3, Corn };

KOKKOS_FORCEINLINE_FUNCTION
CellLocation DirectionToFaceID(int dir) {
  switch (dir) {
  case parthenon::X1DIR:
    return CellLocation::Face1;
  case parthenon::X2DIR:
    return CellLocation::Face2;
  case parthenon::X3DIR:
    return CellLocation::Face3;
  }
  // TODO(JCD): insert the appropriate parthenon fail thing that's safe on device
  return CellLocation::Cent; // return the right type for the compiler
}

#endif // PHOEBUS_UTILS_CELL_LOCATIONS_HPP_
