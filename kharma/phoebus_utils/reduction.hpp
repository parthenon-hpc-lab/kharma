// © 2022-2023. Triad National Security, LLC. All rights reserved.  This
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

#ifndef PHOEBUS_UTILS_REDUCTION_HPP_
#define PHOEBUS_UTILS_REDUCTION_HPP_

namespace reduction {

#ifdef MPI_PARALLEL

#include <mpi.h>

Real inline Max(const Real &x) {
  Real xmax;
  MPI_Allreduce(&x, &xmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return xmax;
}

Real inline Min(const Real &x) {
  Real xmin;
  MPI_Allreduce(&x, &xmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  return xmin;
}

Real inline Sum(const Real &x) {
  Real xsum;
  MPI_Allreduce(&x, &xsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return xsum;
}

#else

Real inline Max(const Real &x) { return x; }
Real inline Min(const Real &x) { return x; }
Real inline Sum(const Real &x) { return x; }

#endif // MPI_PARALLEL

}; // namespace reduction

#endif // PHOEBUS_UTILS_REDUCTION_HPP_
