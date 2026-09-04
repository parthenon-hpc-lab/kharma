// © 2021-2022. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
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

#ifndef PHOEBUS_UTILS_LINEAR_ALGEBRA_HPP_
#define PHOEBUS_UTILS_LINEAR_ALGEBRA_HPP_

#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

namespace LinearAlgebra {

// TODO(JMM): Replace these with a memset?
template <typename Array>
KOKKOS_INLINE_FUNCTION void SetZero(Array &a, int size) {
  for (int i = 0; i < size; ++i)
    a[i] = 0;
}
template <typename Array>
KOKKOS_INLINE_FUNCTION void SetZero(Array &a, int n2, int n1) {
  for (int i2 = 0; i2 < n2; ++i2) {
    for (int i1 = 0; i1 < n1; ++i1) {
      a[i2][i1] = 0;
    }
  }
}
template <typename Array>
KOKKOS_INLINE_FUNCTION void SetZero(Array &a, int n3, int n2, int n1) {
  for (int i3 = 0; i3 < n3; ++i3) {
    for (int i2 = 0; i2 < n2; ++i2) {
      for (int i1 = 0; i1 < n1; ++i1) {
        a[i3][i2][i1] = 0;
      }
    }
  }
}

// Determinant of a 3x3 matrix
KOKKOS_INLINE_FUNCTION
Real Determinant(const Real A[3][3]) {
  return (A[0][0] * A[1][1] * A[2][2] + A[0][1] * A[1][2] * A[2][0] +
          A[0][2] * A[1][0] * A[2][1] - A[0][2] * A[1][1] * A[2][0] -
          A[0][1] * A[1][0] * A[2][2] - A[0][0] * A[1][2] * A[2][1]);
}

// Taken from https://pharr.org/matt/blog/2019/11/03/difference-of-floats
// meant to reduce floating point rounding error in cancellations,
// returns ab - cd
template <class T>
KOKKOS_INLINE_FUNCTION T DifferenceOfProducts(T a, T b, T c, T d) {
  T cd = c * d;
  T err = std::fma(-c, d, cd); // Round off error correction for cd
  T dop = std::fma(a, b, -cd);
  return dop + err;
}

template <class T>
KOKKOS_INLINE_FUNCTION auto matrixInverse3x3(T A, T &Ainv) ->
    typename std::remove_reference<decltype(A(0, 0))>::type {
  Ainv(0, 0) = DifferenceOfProducts(A(1, 1), A(2, 2), A(1, 2), A(2, 1));
  Ainv(1, 0) = -DifferenceOfProducts(A(1, 0), A(2, 2), A(1, 2), A(2, 0));
  Ainv(2, 0) = DifferenceOfProducts(A(1, 0), A(2, 1), A(1, 1), A(2, 0));
  Ainv(0, 1) = -DifferenceOfProducts(A(0, 1), A(2, 2), A(0, 2), A(2, 1));
  Ainv(1, 1) = DifferenceOfProducts(A(0, 0), A(2, 2), A(0, 2), A(2, 0));
  Ainv(2, 1) = -DifferenceOfProducts(A(0, 0), A(2, 1), A(0, 1), A(2, 0));
  Ainv(0, 2) = DifferenceOfProducts(A(0, 1), A(1, 2), A(0, 2), A(1, 1));
  Ainv(1, 2) = -DifferenceOfProducts(A(0, 0), A(1, 2), A(0, 2), A(1, 0));
  Ainv(2, 2) = DifferenceOfProducts(A(0, 0), A(1, 1), A(0, 1), A(1, 0));
  const auto det =
      std::fma(A(0, 0), Ainv(0, 0),
               DifferenceOfProducts(A(2, 0), Ainv(2, 0), A(1, 0), -Ainv(1, 0)));
  const auto invDet = 1.0 / det;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Ainv(i, j) *= invDet;
    }
  }
  return det;
}

template <class T>
KOKKOS_INLINE_FUNCTION void SolveAxb2x2(T A[2][2], T b[2], T x[2]) {
  const auto invDet = 1.0 / DifferenceOfProducts(A[0][0], A[1][1], A[0][1], A[1][0]);
  x[0] = invDet * DifferenceOfProducts(A[1][1], b[0], A[0][1], b[1]);
  x[1] = invDet * DifferenceOfProducts(A[0][0], b[1], A[1][0], b[0]);
}

KOKKOS_INLINE_FUNCTION
Real MINOR(Real m[16], int r0, int r1, int r2, int c0, int c1, int c2) {
  return m[4 * r0 + c0] *
             (m[4 * r1 + c1] * m[4 * r2 + c2] - m[4 * r2 + c1] * m[4 * r1 + c2]) -
         m[4 * r0 + c1] *
             (m[4 * r1 + c0] * m[4 * r2 + c2] - m[4 * r2 + c0] * m[4 * r1 + c2]) +
         m[4 * r0 + c2] *
             (m[4 * r1 + c0] * m[4 * r2 + c1] - m[4 * r2 + c0] * m[4 * r1 + c1]);
}

KOKKOS_INLINE_FUNCTION
void adjoint(Real m[16], Real adjOut[16]) {
  adjOut[0] = MINOR(m, 1, 2, 3, 1, 2, 3);
  adjOut[1] = -MINOR(m, 0, 2, 3, 1, 2, 3);
  adjOut[2] = MINOR(m, 0, 1, 3, 1, 2, 3);
  adjOut[3] = -MINOR(m, 0, 1, 2, 1, 2, 3);

  adjOut[4] = -MINOR(m, 1, 2, 3, 0, 2, 3);
  adjOut[5] = MINOR(m, 0, 2, 3, 0, 2, 3);
  adjOut[6] = -MINOR(m, 0, 1, 3, 0, 2, 3);
  adjOut[7] = MINOR(m, 0, 1, 2, 0, 2, 3);

  adjOut[8] = MINOR(m, 1, 2, 3, 0, 1, 3);
  adjOut[9] = -MINOR(m, 0, 2, 3, 0, 1, 3);
  adjOut[10] = MINOR(m, 0, 1, 3, 0, 1, 3);
  adjOut[11] = -MINOR(m, 0, 1, 2, 0, 1, 3);

  adjOut[12] = -MINOR(m, 1, 2, 3, 0, 1, 2);
  adjOut[13] = MINOR(m, 0, 2, 3, 0, 1, 2);
  adjOut[14] = -MINOR(m, 0, 1, 3, 0, 1, 2);
  adjOut[15] = MINOR(m, 0, 1, 2, 0, 1, 2);
}

KOKKOS_INLINE_FUNCTION
Real determinant(Real m[16]) {
  return m[0] * MINOR(m, 1, 2, 3, 1, 2, 3) - m[1] * MINOR(m, 1, 2, 3, 0, 2, 3) +
         m[2] * MINOR(m, 1, 2, 3, 0, 1, 3) - m[3] * MINOR(m, 1, 2, 3, 0, 1, 2);
}

KOKKOS_INLINE_FUNCTION
void matrixInverse4x4(Real mat[4][4], Real matinv[4][4]) {
  adjoint(&(mat[0][0]), &(matinv[0][0]));

  Real det = determinant(&(mat[0][0]));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      matinv[i][j] /= det;
    }
  }
}

} // namespace LinearAlgebra

#endif // PHOEBUS_UTILS_LINEAR_ALGEBRA_HPP_
