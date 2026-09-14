#ifndef PHOEBUS_UTILS_ROBUST_HPP_
#define PHOEBUS_UTILS_ROBUST_HPP_

#include <kokkos_abstraction.hpp>
#include <limits>

namespace robust {

template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto LARGE() {
  return 0.1 * std::numeric_limits<T>::max();
}
template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto SMALL() {
  return 10 * std::numeric_limits<T>::min();
}
template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto EPS() {
  return 10 * std::numeric_limits<T>::epsilon();
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION auto make_positive(const T val) {
  return std::max(val, EPS<T>());
}

KOKKOS_FORCEINLINE_FUNCTION
Real make_bounded(const Real val, const Real vmin, const Real vmax) {
  return std::min(std::max(val, vmin + EPS()), vmax * (1.0 - EPS()));
}

template <typename T>
KOKKOS_INLINE_FUNCTION int sgn(const T &val) {
  return (T(0) <= val) - (val < T(0));
}
template <typename A, typename B>
KOKKOS_INLINE_FUNCTION auto ratio(const A &a, const B &b) {
  const B sgn = b >= 0 ? 1 : -1;
  return a / (b + sgn * SMALL<B>());
}
} // namespace robust

#endif // PHOEBUS_UTILS_ROBUST_HPP_
