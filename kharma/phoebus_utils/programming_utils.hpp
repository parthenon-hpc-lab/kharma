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

#ifndef PHOEBUS_UTILS_PROGRAMMING_UTILS_HPP_
#define PHOEBUS_UTILS_PROGRAMMING_UTILS_HPP_

#include <algorithm>
#include <vector>

namespace programming {

template <class SPECIAL, template <class...> class TEMPL>
struct is_specialization_of : public std::false_type {};

template <template <class...> class TEMPL, class... TPARAMS>
struct is_specialization_of<TEMPL<TPARAMS...>, TEMPL> : public std::true_type {};

template <typename T>
bool soft_equiv(const T x, const T y, const T tolerance = static_cast<T>(1.e-10f)) {
  return (2. * fabs(y - x) < tolerance * (std::fabs(x) + std::fabs(y)));
}

} // namespace programming

#endif // PHOEBUS_UTILS_PROGRAMMING_UTILS_HPP_
