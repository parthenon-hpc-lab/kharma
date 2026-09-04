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

#ifndef PHOEBUS_UTILS_VARIABLES_HPP_
#define PHOEBUS_UTILS_VARIABLES_HPP_

//#include "compile_constants.hpp"

#include <pack/sparse_pack/sparse_pack.hpp>
#include <parthenon/package.hpp>

#define VARIABLE(ns, varname)                                                            \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #ns "." #varname; }                               \
  }

#define VARIABLE_NONS(varname)                                                           \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #varname; }                                       \
  }

#define VARIABLE_CUSTOM(varname, varstring)                                              \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #varstring; }                                     \
  }

#define TENSOR_SWARM(type, ns, varname, ...)                                             \
  struct varname : public parthenon::swarm_variable_names::base_t<type, __VA_ARGS__> {   \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::swarm_variable_names::base_t<type, __VA_ARGS__>(                    \
              std::forward<Ts>(args)...) {}                                              \
    static std::string name() { return #ns "." #varname; }                               \
  }

#define TENSOR_VARIABLE(ns, varname, ...)                                                \
  struct varname : public parthenon::variable_names::base_t<false, __VA_ARGS__> {        \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false, __VA_ARGS__>(                         \
              std::forward<Ts>(args)...) {}                                              \
    static std::string name() { return #ns "." #varname; }                               \
  }

using parthenon::variable_names::ANYDIM;

namespace geometric_variables {
VARIABLE_CUSTOM(cell_coords, g.c.coord);
VARIABLE_CUSTOM(node_coords, g.n.coord);
} // namespace geometric_variables

namespace phoebus {
template <typename Data, typename... Ts>
auto MakePackDescriptor(Data *rc) {
  parthenon::Mesh *pm = rc->GetMeshPointer();
  parthenon::StateDescriptor *resolved_pkgs = pm->resolved_packages.get();
  return parthenon::MakePackDescriptor<Ts...>(resolved_pkgs);
}
} // namespace phoebus

#endif // PHOEBUS_UTILS_VARIABLES_HPP_
