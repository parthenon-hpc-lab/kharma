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

#ifndef GEOMETRY_SPH2CART_HPP_
#define GEOMETRY_SPH2CART_HPP_

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
#include "phoebus_utils/robust.hpp"

namespace Geometry {

// Maps a geometry in spherical coordinates to one in Cartesian
// coordinates in the standard way:
// x = r sin(theta) cos(phi)
// y = r sin(theta) sin(phi)
// z = r cos(theta)
// Since this object maps the GEOMETRY object from sph to cart, it actually
// maps the coordinates themselves the otehr way: cart -> sph:
// r = sqrt(x**2 + y**2 + z**2)
// th = acos(z/r)
// ph = atan(y / x)
class SphericalToCartesian {
 public:
  SphericalToCartesian() = default;
  KOKKOS_INLINE_FUNCTION
  void operator()(Real X1, Real X2, Real X3, Real C[NDSPACE], Real Jcov[NDSPACE][NDSPACE],
                  Real Jcon[NDSPACE][NDSPACE]) const {
    using robust::ratio;
    const Real x = X1;
    const Real y = X2;
    const Real z = X3;

    Real r, th, ph, r2, sth, cth, sph, cph;
    GetCoordsAndTranscendentals(x, y, z, r, th, ph, r2, sth, cth, sph, cph);

    C[0] = r;
    C[1] = th;
    C[2] = ph;

    const Real rho2 = x * x + y * y;
    const Real rho = std::sqrt(rho2);

    // These are dx^{mu'}/dx^{mu}
    // convention is mu' is first index, mu is second
    // Jcon has x^{mu'} = {x,y,z}
    // Jcov has x^{mu'} = {r,th,ph}

    // Jcov
    /*
    Jcov[0][0] = ratio(x, r);             // dr/dx
    Jcov[0][1] = ratio(y, r);             // dr/dy
    Jcov[0][2] = ratio(z, r);             // dr/dz
    Jcov[1][0] = ratio(x * z, r2 * rho);  // dth/dx
    Jcov[1][1] = ratio(y * z, r2 * rho2); // dth/dy
    Jcov[1][2] = -ratio(rho, r2);         // dth/dz
    Jcov[2][0] = -ratio(y, rho2);         // dph/dx
    Jcov[2][1] = ratio(x, rho2);          // dph/dy
    Jcov[2][2] = 0;                       // dph/dz
    */
    // JMM: The version with all spherical quantities is substantially
    // better-behaved because these quantities cancel out with the
    // quantities in the metric.
    Jcov[0][0] = sth * cph;
    Jcov[0][1] = sth * sph;
    Jcov[0][2] = cth;
    Jcov[1][0] = ratio(cth * cph, r);
    Jcov[1][1] = ratio(cth * sph, r);
    Jcov[1][2] = -ratio(sth, r);
    Jcov[2][0] = -ratio(sph, r * sth);
    Jcov[2][1] = ratio(cph, r * sth);
    Jcov[2][2] = 0;

    // Jcon
    Jcon[0][0] = sth * cph;      // dx/dr
    Jcon[0][1] = r * cth * cph;  // dx/dth
    Jcon[0][2] = -r * sth * sph; // dx/dph
    Jcon[1][0] = sth * sph;      // dy/dr
    Jcon[1][1] = r * cth * sph;  // dy/dth
    Jcon[1][2] = r * sth * cph;  // dy/dph
    Jcon[2][0] = cth;            // dz/dr
    Jcon[2][1] = -r * sth;       // dz/dth
    Jcon[2][2] = 0;              // dz/dph
  }

  KOKKOS_INLINE_FUNCTION
  void GetCoordsAndTranscendentals(const Real x, const Real y, const Real z, Real &r,
                                   Real &th, Real &ph, Real &r2, Real &sth, Real &cth,
                                   Real &sph, Real &cph) const {
    using robust::ratio;
    r2 = robust::make_positive(x * x + y * y + z * z);
    r = std::sqrt(r2);
    th = robust::make_positive(std::acos(ratio(z, r)));
    ph = std::atan2(y, x);
    sth = std::sin(th);
    cth = std::cos(th);
    sph = std::sin(ph);
    cph = std::cos(ph);
  }

  KOKKOS_INLINE_FUNCTION
  void GetCoordsAndDerivatives(const Real x, const Real y, const Real z, const Real dx,
                               const Real dy, const Real dz, Real &r, Real &th, Real &ph,
                               Real &dr, Real &dth, Real &dph) const {
    Real r2, sth, cth, sph, cph;
    GetCoordsAndTranscendentals(x, y, z, r, th, ph, r2, sth, cth, sph, cph);
    dr = dz * cth + sth * (dx * cph + dy * sph);
    dth = r * (-dz * sth + cth * (dx * cph + dy * sph));
    dph = r * sth * (dy * cph - dx * sph);
  }
};

template <>
SphericalToCartesian GetTransformation<SphericalToCartesian>(StateDescriptor *pkg);

} // namespace Geometry

#endif // GEOMETRY_SPH2CART_HPP_
