// © 2021-2023. Triad National Security, LLC. All rights reserved.  This
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

#ifndef GEOMETRY_MCKINNEY_GAMMIE_RYAN_HPP_
#define GEOMETRY_MCKINNEY_GAMMIE_RYAN_HPP_

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

// Modification to a spherical coordinate system, adapted from
// HARM, ebhlight, nubhlight
// https://github.com/AFD-Illinois/ebhlight
// https://github.com/lanl/nubhlight
// first presented in McKinney and Gammie, ApJ 611:977-995, 2004,
// where it was called MKS or Modified Kerr-Schild.
// Includes the derefined poles of the FMKS coordinates.
// This class should likely be used with the "Store" option
// Can modify any "spherical" coordinate system.
// Usually applied to Kerr-Schild to form the FMKS coordinate system
// Use in conjunction with the Modified coordinate system.
class McKinneyGammieRyan {
 public:
  McKinneyGammieRyan()
      : derefine_poles_(true), h_(0.3), xt_(0.82), alpha_(14), x0_(0), smooth_(0.5),
        norm_(GetNorm_(alpha_, xt_)), hexp_br_(1000.), hexp_nsq_(4.0), hexp_csq_(4.0) {}
  McKinneyGammieRyan(Real x0) // this is the most common use-case
      : derefine_poles_(true), h_(0.3), xt_(0.82), alpha_(14), x0_(x0), smooth_(0.5),
        norm_(GetNorm_(alpha_, xt_)), hexp_br_(1000.), hexp_nsq_(4.0), hexp_csq_(4.0) {}
  McKinneyGammieRyan(bool derefine_poles, Real h, Real xt, Real alpha, Real x0,
                     Real smooth, Real hexp_br, Real hexp_nsq, Real hexp_csq)
      : derefine_poles_(derefine_poles), h_(h), xt_(xt), alpha_(alpha), x0_(x0),
        smooth_(smooth), norm_(GetNorm_(alpha_, xt_)), hexp_br_(hexp_br),
        hexp_nsq_(hexp_nsq), hexp_csq_(hexp_csq) {}
  McKinneyGammieRyan(bool derefine_poles, Real h, Real xt, Real alpha, Real x0,
                     Real smooth)
      : derefine_poles_(derefine_poles), h_(h), xt_(xt), alpha_(alpha), x0_(x0),
        smooth_(smooth), norm_(GetNorm_(alpha_, xt_)), hexp_br_(1000.), hexp_nsq_(4.0),
        hexp_csq_(4.0) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(Real X1, Real X2, Real X3, Real C[NDSPACE], Real Jcov[NDSPACE][NDSPACE],
                  Real Jcon[NDSPACE][NDSPACE]) const {
    using robust::ratio;
    Real y, th, thJ;

    const Real r = r_(X1);
    th_(X1, X2, y, th);
    C[0] = r;
    C[1] = th;
    C[2] = X3;

    // const Real drdX1 = std::exp(X1);
    const Real hexp_dr = X1 - hexp_br_;
    const Real drdX1 =
        std::exp(X1 + (hexp_dr > 0.) * hexp_csq_ * std::pow(hexp_dr, hexp_nsq_)) *
        (1. + (hexp_dr > 0.) * hexp_csq_ * hexp_nsq_ * std::pow(hexp_dr, hexp_nsq_ - 1.));
    const Real dthGdX2 = M_PI + M_PI * (1 - h_) * std::cos(2 * M_PI * X2);
    Real dthdX1, dthdX2;
    if (derefine_poles_) {
      thJ_(X2, y, thJ);
      const Real thG = thG_(X2);
      const Real dydX2 = 2.;
      const Real dthJdy = norm_ * (1 + std::pow(y / xt_, alpha_));
      const Real dthJdX2 = dthJdy * dydX2;
      const Real fac = x0_ - X1;
      dthdX1 = -smooth_ * (thJ - thG) * std::exp(smooth_ * fac);
      dthdX2 = dthGdX2 + std::exp(smooth_ * fac) * (dthJdX2 - dthGdX2);
    } else {
      dthdX1 = 0.0;
      dthdX2 = dthGdX2;
    }

    LinearAlgebra::SetZero(Jcov, NDSPACE, NDSPACE);
    LinearAlgebra::SetZero(Jcon, NDSPACE, NDSPACE);
    // Jcov
    Jcov[0][0] = drdX1;  // r
    Jcov[1][0] = dthdX1; // th
    Jcov[1][1] = dthdX2;
    Jcov[2][2] = 1.; // phi
    // Jcon
    Jcon[0][0] = ratio(1., drdX1);               // r
    Jcon[1][0] = -ratio(dthdX1, drdX1 * dthdX2); // th
    Jcon[1][1] = ratio(1., dthdX2);
    Jcon[2][2] = 1.; // phi
  }

  KOKKOS_INLINE_FUNCTION
  Real bl_radius(const Real x1) const { return r_(x1); }
  KOKKOS_INLINE_FUNCTION
  Real bl_theta(const Real x1, const Real x2) const {
    Real y, th;
    th_(x1, x2, y, th);
    return th;
  }

  KOKKOS_INLINE_FUNCTION
  void bl_to_ks(const Real x1, const Real a, Real *ucon_bl, Real *ucon_ks) const {
    Real trans[NDFULL][NDFULL];
    LinearAlgebra::SetZero(trans, NDFULL, NDFULL);
    const Real r = r_(x1);
    const Real idenom = 1.0 / (r * r - 2.0 * r + a * a);
    trans[0][0] = 1.0;
    trans[0][1] = 2.0 * r * idenom;
    trans[1][1] = 1.0;
    trans[2][2] = 1.0;
    trans[3][1] = a * idenom;
    trans[3][3] = 1.0;
    LinearAlgebra::SetZero(ucon_ks, NDFULL);
    SPACETIMELOOP2(mu, nu) { ucon_ks[mu] += trans[mu][nu] * ucon_bl[nu]; }
  }

  KOKKOS_INLINE_FUNCTION
  void bl_to_fmks(const Real x1, const Real x2, const Real x3, const Real a,
                  Real *ucon_bl, Real *ucon_fmks) const {
    Real ucon_ks[NDFULL];
    bl_to_ks(x1, a, ucon_bl, ucon_ks);

    Real c[NDSPACE];
    Real Jcov[NDSPACE][NDSPACE];
    Real Jcon[NDSPACE][NDSPACE];
    (*this)(x1, x2, x3, c, Jcov, Jcon);
    Real trans[NDFULL][NDFULL];
    LinearAlgebra::SetZero(trans, NDFULL, NDFULL);
    trans[0][0] = 1.0;
    SPACELOOP2(i, j) { trans[i + 1][j + 1] = Jcon[i][j]; }
    LinearAlgebra::SetZero(ucon_fmks, NDFULL);
    SPACETIMELOOP2(mu, nu) { ucon_fmks[mu] += trans[mu][nu] * ucon_ks[nu]; }
  }

 private:
  KOKKOS_INLINE_FUNCTION
  Real r_(const Real X1) const {
    const Real hexp_dr = X1 - hexp_br_;
    return std::exp(X1 + (hexp_dr > 0.) * hexp_csq_ * std::pow(hexp_dr, hexp_nsq_));
  }
  KOKKOS_INLINE_FUNCTION
  void th_(const Real X1, const Real X2, Real &y, Real &th) const {
    Real thG = thG_(X2);
    if (derefine_poles_) {
      Real thJ;
      thJ_(X2, y, thJ);
      const Real fac = x0_ - X1;
      th = thG + std::exp(smooth_ * fac) * (thJ - thG);
    } else {
      y = 0;
      th = thG;
    }
    // coordinate singularity fix at the poles. Avoid theta = 0.
    constexpr Real SINGSMALL = 1.e-20;
    if (std::fabs(th) < SINGSMALL) {
      if (th >= 0.)
        th = SINGSMALL;
      else if (th < 0)
        th = -SINGSMALL;
    }
    if (std::fabs(M_PI - th) < SINGSMALL) {
      if (th >= M_PI)
        th = M_PI + SINGSMALL;
      else if (th < M_PI)
        th = M_PI - SINGSMALL;
    }
  }
  KOKKOS_INLINE_FUNCTION
  Real thG_(const Real X2) const {
    return M_PI * X2 + ((1. - h_) / 2.) * std::sin(2. * M_PI * X2);
  }
  KOKKOS_INLINE_FUNCTION
  void thJ_(const Real X2, Real &y, Real &thJ) const {
    y = 2. * X2 - 1.;
    thJ = norm_ * y * (1. + std::pow(y / xt_, alpha_) / (alpha_ + 1.)) + 0.5 * M_PI;
  }
  KOKKOS_INLINE_FUNCTION
  Real GetNorm_(Real alpha, Real xt) const {
    return 0.5 * M_PI * 1. / (1. + 1. / (alpha + 1.) * 1. / std::pow(xt, alpha_));
  }
  bool derefine_poles_ = true;
  Real h_ = 0.3;
  Real xt_ = 0.82;
  Real alpha_ = 14.;
  Real x0_ = 0; // start point of smooth region
  Real smooth_ = 0.5;
  Real norm_;
  Real hexp_br_ = 1000.;
  Real hexp_nsq_ = 4.;
  Real hexp_csq_ = 4.;
};

template <>
McKinneyGammieRyan GetTransformation<McKinneyGammieRyan>(StateDescriptor *pkg);

} // namespace Geometry

#endif // GEOMETRY_MCKINNEY_GAMMIE_RYAN_HPP_
