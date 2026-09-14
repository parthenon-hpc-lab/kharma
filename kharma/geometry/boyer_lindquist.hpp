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

#ifndef GEOMETRY_BOYER_LINDQUIST_HPP_
#define GEOMETRY_BOYER_LINDQUIST_HPP_

#include <array>
#include <cmath>

// Parthenon includes
#include <kokkos_abstraction.hpp>

// phoebus includes
#include "geometry/analytic_system.hpp"
#include "geometry/cached_system.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/linear_algebra.hpp"
#include "phoebus_utils/robust.hpp"

namespace Geometry {

// Boyer-Lindquist coordinates
// singular at the horizon
// Formulae computed with Mathematica
// Mathematica notebooks stored
// Assumes WLOG that G = c = M = 1.
// TODO(JMM): Should we modify to accept arbitrary M?
class BoyerLindquist {
 public:
  BoyerLindquist() = default;
  KOKKOS_INLINE_FUNCTION
  BoyerLindquist(Real a) : dx_(1e-8), a_(a), a2_(a * a) {}
  KOKKOS_INLINE_FUNCTION
  BoyerLindquist(Real a, Real dx) : dx_(dx), a_(a), a2_(a * a) {}
  KOKKOS_INLINE_FUNCTION
  Real Lapse(Real X0, Real X1, Real X2, Real X3) const {
    // sqrt(-1 / g^{00})
    const Real r = std::abs(X1);
    const Real th = X2;
    Real r2, r3, sth, cth, DD, mu;
    ComputeDeltaMu_(r, th, r2, r3, sth, cth, DD, mu);
    const Real rm2 = r * mu - 2;
    const Real a2 = a2_;
    const Real num = a2 * (2 * r * mu - 4) +
                     r2 * (-4 + r * mu * (2 - 2 * DD + r * DD * mu)) + 4 * a2 * sth * sth;
    const Real den = r3 * DD * mu * rm2;
    return std::sqrt(robust::ratio(num, den));
  }
  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(Real X0, Real X1, Real X2, Real X3, Real beta[NDSPACE]) const {
    LinearAlgebra::SetZero(beta, NDSPACE);
    const Real r = std::abs(X1);
    const Real th = X2;
    const Real sth = std::sin(th);
    const Real cth = std::cos(th);
    // -g^{33}/g^{00} = g^{33}*alpha^2
    const Real num = -2 * a_ * r * sth * sth;
    const Real den = (r - 2) * r + a2_ * cth * cth;
    beta[3 - 1] = robust::ratio(num, den);
  }
  // TODO(JMM): SpacetimeMetric and metric could be combined?
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(Real X0, Real X1, Real X2, Real X3, Real g[NDFULL][NDFULL]) const {
    using namespace Utils;
    using namespace robust;
    const Real r = std::abs(X1);
    const Real th = X2;
    Real r2, r3, sth, cth, DD, mu;
    ComputeDeltaMu_(r, th, r2, r3, sth, cth, DD, mu);
    const Real sth2 = sth * sth;
    LinearAlgebra::SetZero(g, NDFULL, NDFULL);
    g[0][0] = -(1.0 - ratio(2.0, r * mu));
    g[0][3] = g[3][0] = -ratio(2.0 * a_ * sth2, r * mu);
    g[1][1] = mu / DD;
    g[2][2] = r2 * mu;
    g[3][3] = r2 * sth2 * (1.0 + ratio(a2_, r2) + ratio(2.0 * a2_ * sth2, r3 * mu));
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(Real X0, Real X1, Real X2, Real X3,
                              Real g[NDFULL][NDFULL]) const {
    using namespace Utils;
    using namespace robust;
    const Real r = std::abs(X1);
    const Real th = X2;
    Real r2, r3, sth, cth, DD, mu;
    ComputeDeltaMu_(r, th, r2, r3, sth, cth, DD, mu);
    const Real r3dm = r3 * DD * mu;
    const Real rm = r * mu;
    const Real rm2 = rm - 2;
    const Real sth2 = sth * sth;
    const Real common_denom =
        2 * a2_ * rm2 + r2 * (-4 + rm * (2 + DD * rm2)) + 4 * a2_ * sth2;
    LinearAlgebra::SetZero(g, NDFULL, NDFULL);
    g[0][0] = ratio(-r3dm * rm2, common_denom);
    g[0][3] = g[3][0] =
        -ratio(a_ * r3dm * sth2,
               a2_ * rm2 + r2 * (-2 + rm * (1 - DD + 0.5 * DD * rm)) + 2 * a2_ * sth2);
    g[1][1] = ratio(mu, DD);
    g[2][2] = r2 * mu;
    g[3][3] = ratio(r3dm * (2 * a2_ + r2 * (2 + DD * rm)) * sth2, common_denom);
  }
  KOKKOS_INLINE_FUNCTION
  void Metric(Real X0, Real X1, Real X2, Real X3, Real g[NDSPACE][NDSPACE]) const {
    using namespace Utils;
    using namespace robust;
    const Real r = std::abs(X1);
    const Real th = X2;
    Real r2, r3, sth, cth, DD, mu;
    ComputeDeltaMu_(r, th, r2, r3, sth, cth, DD, mu);
    const Real r3dm = r3 * DD * mu;
    LinearAlgebra::SetZero(g, NDSPACE, NDSPACE);
    g[0][0] = ratio(DD, mu);
    g[1][1] = ratio(1., r2 * mu);
    g[2][2] = ratio((r * mu - 2) * sth * sth, r3dm);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricInverse(Real X0, Real X1, Real X2, Real X3, Real g[NDSPACE][NDSPACE]) const {
    using namespace Utils;
    using namespace robust;
    const Real r = std::abs(X1);
    const Real th = X2;
    Real r2, r3, sth, cth, DD, mu;
    ComputeDeltaMu_(r, th, r2, r3, sth, cth, DD, mu);
    const Real rm = r * mu;
    const Real rm2 = rm - 2;
    const Real sth2 = sth * sth;
    const Real r3dm = r3 * DD * mu;
    const Real denom = 2 * a2_ * rm2 + r2 * (-4 + rm * (2 + DD * rm2)) + 4 * a2_ * sth2;

    // TODO(JMM): This is a lot less performant but much cleaner
    const Real alpha = Lapse(X0, X1, X2, X3);
    Real beta[NDSPACE];
    ContravariantShift(X0, X1, X2, X3, beta);

    LinearAlgebra::SetZero(g, NDSPACE, NDSPACE);
    g[0][0] = ratio(mu, DD);
    g[1][1] = r2 * mu;
    g[2][2] = ratio(r3dm * (2 * a2_ + r2 * (2 + DD * rm)) * sth2, denom);
    g[2][2] += ratio(beta[2] * beta[2], alpha * alpha);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetGamma(Real X0, Real X1, Real X2, Real X3) const {
    using namespace Utils;
    using namespace robust;
    const Real r = std::abs(X1);
    const Real th = X2;
    Real r2, r3, sth, cth, DD, mu;
    ComputeDeltaMu_(r, th, r2, r3, sth, cth, DD, mu);
    const Real r5 = r2 * r3;
    const Real mu3 = mu * mu * mu;
    const Real sth2 = sth * sth;
    // TODO(JMM): This is a little silly. Could factor to remove a few
    // multiplications
    return std::sqrt(std::abs(ratio(r * mu - 2, r5 * mu3 * sth2)));
  }
  KOKKOS_INLINE_FUNCTION
  Real DetG(Real X0, Real X1, Real X2, Real X3) const {
    // TODO(JMM): Less performant then writing it out. But much saner.
    const Real dgam = DetGamma(X0, X1, X2, X3);
    const Real alpha = Lapse(X0, X1, X2, X3);
    return alpha * dgam;
  }
  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(Real X0, Real X1, Real X2, Real X3,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    Utils::SetConnectionCoeffByFD(*this, Gamma, X0, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(Real X0, Real X1, Real X2, Real X3,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    using namespace Utils;
    using namespace robust;
    LinearAlgebra::SetZero(dg, NDFULL, NDFULL, NDFULL);
    const Real r = std::abs(X1);
    const Real th = X2;
    Real r2, r3, sth, cth, DD, MU;
    ComputeDeltaMu_(r, th, r2, r3, sth, cth, DD, MU);
    const Real c2th = std::cos(2 * th);
    const Real s2th = std::sin(2 * th);
    const Real cth2 = cth * cth;
    const Real cth4 = cth2 * cth2;
    const Real sth2 = sth * sth;
    const Real a3 = a2_ * a_;
    const Real a4 = a3 * a_;
    const Real a6 = a4 * a2_;
    const Real r6 = r3 * r3;
    const Real rm2 = r - 2;
    const Real r3m4 = 3 * r - 4;
    const Real common_denom = r2 + a2_ * cth2;
    const Real denom1 = common_denom * common_denom;
    Real denom2 = (a2_ + r * (r - 2)) * common_denom;
    denom2 *= denom2;

    dg[0][0][1] = ratio(-a6 + 2 * r6 + a2_ * r3 * r3m4 +
                            a2_ * (-a4 - 2 * a2_ * r2 + (4 - r) * r3) * c2th,
                        denom2);
    dg[0][3][1] = dg[3][0][1] =
        2 * a_ * ratio(r2 * (a2_ + r * r3m4) + a2_ * (r - a_) * (r + a_) * cth2, denom2);
    dg[1][1][1] = 2 * ratio(r * (r - a2_) + a2_ * (r - 1) * cth2, denom1);
    dg[2][2][1] = -ratio(2 * r, denom1);
    dg[3][3][1] =
        ratio(2 * r2 * (a2_ - rm2 * rm2 * r) - 2 * a2_ * (a2_ + r2 * (2 * r - 3)) * cth2 -
                  2 * a4 * (r - 1) * cth4,
              denom2 * sth2);

    dg[0][0][2] = -ratio(4 * a2_ * r * (a2_ + r2) * cth * sth, denom2);
    dg[0][3][2] = dg[3][0][2] = -ratio(4 * a3 * r * cth * sth, denom2);
    dg[1][1][2] = ratio(a2_ * (a2_ + r * (r - 2)) * s2th, denom1);
    dg[2][2][2] = ratio(a2_ * s2th, denom1);

    // g[3][3][2] is its own beast
    const Real a2pr2 = a2_ + r2;
    const Real num = -a2pr2 * cth *
                     (3 * a4 + 8 * a2_ * r2 + 8 * rm2 * r3 +
                      4 * a2_ * (a2_ + 2 * rm2 * r) * c2th + a4 * std::cos(4 * th)) *
                     sth;
    Real den = (a2_ + 2 * r2 + a2_ * c2th) * sth2;
    den *= den;
    den *= (a2_ + rm2 * r) * a2pr2;
    dg[3][3][2] = ratio(num, den);
  }

  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const {
    Utils::SetGradLnAlphaByFD(*this, dx_, X0, X1, X2, X3, da);
  }

  KOKKOS_INLINE_FUNCTION
  void Coords(Real X0, Real X1, Real X2, Real X3, Real C[NDFULL]) const {
    const Real r = std::abs(X1);
    const Real sth = std::sin(X2);
    const Real cth = std::cos(X2);
    const Real sph = std::sin(X3);
    const Real cph = std::cos(X3);
    C[0] = X0;
    C[1] = r * sth * cph;
    C[2] = r * sth * sph;
    C[3] = r * cth;
  }

 private:
  KOKKOS_INLINE_FUNCTION
  void ComputeInternals_(const Real r, const Real th, Real &bl_fact, Real &cth2,
                         Real &sth2) const {
    cth2 = std::cos(th);
    cth2 *= cth2;
    sth2 = 1.0 - cth2;
    bl_fact = r * r + a2_ * cth2;
  }
  KOKKOS_INLINE_FUNCTION
  void ComputeDeltaMu_(Real r, Real th, Real &r2, Real &r3, Real &sth, Real &cth,
                       Real &DD, Real &mu) const {
    using namespace Utils;
    using namespace robust;
    sth = std::sin(th);
    cth = std::cos(th);
    /*
    if (std::abs(sth) < EPS()) {
      sth = sgn(sth) * EPS();
    }
    if (std::abs(cth) < EPS()) {
      cth = sgn(cth) * EPS();
    }
    */
    r2 = r * r;
    r3 = r2 * r;
    DD = 1. - ratio(2., r) + ratio(a2_, r2);
    mu = 1. + ratio(a2_ * cth * cth, r2);
  }
  Real dx_ = 1e-8;
  Real a_ = 0;
  Real a2_;
};

using BoyerLindquistMeshBlock = Analytic<BoyerLindquist, IndexerMeshBlock>;
using BoyerLindquistMesh = Analytic<BoyerLindquist, IndexerMesh>;

using CBoyerLindquistMeshBlock =
    CachedOverMeshBlock<Analytic<BoyerLindquist, IndexerMeshBlock>>;
using CBoyerLindquistMesh = CachedOverMesh<Analytic<BoyerLindquist, IndexerMesh>>;

template <>
void Initialize<BoyerLindquistMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);
template <>
void Initialize<CBoyerLindquistMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);

} // namespace Geometry

#endif // GEOMETRY_BOYER_LINDQUIST_HPP_
