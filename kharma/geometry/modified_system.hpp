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

#ifndef GEOMETRY_MODIFIED_SYSTEM_HPP_
#define GEOMETRY_MODIFIED_SYSTEM_HPP_

#include <array>
#include <cmath>

// Parthenon includes
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

// phoebus includes
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/linear_algebra.hpp"
#include "phoebus_utils/robust.hpp"

namespace Geometry {

// A "modifier" that allows one to apply coordinate transformations to
// the spatial component an analytic coordinate system.
// For now, only the spatial component can be modified.
// i.e., the lapse and shift cannot be modified.
// Takes a coordinate system to modify,
// a functor that maps X1, X2, X3 to their new values,
// and to the Jacobian at that location.
// Jcon is defined as map from old coords to new for contravariant vectors
// Jcov is the same for covariant
// The Transformation functor should have a function
// void Operator()(Real X1, Real X2, Real X3,
//                 Real C[NDSPACE],
//                 Real Jcov[NDSPACE][NDSPACE],
//                 Real Jcon[NDSPACE][NDSPACE]) const;
//
// The TIME_DEPENDENT template parameter DOES NOT mean the
// transformation is time-dependent. It means the metric is.
// TODO(JMM): A time-dependent version of this may be worth pursuing
// eventually.
template <typename System, typename Transformation, bool TIME_DEPENDENT = false>
class Modified {
 public:
  Modified() = default;
  template <typename... Args>
  Modified(const Transformation &GetTransformation, Args... args)
      : dx_(1e-10), GetTransformation_(GetTransformation),
        s_(std::forward<Args>(args)...) {}
  template <typename... Args>
  Modified(const Real dx, const Transformation &GetTransformation, Args... args)
      : dx_(dx), GetTransformation_(GetTransformation), s_(std::forward<Args>(args)...) {}

  KOKKOS_INLINE_FUNCTION
  Real Lapse(Real X0, Real X1, Real X2, Real X3) const {
    Real C[NDSPACE];
    Real Jcov[NDSPACE][NDSPACE];
    Real Jcon[NDSPACE][NDSPACE];
    GetTransformation_(X1, X2, X3, C, Jcov, Jcon);
    return s_.Lapse(X0, C[0], C[1], C[2]);
  }

  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(Real X0, Real X1, Real X2, Real X3, Real beta[NDSPACE]) const {
    Real beta0[NDSPACE];
    Real C[NDSPACE];
    Real Jcov[NDSPACE][NDSPACE];
    Real Jcon[NDSPACE][NDSPACE];
    GetTransformation_(X1, X2, X3, C, Jcov, Jcon);
    s_.ContravariantShift(X0, C[0], C[1], C[2], beta0);
    LinearAlgebra::SetZero(beta, NDSPACE);
    SPACELOOP(i) {
      SPACELOOP(j) { beta[i] += Jcon[i][j] * beta0[j]; }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(Real X0, Real X1, Real X2, Real X3, Real g[NDFULL][NDFULL]) const {

    Real g0[NDFULL][NDFULL];
    Real C[NDSPACE];
    Real Jcov[NDSPACE][NDSPACE];
    Real Jcon[NDSPACE][NDSPACE];
    GetTransformation_(X1, X2, X3, C, Jcov, Jcon);
    s_.SpacetimeMetric(X0, C[0], C[1], C[2], g0);
    LinearAlgebra::SetZero(g, NDFULL, NDFULL);
    SPACETIMELOOP(mu) {
      SPACETIMELOOP(nu) {
        SPACETIMELOOP(lam) {
          SPACETIMELOOP(kap) {
            g[mu][nu] += g0[lam][kap] * S2ST_(Jcov, lam, mu) * S2ST_(Jcov, kap, nu);
          }
        }
      }
    }
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(Real X0, Real X1, Real X2, Real X3,
                              Real g[NDFULL][NDFULL]) const {
    Real g0[NDFULL][NDFULL];
    Real C[NDSPACE];
    Real Jcov[NDSPACE][NDSPACE];
    Real Jcon[NDSPACE][NDSPACE];
    GetTransformation_(X1, X2, X3, C, Jcov, Jcon);
    s_.SpacetimeMetricInverse(X0, C[0], C[1], C[2], g0);
    LinearAlgebra::SetZero(g, NDFULL, NDFULL);
    SPACETIMELOOP(mu) {
      SPACETIMELOOP(nu) {
        SPACETIMELOOP(lam) {
          SPACETIMELOOP(kap) {
            g[mu][nu] += g0[lam][kap] * S2ST_(Jcon, mu, lam) * S2ST_(Jcon, nu, kap);
          }
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void Metric(Real X0, Real X1, Real X2, Real X3, Real g[NDSPACE][NDSPACE]) const {

    Real g0[NDSPACE][NDSPACE];
    Real C[NDSPACE];
    Real Jcov[NDSPACE][NDSPACE];
    Real Jcon[NDSPACE][NDSPACE];
    GetTransformation_(X1, X2, X3, C, Jcov, Jcon);
    s_.Metric(X0, C[0], C[1], C[2], g0);
    LinearAlgebra::SetZero(g, NDSPACE, NDSPACE);
    SPACELOOP(i) {
      SPACELOOP(j) {
        SPACELOOP(k) {
          SPACELOOP(l) { g[i][j] += g0[k][l] * Jcov[k][i] * Jcov[l][j]; }
        }
      }
    }
  }
  KOKKOS_INLINE_FUNCTION
  void MetricInverse(Real X0, Real X1, Real X2, Real X3, Real g[NDSPACE][NDSPACE]) const {

    Real g0[NDSPACE][NDSPACE];
    Real C[NDSPACE];
    Real Jcov[NDSPACE][NDSPACE];
    Real Jcon[NDSPACE][NDSPACE];
    GetTransformation_(X1, X2, X3, C, Jcov, Jcon);
    s_.MetricInverse(X0, C[0], C[1], C[2], g0);
    LinearAlgebra::SetZero(g, NDSPACE, NDSPACE);
    SPACELOOP(i) {
      SPACELOOP(j) {
        SPACELOOP(k) {
          SPACELOOP(l) { g[i][j] += g0[k][l] * Jcon[i][k] * Jcon[j][l]; }
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  Real DetGamma(Real X0, Real X1, Real X2, Real X3) const {
    Real C[NDSPACE];
    Real Jcov[NDSPACE][NDSPACE];
    Real Jcon[NDSPACE][NDSPACE];
    GetTransformation_(X1, X2, X3, C, Jcov, Jcon);
    Real detJ = LinearAlgebra::Determinant(Jcov);
    return std::abs(s_.DetGamma(X0, C[0], C[1], C[2]) * detJ);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetG(Real X0, Real X1, Real X2, Real X3) const {
    Real C[NDSPACE];
    Real Jcov[NDSPACE][NDSPACE];
    Real Jcon[NDSPACE][NDSPACE];
    GetTransformation_(X1, X2, X3, C, Jcov, Jcon);
    Real detJ = LinearAlgebra::Determinant(Jcov);
    return std::abs(s_.DetG(X0, C[0], C[1], C[2]) * detJ);
  }

  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(Real X0, Real X1, Real X2, Real X3,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    Utils::SetConnectionCoeffByFD(*this, Gamma, X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(Real X0, Real X1, Real X2, Real X3,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    Utils::SetMetricGradientByFD(*this, dx_, X0, X1, X2, X3, dg);
    // If time-dependent, assume coordinate transformation is time
    // indepedent, pull dg_{mu nu}/dt from underlying system, and
    // transform mu and nu
    if (TIME_DEPENDENT) {
      Real dg0[NDFULL][NDFULL][NDFULL];
      Real C[NDSPACE];
      Real Jcon[NDSPACE][NDSPACE];
      Real Jcov[NDSPACE][NDSPACE];
      GetTransformation_(X1, X2, X3, C, Jcov, Jcon);
      s_.MetricDerivative(X0, X1, X2, X3, dg0);
      SPACETIMELOOP2(mu, nu) {
        dg[mu][nu][0] = 0;
        SPACETIMELOOP2(lam, kap) {
          dg[mu][nu][0] += dg0[lam][kap][0] * S2ST_(Jcon, mu, lam) * S2ST_(Jcon, nu, kap);
        }
      }
    }
  }
  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const {
    Utils::SetGradLnAlphaByFD(*this, dx_, X0, X1, X2, X3, da);
    // If time-dependent, assume coordinate transformation is time
    // indepedent, pull dalpha/dt from underlying system
    if (TIME_DEPENDENT) {
      Real da0[NDFULL];
      Real C[NDSPACE];
      Real Jcon[NDSPACE][NDSPACE];
      Real Jcov[NDSPACE][NDSPACE];
      GetTransformation_(X1, X2, X3, C, Jcov, Jcon);
      s_.GradLnAlpha(X0, X1, X2, X3, da0);
      da[0] = da0[0];
    }
  }

  KOKKOS_INLINE_FUNCTION
  void Coords(Real X0, Real X1, Real X2, Real X3, Real C[NDFULL]) const {
    Real Cnew[NDSPACE];
    Real Jcov[NDSPACE][NDSPACE];
    Real Jcon[NDSPACE][NDSPACE];
    GetTransformation_(X1, X2, X3, Cnew, Jcov, Jcon);
    s_.Coords(X0, Cnew[0], Cnew[1], Cnew[2], C);
  }

 private:
  KOKKOS_INLINE_FUNCTION
  Real S2ST_(const Real A[NDSPACE][NDSPACE], int mu, int nu) const {
    if (mu == 0 || nu == 0) {
      return (mu == nu) ? 1 : 0;
    } else {
      return A[mu - 1][nu - 1];
    }
  }
  Real dx_ = 1e-10;                  // finite differences dx
  System s_;                         // underlying coordinate system
  Transformation GetTransformation_; // transformation operator
};

} // namespace Geometry

#endif // GEOMETRY_MODIFIED_SYSTEM_HPP_
