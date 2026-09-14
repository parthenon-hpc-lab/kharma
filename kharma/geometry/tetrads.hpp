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

#ifndef GEOMETRY_TETRADS_HPP_
#define GEOMETRY_TETRADS_HPP_

#include "phoebus_utils/robust.hpp"
#include <parthenon/package.hpp>

namespace Geometry {

class Tetrads {
 public:
  KOKKOS_FUNCTION
  Tetrads(const double Ucon[NDFULL], const double Trial[NDFULL],
          const double Gcov[NDFULL][NDFULL]) {
    ConstructTetrads_(Ucon, Trial, Gcov);
  }

  KOKKOS_FUNCTION
  Tetrads(const double Ucon[NDFULL], const double Gcov[NDFULL][NDFULL]) {
    Real Trial[NDFULL] = {0, 1, 0, 0};
    ConstructTetrads_(Ucon, Trial, Gcov);
  }

  KOKKOS_INLINE_FUNCTION
  void CoordToTetradCon(const Real VCoord[NDFULL], Real VTetrad[NDFULL]) {
    SPACETIMELOOP(mu) {
      VTetrad[mu] = 0.;
      SPACETIMELOOP(nu) { VTetrad[mu] += Ecov_[mu][nu] * VCoord[nu]; }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void CoordToTetradCov(const Real VCoord[NDFULL], Real VTetrad[NDFULL]) {
    SPACETIMELOOP(mu) {
      VTetrad[mu] = 0.;
      SPACETIMELOOP(nu) { VTetrad[mu] += Econ_[mu][nu] * VCoord[nu]; }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void TetradToCoordCon(const Real VTetrad[NDFULL], Real VCoord[NDFULL]) {
    SPACETIMELOOP(mu) {
      VCoord[mu] = 0.;
      SPACETIMELOOP(nu) { VCoord[mu] += Econ_[nu][mu] * VTetrad[nu]; }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void TetradToCoordCov(const Real VTetrad[NDFULL], Real VCoord[NDFULL]) {
    SPACETIMELOOP(mu) {
      VCoord[mu] = 0.;
      SPACETIMELOOP(nu) { VCoord[mu] += Ecov_[nu][mu] * VTetrad[nu]; }
    }
  }

 private:
  KOKKOS_FUNCTION
  void ConstructTetrads_(const double Ucon[NDFULL], const double Trial[NDFULL],
                         const double Gcov[NDFULL][NDFULL]) {

    Real X1ness = 0.;
    Real X2ness = 0.;
    Real X3ness = 0.;
    SPACETIMELOOP(mu) {
      X1ness += Gcov[1][mu] * Trial[mu] / sqrt(fabs(Gcov[1][1]));
      X2ness += Gcov[2][mu] * Trial[mu] / sqrt(fabs(Gcov[2][2]));
      X3ness += Gcov[3][mu] * Trial[mu] / sqrt(fabs(Gcov[3][3]));
    }
    X1ness = fabs(X1ness);
    X2ness = fabs(X2ness);
    X3ness = fabs(X3ness);

    // Normalize Trial vector
    double norm = 0.;
    SPACETIMELOOP(mu) SPACETIMELOOP(nu) { norm += Trial[mu] * Trial[nu] * Gcov[mu][nu]; }

    // Time component parallel to u^{\mu}
    SPACETIMELOOP(mu) { Econ_[0][mu] = Ucon[mu]; }

    // Use X1, X2, X3. Then, whichever is closest to Trial, overwrite.
    SPACETIMELOOP(mu) {
      Econ_[1][mu] = Geometry::Utils::KroneckerDelta(mu, 1);
      Econ_[2][mu] = Geometry::Utils::KroneckerDelta(mu, 2);
      Econ_[3][mu] = Geometry::Utils::KroneckerDelta(mu, 3);
    }

    if (norm > robust::EPS()) {
      // We can use the Trial vector
      if (X1ness > X2ness && X1ness > X3ness) {
        // Trial vector is closest to X1. Overwrite
        SPACETIMELOOP(mu) { Econ_[1][mu] = Trial[mu]; }
      } else if (X2ness >= X1ness && X2ness > X3ness) {
        // Trial vector is closest to X2. Overwrite
        SPACETIMELOOP(mu) { Econ_[2][mu] = Trial[mu]; }
      } else { // Trial vector is closest X3. Overwrite
        SPACETIMELOOP(mu) { Econ_[3][mu] = Trial[mu]; }
      }
    }

    // Gram-Schmidt and normalization
    Normalize_(Econ_[0], Gcov);
    ProjectOut_(Econ_[1], Econ_[0], Gcov);
    Normalize_(Econ_[1], Gcov);
    ProjectOut_(Econ_[2], Econ_[0], Gcov);
    ProjectOut_(Econ_[2], Econ_[1], Gcov);
    Normalize_(Econ_[2], Gcov);
    ProjectOut_(Econ_[3], Econ_[0], Gcov);
    ProjectOut_(Econ_[3], Econ_[1], Gcov);
    ProjectOut_(Econ_[3], Econ_[2], Gcov);
    Normalize_(Econ_[3], Gcov);

    // Make covariant version
    SPACETIMELOOP(mu) { Geometry::Utils::Lower(Econ_[mu], Gcov, Ecov_[mu]); }
    SPACETIMELOOP(mu) { Ecov_[0][mu] *= -1.; }
  }
  KOKKOS_INLINE_FUNCTION
  void Normalize_(double Vcon[NDFULL], const double Gcov[NDFULL][NDFULL]) {
    double norm = 0.;
    SPACETIMELOOP(mu) SPACETIMELOOP(nu) { norm += Vcon[mu] * Vcon[nu] * Gcov[mu][nu]; }

    norm = sqrt(fabs(norm));
    SPACETIMELOOP(mu) { Vcon[mu] /= norm; }
  }

  KOKKOS_INLINE_FUNCTION
  void ProjectOut_(double Vcona[NDFULL], double Vconb[NDFULL],
                   const double Gcov[NDFULL][NDFULL]) {
    double Vconb_sq = 0.;
    SPACETIMELOOP(mu) SPACETIMELOOP(nu) {
      Vconb_sq += Vconb[mu] * Vconb[nu] * Gcov[mu][nu];
    }

    double adotb = 0.;
    SPACETIMELOOP(mu) SPACETIMELOOP(nu) { adotb += Vcona[mu] * Vconb[nu] * Gcov[mu][nu]; }

    SPACETIMELOOP(mu) { Vcona[mu] -= Vconb[mu] * adotb / Vconb_sq; }
  }

  Real Econ_[NDFULL][NDFULL];
  Real Ecov_[NDFULL][NDFULL];
};

} // namespace Geometry

#endif // GEOMETRY_TETRADS_HPP_
