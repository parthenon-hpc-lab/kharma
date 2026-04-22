/*
 *  File: tetrads.hpp
 *
 *  BSD 3-Clause License
 *
 *  Copyright (c) 2025, Iris contributors
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

// Iris headers
#include "variables.hpp"

// KHARMA headers
#include "decs.hpp"
#include "matrix.hpp"
#include "types.hpp"

// TODO lots of this should be GReal
// TODO overlap here with matrix.hpp

KOKKOS_INLINE_FUNCTION void flip_index(const double ucon[GR_DIM], const double Gcov[GR_DIM][GR_DIM], double ucov[GR_DIM])
{
    DLOOP1 ucov[mu] = 0.;
    DLOOP2 ucov[mu] += Gcov[mu][nu] * ucon[nu];
}

/* assumes gcov has been set first; returns sqrt{|g|} */
KOKKOS_INLINE_FUNCTION double gdet_func(const double gcov[GR_DIM][GR_DIM])
{
    Real gcon[GR_DIM][GR_DIM];
    return m::sqrt(m::abs(invert(&gcov[0][0], &gcon[0][0])));
}
KOKKOS_INLINE_FUNCTION double gcon_func(const double gcov[GR_DIM][GR_DIM], double gcon[GR_DIM][GR_DIM])
{
    return m::sqrt(m::abs(invert(&gcov[0][0], &gcon[0][0])));
}

/* input and vectors are contravariant (index up) */
KOKKOS_INLINE_FUNCTION void tetrad_to_coordinate(const double Econ[GR_DIM][GR_DIM], const double K_tetrad[GR_DIM],
                          double K[GR_DIM])
{
    DLOOP1 K[mu] = 0.;
    DLOOP2 K[mu] += Econ[nu][mu] * K_tetrad[nu];
}

/*
 * Copy the trial vector into a tetrad basis vector,
 * checking to see if it is null, and if it is null
 * setting to some default value
 */
KOKKOS_INLINE_FUNCTION void set_Econ_from_trial(const double trial[GR_DIM], const int &defdir, double Econ[GR_DIM])
{
    double norm = 0.;
    VLOOP norm += fabs(trial[v+1]);

    if (norm <= SMALL_NUM) // bad trial vector; default to defdir
        DLOOP1 Econ[mu] = (mu == defdir);
    else
        DLOOP1 Econ[mu] = trial[mu];
}

/*
 * Normalize input vector so that |v . v| = 1
 * Overwrites input
 */
KOKKOS_INLINE_FUNCTION void normalize(double vcon[GR_DIM], const double Gcov[GR_DIM][GR_DIM])
{
  double norm = 0.;
  DLOOP2 norm += vcon[mu] * vcon[nu] * Gcov[mu][nu];
  norm = m::sqrt(m::abs(norm));

  DLOOP1 vcon[mu] /= norm;
}

/* normalize null vector in a tetrad frame */
KOKKOS_INLINE_FUNCTION void null_normalize(double Kcon[GR_DIM], double fnorm)
{
  double inorm =
    m::sqrt(Kcon[1] * Kcon[1] + Kcon[2] * Kcon[2] + Kcon[3] * Kcon[3]);

  Kcon[0] = fnorm;
  Kcon[1] *= fnorm / inorm;
  Kcon[2] *= fnorm / inorm;
  Kcon[3] *= fnorm / inorm;
}

/*
 * project out vconb from vcona
 * both arguments are index up (contravariant)
 * covariant metric is third argument.
 * overwrite the first argument on return
 */
KOKKOS_INLINE_FUNCTION void project_out(double vcona[GR_DIM], const double vconb[GR_DIM],
                                        const double Gcov[GR_DIM][GR_DIM])
{
  double vconb_sq = 0.;
  DLOOP2 vconb_sq += vconb[mu] * vconb[nu] * Gcov[mu][nu];

  double adotb = 0.;
  DLOOP2 adotb += vcona[mu] * vconb[nu] * Gcov[mu][nu];

  DLOOP1 vcona[mu] -= vconb[mu] * adotb / vconb_sq;
}

/* the completely antisymmetric symbol; not a tensor
 in the coordinate basis */
KOKKOS_INLINE_FUNCTION int levi_civita(const int &i, const int &j, const int &k, const int &l)
{
    int index[4], n, do_sort, n_perm, val, n_swap;
    if (i == j || i == k || i == l || j == k || j == l || k == l) {
        return 0;
    } else {
        index[0] = i;
        index[1] = j;
        index[2] = k;
        index[3] = l;
        do_sort = 1;
        n_perm = 0;
        while (do_sort) {
            n_swap = 0;
            for (n = 0; n < GR_DIM - 1; n++) {
                if (index[n] > index[n + 1]) {
                    n_perm++;
                    n_swap++;
                    val = index[n];
                    index[n] = index[n + 1];
                    index[n + 1] = val;
                }
            }
            do_sort = n_swap;
        }
        return (n_perm % 2) ? -1 : 1;
    }
}

/*
 * Check the handedness of a tetrad basis.
 * Basis is assumed to be in form e^\mu_{(a)} = Econ[a][mu]
 * levi_(ijkl) e0^i e1^j e2^k e3^l will be +1 if spatial
 * components are right-handed, -1 if left-handed.
 * experience suggests that roundoff produces errors of
 * order 10^{-12} in the result.
 */
KOKKOS_INLINE_FUNCTION double check_handedness(const double Econ[GR_DIM][GR_DIM])
{
    /* check handedness */
    double dot = 0.;
    DLOOP4 dot += levi_civita(mu, nu, lam, kap) * Econ[0][mu]  * Econ[1][nu]
                                                * Econ[2][lam] * Econ[3][kap];
    return dot;
}

/*
 * econ/ecov index key:
 * Econ[k][l]
 * k: index attached to tetrad basis
 * index down
 * l: index attached to coordinate basis
 * index up
 * Ecov switches both indices
 *
 * make orthonormal basis for plasma frame.
 * e^0 along U
 * e^2 along b
 * e^3 along spatial part of K
 *
 * Returns flag for whether the tetrad is suspicious.
 * Ideally ipole should crash on these errors but there are a lot of corner cases...
 */
KOKKOS_INLINE_FUNCTION int make_plasma_tetrad(const double Ucon[GR_DIM], const double Kcon[GR_DIM],
                        const double Bcon[GR_DIM], const double Gcov[GR_DIM][GR_DIM],
                        double Econ[GR_DIM][GR_DIM], double Ecov[GR_DIM][GR_DIM])
{
    // Modified Gram-Schmidt method to produce e^n orthogonal
    // e^1 is wholly determined by orthonormality
    constexpr double ones[GR_DIM] = {1., 1., 1., 1.};
    set_Econ_from_trial(Ucon, 0, Econ[0]);
    set_Econ_from_trial(ones, 1, Econ[1]);
    set_Econ_from_trial(Bcon, 2, Econ[2]);
    set_Econ_from_trial(Kcon, 3, Econ[3]);

    // (Re)orthogonalize
    // Repeating the orthogonalizations below wouldn't be necessary in
    // exact math, of course, but what if we have numbers on the order
    // of machine accuracy (epsilon)? Consider a matrix A:
    // A = [2 1]
    //     [d 0]
    //     [0 d]
    // where d is O(sqrt(epsilon)), so that generally N + d^2 = N when
    // represented in double precision.  Say we want an orthogonal form of A,
    // the matrix Q.
    // Orthonormalized with Gram-Schmidt and throwing away d^2:
    // Q2= [1        0     ]
    //     [d/2  -1/sqrt(5)]
    //     [0     2/sqrt(5)]
    // But if we orthogonalize Q2 again, we get:
    // Q = [1   d/(2 sqrt(5))]
    //     [d/2  -1/sqrt(5)  ]
    //     [0     2/sqrt(5)  ]
    // While there are more complex & faster reorthogonalization methods, simply
    // repeating the orthogonalization exactly once for each vector reaches near
    // optimal accuracy, see Giraud and Langou 2003 for background and
    // Giraud et al. '05 for the relevant analysis

    // Note we choose the order carefully to preserve e^0 == ucon perfectly,
    // and e^3 ~= Kcon as closely as possible.
    normalize(Econ[0], Gcov);
    project_out(Econ[3], Econ[0], Gcov);
    project_out(Econ[3], Econ[0], Gcov);
    normalize(Econ[3], Gcov);
    project_out(Econ[2], Econ[0], Gcov);
    project_out(Econ[2], Econ[3], Gcov);
    project_out(Econ[2], Econ[0], Gcov);
    project_out(Econ[2], Econ[3], Gcov);
    normalize(Econ[2], Gcov);
    project_out(Econ[1], Econ[0], Gcov);
    project_out(Econ[1], Econ[2], Gcov);
    project_out(Econ[1], Econ[3], Gcov);
    project_out(Econ[1], Econ[0], Gcov);
    project_out(Econ[1], Econ[2], Gcov);
    project_out(Econ[1], Econ[3], Gcov);
    normalize(Econ[1], Gcov);

    int oddflag = 0;

    /* check handedness */
    double gdet = gdet_func(Gcov);
    if (gdet < 0.) {
        oddflag |= 16;
    }

    // less restrictive condition on geometry for eKS coordinates which are
    // used when the exotic is expected.
    // TODO configurable tolerance
    double dot = gdet * check_handedness(Econ);
    if (fabs(fabs(dot) - 1.) > 1.e-10) {
        oddflag |= 1;
    }

    /* we expect dot = 1. for right-handed system.
    If not, flip Econ[1] to make system right-handed. */
    if (dot < 0.) {
        DLOOP1 Econ[1][mu] *= -1.;
    }

    /*** done w/ basis vector 3 ***/

    // Lower into Ecov
    DLOOP1 flip_index(Econ[mu], Gcov, Ecov[mu]);

    // then raise tetrad basis index
    DLOOP1 Ecov[0][mu] *= -1.;

    // For paranoia could run check_ortho here

    // TODO if DEBUG etc
    // if (oddflag) {
    //     printf("ODDFLAG in make_plasma_tetrad: %d dot: %g\n", oddflag, dot);
    //     printf("Ucon: %g %g %g %g\n", Ucon[0], Ucon[1], Ucon[2], Ucon[3]);
    //     printf("Bcon: %g %g %g %g\n", Bcon[0], Bcon[1], Bcon[2], Bcon[3]);
    //     printf("Kcon: %g %g %g %g\n", Kcon[0], Kcon[1], Kcon[2], Kcon[3]);
    //     printf("Econ[0]: %g %g %g %g\n", Econ[0][0], Econ[0][1], Econ[0][2], Econ[0][3]);
    //     printf("Econ[1]: %g %g %g %g\n", Econ[1][0], Econ[1][1], Econ[1][2], Econ[1][3]);
    //     printf("Econ[2]: %g %g %g %g\n", Econ[2][0], Econ[2][1], Econ[2][2], Econ[2][3]);
    //     printf("Econ[3]: %g %g %g %g\n", Econ[3][0], Econ[3][1], Econ[3][2], Econ[3][3]);
    // }

    return oddflag;
}

KOKKOS_INLINE_FUNCTION void N_to_tetrad(const Kokkos::complex<double> N_coord[GR_DIM][GR_DIM],
    const double Ecov[GR_DIM][GR_DIM], Kokkos::complex<double> N_tetrad[GR_DIM][GR_DIM])
{
    DLOOP2 N_tetrad[mu][nu] = Kokkos::complex<double>(0., 0.);
    DLOOP4 N_tetrad[mu][nu] += N_coord[lam][kap] * Ecov[mu][lam] * Ecov[nu][kap];
}
// This is a little different than above, it uses Econ.T
KOKKOS_INLINE_FUNCTION void N_to_coord(const Kokkos::complex<double> N_tetrad[GR_DIM][GR_DIM],
    const double Econ[GR_DIM][GR_DIM], Kokkos::complex<double> N_coord[GR_DIM][GR_DIM])
{
    DLOOP2 N_coord[mu][nu] = Kokkos::complex<double>(0., 0.);
    DLOOP4 N_coord[mu][nu] += N_tetrad[lam][kap] * Econ[lam][mu] * Econ[kap][nu];
}

// Split N into real/imag for Parthenon
KOKKOS_INLINE_FUNCTION void write_N(const Kokkos::complex<double> N_coord[GR_DIM][GR_DIM],
            const ParArrayND<Real> &Nr, const ParArrayND<Real> &Ni, const int &n)
{
    DLOOP2 {
        Nr(mu, nu, n) = N_coord[mu][nu].real();
        Ni(mu, nu, n) = N_coord[mu][nu].imag();
    }
}
template <typename T>
KOKKOS_INLINE_FUNCTION void write_N(Kokkos::complex<double> N_coord[GR_DIM][GR_DIM], T &pack,
                                    const int &b, const int &n)
{
    DLOOP2 {
        pack(b, rays::Nr(4*mu + nu), n) = N_coord[mu][nu].real();
        pack(b, rays::Ni(4*mu + nu), n) = N_coord[mu][nu].imag();
    }
}
KOKKOS_INLINE_FUNCTION void read_N(const ParArrayND<Real> &Nr, const ParArrayND<Real> &Ni, const int &n,
                                    Kokkos::complex<double> N_coord[GR_DIM][GR_DIM])
{
    DLOOP2 N_coord[mu][nu] = Kokkos::complex<double>(Nr(mu, nu, n), Ni(mu, nu, n));
}
template <typename T>
KOKKOS_INLINE_FUNCTION void read_N(T &pack, const int &b, const int &n,
                                    Kokkos::complex<double> N_coord[GR_DIM][GR_DIM])
{
    DLOOP2 N_coord[mu][nu] = Kokkos::complex<double>(pack(b, rays::Nr(4*mu + nu), n), pack(b, rays::Ni(4*mu + nu), n));
}

KOKKOS_INLINE_FUNCTION void stokes_to_N(const double &SI, const double &SQ, const double &SU,
                      const double &SV, Kokkos::complex<double> N[GR_DIM][GR_DIM])
{
    N[1][1].real(SI + SQ);
    N[1][2] = Kokkos::complex<double>(SU, -SV);
    N[2][1] = Kokkos::complex<double>(SU,  SV);
    N[2][2].real(SI - SQ);
}

KOKKOS_INLINE_FUNCTION void N_to_stokes(const Kokkos::complex<double> N[GR_DIM][GR_DIM],
                                             double &SI, double &SQ, double &SU, double &SV)
{
    SI = (N[1][1] + N[2][2]).real() / 2;
    SQ = (N[1][1] - N[2][2]).real() / 2;
    SU = (N[1][2] + N[2][1]).real() / 2;
    SV = (N[2][1] - N[1][2]).imag() / 2;
}

// KOKKOS_INLINE_FUNCTION void print_stokes_from_N_coord(
//                                             const Kokkos::complex<double> N_coord[GR_DIM][GR_DIM])
// {

//     make_plasma_tetrad();
//     N_to_tetrad(N_coord, Ecov, N_tetrad);
//     N_to_stokes();
// }
