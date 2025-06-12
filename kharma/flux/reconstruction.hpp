/* 
 *  File: reconstruction.hpp
 *  
 *  BSD 3-Clause License
 *  
 *  Copyright (c) 2020, AFD Group at UIUC
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

#include "decs.hpp"

#include "reconstruct/dc_inline.hpp"
#include "plm_inline.hpp"

using namespace parthenon;


namespace KReconstruction
{
constexpr Real EPS = 1.e-26;

// Enum for all supported reconstruction types.
enum class Type{donor_cell=0, donor_cell_c, linear_mc, linear_vl, ppm, ppmx, mp5, weno5, weno5_lower_edges, weno5_lower_poles, weno5_linear};

// Component functions
KOKKOS_FORCEINLINE_FUNCTION Real mc(const Real dm, const Real dp)
{
    const Real r = (m::abs(dp) > 0. ? dm/dp : 2.0);
    return m::max(0.0, m::min(2.0, m::min(2*r,0.5*(1+r))));
}

KOKKOS_FORCEINLINE_FUNCTION Real mc(const Real dm, const Real dp, const Real alpha)
{
  const Real dc = (dm * dp > 0.0) * 0.5 * (dm + dp);
  return m::copysign(
      m::min(m::abs(dc), alpha * m::min(m::abs(dm), m::abs(dp))), dc);
}

// TODO make this a function w/forceinline
#define MINMOD(a, b) ((a) * (b) > 0.0 ? (m::abs(a) < m::abs(b) ? (a) : (b)) : 0.0)

KOKKOS_FORCEINLINE_FUNCTION double Median(double a, double b, double c)
{
    return (a + MINMOD(b - a, c - a));
}

// "left" and "right" here are relative to zone centers, so the pencil calls will switch them later.
#define RECONSTRUCT_ONE_ARGS const Real& x1, const Real& x2, const Real& x3, const Real& x4, const Real& x5, Real &lout, Real &rout
#define RECONSTRUCT_ONE_LEFT_ARGS const Real& x1, const Real& x2, const Real& x3, const Real& x4, const Real& x5, Real &lout
#define RECONSTRUCT_ONE_RIGHT_ARGS const Real& x1, const Real& x2, const Real& x3, const Real& x4, const Real& x5, Real &rout

// Single-element implementations:
template<Type recon_type>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct(RECONSTRUCT_ONE_ARGS) {}

// TODO better defaults?  As-is, forgetting to implement these causes problems!
template<Type recon_type>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_left(RECONSTRUCT_ONE_LEFT_ARGS) {}
template<Type recon_type>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_right(RECONSTRUCT_ONE_RIGHT_ARGS) {}

// Donor-cell
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct<Type::donor_cell_c>(RECONSTRUCT_ONE_ARGS)
{
    rout = x3;
    lout = x3;
}
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_left<Type::donor_cell_c>(RECONSTRUCT_ONE_LEFT_ARGS)
{
    lout = x3;
}
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_right<Type::donor_cell_c>(RECONSTRUCT_ONE_RIGHT_ARGS)
{
    rout = x3;
}

// Linear
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct<Type::linear_mc>(RECONSTRUCT_ONE_ARGS)
{
    const Real dq = mc(x3 - x2, x4 - x3)*(x4 - x3);
    rout = x3 + 0.5*dq;
    lout = x3 - 0.5*dq;
}
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_left<Type::linear_mc>(RECONSTRUCT_ONE_LEFT_ARGS)
{
    lout = x3 - 0.5*(mc(x3 - x2, x4 - x3)*(x4 - x3));
}
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_right<Type::linear_mc>(RECONSTRUCT_ONE_RIGHT_ARGS)
{
    rout = x3 + 0.5*(mc(x3 - x2, x4 - x3)*(x4 - x3));
}

// WENO5 (no linearization)
// Adapted from implementation in iharm3d originally by Monika Moscibrodzka
// References: Tchekhovskoy et al. 2007 (T07), Shu 2011 (S11)
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct<Type::weno5>(RECONSTRUCT_ONE_ARGS)
{
    // Smoothness indicators, T07 A18 or S11 8
    Real beta[3], c1, c2;
    c1 = x1 - 2.*x2 + x3; c2 = x1 - 4.*x2 + 3.*x3;
    beta[0] = (13./12.)*c1*c1 + (1./4.)*c2*c2;
    c1 = x2 - 2.*x3 + x4; c2 = x4 - x2;
    beta[1] = (13./12.)*c1*c1 + (1./4.)*c2*c2;
    c1 = x3 - 2.*x4 + x5; c2 = x5 - 4.*x4 + 3.*x3;
    beta[2] = (13./12.)*c1*c1 + (1./4.)*c2*c2;

    // Nonlinear weights S11 9
    Real den[3] = {EPS + beta[0], EPS + beta[1], EPS + beta[2]};
    den[0] *= den[0]; den[1] *= den[1]; den[2] *= den[2];

    Real wtr[3] = {(1./16.)/den[0], (5./8. )/den[1], (5./16.)/den[2]};
    Real Wr = wtr[0] + wtr[1] + wtr[2];

    Real wtl[3] = {(1./16.)/den[2], (5./8. )/den[1], (5./16.)/den[0]};
    Real Wl = wtl[0] + wtl[1] + wtl[2];

    // S11 1, 2, 3
    lout = ((3./8.)*x5 - (5./4.)*x4 + (15./8.)*x3)*(wtl[0] / Wl) +
            ((-1./8.)*x4 + (3./4.)*x3 + (3./8.)*x2)*(wtl[1] / Wl) +
            ((3./8.)*x3 + (3./4.)*x2 - (1./8.)*x1)*(wtl[2] / Wl);
    rout = ((3./8.)*x1 - (5./4.)*x2 + (15./8.)*x3)*(wtr[0] / Wr) +
            ((-1./8.)*x2 + (3./4.)*x3 + (3./8.)*x4)*(wtr[1] / Wr) +
            ((3./8.)*x3 + (3./4.)*x4 - (1./8.)*x5)*(wtr[2] / Wr);
}
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_left<Type::weno5>(RECONSTRUCT_ONE_LEFT_ARGS)
{
    // Smoothness indicators, T07 A18 or S11 8
    Real beta[3], c1, c2;
    c1 = x1 - 2.*x2 + x3; c2 = x1 - 4.*x2 + 3.*x3;
    beta[0] = (13./12.)*c1*c1 + (1./4.)*c2*c2;
    c1 = x2 - 2.*x3 + x4; c2 = x4 - x2;
    beta[1] = (13./12.)*c1*c1 + (1./4.)*c2*c2;
    c1 = x3 - 2.*x4 + x5; c2 = x5 - 4.*x4 + 3.*x3;
    beta[2] = (13./12.)*c1*c1 + (1./4.)*c2*c2;

    // Nonlinear weights S11 9
    Real den[3] = {EPS + beta[0], EPS + beta[1], EPS + beta[2]};
    den[0] *= den[0]; den[1] *= den[1]; den[2] *= den[2];

    Real wtl[3] = {(1./16.)/den[2], (5./8. )/den[1], (5./16.)/den[0]};
    Real Wl = wtl[0] + wtl[1] + wtl[2];

    // S11 1, 2, 3
    lout = ((3./8.)*x5 - (5./4.)*x4 + (15./8.)*x3)*(wtl[0] / Wl) +
            ((-1./8.)*x4 + (3./4.)*x3 + (3./8.)*x2)*(wtl[1] / Wl) +
            ((3./8.)*x3 + (3./4.)*x2 - (1./8.)*x1)*(wtl[2] / Wl);
}
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_right<Type::weno5>(RECONSTRUCT_ONE_RIGHT_ARGS)
{
    // Smoothness indicators, T07 A18 or S11 8
    Real beta[3], c1, c2;
    c1 = x1 - 2.*x2 + x3; c2 = x1 - 4.*x2 + 3.*x3;
    beta[0] = (13./12.)*c1*c1 + (1./4.)*c2*c2;
    c1 = x2 - 2.*x3 + x4; c2 = x4 - x2;
    beta[1] = (13./12.)*c1*c1 + (1./4.)*c2*c2;
    c1 = x3 - 2.*x4 + x5; c2 = x5 - 4.*x4 + 3.*x3;
    beta[2] = (13./12.)*c1*c1 + (1./4.)*c2*c2;

    // Nonlinear weights S11 9
    Real den[3] = {EPS + beta[0], EPS + beta[1], EPS + beta[2]};
    den[0] *= den[0]; den[1] *= den[1]; den[2] *= den[2];

    Real wtr[3] = {(1./16.)/den[0], (5./8. )/den[1], (5./16.)/den[2]};
    Real Wr = wtr[0] + wtr[1] + wtr[2];

    rout = ((3./8.)*x1 - (5./4.)*x2 + (15./8.)*x3)*(wtr[0] / Wr) +
            ((-1./8.)*x2 + (3./4.)*x3 + (3./8.)*x4)*(wtr[1] / Wr) +
            ((3./8.)*x3 + (3./4.)*x4 - (1./8.)*x5)*(wtr[2] / Wr);
}

// Linearized WENO, stolen from Phoebus
// Note lout/rout are SWITCHED until output to aid comparison with Phoebus,
// which uses the opposite L/R convention in per-zone calculations
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct<Type::weno5_linear>(RECONSTRUCT_ONE_ARGS)
{
    constexpr Real w5alpha[3][3] = {{1.0 / 3.0, -7.0 / 6.0, 11.0 / 6.0},
                                    {-1.0 / 6.0, 5.0 / 6.0, 1.0 / 3.0},
                                    {1.0 / 3.0, 5.0 / 6.0, -1.0 / 6.0}};
    constexpr Real w5gamma[3] = {0.1, 0.6, 0.3};
    constexpr Real eps = 1e-100;
    constexpr Real thirteen_thirds = 13.0 / 3.0;

    Real a = x1 - 2 * x2 + x3;
    Real b = x1 - 4.0 * x2 + 3.0 * x3;
    Real beta0 = thirteen_thirds * a * a + b * b + eps;
    a = x2 - 2.0 * x3 + x4;
    b = x4 - x2;
    Real beta1 = thirteen_thirds * a * a + b * b + eps;
    a = x3 - 2.0 * x4 + x5;
    b = x5 - 4.0 * x4 + 3.0 * x3;
    Real beta2 = thirteen_thirds * a * a + b * b + eps;
    const Real tau5 = m::abs(beta2 - beta0);

    beta0 = (beta0 + tau5) / beta0;
    beta1 = (beta1 + tau5) / beta1;
    beta2 = (beta2 + tau5) / beta2;

    Real w0 = w5gamma[0] * beta0 + eps;
    Real w1 = w5gamma[1] * beta1 + eps;
    Real w2 = w5gamma[2] * beta2 + eps;
    Real wsum = 1.0 / (w0 + w1 + w2);
    rout = w0 * (w5alpha[0][0] * x1 + w5alpha[0][1] * x2 + w5alpha[0][2] * x3);
    rout += w1 * (w5alpha[1][0] * x2 + w5alpha[1][1] * x3 + w5alpha[1][2] * x4);
    rout += w2 * (w5alpha[2][0] * x3 + w5alpha[2][1] * x4 + w5alpha[2][2] * x5);
    rout *= wsum;
    const Real alpha_r =
        3.0 * wsum * w0 * w1 * w2 /
            (w5gamma[2] * w0 * w1 + w5gamma[1] * w0 * w2 + w5gamma[0] * w1 * w2) +
        eps;

    w0 = w5gamma[0] * beta2 + eps;
    w1 = w5gamma[1] * beta1 + eps;
    w2 = w5gamma[2] * beta0 + eps;
    wsum = 1.0 / (w0 + w1 + w2);
    lout = w0 * (w5alpha[0][0] * x5 + w5alpha[0][1] * x4 + w5alpha[0][2] * x3);
    lout += w1 * (w5alpha[1][0] * x4 + w5alpha[1][1] * x3 + w5alpha[1][2] * x2);
    lout += w2 * (w5alpha[2][0] * x3 + w5alpha[2][1] * x2 + w5alpha[2][2] * x1);
    lout *= wsum;
    const Real alpha_l =
        3.0 * wsum * w0 * w1 * w2 /
            (w5gamma[2] * w0 * w1 + w5gamma[1] * w0 * w2 + w5gamma[0] * w1 * w2) +
        eps;

    Real dq = x4 - x3;
    dq = mc(x3 - x2, dq, 2.0);

    const Real alpha_lin = clip(2.0 * alpha_r * alpha_l / (alpha_r + alpha_l), 0.0, 1.0);
    rout = alpha_lin * rout + (1.0 - alpha_lin) * (x3 + 0.5 * dq);
    lout = alpha_lin * lout + (1.0 - alpha_lin) * (x3 - 0.5 * dq);
}
// TODO(BSP) Breaking out l & r probably doesn't save us much time on this one,
// but we could
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_left<Type::weno5_linear>(RECONSTRUCT_ONE_LEFT_ARGS)
{
    Real null;
    reconstruct<Type::weno5_linear>(x1, x2, x3, x4, x5, lout, null);
}
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_right<Type::weno5_linear>(RECONSTRUCT_ONE_RIGHT_ARGS)
{
    Real null;
    reconstruct<Type::weno5_linear>(x1, x2, x3, x4, x5, null, rout);
}

// MP5, lifted shamelessly from Phoebus, itself from nubhlight, originally from PLUTO
// How long can we keep this going?
KOKKOS_FORCEINLINE_FUNCTION double mp5_subcalc(double Fjm2, double Fjm1, double Fj, double Fjp1, double Fjp2)
{
  double f, d2, d2p, d2m;
  double dMMm, dMMp;
  double scrh1, scrh2, min, max;
  double fAV, fMD, fLC, fUL, fMP;
  constexpr double alpha = 4.0, epsm = 1.e-12;

  f = 2.0 * Fjm2 - 13.0 * Fjm1 + 47.0 * Fj + 27.0 * Fjp1 - 3.0 * Fjp2;
  f /= 60.0;

  fMP = Fj + MINMOD(Fjp1 - Fj, alpha * (Fj - Fjm1));

  if ((f - Fj) * (f - fMP) <= epsm) return f;

  d2m = Fjm2 + Fj - 2.0 * Fjm1; // Eqn. 2.19
  d2 = Fjm1 + Fjp1 - 2.0 * Fj;
  d2p = Fj + Fjp2 - 2.0 * Fjp1; // Eqn. 2.19

  scrh1 = MINMOD(4.0 * d2 - d2p, 4.0 * d2p - d2);
  scrh2 = MINMOD(d2, d2p);
  dMMp = MINMOD(scrh1, scrh2); // Eqn. 2.27
  scrh1 = MINMOD(4.0 * d2m - d2, 4.0 * d2 - d2m);
  scrh2 = MINMOD(d2, d2m);
  dMMm = MINMOD(scrh1, scrh2); // Eqn. 2.27

  fUL = Fj + alpha * (Fj - Fjm1);                   // Eqn. 2.8
  fAV = 0.5 * (Fj + Fjp1);                          // Eqn. 2.16
  fMD = fAV - 0.5 * dMMp;                           // Eqn. 2.28
  fLC = 0.5 * (3.0 * Fj - Fjm1) + 4.0 / 3.0 * dMMm; // Eqn. 2.29

  scrh1 = m::min(m::min(Fj, Fjp1), fMD);
  scrh2 = m::min(m::min(Fj, fUL), fLC);
  min = m::max(scrh1, scrh2); // Eqn. (2.24a)

  scrh1 = m::max(m::max(Fj, Fjp1), fMD);
  scrh2 = m::max(m::max(Fj, fUL), fLC);
  max = m::min(scrh1, scrh2); // Eqn. 2.24b

  f = Median(f, min, max); // Eqn. 2.26
  return f;
}
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct<Type::mp5>(RECONSTRUCT_ONE_ARGS)
{
    lout = mp5_subcalc(x5, x4, x3, x2, x1);
    rout = mp5_subcalc(x1, x2, x3, x4, x5);
}
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_left<Type::mp5>(RECONSTRUCT_ONE_LEFT_ARGS)
{
    lout = mp5_subcalc(x5, x4, x3, x2, x1);
}
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_right<Type::mp5>(RECONSTRUCT_ONE_RIGHT_ARGS)
{
    rout = mp5_subcalc(x1, x2, x3, x4, x5);
}


/**
 * PPM reconstruction, stolen from AthenaK complete with description
 * 
 * Original PPM (Colella & Woodward) parabolic reconstruction.  Returns
 * interpolated values at L/R edges of cell i, that is ql(i+1) and qr(i). Works for
 * reconstruction in any dimension by passing in the appropriate q_im2,...,q _ip2.
 */
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct<Type::ppm>(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
                                                    const Real &q_ip2, Real &qlv, Real &qrv)
{
  //---- Interpolate L/R values (CS eqn 16, PH 3.26 and 3.27) ----
  // qlv = q at left  side of cell-center = q[i-1/2] = a_{j,-} in CS
  // qrv = q at right side of cell-center = q[i+1/2] = a_{j,+} in CS
  qlv = (7.*(q_i + q_im1) - (q_im2 + q_ip1))/12.0;
  qrv = (7.*(q_i + q_ip1) - (q_im1 + q_ip2))/12.0;

  //---- limit qrv and qlv to neighboring cell-centered values (CS eqn 13) ----
  qlv = m::max(qlv, m::min(q_i, q_im1));
  qlv = m::min(qlv, m::max(q_i, q_im1));
  qrv = m::max(qrv, m::min(q_i, q_ip1));
  qrv = m::min(qrv, m::max(q_i, q_ip1));

  //--- monotonize interpolated L/R states (CS eqns 14, 15) ---
  Real qc = qrv - q_i;
  Real qd = qlv - q_i;
  if ((qc*qd) >= 0.0) {
    qlv = q_i;
    qrv = q_i;
  } else {
    if (m::abs(qc) >= 2.0*m::abs(qd)) {
      qrv = q_i - 2.0*qd;
    }
    if (m::abs(qd) >= 2.0*m::abs(qc)) {
      qlv = q_i - 2.0*qc;
    }
  }
}
// TODO(BSP) We also probably don't save much splitting here, but worth a shot?
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_left<Type::ppm>(RECONSTRUCT_ONE_LEFT_ARGS)
{
    Real null;
    reconstruct<Type::ppm>(x1, x2, x3, x4, x5, lout, null);
}
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_right<Type::ppm>(RECONSTRUCT_ONE_RIGHT_ARGS)
{
    Real null;
    reconstruct<Type::ppm>(x1, x2, x3, x4, x5, null, rout);
}

/**
 * PPMX extremum-preserving PPM, stolen from AthenaK complete with description
 * 
 * PPM parabolic reconstruction with Colella & Sekora limiters.  Returns
 * interpolated values at L/R edges of cell i, that is ql(i+1) and qr(i). Works for
 * reconstruction in any dimension by passing in the appropriate q_im2,...,q _ip2.
 */
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct<Type::ppmx>(const Real &q_im2, const Real &q_im1,
        const Real &q_i, const Real &q_ip1, const Real &q_ip2, Real &qlv, Real &qrv) {
  //---- Compute L/R values (CS eqns 12-15, PH 3.26 and 3.27) ----
  // qlv = q at left  side of cell-center = q[i-1/2] = a_{j,-} in CS
  // qrv = q at right side of cell-center = q[i+1/2] = a_{j,+} in CS
  qlv = (7.*(q_i + q_im1) - (q_im2 + q_ip1))/12.0;
  qrv = (7.*(q_i + q_ip1) - (q_im1 + q_ip2))/12.0;

  //---- Apply CS monotonicity limiters to qrv and qlv ----
  // approximate second derivatives at i-1/2 (PH 3.35)
  // KGF: add the off-center quantities first to preserve FP symmetry
  Real d2qc = 3.0*((q_im1 + q_i) - 2.0*qlv);
  Real d2ql = (q_im2 + q_i  ) - 2.0*q_im1;
  Real d2qr = (q_im1 + q_ip1) - 2.0*q_i;

  // limit second derivative (PH 3.36)
  Real d2qlim = 0.0;
  Real lim_slope = m::min(m::abs(d2ql),m::abs(d2qr));
  if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
    d2qlim = SIGN(d2qc)*m::min(1.25*lim_slope,m::abs(d2qc));
  }
  if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
    d2qlim = SIGN(d2qc)*m::min(1.25*lim_slope,m::abs(d2qc));
  }
  // compute limited value for qlv (PH 3.33 and 3.34)
  if (((q_im1 - qlv)*(q_i - qlv)) > 0.0) {
    qlv = 0.5*(q_i + q_im1) - d2qlim/6.0;
  }

  // approximate second derivatives at i+1/2 (PH 3.35)
  // KGF: add the off-center quantities first to preserve FP symmetry
  d2qc = 3.0*((q_i + q_ip1) - 2.0*qrv);
  d2ql = d2qr;
  d2qr = (q_i + q_ip2) - 2.0*q_ip1;

  // limit second derivative (PH 3.36)
  d2qlim = 0.0;
  lim_slope = m::min(m::abs(d2ql),m::abs(d2qr));
  if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
    d2qlim = SIGN(d2qc)*m::min(1.25*lim_slope,m::abs(d2qc));
  }
  if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
    d2qlim = SIGN(d2qc)*m::min(1.25*lim_slope,m::abs(d2qc));
  }
  // compute limited value for qrv (PH 3.33 and 3.34)
  if (((q_i - qrv)*(q_ip1 - qrv)) > 0.0) {
    qrv = 0.5*(q_i + q_ip1) - d2qlim/6.0;
  }

  //---- identify extrema, use smooth extremum limiter ----
  // CS 20 (missing "OR"), and PH 3.31
  Real qa = (qrv - q_i)*(q_i - qlv);
  Real qb = (q_im1 - q_i)*(q_i - q_ip1);
  if (qa <= 0.0 || qb <= 0.0) {
    // approximate secnd derivates (PH 3.37)
    // KGF: add the off-center quantities first to preserve FP symmetry
    Real d2q  = 6.0*(qlv + qrv - 2.0*q_i);
    Real d2qc = (q_im1 + q_ip1) - 2.0*q_i;
    Real d2ql = (q_im2 + q_i  ) - 2.0*q_im1;
    Real d2qr = (q_i   + q_ip2) - 2.0*q_ip1;

    // limit second derivatives (PH 3.38)
    d2qlim = 0.0;
    lim_slope = m::min(m::abs(d2ql),m::abs(d2qr));
    lim_slope = m::min(m::abs(d2qc),lim_slope);
    if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0 && d2q > 0.0) {
      d2qlim = SIGN(d2q)*m::min(1.25*lim_slope,m::abs(d2q));
    }
    if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0 && d2q < 0.0) {
      d2qlim = SIGN(d2q)*m::min(1.25*lim_slope,m::abs(d2q));
    }

    // limit L/R states at extrema (PH 3.39)
    Real rho = 0.0;
    if ( m::abs(d2q) > (1.0e-12)*m::max( m::abs(q_im1), m::max(m::abs(q_i),m::abs(q_ip1))) ) {
      // Limiter is not sensitive to round-off error.  Use limited slope
      rho = d2qlim/d2q;
    }
    qlv = q_i + (qlv - q_i)*rho;
    qrv = q_i + (qrv - q_i)*rho;
  } else {
    // Monotonize again, away from extrema (CW eqn 1.10, PH 3.32)
    Real qc = qrv - q_i;
    Real qd = qlv - q_i;
    if (m::abs(qc) >= 2.0*m::abs(qd)) {
      qrv = q_i - 2.0*qd;
    }
    if (m::abs(qd) >= 2.0*m::abs(qc)) {
      qlv = q_i - 2.0*qc;
    }
  }
}
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_left<Type::ppmx>(RECONSTRUCT_ONE_LEFT_ARGS)
{
    Real null;
    reconstruct<Type::ppmx>(x1, x2, x3, x4, x5, lout, null);
}
template<>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_right<Type::ppmx>(RECONSTRUCT_ONE_RIGHT_ARGS)
{
    Real null;
    reconstruct<Type::ppmx>(x1, x2, x3, x4, x5, null, rout);
}


// Row-wise implementations
// Note that "L" and "R" refer to the sides of the *face*
// ql(1) is to the right of the first zone center, but corresponds to the face value reconstructed from the left
// qr(1) is then the value at that face reconstructed from the right
// This is *opposite* the single-zone convention (or rather, offset from it to the faces):
// so weirdly, ReconstructX2l calls reconstruct_right.  Get it?
#define RECONSTRUCT_ROW_INPUT parthenon::team_mbr_t const &member, const int& k, const int& j, \
                              const int& il, const int& iu, const VariablePack<Real> &q,
#define RECONSTRUCT_ROW_ARGS RECONSTRUCT_ROW_INPUT ScratchPad2D<Real> &ql, ScratchPad2D<Real> &qr
#define RECONSTRUCT_ROW_LEFT_ARGS RECONSTRUCT_ROW_INPUT ScratchPad2D<Real> &ql
#define RECONSTRUCT_ROW_RIGHT_ARGS RECONSTRUCT_ROW_INPUT ScratchPad2D<Real> &qr

// TODO(BSP) I'm sure these could be shorter with more C++ magic
template <Type recon_type>
KOKKOS_FORCEINLINE_FUNCTION void ReconstructX1(RECONSTRUCT_ROW_ARGS)
{
    for (int p = 0; p <= q.GetDim(4) - 1; ++p) {
        parthenon::par_for_inner(member, il, iu,
            KOKKOS_LAMBDA (const int& i) {
                reconstruct<recon_type>(
                    q(p, k, j, i - 2),
                    q(p, k, j, i - 1),
                    q(p, k, j, i),
                    q(p, k, j, i + 1),
                    q(p, k, j, i + 2),
                    qr(p, i), ql(p, i+1));
            }
        );
    }
}
template <Type recon_type>
KOKKOS_FORCEINLINE_FUNCTION void ReconstructX2l(RECONSTRUCT_ROW_LEFT_ARGS)
{
    for (int p = 0; p <= q.GetDim(4) - 1; ++p) {
        parthenon::par_for_inner(member, il, iu,
            KOKKOS_LAMBDA (const int& i) {
                reconstruct_right<recon_type>(
                    q(p, k, j - 2, i),
                    q(p, k, j - 1, i),
                    q(p, k, j, i),
                    q(p, k, j + 1, i),
                    q(p, k, j + 2, i),
                    ql(p, i));
            }
        );
    }
}
template <Type recon_type>
KOKKOS_FORCEINLINE_FUNCTION void ReconstructX2r(RECONSTRUCT_ROW_RIGHT_ARGS)
{
    for (int p = 0; p <= q.GetDim(4) - 1; ++p) {
        parthenon::par_for_inner(member, il, iu,
            KOKKOS_LAMBDA (const int& i) {
                reconstruct_left<recon_type>(
                    q(p, k, j - 2, i),
                    q(p, k, j - 1, i),
                    q(p, k, j, i),
                    q(p, k, j + 1, i),
                    q(p, k, j + 2, i),
                    qr(p, i));
            }
        );
    }
}
template <Type recon_type>
KOKKOS_FORCEINLINE_FUNCTION void ReconstructX3l(RECONSTRUCT_ROW_LEFT_ARGS)
{
    for (int p = 0; p <= q.GetDim(4) - 1; ++p) {
        parthenon::par_for_inner(member, il, iu,
            KOKKOS_LAMBDA (const int& i) {
                reconstruct_right<recon_type>(
                    q(p, k - 2, j, i),
                    q(p, k - 1, j, i),
                    q(p, k, j, i),
                    q(p, k + 1, j, i),
                    q(p, k + 2, j, i),
                    ql(p, i));
            }
        );
    }
}
template <Type recon_type>
KOKKOS_FORCEINLINE_FUNCTION void ReconstructX3r(RECONSTRUCT_ROW_RIGHT_ARGS)
{
    for (int p = 0; p <= q.GetDim(4) - 1; ++p) {
        parthenon::par_for_inner(member, il, iu,
            KOKKOS_LAMBDA (const int& i) {
                reconstruct_left<recon_type>(
                    q(p, k - 2, j, i),
                    q(p, k - 1, j, i),
                    q(p, k, j, i),
                    q(p, k + 1, j, i),
                    q(p, k + 2, j, i),
                    qr(p, i));
            }
        );
    }
}


/**
 * Templated calls to different reconstruction algorithms
 * This is basically a compile-time 'if' or 'switch' statement, where all the options get generated
 * at compile-time (see driver.cpp for the different instantiations)
 */
template <Type recon_type, int dir>
KOKKOS_FORCEINLINE_FUNCTION void ReconstructRow(parthenon::team_mbr_t& member, const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    if constexpr (dir == X1DIR) {
        ReconstructX1<recon_type>(member, k, j, is_l, ie_l, P, ql, qr);
    } else if constexpr (dir == X2DIR) {
        ReconstructX2l<recon_type>(member, k, j - 1, is_l, ie_l, P, ql);
        ReconstructX2r<recon_type>(member, k, j, is_l, ie_l, P, qr);
    } else {
        ReconstructX3l<recon_type>(member, k - 1, j, is_l, ie_l, P, ql);
        ReconstructX3r<recon_type>(member, k, j, is_l, ie_l, P, qr);
    }
}

// Donor cell: Parthenon already implemented the row versions, so we call through.
template <>
KOKKOS_FORCEINLINE_FUNCTION void ReconstructRow<Type::donor_cell, X1DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    DonorCellX1(member, k, j, is_l, ie_l, P, ql, qr);
}
template <>
KOKKOS_FORCEINLINE_FUNCTION void ReconstructRow<Type::donor_cell, X2DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    DonorCellX2(member, k, j - 1, is_l, ie_l, P, ql, q_u);
    DonorCellX2(member, k, j, is_l, ie_l, P, q_u, qr);
}
template <>
KOKKOS_FORCEINLINE_FUNCTION void ReconstructRow<Type::donor_cell, X3DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    DonorCellX3(member, k - 1, j, is_l, ie_l, P, ql, q_u);
    DonorCellX3(member, k, j, is_l, ie_l, P, q_u, qr);
}

// Linear from Parthenon, w/simplified van Leer limiting
template <>
KOKKOS_FORCEINLINE_FUNCTION void ReconstructRow<Type::linear_vl, X1DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    // Extra scratch space for Parthenon's VL limiter stuff
    ScratchPad2D<Real>  qc(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> dql(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> dqr(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> dqm(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    PiecewiseLinearX1(member, k, j, is_l, ie_l, P, ql, qr, qc, dql, dqr, dqm);
}
template <>
KOKKOS_FORCEINLINE_FUNCTION void ReconstructRow<Type::linear_vl, X2DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    // Extra scratch space for Parthenon's VL limiter stuff
    ScratchPad2D<Real>  qc(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> dql(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> dqr(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> dqm(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    PiecewiseLinearX2(member, k, j - 1, is_l, ie_l, P, ql, q_u, qc, dql, dqr, dqm);
    PiecewiseLinearX2(member, k, j, is_l, ie_l, P, q_u, qr, qc, dql, dqr, dqm);
}
template <>
KOKKOS_FORCEINLINE_FUNCTION void ReconstructRow<Type::linear_vl, X3DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    // Extra scratch space for Parthenon's VL limiter stuff
    ScratchPad2D<Real>  qc(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> dql(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> dqr(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> dqm(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    PiecewiseLinearX3(member, k - 1, j, is_l, ie_l, P, ql, q_u, qc, dql, dqr, dqm);
    PiecewiseLinearX3(member, k, j, is_l, ie_l, P, q_u, qr, qc, dql, dqr, dqm);
}

// WENO5 lowered edges:
// Linear X1 reconstruction near X1 boundaries
template <>
KOKKOS_FORCEINLINE_FUNCTION void ReconstructRow<Type::weno5_lower_edges, X1DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    // This prioiritizes using the same-order fluxes on faces rather than for cells.
    // Neither is transparently wrong (afaict) but this feels nicer
    constexpr int o = 5; // offset
    ReconstructX1<Type::weno5>(member, k, j, is_l+o, ie_l-o, P, ql, qr);
    ReconstructX1<Type::linear_mc>(member, k, j, is_l, is_l+o-1, P, ql, qr);
    ReconstructX1<Type::linear_mc>(member, k, j, ie_l-o+1, ie_l, P, ql, qr);
}
template <>
KOKKOS_FORCEINLINE_FUNCTION void ReconstructRow<Type::weno5_lower_edges, X2DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    ReconstructRow<Type::weno5, X2DIR>(member, P, k, j, is_l, ie_l, ql, qr);
}
template <>
KOKKOS_FORCEINLINE_FUNCTION void ReconstructRow<Type::weno5_lower_edges, X3DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    ReconstructRow<Type::weno5, X3DIR>(member, P, k, j, is_l, ie_l, ql, qr);
}

// WENO5 lowered poles:
// Linear X2 reconstruction near X2 boundaries
template <>
KOKKOS_FORCEINLINE_FUNCTION void ReconstructRow<Type::weno5_lower_poles, X1DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    constexpr int o = 6;
    if (j > o && j < P.GetDim(2) - o) {
        ReconstructRow<Type::weno5, X1DIR>(member, P, k, j, is_l, ie_l, ql, qr);
    } else {
        ReconstructRow<Type::linear_mc, X1DIR>(member, P, k, j, is_l, ie_l, ql, qr);
    }
}
template <>
KOKKOS_FORCEINLINE_FUNCTION void ReconstructRow<Type::weno5_lower_poles, X2DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    // This prioiritizes using the same fluxes on faces rather than for cells.
    // Neither is transparently wrong (afaict) but this feels nicer
    // Use first 2 physical rows to prevent high-order recon reaching from rank 2 across pole
    constexpr int o = 6;
    if (j > o && j < P.GetDim(2) - o) {
        ReconstructX2l<Type::weno5>(member, k, j - 1, is_l, ie_l, P, ql);
        ReconstructX2r<Type::weno5>(member, k, j, is_l, ie_l, P, qr);
    } else {
        ReconstructX2l<Type::linear_mc>(member, k, j - 1, is_l, ie_l, P, ql);
        ReconstructX2r<Type::linear_mc>(member, k, j, is_l, ie_l, P, qr);
    }
}
template <>
KOKKOS_FORCEINLINE_FUNCTION void ReconstructRow<Type::weno5_lower_poles, X3DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    constexpr int o = 6;
    if (j > o && j < P.GetDim(2) - o) {
        ReconstructRow<Type::weno5, X3DIR>(member, P, k, j, is_l, ie_l, ql, qr);
    } else {
        ReconstructRow<Type::linear_mc, X3DIR>(member, P, k, j, is_l, ie_l, ql, qr);
    }
}

/**
 * Versions computing just the (limited) slope, for linear reconstructions.
 * Used for gradient calculations needed to implement Extended GRMHD.
 */
template <Type Recon>
KOKKOS_FORCEINLINE_FUNCTION Real slope_limit(Real x1, Real x2, Real x3, Real dx);
// Linear MC slope limiter
template <>
KOKKOS_FORCEINLINE_FUNCTION Real slope_limit<Type::linear_mc>(Real x1, Real x2, Real x3, Real dx)
{
    const Real Dqm = 2 * (x2 - x1) / dx;
    const Real Dqp = 2 * (x3 - x2) / dx;
    const Real Dqc = 0.5 * (x3 - x1) / dx;

    if (Dqm * Dqp <= 0) {
        return 0;
    } else {
        if ((m::abs(Dqm) < m::abs(Dqp)) && (m::abs(Dqm) < m::abs(Dqc))) {
            return Dqm;
        } else if (m::abs(Dqp) < m::abs(Dqc)) {
            return Dqp;
        } else {
            return Dqc;
        }
    }
}
// Linear Van Leer slope limiter
template <>
KOKKOS_FORCEINLINE_FUNCTION Real slope_limit<Type::linear_vl>(Real x1, Real x2, Real x3, Real dx)
{
    const Real Dqm = (x2 - x1) / dx;
    const Real Dqp = (x3 - x2) / dx;

    const Real extrema = Dqm * Dqp;

    if (extrema <= 0) {
        return 0;
    } else {
        return (2 * extrema / (Dqm + Dqp)); 
    }
}

/**
 * Run slope_limit in direction 'dir' using limiter 'recon'
 */
template <Type recon, int dir>
KOKKOS_FORCEINLINE_FUNCTION Real slope_calc(const GRCoordinates& G, const VariablePack<Real>& P,
                                              const int& p, const int& k, const int& j, const int& i);
// And six implementations.  Why can't you partial-specialize functions?  Why?
template <>
KOKKOS_FORCEINLINE_FUNCTION Real slope_calc<Type::linear_mc, X1DIR>(const GRCoordinates& G, const VariablePack<Real>& P,
                                              const int& p, const int& k, const int& j, const int& i)
{
    return slope_limit<Type::linear_mc>(P(p, k, j, i-1), P(p, k, j, i), P(p, k, j, i+1), G.Dxc<X1DIR>(i));
}
template <>
KOKKOS_FORCEINLINE_FUNCTION Real slope_calc<Type::linear_mc, X2DIR>(const GRCoordinates& G, const VariablePack<Real>& P,
                                              const int& p, const int& k, const int& j, const int& i)
{
    return slope_limit<Type::linear_mc>(P(p, k, j-1, i), P(p, k, j, i), P(p, k, j+1, i), G.Dxc<X2DIR>(j));
}
template <>
KOKKOS_FORCEINLINE_FUNCTION Real slope_calc<Type::linear_mc, X3DIR>(const GRCoordinates& G, const VariablePack<Real>& P,
                                              const int& p, const int& k, const int& j, const int& i)
{
    return slope_limit<Type::linear_mc>(P(p, k-1, j, i), P(p, k, j, i), P(p, k+1, j, i), G.Dxc<X3DIR>(k));
}
template <>
KOKKOS_FORCEINLINE_FUNCTION Real slope_calc<Type::linear_vl, X1DIR>(const GRCoordinates& G, const VariablePack<Real>& P,
                                              const int& p, const int& k, const int& j, const int& i)
{
    return slope_limit<Type::linear_vl>(P(p, k, j, i-1), P(p, k, j, i), P(p, k, j, i+1), G.Dxc<X1DIR>(i));
}
template <>
KOKKOS_FORCEINLINE_FUNCTION Real slope_calc<Type::linear_vl, X2DIR>(const GRCoordinates& G, const VariablePack<Real>& P,
                                              const int& p, const int& k, const int& j, const int& i)
{
    return slope_limit<Type::linear_vl>(P(p, k, j-1, i), P(p, k, j, i), P(p, k, j+1, i), G.Dxc<X2DIR>(j));
}
template <>
KOKKOS_FORCEINLINE_FUNCTION Real slope_calc<Type::linear_vl, X3DIR>(const GRCoordinates& G, const VariablePack<Real>& P,
                                              const int& p, const int& k, const int& j, const int& i)
{
    return slope_limit<Type::linear_vl>(P(p, k-1, j, i), P(p, k, j, i), P(p, k+1, j, i), G.Dxc<X3DIR>(k));
}

} // namespace KReconstruction
