/* 
 *  File: file.hpp
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
#include "coefficients.hpp"
#include "ipolarray.hpp"
#include "model_utils.hpp"
#include "tetrads.hpp"

// KHARMA headers
#include "decs.hpp"
#include "interpolation.hpp"
#include "matrix.hpp"
#include "types.hpp"

using namespace parthenon;

/**
 */
namespace File {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus InitMeshBlock(MeshBlock *pmb);

KOKKOS_INLINE_FUNCTION void clip_x(const GReal X[GR_DIM], GReal Xclip[GR_DIM])
{
    DLOOP1 Xclip[mu] = X[mu];
    // TODO check X1? Outside the domain we just don't process emission
    if (Xclip[2] < 0 || Xclip[2] > M_PI) printf("BAD X2\n");

    // TODO accommodate X[3] != phi
    while (Xclip[3] > 2*M_PI) Xclip[3] -= 2*M_PI;
    while (Xclip[3] < 0.) Xclip[3] += 2*M_PI;
}

KOKKOS_INLINE_FUNCTION Real lorentz_calc(const GReal Gcov[GR_DIM][GR_DIM], const Real uv[NVEC])
{
    const Real qsq = Gcov[1][1] * uv[V1] * uv[V1] +
                    Gcov[2][2] * uv[V2] * uv[V2] +
                    Gcov[3][3] * uv[V3] * uv[V3] +
                    2. * (Gcov[1][2] * uv[V1] * uv[V2] +
                        Gcov[1][3] * uv[V1] * uv[V3] +
                        Gcov[2][3] * uv[V2] * uv[V3]);

    return m::sqrt(1. + qsq);
}

KOKKOS_INLINE_FUNCTION void get_fourvectors(const GRCoordinates &G, const double X[GR_DIM],
                                            const VariablePack<Real> &P, const VarMap &m_p,
                                            double Ucon[GR_DIM], double Ucov[GR_DIM], double Bcon[GR_DIM], double Bcov[GR_DIM])
{
    const auto &coords = G.coords;
    int i, j, k;
    GReal del[4];
    GReal startx[GR_DIM] = {0., G.Xf<1>(0), G.Xf<2>(0), G.Xf<3>(0)};
    GReal dx[GR_DIM] = {0., G.Dxc<1>(), G.Dxc<2>(), G.Dxc<3>()};
    GReal Xclip[GR_DIM];
    clip_x(X, Xclip);
    Interpolation::Xtoijk(Xclip, startx, dx, i, j, k, del);

    // Interpolate for B, U
    Real uvec[NVEC], B_P[NVEC];
    B_P[V1] = Interpolation::linear(i, j, k, del, m_p.B1, P);
    B_P[V2] = Interpolation::linear(i, j, k, del, m_p.B2, P);
    B_P[V3] = Interpolation::linear(i, j, k, del, m_p.B3, P);
    uvec[V1] = Interpolation::linear(i, j, k, del, m_p.U1, P);
    uvec[V2] = Interpolation::linear(i, j, k, del, m_p.U2, P);
    uvec[V3] = Interpolation::linear(i, j, k, del, m_p.U3, P);

    double Gcov[GR_DIM][GR_DIM], Gcon[GR_DIM][GR_DIM];
    coords.gcov_native(Xclip, Gcov);
    coords.gcon_from_gcov(Gcov, Gcon);

    const Real gamma = lorentz_calc(Gcov, uvec);
    const Real alpha = 1. / m::sqrt(-Gcon[0][0]);

    Ucon[0] = gamma / alpha;
    VLOOP Ucon[v+1] = uvec[v] - gamma * alpha * Gcon[0][v+1];

    flip_index(Ucon, Gcov, Ucov);

    // This fn is guaranteed to have B values
    Bcon[0] = 0;
    VLOOP Bcon[0]  += B_P[v] * Ucov[v+1];
    VLOOP Bcon[v+1] = (B_P[v] + Bcon[0] * Ucon[v+1]) / Ucon[0];

    flip_index(Bcon, Gcov, Bcov);
}

KOKKOS_INLINE_FUNCTION void get_params(const GRCoordinates &G,
                                        const double X[GR_DIM], const double Kcon[GR_DIM],
                                        const double Ucon[GR_DIM], const double Ucov[GR_DIM],
                                        const double Bcon[GR_DIM], const double Bcov[GR_DIM],
                                        const double &RHO_unit, const double &B_unit,
                                        const VariablePack<Real> &P, const VarMap &m_p,
                                        FitParams &params)
{
    int i, j, k;
    GReal del[4];
    GReal startx[GR_DIM] = {0., G.Xf<1>(0), G.Xf<2>(0), G.Xf<3>(0)};
    GReal dx[GR_DIM] = {0., G.Dxc<1>(), G.Dxc<2>(), G.Dxc<3>()};
    GReal Xclip[GR_DIM];
    clip_x(X, Xclip);
    Interpolation::Xtoijk(Xclip, startx, dx, i, j, k, del);

    double rho = Interpolation::linear(i, j, k, del, m_p.RHO, P);
    double u = Interpolation::linear(i, j, k, del, m_p.UU, P);
    // printf("Interpolated X %g %g %g %g i=%d j=%d k=%d\n", Xclip[0], Xclip[1], Xclip[2], Xclip[3], i, j, k);
    // printf("Interpolated rho=%g u=%g\n", rho, u);

    // USER PARAMS:
    params.nu = get_fluid_nu(Kcon, Ucov);
    double bsq = dot(Bcon, Bcov);
    double B = m::sqrt(bsq);
    params.magnetic_field = B * B_unit;
    params.electron_density = rho * RHO_unit/(MP+ME); // TODO He
    params.observer_angle = m::acos(get_bk_mu(Kcon, Ucov, Bcov, B));

    // TODO PARAMETERS
    const double game = 4./3, gamp = 5./3, gam = 13./9;
    const double beta_crit = 1.0, trat_large = 3.0, trat_small = 3.0;

    // Get electron temp (TODO move ofc)
    if (rho > 0 && bsq > 0. && bsq/rho < 1.0) {
        const double beta_m = u * (gam - 1.) / (0.5 * bsq);
        const double betasq = beta_m * beta_m / (beta_crit * beta_crit);
        const double trat = trat_large * betasq / (1. + betasq) + trat_small /(1. + betasq);
        params.theta_e = (MP/ME) * (game - 1.) * (gamp - 1.)
                                / ((gamp - 1.) + (game-1.) * trat)
                        * u/rho;
        //params.theta_e = 2. * (MP/ME) * u / (15. * rho);
    } else {
        params.theta_e = 0.;
        params.electron_density = 0.;
    }

    // Options for fits (TODO separate function)
    params.approximate_bessels = false;
    params.fit_select = FitSelect::dexter;
    params.electrons = ElectronDist::maxwell_juettner;
}

}
