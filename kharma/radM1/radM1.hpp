/*
 *  File: radM1.hpp
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

#include "gr_coordinates.hpp"
#include "grmhd_functions.hpp"
#include "kharma_utils.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

namespace RadM1
{

// Denote implicit solve failures (rflags)
// This enum should grow to cover any potential flags
enum class StatusImplicitStep { success = 0, mhdsolve, radsolve, bothsolve, failure };

static const std::map<int, std::string> status_names = {
    {(int)StatusImplicitStep::mhdsolve,
        "RadM1 MHD Solve Failure"}, // flag that means that the MHD inversion failed (but
                                    // rad solve worked)
    {(int)StatusImplicitStep::radsolve,
        "RadM1 Radiation Solve Failure"}, // flag that means that the radiation solve
                                          // failed (but mhd solve worked)
    {(int)StatusImplicitStep::failure, "RadM1 Step Failure"}};

TaskStatus BlockPtoU(MeshBlockData<Real>* rc, IndexDomain domain, bool coarse = false);
/**
 * Initialize the radM1 package with several options from the input deck
 */
std::shared_ptr<KHARMAPackage> Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages);

/**
 * Perform the implicit solve for radiation and plasma coupled. For now, only 4D
 * implemented.
 */
TaskStatus Step(MeshData<Real>* md_full_init, MeshData<Real>* md_sub_init,
    MeshData<Real>* md_sub_final, const Real dt);

/**
 * Convert from conserved to primitive variables for the radiation field.
 */
TaskStatus BlockUtoP(MeshBlockData<Real>* rc, IndexDomain domain, bool coarse = false);

/**
 * Apply floors to the radiation energy variables.
 */
void ApplyRadM1Floors(MeshBlockData<Real>* rc, IndexDomain domain);

/**
 * Anything printed post-step
 */
TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real>* md);

// Opacity model selector for calc_kabs/calc_kscattering/ComputeCovariantFourForce.
// Default is the eventual singularity-opac slot! Add new cases here as more
// tests need their own opacity law, without causing issues with the default.
enum class OpacityModel : int { Default = 0, ShocktubeConstant = 1 };

KOKKOS_INLINE_FUNCTION Real calc_kabs(
    Real rho, Real T, int opacity_model, Real shocktube_kappa_rho)
{
    Real kappa_rho;
    if (opacity_model == (int)OpacityModel::ShocktubeConstant) {
        kappa_rho = shocktube_kappa_rho;
    } else {
        kappa_rho = 0.08;
    }
    return m::min(rho * kappa_rho, 1.e5);
}

KOKKOS_INLINE_FUNCTION Real calc_kscattering(
    Real rho, Real T, int opacity_model, Real shocktube_kappa_scat)
{
    if (opacity_model == (int)OpacityModel::ShocktubeConstant) {
        return shocktube_kappa_scat;
    }
    return 0.0;
}

// Global Lorentz Factor for Radiation
template<typename Global>
KOKKOS_INLINE_FUNCTION Real lorentz_calc_rad(const GRCoordinates& G, const Global& P,
    const VarMap& m, const int& k, const int& j, const int& i, const Loci loc)
{
    Real qsq =
        G.gcov(loc, j, i, 1, 1) * P(m.U1_RAD, k, j, i) * P(m.U1_RAD, k, j, i) +
        G.gcov(loc, j, i, 2, 2) * P(m.U2_RAD, k, j, i) * P(m.U2_RAD, k, j, i) +
        G.gcov(loc, j, i, 3, 3) * P(m.U3_RAD, k, j, i) * P(m.U3_RAD, k, j, i) +
        2. * (G.gcov(loc, j, i, 1, 2) * P(m.U1_RAD, k, j, i) * P(m.U2_RAD, k, j, i) +
                 G.gcov(loc, j, i, 1, 3) * P(m.U1_RAD, k, j, i) * P(m.U3_RAD, k, j, i) +
                 G.gcov(loc, j, i, 2, 3) * P(m.U2_RAD, k, j, i) * P(m.U3_RAD, k, j, i));
    return m::sqrt(1. + qsq);
}

// Local Lorentz Factor for Radiation
template<typename Local>
KOKKOS_INLINE_FUNCTION Real lorentz_calc_rad(const GRCoordinates& G, const Local& P,
    const VarMap& m, const int& j, const int& i, const Loci loc)
{
    Real qsq = G.gcov(loc, j, i, 1, 1) * P(m.U1_RAD) * P(m.U1_RAD) +
               G.gcov(loc, j, i, 2, 2) * P(m.U2_RAD) * P(m.U2_RAD) +
               G.gcov(loc, j, i, 3, 3) * P(m.U3_RAD) * P(m.U3_RAD) +
               2. * (G.gcov(loc, j, i, 1, 2) * P(m.U1_RAD) * P(m.U2_RAD) +
                        G.gcov(loc, j, i, 1, 3) * P(m.U1_RAD) * P(m.U3_RAD) +
                        G.gcov(loc, j, i, 2, 3) * P(m.U2_RAD) * P(m.U3_RAD));
    return m::sqrt(1. + qsq);
}

// Global ucon for Radiation
template<typename Global>
KOKKOS_INLINE_FUNCTION void calc_ucon_rad(const GRCoordinates& G, const Global& P,
    const VarMap& m, const int& k, const int& j, const int& i, const Loci loc,
    Real ucon[GR_DIM])
{
    const Real gamma = lorentz_calc_rad(G, P, m, k, j, i, loc);
    const Real alpha = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));
    ucon[0] = gamma / alpha;
    VLOOP
        ucon[v + 1] =
            P(m.U1_RAD + v, k, j, i) - gamma * alpha * G.gcon(loc, j, i, 0, v + 1);
}

// Local ucon for Radiation
template<typename Local>
KOKKOS_INLINE_FUNCTION void calc_ucon_rad(const GRCoordinates& G, const Local& P,
    const VarMap& m, const int& j, const int& i, const Loci loc, Real ucon[GR_DIM])
{
    const Real gamma = lorentz_calc_rad(G, P, m, j, i, loc);
    const Real alpha = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));
    ucon[0] = gamma / alpha;
    VLOOP
        ucon[v + 1] = P(m.U1_RAD + v) - gamma * alpha * G.gcon(loc, j, i, 0, v + 1);
}

KOKKOS_INLINE_FUNCTION Real lorentz_calc_rad(
    const GRCoordinates& G, const Real P[4], const int& j, const int& i)
{
    Real qsq = G.gcov(Loci::center, j, i, 1, 1) * P[1] * P[1] +
               G.gcov(Loci::center, j, i, 2, 2) * P[2] * P[2] +
               G.gcov(Loci::center, j, i, 3, 3) * P[3] * P[3] +
               2. * (G.gcov(Loci::center, j, i, 1, 2) * P[1] * P[2] +
                        G.gcov(Loci::center, j, i, 1, 3) * P[1] * P[3] +
                        G.gcov(Loci::center, j, i, 2, 3) * P[2] * P[3]);
    return m::sqrt(1. + qsq);
}

// Local ucon for Radiation
KOKKOS_INLINE_FUNCTION void calc_ucon_rad(const GRCoordinates& G, const Real P[4],
    const int& j, const int& i, Real ucon[GR_DIM])
{
    const Real gamma = lorentz_calc_rad(G, P, j, i);
    const Real alpha = 1. / m::sqrt(-G.gcon(Loci::center, j, i, 0, 0));
    ucon[0] = gamma / alpha;
    VLOOP
        ucon[v + 1] = P[1 + v] - gamma * alpha * G.gcon(Loci::center, j, i, 0, v + 1);
}

// M1 Tensor construction (Global)
// This will give you R^mu_dir
KOKKOS_INLINE_FUNCTION void calc_tensor(const GRCoordinates& G, const Real P[4],
    const int& dir, const int& j, const int& i, Real R_dir_mu[GR_DIM])
{
    Real Erf = P[0];
    Real ucon_rad[GR_DIM];
    calc_ucon_rad(G, P, j, i, ucon_rad);

    Real R_con_dir[GR_DIM];
    for (int nu = 0; nu < 4; ++nu) {
        R_con_dir[nu] = (4.0 / 3.0) * Erf * ucon_rad[dir] * ucon_rad[nu] +
                        (1.0 / 3.0) * Erf * G.gcon(Loci::center, j, i, dir, nu);
    }

    G.lower(R_con_dir, R_dir_mu, 0, j, i, Loci::center);
}

// M1 Tensor construction (Global)
// This will give you R^mu_dir
KOKKOS_INLINE_FUNCTION void calc_tensor(const GRCoordinates& G,
    const VariablePack<Real>& P, const VarMap& m_p, const int& dir, const int& k,
    const int& j, const int& i, const Loci loc, Real R_dir_mu[GR_DIM])
{
    Real Erf = P(m_p.UU_RAD, k, j, i);
    Real ucon_rad[GR_DIM];
    calc_ucon_rad(G, P, m_p, k, j, i, loc, ucon_rad);

    Real R_con_dir[GR_DIM];
    for (int nu = 0; nu < 4; ++nu) {
        R_con_dir[nu] = (4.0 / 3.0) * Erf * ucon_rad[dir] * ucon_rad[nu] +
                        (1.0 / 3.0) * Erf * G.gcon(loc, j, i, dir, nu);
    }

    G.lower(R_con_dir, R_dir_mu, k, j, i, loc);
}

// M1 Tensor construction (Local)
// This will give you R^mu_dir
template<typename Local>
KOKKOS_INLINE_FUNCTION void calc_tensor(const GRCoordinates& G, const Local& P,
    const VarMap& m_p, const int& dir, const int& j, const int& i, const Loci loc,
    Real R_dir_mu[GR_DIM])
{
    Real Erf = P(m_p.UU_RAD);
    Real ucon_rad[GR_DIM];
    calc_ucon_rad(G, P, m_p, j, i, loc, ucon_rad);

    Real R_con_dir[GR_DIM];
    for (int nu = 0; nu < 4; ++nu) {
        R_con_dir[nu] = (4.0 / 3.0) * Erf * ucon_rad[dir] * ucon_rad[nu] +
                        (1.0 / 3.0) * Erf * G.gcon(loc, j, i, dir, nu);
    }

    G.lower(R_con_dir, R_dir_mu, 0, j, i, loc); // Note: Assuming k=0 for local slices
}

KOKKOS_INLINE_FUNCTION void initialize_radiation_pressure(Real UU, Real& UU_rad)
{
    // Here we assume that Pgas + Prad = Ptot
    // This translates to rho * T + 1/3 a_rad * T^4 - Ptot = 0
    // The derivative gives us rho + 4/3 a_rad * T^3 = 0, which we can use to find the
    // root of the equation and solve for T given rho and Ptot.
    //  This should be done if we're simulating high accretion rates, bnecause then we
    //  should not start with a low radiation pressure, but for all purposes we are gonna
    //  assume here that the radiation pressure is negligible at the start of the
    //  simulation, so we can just set it to a small value.

    // radiation pressure is 0.1% of the gas pressure at the start of the simulation.
    UU_rad = UU * 0.001;

    return;
}

// TODO import these from matrix.hpp
KOKKOS_INLINE_FUNCTION
void adjoint(Real m[16], Real adjOut[16])
{
    adjOut[0] = MINOR(m, 1, 2, 3, 1, 2, 3);
    adjOut[1] = -MINOR(m, 0, 2, 3, 1, 2, 3);
    adjOut[2] = MINOR(m, 0, 1, 3, 1, 2, 3);
    adjOut[3] = -MINOR(m, 0, 1, 2, 1, 2, 3);

    adjOut[4] = -MINOR(m, 1, 2, 3, 0, 2, 3);
    adjOut[5] = MINOR(m, 0, 2, 3, 0, 2, 3);
    adjOut[6] = -MINOR(m, 0, 1, 3, 0, 2, 3);
    adjOut[7] = MINOR(m, 0, 1, 2, 0, 2, 3);

    adjOut[8] = MINOR(m, 1, 2, 3, 0, 1, 3);
    adjOut[9] = -MINOR(m, 0, 2, 3, 0, 1, 3);
    adjOut[10] = MINOR(m, 0, 1, 3, 0, 1, 3);
    adjOut[11] = -MINOR(m, 0, 1, 2, 0, 1, 3);

    adjOut[12] = -MINOR(m, 1, 2, 3, 0, 1, 2);
    adjOut[13] = MINOR(m, 0, 2, 3, 0, 1, 2);
    adjOut[14] = -MINOR(m, 0, 1, 3, 0, 1, 2);
    adjOut[15] = MINOR(m, 0, 1, 2, 0, 1, 2);
}

KOKKOS_INLINE_FUNCTION
Real MINOR(Real m[16], int r0, int r1, int r2, int c0, int c1, int c2)
{
    return m[4 * r0 + c0] *
               (m[4 * r1 + c1] * m[4 * r2 + c2] - m[4 * r2 + c1] * m[4 * r1 + c2]) -
           m[4 * r0 + c1] *
               (m[4 * r1 + c0] * m[4 * r2 + c2] - m[4 * r2 + c0] * m[4 * r1 + c2]) +
           m[4 * r0 + c2] *
               (m[4 * r1 + c0] * m[4 * r2 + c1] - m[4 * r2 + c0] * m[4 * r1 + c1]);
}

KOKKOS_INLINE_FUNCTION
Real determinant(Real m[16])
{
    return m[0] * MINOR(m, 1, 2, 3, 1, 2, 3) - m[1] * MINOR(m, 1, 2, 3, 0, 2, 3) +
           m[2] * MINOR(m, 1, 2, 3, 0, 1, 3) - m[3] * MINOR(m, 1, 2, 3, 0, 1, 2);
}

KOKKOS_INLINE_FUNCTION
void matrixInverse4x4(Real mat[4][4], Real matinv[4][4])
{
    adjoint(&(mat[0][0]), &(matinv[0][0]));

    Real det = determinant(&(mat[0][0]));
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            matinv[i][j] /= det;
        }
    }
}

}
