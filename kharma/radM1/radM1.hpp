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
#include "utils/constants.hpp"

#include <parthenon/parthenon.hpp>

using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;

namespace RadM1
{

struct UnitScales {
    Real length_cgs;
    Real time_cgs;
    Real mass_cgs;
    Real energy_cgs;
    Real temperature_cgs;
};

// Denote implicit solve failures (rflags)
// This enum should grow to cover any potential flags
enum class StatusImplicitStep { success = 0, mhdsolve, radsolve, bothsolve, failure, onedfallback };

static const std::map<int, std::string> status_names = {
    {(int)StatusImplicitStep::mhdsolve,
        "RadM1 MHD Solve Failure"}, // flag that means that the MHD inversion failed (but
                                    // rad solve worked)
    {(int)StatusImplicitStep::radsolve,
        "RadM1 Radiation Solve Failure"}, // flag that means that the radiation solve
                                          // failed (but mhd solve worked)
    {(int)StatusImplicitStep::failure, "RadM1 Step Failure"},
    {(int)StatusImplicitStep::onedfallback,
        "RadM1 4D Solver Fell Back to 1D"}}; // flag that means the 4D Newton solve didn't
                                              // converge/failed and the 1D fallback solver
                                              // was used instead

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
TaskStatus Step(MeshData<Real>* md_sub_init,
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

// Opacity model selector for calc_kabs/calc_kscattering/compute_covariant_fourforce.
enum class OpacityModel : int { Default = 0, ShocktubeConstant = 1, Bondi = 2, BeamLight = 3};


KOKKOS_INLINE_FUNCTION Real calc_kabs(Real rho, Real T, int opacity_model,
    Real shocktube_kappa_rho, const UnitScales& units_cgs)
{
    if (opacity_model == (int)OpacityModel::ShocktubeConstant) {
        return m::min(rho * shocktube_kappa_rho, 1.e5);
    } else if (opacity_model == (int)OpacityModel::Bondi) {
        // Thermal bremsstrahlung, McKinney et al. 2014 eq. 91.
        const Real T_cgs = m::abs(T) * units_cgs.temperature_cgs;
        const Real rho_cgs = rho * units_cgs.mass_cgs / (units_cgs.length_cgs * units_cgs.length_cgs * units_cgs.length_cgs);
        const Real kappa_a_cgs =
            1.7e-25 * m::pow(T_cgs, -3.5) * m::pow(pc::mp, -2.0) * rho_cgs * rho_cgs;
        //make it scale free
        return kappa_a_cgs * units_cgs.length_cgs;
    } else if (opacity_model == (int)OpacityModel::BeamLight) {
        return 0.0;
    
    } else {
        // TODO: singularity-opac
        return m::min(rho * 0.08, 1.e5);
    }
}

KOKKOS_INLINE_FUNCTION Real calc_kscattering(Real rho, Real T, int opacity_model,
    Real shocktube_kappa_scat, const UnitScales& units_cgs)
{
    if (opacity_model == (int)OpacityModel::ShocktubeConstant) {
        return shocktube_kappa_scat;
    } else if (opacity_model == (int)OpacityModel::BeamLight) {
        return 0.0;
    } else if (opacity_model == (int)OpacityModel::Bondi) {
        const Real rho_cgs = rho * units_cgs.mass_cgs / (units_cgs.length_cgs * units_cgs.length_cgs * units_cgs.length_cgs);
        const Real kappa_sc_cgs = 0.4 * rho_cgs;
        //make it scale free
        return kappa_sc_cgs * units_cgs.length_cgs;
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

}
