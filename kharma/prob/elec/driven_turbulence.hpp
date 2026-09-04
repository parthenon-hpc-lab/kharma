/*
 *  File: driven_turbulence.hpp
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
#include "gaussian.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

using namespace parthenon;

TaskStatus InitializeDrivenTurbulence(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;

    const Real gam = pmb->packages.Get("eos")->Param<Real>("gm1") + 1.0;
    const Real rho0 = pin->GetOrAddReal("driven_turbulence", "rho", 1.0);
    const Real cs0 = pin->GetOrAddReal("driven_turbulence", "cs0", 8.6e-4);
    const Real dt_kick = pin->GetOrAddReal("driven_turbulence", "dt_kick", 1);
    const Real edot_frac = pin->GetOrAddReal("driven_turbulence", "edot_frac", 0.5);
    const Real x1min = pin->GetOrAddReal("parthenon/mesh", "x1min", 0);
    const Real x1max = pin->GetOrAddReal("parthenon/mesh", "x1max", 1);
    const Real x2min = pin->GetOrAddReal("parthenon/mesh", "x2min", 0);
    const Real x2max = pin->GetOrAddReal("parthenon/mesh", "x2max", 1);
    const Real x3min = pin->GetOrAddReal("parthenon/mesh", "x3min", -1);
    const Real x3max = pin->GetOrAddReal("parthenon/mesh", "x3max", 1);

    const Real edot = edot_frac * rho0 * pow(cs0, 3);
    const Real counter = 0.;
    if (!(pmb->packages.Get("GRMHD")->AllParams().hasKey("drive_edot")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("drive_edot", edot);
    if (!(pmb->packages.Get("GRMHD")->AllParams().hasKey("counter")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("counter", counter, true);
    const Real lx1 = x1max - x1min;
    const Real lx2 = x2max - x2min;
    if (!(pmb->packages.Get("GRMHD")->AllParams().hasKey("lx1")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("lx1", lx1);
    if (!(pmb->packages.Get("GRMHD")->AllParams().hasKey("lx2")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("lx2", lx2);
    // adding for later use in create_grf
    if (!(pmb->packages.Get("GRMHD")->AllParams().hasKey("dt_kick")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("dt_kick", dt_kick);

    const Real u0 = cs0 * cs0 * rho0 / (gam - 1) / gam; // from flux_functions.hpp
    IndexRange myib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange myjb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange mykb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    pmb->par_for("driven_turb_rho_u_init", mykb.s, mykb.e, myjb.s, myjb.e, myib.s, myib.e,
                 KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            rho(k, j, i) = rho0;
            u(k, j, i) = u0;
        });

    return TaskStatus::complete;
}

/**
 * This applies a turbulent Gaussian random "kick" every dt_kick units of simulation time
 * It is only called after the last sub-step, so this splits nicely with the fluid
 * evolution operator.
 */
void ApplyDrivingTurbulence(MeshBlockData<Real>* rc)
{
    auto pmb = rc->GetBlockPointer();
    const IndexRange myib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    const IndexRange myjb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange mykb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    // Gaussian random field:
    const auto& G = pmb->coords;
    GridScalar rho = rc->Get("prims.rho").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    GridVector B_P = rc->Get("prims.B").data;
    GridVector grf_normalized = rc->Get("grf_normalized").data;
    const Real t = pmb->packages.Get("Globals")->Param<Real>("time");
    Real counter = pmb->packages.Get("GRMHD")->Param<Real>("counter");
    const Real dt_kick = pmb->packages.Get("GRMHD")->Param<Real>("dt_kick");
    if (counter < t) {
        counter += dt_kick;
        pmb->packages.Get("GRMHD")->UpdateParam<Real>("counter", counter);
        printf("Kick applied at time %.32f\n", t);

        const Real lx1 = pmb->packages.Get("GRMHD")->Param<Real>("lx1");
        const Real lx2 = pmb->packages.Get("GRMHD")->Param<Real>("lx2");
        const Real edot = pmb->packages.Get("GRMHD")->Param<Real>("drive_edot");
        GridScalar alfven_speed = rc->Get("alfven_speed").data;

        int Nx1 = pmb->cellbounds.ncellsi(IndexDomain::interior);
        int Nx2 = pmb->cellbounds.ncellsj(IndexDomain::interior);
        Real* dv0 = (Real*)malloc(sizeof(Real) * Nx1 * Nx2);
        Real* dv1 = (Real*)malloc(sizeof(Real) * Nx1 * Nx2);
        create_grf(Nx1, Nx2, lx1, lx2, dv0, dv1);

        Real mean_velocity_num0 = 0;
        Kokkos::Sum<Real> mean_velocity_num0_reducer(mean_velocity_num0);
        Real mean_velocity_num1 = 0;
        Kokkos::Sum<Real> mean_velocity_num1_reducer(mean_velocity_num1);
        Real tot_mass = 0;
        Kokkos::Sum<Real> tot_mass_reducer(tot_mass);
        pmb->par_reduce(
            "forced_mhd_normal_kick_centering_mean_vel0", mykb.s, mykb.e, myjb.s, myjb.e,
            myib.s, myib.e,
            KOKKOS_LAMBDA(const int& k, const int& j, const int& i, double& local_result)
            {
                Real cell_mass = (rho(k, j, i) * G.Dxc<3>(k) * G.Dxc<2>(j) * G.Dxc<1>(i));
                local_result += cell_mass * dv0[(i - 4) * Nx1 + (j - 4)];
            },
            mean_velocity_num0_reducer);
        pmb->par_reduce(
            "forced_mhd_normal_kick_centering_mean_vel1", mykb.s, mykb.e, myjb.s, myjb.e,
            myib.s, myib.e,
            KOKKOS_LAMBDA(const int& k, const int& j, const int& i, double& local_result)
            {
                Real cell_mass = (rho(k, j, i) * G.Dxc<3>(k) * G.Dxc<2>(j) * G.Dxc<1>(i));
                local_result += cell_mass * dv1[(i - 4) * Nx1 + (j - 4)];
            },
            mean_velocity_num1_reducer);
        pmb->par_reduce(
            "forced_mhd_normal_kick_centering_tot_mass", mykb.s, mykb.e, myjb.s, myjb.e,
            myib.s, myib.e,
            KOKKOS_LAMBDA(const int& k, const int& j, const int& i, double& local_result)
            {
                local_result += (rho(k, j, i) * G.Dxc<3>(k) * G.Dxc<2>(j) * G.Dxc<1>(i));
            },
            tot_mass_reducer);
        Real mean_velocity0 = mean_velocity_num0 / tot_mass;
        Real mean_velocity1 = mean_velocity_num1 / tot_mass;
#pragma omp parallel for simd collapse(2)
        for (size_t i = 0; i < Nx1; i++) {
            for (size_t j = 0; j < Nx2; j++) {
                dv0[i * Nx1 + j] -= mean_velocity0;
                dv1[i * Nx1 + j] -= mean_velocity1;
            }
        }

        Real Bhalf = 0;
        Real A = 0;
        Real init_e = 0;
        Kokkos::Sum<Real> Bhalf_reducer(Bhalf);
        Kokkos::Sum<Real> A_reducer(A);
        Kokkos::Sum<Real> init_e_reducer(init_e);
        pmb->par_reduce(
            "forced_mhd_normal_kick_normalization_Bhalf", mykb.s, mykb.e, myjb.s, myjb.e,
            myib.s, myib.e,
            KOKKOS_LAMBDA(const int& k, const int& j, const int& i, double& local_result)
            {
                Real cell_mass = (rho(k, j, i) * G.Dxc<3>(k) * G.Dxc<2>(j) * G.Dxc<1>(i));
                local_result +=
                    cell_mass * (dv0[(i - 4) * Nx1 + (j - 4)] * uvec(0, k, j, i) +
                                    dv1[(i - 4) * Nx1 + (j - 4)] * uvec(1, k, j, i));
            },
            Bhalf_reducer);
        pmb->par_reduce(
            "forced_mhd_normal_kick_normalization_A", mykb.s, mykb.e, myjb.s, myjb.e,
            myib.s, myib.e,
            KOKKOS_LAMBDA(const int& k, const int& j, const int& i, double& local_result)
            {
                Real cell_mass = (rho(k, j, i) * G.Dxc<3>(k) * G.Dxc<2>(j) * G.Dxc<1>(i));
                local_result += cell_mass * (pow(dv0[(i - 4) * Nx1 + (j - 4)], 2) +
                                                pow(dv1[(i - 4) * Nx1 + (j - 4)], 2));
            },
            A_reducer);
        pmb->par_reduce(
            "forced_mhd_normal_kick_init_e", mykb.s, mykb.e, myjb.s, myjb.e, myib.s,
            myib.e,
            KOKKOS_LAMBDA(const int& k, const int& j, const int& i, double& local_result)
            {
                Real cell_mass = (rho(k, j, i) * G.Dxc<3>(k) * G.Dxc<2>(j) * G.Dxc<1>(i));
                local_result += 0.5 * cell_mass *
                                (pow(uvec(0, k, j, i), 2) + pow(uvec(1, k, j, i), 2));
            },
            init_e_reducer);

        Real norm_const = (-Bhalf + pow(pow(Bhalf, 2) + A * 2 * dt_kick * edot, 0.5)) /
                          A; // going from k:(0, 0), j:(4, 515), i:(4, 515) inclusive
        pmb->par_for("forced_mhd_normal_kick_setting", mykb.s, mykb.e, myjb.s, myjb.e,
            myib.s, myib.e,
            KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
            {
                grf_normalized(0, k, j, i) = (dv0[(i - 4) * Nx1 + (j - 4)] * norm_const);
                grf_normalized(1, k, j, i) = (dv1[(i - 4) * Nx1 + (j - 4)] * norm_const);
                uvec(0, k, j, i) += grf_normalized(0, k, j, i);
                uvec(1, k, j, i) += grf_normalized(1, k, j, i);
                FourVectors Dtmp;
                GRMHD::calc_4vecs(G, uvec, B_P, k, j, i, Loci::center, Dtmp);
                Real bsq = dot(Dtmp.bcon, Dtmp.bcov);
                alfven_speed(k, j, i) =
                    bsq / rho(k, j, i); // saving alfven speed for analysis purposes
            });

        Real finl_e = 0;
        Kokkos::Sum<Real> finl_e_reducer(finl_e);
        pmb->par_reduce(
            "forced_mhd_normal_kick_finl_e", mykb.s, mykb.e, myjb.s, myjb.e, myib.s,
            myib.e,
            KOKKOS_LAMBDA(const int& k, const int& j, const int& i, double& local_result)
            {
                Real cell_mass = (rho(k, j, i) * G.Dxc<3>(k) * G.Dxc<2>(j) * G.Dxc<1>(i));
                local_result += 0.5 * cell_mass *
                                (pow(uvec(0, k, j, i), 2) + pow(uvec(1, k, j, i), 2));
            },
            finl_e_reducer);
        printf("%.32f\n", A);
        printf("%.32f\n", Bhalf);
        printf("%.32f\n", norm_const);
        printf("%.32f\n", (finl_e - init_e) / dt_kick);
        free(dv0);
        free(dv1);
    }
}
