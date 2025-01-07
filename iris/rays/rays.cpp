/*
 *  File: rays.cpp
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
#include "rays.hpp"

// Iris headers
#include "model.hpp"
#include "tetrads.hpp"
#include "units.hpp"

#include <parthenon/parthenon.hpp>

std::shared_ptr<StateDescriptor> Rays::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Rays");
    Params &params = pkg->AllParams();

    int max_nstep = pin->GetOrAddInteger("geodesics", "max_nstep", 50000); // ipole's value
    params.Add("max_nstep", max_nstep);
    GReal eps = pin->GetOrAddReal("geodesics", "eps", 0.01); // ipole's value
    params.Add("eps", eps);
    GReal conn_delta = pin->GetOrAddReal("geodesics", "conn_delta", 1e-7); // ipole's value
    params.Add("conn_delta", conn_delta);
    bool store_paths = !pin->GetOrAddBoolean("geodesics", "recalculate_paths", false);
    params.Add("store_paths", store_paths);

    // Store polarized *transport* separaete from *model* in case we want to 
    bool polarized = pin->GetBoolean("model", "polarized");
    params.Add("polarized", polarized);

    // Add swarm of tracer particles
    std::string swarm_name = "rays";
    Metadata swarm_metadata({Metadata::Provides, Metadata::None});
    pkg->AddSwarm(swarm_name, swarm_metadata);
    // Time/x0.  Traditional name from Parthenon codes...
    Metadata swarmreal({Metadata::Real});
    pkg->AddSwarmValue("t", swarm_name, swarmreal);
    // Path length
    pkg->AddSwarmValue("path_len", swarm_name, swarmreal);

    Metadata swarmint({Metadata::Integer});
    // Keep track of host camera
    pkg->AddSwarmValue("camera_id", swarm_name, swarmint);
    // And pixel location
    pkg->AddSwarmValue("i", swarm_name, swarmint);
    pkg->AddSwarmValue("j", swarm_name, swarmint);
    // Number of geodesic steps
    pkg->AddSwarmValue("nstep_geo", swarm_name, swarmint);
    //pkg->AddSwarmValue("nstep_rad", swarm_name, swarmint); // please make this separate
    // Number of geodesic steps
    pkg->AddSwarmValue("stop_flag", swarm_name, swarmint);

    // Wavevector 4-vector
    Metadata swarmvec({Metadata::Real}, std::vector<int>{4});
    pkg->AddSwarmValue("k", swarm_name, swarmvec);

    if (store_paths) {
        // Ray path X, K
        Metadata swarmpath({Metadata::Real}, std::vector<int>{GR_DIM*(max_nstep+1)});
        Metadata swarmpathlen({Metadata::Real}, std::vector<int>{max_nstep+1});
        pkg->AddSwarmValue("xpath", swarm_name, swarmpath);
        pkg->AddSwarmValue("kpath", swarm_name, swarmpath);
        pkg->AddSwarmValue("dlpath", swarm_name, swarmpathlen);
    }

    // Radiation parameters
    pkg->AddSwarmValue("I", swarm_name, swarmreal);
    if (polarized) { // TODO if pol etc
        // Full radiation tensor N (Gammie & Leung)
        // We'll always pull the stokes parameters from these locally
        Metadata swarmtensor({Metadata::Real}, std::vector<int>{4,4});
        pkg->AddSwarmValue("Nr", swarm_name, swarmtensor);
        pkg->AddSwarmValue("Ni", swarm_name, swarmtensor);
    }

    return pkg;
}

// TODO Elliptic analytic solutions! Then we don't need to worry about trace-back!

// Solve just the geodesic equation backward, until defined stopping points
TaskStatus Rays::TraceGeodesicsUntilStop(MeshBlock* pmb)
{
    PARTHENON_INSTRUMENT

    auto swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get("rays");
    auto pkg = pmb->packages.Get("Rays");
    const auto max_nstep = pkg->Param<int>("max_nstep");
    const auto eps = pkg->Param<GReal>("eps");
    const auto conn_delta = pkg->Param<GReal>("conn_delta");
    const auto polarized = pkg->Param<bool>("polarized");
    const auto store_paths = pkg->Param<bool>("store_paths");

    const auto stop_condition = pmb->packages.Get("Model")->Param<Model::StopCondition>("stop_condition");

    int max_active_index = swarm->GetMaxActiveIndex();

    auto &x0 = swarm->Get<Real>("t").Get();
    auto &x1 = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &x2 = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &x3 = swarm->Get<Real>(swarm_position::z::name()).Get();
    auto &k = swarm->Get<Real>("k").Get();
    // Tracing data
    auto &nstep = swarm->Get<int>("nstep_geo").Get();
    auto &pathlen = swarm->Get<Real>("path_len").Get();
    auto &stopflag = swarm->Get<int>("stop_flag").Get();
    // Path data
    auto &xpath = (store_paths) ? swarm->Get<Real>("xpath").Get() : swarm->Get<Real>("k").Get();
    auto &kpath = (store_paths) ? swarm->Get<Real>("kpath").Get() : swarm->Get<Real>("k").Get();
    auto &dlpath = (store_paths) ? swarm->Get<Real>("dlpath").Get() : swarm->Get<Real>("k").Get();

    auto swarm_d = swarm->GetDeviceContext();

    const CoordinateEmbedding coords = pmb->coords.coords;

    // TODO implement better than RK2
    pmb->par_for(
        PARTHENON_AUTO_LABEL, 0, max_active_index, KOKKOS_LAMBDA(const int &n) {
            if (swarm_d.IsActive(n)) {
                nstep(n) = 0;
                pathlen(n) = 0.0;
                double X[GR_DIM] = {x0(n), x1(n), x2(n), x3(n)};
                double Kcon[GR_DIM] = {k(0, n), k(1, n), k(2, n), k(3, n)};
                if (store_paths) {
                    // nstep zero
                    DLOOP1 xpath(mu, n) = X[mu];
                    DLOOP1 kpath(mu, n) = Kcon[mu];
                }
                double Xi[GR_DIM]; // need Kconi?
                do {
                    nstep(n)++;
                    DLOOP1 Xi[mu] = X[mu];

                    double dl = geodesic_step(coords, X, Kcon, eps, conn_delta, true);
                    pathlen(n) += dl;

                    if (store_paths) {
                        DLOOP1 {
                            xpath(4*nstep(n) + mu, n) = X[mu];
                            kpath(4*nstep(n) + mu, n) = Kcon[mu];
                        }
                        dlpath(nstep(n), n) = dl;
                    }
                } while (!Model::stop_backward_integration(Xi[1], Xi[2], Xi[3], X[1], X[2], X[3],
                                                        Kcon[1], stop_condition, stopflag(n))
                    && nstep(n) < max_nstep);
                // Set the new position and wavevector
                x0(n) = X[0];
                x1(n) = X[1];
                x2(n) = X[2];
                x3(n) = X[3];
                k(0,n) = Kcon[0];
                k(1,n) = Kcon[1];
                k(2,n) = Kcon[2];
                k(3,n) = Kcon[3];
            }
        }
    );
    Kokkos::fence();
    std::cerr << "Finished back trace" << std::endl;

    return TaskStatus::complete;
}

// Solve the unpolarized transfer equation
TaskStatus Rays::TraceRaysToCameras(MeshBlock* pmb)
{
    PARTHENON_INSTRUMENT

    auto swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get("rays");
    auto pkg = pmb->packages.Get("Rays");
    const auto max_nstep = pkg->Param<int>("max_nstep");
    const auto eps = pkg->Param<GReal>("eps");
    const auto conn_delta = pkg->Param<GReal>("conn_delta");
    const auto polarized = pkg->Param<bool>("polarized");
    const auto store_paths = pkg->Param<bool>("store_paths");

    auto &model = pmb->packages.Get("Model")->AllParams();
    const auto L_unit = model.Get<double>("L_unit");
    const auto RHO_unit = model.Get<double>("RHO_unit");
    const auto B_unit = model.Get<double>("B_unit");
    const auto radiative = model.Get<bool>("radiative");
    const auto stop_condition = model.Get<Model::StopCondition>("stop_condition");

    // Intensity is conserved in vacuum
    if (!polarized && !radiative) {
        return TaskStatus::complete;
    }

    int max_active_index = swarm->GetMaxActiveIndex();

    auto &x0 = swarm->Get<Real>("t").Get();
    auto &x1 = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &x2 = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &x3 = swarm->Get<Real>(swarm_position::z::name()).Get();
    auto &k = swarm->Get<Real>("k").Get();
    // Radiation
    auto &Nr = swarm->Get<Real>("Nr").Get();
    auto &Ni = swarm->Get<Real>("Ni").Get();
    // Tracing data
    auto &nstep = swarm->Get<int>("nstep_geo").Get();
    // Path data
    auto &xpath = (store_paths) ? swarm->Get<Real>("xpath").Get() : swarm->Get<Real>("k").Get();
    auto &kpath = (store_paths) ? swarm->Get<Real>("kpath").Get() : swarm->Get<Real>("k").Get();
    auto &dlpath = (store_paths) ? swarm->Get<Real>("dlpath").Get() : swarm->Get<Real>("k").Get();

    auto swarm_d = swarm->GetDeviceContext();

    const GRCoordinates &G = pmb->coords;
    const CoordinateEmbedding &coords = pmb->coords.coords;

    PackIndexMap prims_map;
    auto P = pmb->meshblock_data.Get()->PackVariables(std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prims_map);
    const VarMap m_p(prims_map, false);
    printf("P size: %d %d %d %d\n", P.GetDim(1), P.GetDim(2), P.GetDim(3), P.GetDim(4));

    // TODO implement better than RK2
    pmb->par_for(PARTHENON_AUTO_LABEL, 0, max_active_index,
        KOKKOS_LAMBDA(const int &n) {
            if (swarm_d.IsActive(n)) {
                // Read N
                Kokkos::complex<double> N_coord[GR_DIM][GR_DIM];
                read_N(Nr, Ni, n, N_coord);

                // Evolve N
                if (store_paths) {
                    for (int step = nstep(n); step >= 1; step--) {
                        double Xi[GR_DIM]    = {xpath(4*step, n), xpath(4*step+1, n), xpath(4*step+2, n), xpath(4*step+3, n)};
                        double Kconi[GR_DIM] = {kpath(4*step, n), kpath(4*step+1, n), kpath(4*step+2, n), kpath(4*step+3, n)};
                        double Xf[GR_DIM]    = {xpath(4*(step-1), n), xpath(4*(step-1)+1, n), xpath(4*(step-1)+2, n), xpath(4*(step-1)+3, n)};
                        double Kconf[GR_DIM] = {kpath(4*(step-1), n), kpath(4*(step-1)+1, n), kpath(4*(step-1)+2, n), kpath(4*(step-1)+3, n)};
                        // printf("Path %d step %d: %g %g %g %g to %g %g %g %g, %g %g %g %g to %g %g %g %g\n", n, step,
                        //         Xi[0], Xi[1], Xi[2], Xi[3], Xf[0], Xf[1], Xf[2], Xf[3],
                        //         Kconi[0], Kconi[1], Kconi[2], Kconi[3], Kconf[0], Kconf[1], Kconf[2], Kconf[3]);
                        path_parallel_transport_step(coords, Xi, Kconi, Xf, Kconf, N_coord, conn_delta, dlpath(step, n));

                        if (radiative && Xf[1] > stop_condition.x1min_geo && Xf[1] < stop_condition.x1max_geo &&
                            Xf[2] > coords.startx(2)+0.01 && Xf[2] < coords.stopx(2)-0.01) {
                            // Add units to dl to process radiation
                            Model::process_radiation(G, N_coord, Xf, Kconf, P, m_p, RHO_unit, B_unit,
                                                    dlpath(step, n) * L_unit * HPL / (ME * CL * CL), false);
                        }
                    }
                    // Last write
                    x0(n) = xpath(0, n);
                    x1(n) = xpath(1, n);
                    x2(n) = xpath(2, n);
                    x3(n) = xpath(3, n);
                    k(0,n) = kpath(0, n);
                    k(1,n) = kpath(1, n);
                    k(2,n) = kpath(2, n);
                    k(3,n) = kpath(3, n);
                } else {
                    int nstep_back = 0;
                    double X[GR_DIM] = {x0(n), x1(n), x2(n), x3(n)};
                    double Kcon[GR_DIM] = {k(0, n), k(1, n), k(2, n), k(3, n)};

                    do {
                        nstep_back++;

                        double dl = geodesic_parallel_transport_step(coords, X, Kcon, N_coord, eps, conn_delta, false);

                        if (radiative && X[1] > stop_condition.x1min_geo && X[1] < stop_condition.x1max_geo &&
                            X[2] > coords.startx(2)+0.01 && X[2] < coords.stopx(2)-0.01) {
                            // Add units to dl to process radiation
                            Model::process_radiation(G, N_coord, X, Kcon, P, m_p, RHO_unit, B_unit,
                                                    dl * L_unit * HPL / (ME * CL * CL), false);
                        }

                    } while (nstep_back < nstep(n)); // Or stop at camera radius

                    // Last write
                    x0(n) = X[0];
                    x1(n) = X[1];
                    x2(n) = X[2];
                    x3(n) = X[3];
                    k(0,n) = Kcon[0];
                    k(1,n) = Kcon[1];
                    k(2,n) = Kcon[2];
                    k(3,n) = Kcon[3];
                }

                // Write N
                write_N(N_coord, Nr, Ni, n);
            }
        }
    );

    // Make sure we're (a) finished and (b) honest about timings
    Kokkos::fence();

    return TaskStatus::complete;
}
