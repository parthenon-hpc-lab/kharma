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

    int max_nstep_geo = pin->GetOrAddInteger("geodesics", "max_nstep", 50000); // ipole's value
    params.Add("max_nstep_geo", max_nstep_geo);
    int max_nstep_rad = pin->GetOrAddInteger("geodesics", "max_nstep_rad", max_nstep_geo); // separate max radiative steps
    params.Add("max_nstep_rad", max_nstep_rad);
    // Max steps per kernel invocation, otherwise long geodesics ruin load balancing
    int max_kernel_nstep_geo = pin->GetOrAddInteger("geodesics", "max_kernel_nstep", 100); // ipole's value
    params.Add("max_kernel_nstep_geo", max_kernel_nstep_geo);
    int max_kernel_nstep_rad = pin->GetOrAddInteger("geodesics", "max_kernel_nstep_rad", max_kernel_nstep_geo); // separate max radiative steps
    params.Add("max_kernel_nstep_rad", max_kernel_nstep_rad);

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
    pkg->AddSwarmValue(rays::t::name(), swarm_name, swarmreal);
    // Path length
    pkg->AddSwarmValue(rays::path_len::name(), swarm_name, swarmreal);

    Metadata swarmint({Metadata::Integer});
    // Keep track of host camera
    pkg->AddSwarmValue(rays::camera_id::name(), swarm_name, swarmint);
    // And pixel location
    pkg->AddSwarmValue(rays::camera_i::name(), swarm_name, swarmint);
    pkg->AddSwarmValue(rays::camera_j::name(), swarm_name, swarmint);
    // Number of geodesic steps
    pkg->AddSwarmValue(rays::nstep_geo::name(), swarm_name, swarmint);
    // Number of radiative steps
    pkg->AddSwarmValue(rays::nstep_rad::name(), swarm_name, swarmint);
    //pkg->AddSwarmValue("nstep_rad", swarm_name, swarmint); // please make this separate
    // Number of geodesic steps
    pkg->AddSwarmValue(rays::stop_flag::name(), swarm_name, swarmint);
    pkg->AddSwarmValue(rays::at_camera::name(), swarm_name, swarmint);

    // Wavevector 4-vector
    Metadata swarmvec({Metadata::Real}, std::vector<int>{4});
    pkg->AddSwarmValue(rays::k::name(), swarm_name, swarmvec);

    if (store_paths) {
        // Ray path X, K
        Metadata swarmpath({Metadata::Real}, std::vector<int>{GR_DIM*(max_nstep_geo+1)});
        Metadata swarmpathlen({Metadata::Real}, std::vector<int>{max_nstep_geo+1});
        pkg->AddSwarmValue(rays::xpath::name(), swarm_name, swarmpath);
        pkg->AddSwarmValue(rays::kpath::name(), swarm_name, swarmpath);
        pkg->AddSwarmValue(rays::dlpath::name(), swarm_name, swarmpathlen);
    }

    // Radiation parameters
    pkg->AddSwarmValue(rays::I::name(), swarm_name, swarmreal);
    if (polarized) { // TODO if pol etc
        // Full radiation tensor N (Gammie & Leung)
        // We'll always pull the stokes parameters from these locally
        Metadata swarmtensor({Metadata::Real}, std::vector<int>{4,4});
        pkg->AddSwarmValue(rays::Nr::name(), swarm_name, swarmtensor);
        pkg->AddSwarmValue(rays::Ni::name(), swarm_name, swarmtensor);
    }

    // Dummy "primitive" variable to coax parthenon to pack our precious geometry
    if (!packages->AllPackages().count("File")) {
        auto m = Metadata({Metadata::Cell, Metadata::Real, Metadata::Derived, Metadata::OneCopy,
                            Metadata::GetUserFlag("Primitive"), Metadata::Restart});
        pkg->AddField("prims.rho.fake", m);
    }

    return pkg;
}

// Count remaining particles at block edges rather than their destinations
template<bool emission>
int Rays::CountLiveParticles(MeshData<Real>* md)
{
    PARTHENON_INSTRUMENT

    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // const auto mesh_size = pmb0->pmy_mesh->mesh_size;
    // const Real x_max_mesh = mesh_size.xmax(X1DIR);
    // const int ng = Globals::nghost;

    std::string swarm_name = "rays";
    static auto desc = MakeSwarmPackDescriptor<rays::t,
                                               swarm_position::x,
                                               swarm_position::y,
                                               swarm_position::z,
                                               rays::path_len>(swarm_name);
    auto pack = desc.GetPack(md);
    static auto int_desc = MakeSwarmPackDescriptor<rays::stop_flag,
                                                   rays::at_camera>(swarm_name);
    auto int_pack = int_desc.GetPack(md);

    int live = 0;
    pmb0->par_reduce(PARTHENON_AUTO_LABEL, 0, int_pack.GetMaxFlatIndex(),
        KOKKOS_LAMBDA(const int &idx, int &local_result) {
            auto [b, n] = int_pack.GetBlockParticleIndices(idx);
            const auto swarm_d = int_pack.GetContext(b);
            if (swarm_d.IsActive(n)) {
                // Now: count everything not marked as finished
                if constexpr (emission) {
                    // Emission rays are retired when they reach the camera (or where it should be)
                    local_result += !int_pack(b, rays::at_camera(), n);
                } else {
                    // A geodesic stop flag not set (still zero) is an active particle
                    local_result += !int_pack(b, rays::stop_flag(), n);
                }
            }
        }
    , Kokkos::Sum<int>(live));

    // If there's one outlier print it
    if (live == 1) {
        pmb0->par_for(PARTHENON_AUTO_LABEL, 0, int_pack.GetMaxFlatIndex(),
            KOKKOS_LAMBDA(const int &idx) {
                auto [b, n] = int_pack.GetBlockParticleIndices(idx);
                const auto swarm_d = int_pack.GetContext(b);
                if (swarm_d.IsActive(n)) {
                    bool print;
                    if constexpr (emission) {
                        print = !int_pack(b, rays::at_camera(), n);
                    } else {
                        print = !int_pack(b, rays::stop_flag(), n);
                    }
                    if (print) {
                        printf("Loner: X: %f %f %f Len: %f\n", pack(b, swarm_position::x(), n),
                                                            pack(b, swarm_position::y(), n),
                                                            pack(b, swarm_position::z(), n),
                                                            pack(b, rays::path_len(), n));
                    }
                }
            }
        );
    }

    std::cout << live << " active particles remaining." << std::endl;

    // Hack to stop runaway/non-moving particles
    // Have to reset it for trace back!
    static int loner_cycles = 0;
    if (live >= 100) loner_cycles = 0;
    if (live <= 2) loner_cycles += 1;
    if (loner_cycles > 5) {
        return 0;
    }

    return live;
}
// There are only two
template int Rays::CountLiveParticles<true>(MeshData<Real>* md);
template int Rays::CountLiveParticles<false>(MeshData<Real>* md);

// Solve just the geodesic equation backward, until defined stopping points incl. block bounds
TaskStatus Rays::TraceGeodesicsBlock(MeshData<Real>* md)
{
    PARTHENON_INSTRUMENT

    // TODO grab pmesh only
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    auto pkg = pmb0->packages.Get("Rays");
    const auto max_nstep = pkg->Param<int>("max_nstep_geo");
    const auto max_kernel_nstep = pkg->Param<int>("max_kernel_nstep_geo");
    const auto eps = pkg->Param<GReal>("eps");
    const auto conn_delta = pkg->Param<GReal>("conn_delta");
    const auto polarized = pkg->Param<bool>("polarized");
    const auto store_paths = pkg->Param<bool>("store_paths");

    const auto stop_condition = pmb0->packages.Get("Model")->Param<Model::StopCondition>("stop_condition");

    std::string swarm_name = "rays";
    static auto desc = MakeSwarmPackDescriptor<rays::t,
                                              swarm_position::x,
                                              swarm_position::y,
                                              swarm_position::z,
                                              rays::path_len,
                                              // Vectors
                                              rays::k,
                                              rays::Nr,
                                              rays::Ni,
                                              rays::xpath,
                                              rays::kpath,
                                              rays::dlpath>(swarm_name);
    auto pack = desc.GetPack(md);

    // If running with MPI, sometimes our rank has no rays right now
    if (pack.GetMaxFlatIndex() < 0) return TaskStatus::complete;

    static auto int_desc = MakeSwarmPackDescriptor<rays::nstep_geo,
                                                   rays::stop_flag>(swarm_name);
    auto int_pack = int_desc.GetPack(md);

    // TODO coordinates in swarmpacks
    auto P = md->PackVariables(std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")});

    const Real x_max_mesh = pmb0->pmy_mesh->mesh_size.xmax(X1DIR);
    const int ng = Globals::nghost;

    // This load is extremely unbalanced, so we chunk out all rays separately
    auto policyBlock = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>(DevExecSpace(),
                                            0, pack.GetMaxFlatIndex(), Kokkos::ChunkSize(1));
    Kokkos::parallel_for(PARTHENON_AUTO_LABEL, policyBlock,
        KOKKOS_LAMBDA(const int &idx) {
            auto [b, n] = pack.GetBlockParticleIndices(idx);
            const auto swarm_d = pack.GetContext(b);
            if (swarm_d.IsActive(n) && !int_pack(b, rays::stop_flag(), n)) {
                // Grab pointers
                const GRCoordinates &G = P.GetCoords(b);
                const CoordinateEmbedding &coords = G.coords;
                // Grab position and wavevector
                double X[GR_DIM] = {pack(b, rays::t(), n),
                                    pack(b, swarm_position::x(), n),
                                    pack(b, swarm_position::y(), n),
                                    pack(b, swarm_position::z(), n)};
                double Kcon[GR_DIM] = {pack(b, rays::k(0), n),
                                    pack(b, rays::k(1), n),
                                    pack(b, rays::k(2), n),
                                    pack(b, rays::k(3), n)};

                // Need the local number to load balance, global for stopping
                int nstep_local = 0;
                int nstep = int_pack(b, rays::nstep_geo(), n);
                if (store_paths) {
                    // nstep zero
                    DLOOP1 pack(b, rays::xpath(4*nstep + mu), n) = X[mu];
                    DLOOP1 pack(b, rays::kpath(4*nstep + mu), n) = Kcon[mu];
                }

                double Xi[GR_DIM]; // need Kconi?

                // ======================== BEGIN ITERATION ========================
                do {
                    // We need this locally and stored
                    nstep++;
                    nstep_local++;

                    DLOOP1 Xi[mu] = X[mu];

                    double dl = geodesic_step(coords, X, Kcon, eps, conn_delta, true);
                    pack(b, rays::path_len(), n) += dl;

                    if (store_paths) {
                        DLOOP1 pack(b, rays::xpath(4*nstep + mu), n) = X[mu];
                        DLOOP1 pack(b, rays::kpath(4*nstep + mu), n) = Kcon[mu];
                        pack(b, rays::dlpath(nstep), n) = dl;
                    }

                } while (!Model::stop_backward_integration(Xi[1], Xi[2], Xi[3], X[1], X[2], X[3],
                                                        Kcon[1], stop_condition, int_pack(b, rays::stop_flag(), n))
                    && nstep < max_nstep && nstep_local < max_kernel_nstep
                    && in_block(G, X, x_max_mesh));
                // ======================== END ITERATION ========================

                // Write back nstep
                int_pack(b, rays::nstep_geo(), n) = nstep;
                // Set the new position and wavevector
                pack(b, rays::t(), n) = X[0];
                pack(b, swarm_position::x(), n) = X[1];
                pack(b, swarm_position::y(), n) = X[2];
                pack(b, swarm_position::z(), n) = phi_of(X[3]);
                DLOOP1 pack(b, rays::k(mu), n) = Kcon[mu];
            }
        }
    );
    Kokkos::fence();
    std::cerr << "Finished block of geodesics" << std::endl;

    return TaskStatus::complete;
}

// Solve the unpolarized transfer equation
template<bool store_paths>
TaskStatus Rays::TraceEmissionBlock(MeshData<Real>* md)
{
    PARTHENON_INSTRUMENT

    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    auto pkg = pmb0->packages.Get("Rays");
    const auto max_nstep = pkg->Param<int>("max_nstep_rad");
    const auto max_kernel_nstep = pkg->Param<int>("max_kernel_nstep_rad");
    const auto eps = pkg->Param<GReal>("eps");
    const auto conn_delta = pkg->Param<GReal>("conn_delta");
    const auto polarized = pkg->Param<bool>("polarized");

    auto &model = pmb0->packages.Get("Model")->AllParams();
    const auto L_unit = model.Get<double>("L_unit");
    const auto RHO_unit = model.Get<double>("RHO_unit");
    const auto B_unit = model.Get<double>("B_unit");
    const auto radiative = model.Get<bool>("radiative");
    const auto stop_condition = model.Get<Model::StopCondition>("stop_condition");

    const Real x_max_mesh = pmb0->pmy_mesh->mesh_size.xmax(X1DIR);
    const int ng = Globals::nghost;

    // Intensity is conserved in vacuum
    if (!polarized && !radiative) {
        return TaskStatus::complete;
    }

    std::string swarm_name = "rays";
    static auto desc = MakeSwarmPackDescriptor<rays::t,
                                              swarm_position::x,
                                              swarm_position::y,
                                              swarm_position::z,
                                              rays::path_len,
                                              // Vectors
                                              rays::k,
                                              rays::Nr,
                                              rays::Ni,
                                              rays::xpath,
                                              rays::kpath,
                                              rays::dlpath>(swarm_name);
    auto pack = desc.GetPack(md);

    // If running with MPI, sometimes our rank has no rays right now
    if (pack.GetMaxFlatIndex() < 0) return TaskStatus::complete;

    static auto int_desc = MakeSwarmPackDescriptor<rays::nstep_geo,
                                                   rays::nstep_rad,
                                                   rays::at_camera>(swarm_name);
    auto int_pack = int_desc.GetPack(md);

    PackIndexMap prims_map;
    auto P = md->PackVariables(std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prims_map);
    const VarMap m_p(prims_map, false);
    //printf("P size: %d %d %d %d\n", P.GetDim(1), P.GetDim(2), P.GetDim(3), P.GetDim(4));

    // This load is extremely unbalanced.
    auto policyBlock = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>(DevExecSpace(),
                                                        0, pack.GetMaxFlatIndex(), Kokkos::ChunkSize(1));
    Kokkos::parallel_for(PARTHENON_AUTO_LABEL, policyBlock,
        KOKKOS_LAMBDA(const int &idx) {
            auto [b, n] = pack.GetBlockParticleIndices(idx);
            const auto swarm_d = pack.GetContext(b);
            if (swarm_d.IsActive(n) && !int_pack(b, rays::at_camera(), n)) {
                // Grab pointers
                const GRCoordinates &G = P.GetCoords(b);
                const CoordinateEmbedding &coords = G.coords;

                // Read N and nsteps
                Kokkos::complex<double> N_coord[GR_DIM][GR_DIM];
                if (polarized)
                    read_N(pack, b, n, N_coord);
                int rstep = int_pack(b, rays::nstep_rad(), n);
                const int gstep = int_pack(b, rays::nstep_geo(), n);
                int nstep_local = 0;

                // Declare X and K, reading is specific to algorithm
                double X[GR_DIM], Kcon[GR_DIM];

                // Evolve N
                if constexpr (store_paths) {
                    double Xi[GR_DIM], Kconi[GR_DIM];
                    // X and K will be final values "Xf"/"Kf"
                    // TODO does if constexpr actually eliminate the register use?

                    do {
                        // Step backwards from the last geometric step
                        const int step = gstep-1 - rstep;
                        DLOOP1 Xi[mu] = pack(b, rays::xpath(4*step + mu), n);
                        DLOOP1 Kconi[mu] = pack(b, rays::kpath(4*step + mu), n);
                        const int next = step - 1; // One step toward camera
                        DLOOP1 X[mu] = pack(b, rays::xpath(4*next + mu), n);
                        DLOOP1 Kcon[mu] = pack(b, rays::kpath(4*next + mu), n);
                        // printf("Path %d step %d: %g %g %g %g to %g %g %g %g, %g %g %g %g to %g %g %g %g\n", n, step,
                        //         Xi[0], Xi[1], Xi[2], Xi[3], X[0], X[1], X[2], X[3],
                        //         Kconi[0], Kconi[1], Kconi[2], Kconi[3], Kcon[0], Kcon[1], Kcon[2], Kcon[3]);

                        if (polarized)
                            path_parallel_transport_step(coords, Xi, Kconi, X, Kcon, N_coord, conn_delta, pack(b, rays::dlpath(step), n));

                        // Only process radiation on the mesh domain -- outside it any interpolation means nothing
                        if (radiative && X[1] > G.Xf<X1DIR>(ng) && X[1] < G.Xf<X1DIR>(G.n1-ng) &&
                            X[1] > stop_condition.x1min_geo && X[1] < stop_condition.x1max_geo &&
                            X[2] > G.Xf<X2DIR>(ng)+0.01 && X[2] < G.Xf<X2DIR>(G.n2-ng)-0.01) {
                            // Add units to dl to process radiation
                            // TODO template this for unpolarized transport etc w/speed
                            Model::process_radiation(G, N_coord, X, Kcon, P(b), m_p, RHO_unit, B_unit,
                                                    pack(b, rays::dlpath(step), n) * L_unit * HPL / (ME * CL * CL), false);
                        }

                        rstep++;
                        nstep_local++;

                    } while (rstep < gstep && rstep < max_nstep && nstep_local < max_kernel_nstep && in_block(G, X, x_max_mesh));
                } else {
                    // Set initial values
                    X[0] = pack(b, rays::t(), n);
                    X[1] = pack(b, swarm_position::x(), n);
                    X[2] = pack(b, swarm_position::y(), n);
                    X[3] = pack(b, swarm_position::z(), n);
                    DLOOP1 Kcon[mu] = pack(b, rays::k(mu), n);

                    do {
                        // Evolve X, Kcon to get subsequent values
                        double dl = geodesic_parallel_transport_step(coords, X, Kcon, N_coord, eps, conn_delta, false);

                        // Only process radiation on the mesh domain -- outside it any interpolation means nothing
                        if (radiative && X[1] > G.Xf<X1DIR>(ng) && X[1] < G.Xf<X1DIR>(G.n1-ng) &&
                            X[1] > stop_condition.x1min_geo && X[1] < stop_condition.x1max_geo &&
                            X[2] > G.Xf<X2DIR>(ng)+0.01 && X[2] < G.Xf<X2DIR>(G.n2-ng)-0.01) {
                            // Add units to dl to process radiation
                            Model::process_radiation(G, N_coord, X, Kcon, P(b), m_p, RHO_unit, B_unit,
                                                    dl * L_unit * HPL / (ME * CL * CL), false);
                        }

                        rstep++;
                        nstep_local++;

                        // TODO also stop at camera radius etc.
                    } while (rstep < gstep && rstep < max_nstep && nstep_local < max_kernel_nstep && in_block(G, X, x_max_mesh));
                }

                // Write X and K
                pack(b, rays::t(), n) = X[0];
                pack(b, swarm_position::x(), n) = X[1];
                pack(b, swarm_position::y(), n) = X[2];
                pack(b, swarm_position::z(), n) = phi_of(X[3]);
                DLOOP1 pack(b, rays::k(mu), n) = Kcon[mu];

                // Write N and nstep, record if we arrived
                if (polarized)
                    write_N(N_coord, pack, b, n);
                int_pack(b, rays::nstep_rad(), n) = rstep;
                if (rstep == gstep || rstep == max_nstep)
                    int_pack(b, rays::at_camera(), n) = 1;
            }
        }
    );

    // Make sure we're (a) finished and (b) honest about timings
    Kokkos::fence();
    std::cerr << "Finished block of emission" << std::endl;

    return TaskStatus::complete;
}
template TaskStatus Rays::TraceEmissionBlock<true>(MeshData<Real>* md);
template TaskStatus Rays::TraceEmissionBlock<false>(MeshData<Real>* md);
