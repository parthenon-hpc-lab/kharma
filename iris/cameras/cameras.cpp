/*
 *  File: cameras.cpp
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
#include "cameras.hpp"

// Iris headers
#include "camera_tetrads.hpp"
#include "rays.hpp"

#include <parthenon/parthenon.hpp>

std::shared_ptr<StateDescriptor> Cameras::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<StateDescriptor>("Cameras");
    Params &params = pkg->AllParams();

    // General options
    // We hope this shouldn't need to be set per-camera within a run...
    auto old_centering = pin->GetOrAddBoolean("camera", "use_old_centering", false);
    params.Add("old_centering", old_centering);

    auto polarized = pin->GetBoolean("model", "polarized");
    params.Add("polarized", polarized);

    auto verbose = pin->GetInteger("debug", "verbose");
    params.Add("verbose", verbose); // TODO stash in separate package?

    // Pull coordinates, which the model constructed for us
    auto coords = packages->Get("Model")->Param<CoordinateEmbedding>("coords");

    std::vector<std::string> vars = {"unpol", "nstep", "pathlen", "stopflag"};
    if (polarized) vars.push_back("stokes");
    pkg->AddParam("vars", vars);

    // Flag for camera variables, used for packing
    Metadata::AddUserFlag("Camera");

    // TODO basic multiple camera blocks camera1 etc
    // TODO if any parameter is list, repeat & generate new cameras...
    // TODO this allocs an output field for every MeshBlock.  Make them sparse or elminate
    std::vector<Camera> cameras;
    std::vector<std::string> camnames = {"camera"};
    for (auto camname : camnames) {
        if (pin->DoesBlockExist(camname)) {
            cameras.push_back(Camera(coords, pin, camname));
            auto camera = cameras.back();

            // For double-checking cameras.  Resembles ipole output
            if (verbose > 0) camera.print();

            // Ideally we'd want nstep and stopflag to be ints, but no support
            // We keep the reducers Real too for simplicity
            // We declare indices backward here to preserve forward indexing in kernels via ()
            Metadata image_meta = Metadata({Metadata::Real, Metadata::None, Metadata::Derived,
                                            Metadata::OneCopy, Metadata::GetUserFlag("Camera")},
                                            std::vector<int>{camera.ny, camera.nx});
            Metadata image_vector_meta = Metadata({Metadata::Real, Metadata::None, Metadata::Derived,
                                            Metadata::OneCopy, Metadata::GetUserFlag("Camera")},
                                            std::vector<int>{4, camera.ny, camera.nx});
            // Add reducers for all cameras among our MPI ranks (1 per rank!) to Params
            // Also add fields, which will be allocated on each block but only written at the end, on block 0
            for (auto var : vars) {
                bool vector = (var == "stokes");
                if (vector) {
                    HostArray3D<Real> camera_reduce =
                        HostArray3D<Real>(camname+"_reduce_"+var, camera.nx, camera.ny, 4);
                    pkg->AddParam(camname+"_reduce_"+var, camera_reduce, true);
                    pkg->AddField(camname+"_"+var, image_vector_meta);
                } else {
                    HostArray2D<Real> camera_reduce =
                        HostArray2D<Real>(camname+"_reduce_"+var, camera.nx, camera.ny);
                    pkg->AddParam(camname+"_reduce_"+var, camera_reduce, true);
                    pkg->AddField(camname+"_"+var, image_meta);
                }
            }
        } else {
            throw std::runtime_error("Camera not defined!");
            // for block in camera blocks...
        }
    }
    params.Add("camnames", camnames);
    params.Add("cameras", cameras);

    return pkg;
}

TaskStatus Cameras::InitGeodesics(MeshBlock* pmb)
{
    PARTHENON_INSTRUMENT

    using namespace std;

    auto pkg = pmb->packages.Get("Cameras");
    auto swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get("rays");
    auto cameras = pkg->Param<std::vector<Camera>>("cameras");
    auto old_centering = pkg->Param<bool>("old_centering");

    auto &x0 = swarm->Get<Real>(rays::t::name()).Get();
    auto &x1 = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &x2 = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &x3 = swarm->Get<Real>(swarm_position::z::name()).Get();
    auto &k = swarm->Get<Real>(rays::k::name()).Get();

    auto &camera_id = swarm->Get<int>(rays::camera_id::name()).Get();

    auto &ipx = swarm->Get<int>(rays::camera_i::name()).Get();
    auto &jpx = swarm->Get<int>(rays::camera_j::name()).Get();

    auto &nstep_geo = swarm->Get<int>(rays::nstep_geo::name()).Get();
    auto &path_len = swarm->Get<Real>(rays::path_len::name()).Get();

    const auto &G = pmb->coords;
    const Real x_max_mesh = pmb->pmy_mesh->mesh_size.xmax(X1DIR);
    const int ng = Globals::nghost; // We pull this so it gets copied to device

    int ncamera = 0;
    for (Camera camera : cameras) {
        // If we're responsible for this camera...
        if (!Rays::in_block_domain(G, camera.X, x_max_mesh)) continue;

        // Create new particles for this camera
        auto newParticlesContext = swarm->AddEmptyParticles(camera.nx * camera.ny);

        // Calculate tetrads *once* for each loc
        GReal Gcov[GR_DIM][GR_DIM];
        auto &coords = pmb->coords.coords;
        coords.gcov_native(camera.X, Gcov);
        double Econ[GR_DIM][GR_DIM];
        double Ecov[GR_DIM][GR_DIM];
        if (old_centering) {
            make_camera_tetrad_old(Gcov, camera.X, Econ, Ecov);
        } else {
            make_camera_tetrad(Gcov, camera.X, Econ, Ecov);
        }
        const double freq_me = camera.frequency * HPL / (ME * CL * CL);

        pmb->par_for(PARTHENON_AUTO_LABEL, 0, newParticlesContext.GetNewParticlesMaxIndex(),
            KOKKOS_LAMBDA(const int &new_n) {
                const int n = newParticlesContext.GetNewParticleIndex(new_n);

                // Create particle at camera
                x0(n) = camera.X[0];
                x1(n) = camera.X[1];
                x2(n) = camera.X[2];
                x3(n) = camera.X[3];

                // Set camera id and pixel
                camera_id(n) = ncamera;
                // Turn our index (among new particles only!) into a pixel location
                int i = new_n % camera.nx;
                int j = new_n / camera.nx;
                ipx(n) = i;
                jpx(n) = j;
                // Reset tracker variables
                nstep_geo(n) = 0;
                path_len(n) = 0.;

                // Construct outgoing wavevector
                // xoff: allow arbitrary offset for e.g. ML training imgs
                // +0.5: project geodesics from px centers
                double dxoff = (i + 0.5 + camera.xoff) / camera.nx - 0.5;
                double dyoff = (j + 0.5 + camera.yoff) / camera.ny - 0.5;
                double Kcon_tetrad[GR_DIM];
                Kcon_tetrad[0] = 0.;
                Kcon_tetrad[1] = (dxoff * cos(camera.rotcam) - dyoff * sin(camera.rotcam)) * camera.fovx;
                Kcon_tetrad[2] = (dxoff * sin(camera.rotcam) + dyoff * cos(camera.rotcam)) * camera.fovy;
                Kcon_tetrad[3] = 1.;

                /* normalize */
                null_normalize(Kcon_tetrad, 1.);

                /* translate into coordinate frame */
                GReal Kcon[GR_DIM];
                tetrad_to_coordinate(Econ, Kcon_tetrad, Kcon);
                k(0, n) = Kcon[0] * freq_me;
                k(1, n) = Kcon[1] * freq_me;
                k(2, n) = Kcon[2] * freq_me;
                k(3, n) = Kcon[3] * freq_me;
            }
        );

        ncamera++;
    }

    return TaskStatus::complete;
}

TaskStatus Cameras::WriteLocalCameras(MeshData<Real>* md)
{
    PARTHENON_INSTRUMENT

    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    auto pkg = pmb0->packages.Get("Cameras");
    const auto old_centering = pkg->Param<bool>("old_centering");
    const auto polarized = pmb0->packages.Get("Model")->Param<bool>("polarized");
    const Real x_max_mesh = pmb0->pmy_mesh->mesh_size.xmax(X1DIR);

    const auto camnames = pkg->Param<std::vector<std::string>>("camnames");
    const auto cameras = pkg->Param<std::vector<Camera>>("cameras");
    const auto vars = pkg->Param<std::vector<std::string>>("vars");

    std::string swarm_name = "rays";
    static auto desc = MakeSwarmPackDescriptor<rays::t,
                                              swarm_position::x,
                                              swarm_position::y,
                                              swarm_position::z,
                                              rays::path_len,
                                              rays::I,
                                              // Vectors
                                              rays::k,
                                              rays::Nr,
                                              rays::Ni>(swarm_name);
    auto pack = desc.GetPack(md);

    static auto int_desc = MakeSwarmPackDescriptor<rays::nstep_geo,
                                                   rays::nstep_rad,
                                                   rays::stop_flag,
                                                   rays::camera_id,
                                                   rays::camera_i,
                                                   rays::camera_j>(swarm_name);
    auto int_pack = int_desc.GetPack(md);

    // One giant pack of all cameras
    // std::vector<std::string> all_camera_vars;
    // for (auto camname : camnames) for (auto var : vars)
    //     all_camera_vars.push_back(camname+"_"+var);
    // const auto &resolved = md->GetMeshPointer()->resolved_packages.get();
    // static auto camera_desc = MakePackDescriptor(resolved, all_camera_vars);
    // auto camera_pack = camera_desc.GetPack(md);
    // auto camera_pack_map = camera_desc.GetMap();

    // Purely for coordinates.  TODO PackDesc so this doesn't cost ever
    auto P = md->PackVariables(std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")});

    // TODO could maybe interleave all cameras?
    // Just need to fetch lots of indices
    int ncamera = 0;
    for (const Camera camera : cameras) {
        // We write any particle from any block into the block-0 camera
        // Then we'll copy that and run an MPI reduction
        std::string camname = camnames[ncamera];

        const double freq3 = m::pow(camera.frequency, 3);

        // Individual packs for cameras
        std::vector<std::string> camera_vars;
        for (auto var : vars)
            camera_vars.push_back(camname+"_"+var);

        const auto &resolved = md->GetMeshPointer()->resolved_packages.get();
        static auto camera_desc = MakePackDescriptor(resolved, camera_vars);
        auto camera_pack = camera_desc.GetPack(md);
        auto camera_pack_map = camera_desc.GetMap();

        // Grab only meshblock 0 of our pack, we'll put all camera data there
        auto *rc0 = md->GetBlockData(0).get();
        auto &unpol = rc0->Get(camname+"_unpol").data;
        auto &pathlen = rc0->Get(camname+"_pathlen").data;
        auto &nstep = rc0->Get(camname+"_nstep").data;
        auto &stopflag = rc0->Get(camname+"_stopflag").data;

        std::cerr << "Writing camera " << camname << std::endl;
        pmb0->par_for(PARTHENON_AUTO_LABEL, 0, pack.GetMaxFlatIndex(),
            KOKKOS_LAMBDA(const int &idx) {
                auto [b, n] = pack.GetBlockParticleIndices(idx);
                const auto swarm_d = pack.GetContext(b);
                if (swarm_d.IsActive(n) && int_pack(b, rays::camera_id(), n) == ncamera) {
                    // TODO detect doubles!!
                    // Pixel location
                    const int &ipx = int_pack(b, rays::camera_i(), n);
                    const int &jpx = int_pack(b, rays::camera_j(), n);

                    // Always write the unpolarized image
                    unpol(ipx, jpx) = pack(b, rays::I(), n) * freq3 * camera.scale;
                    pathlen(ipx, jpx) = pack(b, rays::path_len(), n);
                    // TODO For some reason these crash everything...
                    nstep(ipx, jpx) = static_cast<Real>(int_pack(b, rays::nstep_geo(), n));
                    stopflag(ipx, jpx) = static_cast<Real>(int_pack(b, rays::stop_flag(), n));
                }
            }
        );
        if (polarized) {
            auto &stokes = rc0->Get(camname+"_stokes").data;

            pmb0->par_for(PARTHENON_AUTO_LABEL, 0, pack.GetMaxFlatIndex(),
                KOKKOS_LAMBDA(const int &idx) {
                    auto [b, n] = pack.GetBlockParticleIndices(idx);
                    const auto swarm_d = pack.GetContext(b);

                    if (swarm_d.IsActive(n) && int_pack(b, rays::camera_id(), n) == ncamera) {
                        // Grab pointers
                        const GRCoordinates &G = P.GetCoords(b);
                        const CoordinateEmbedding &coords = G.coords;

                        // Write polarized camera by measuring it in observer frame
                        Kokkos::complex<double> N_coord[GR_DIM][GR_DIM], N_tetrad[GR_DIM][GR_DIM];
                        read_N(pack, b, n, N_coord);
                        double gcov[GR_DIM][GR_DIM], Ecov[GR_DIM][GR_DIM], Econ[GR_DIM][GR_DIM];
                        // To measure at the geodesic endpoint, if different from camera
                        double cX[GR_DIM] = {pack(b, rays::t(), n),
                                             pack(b, swarm_position::x(), n),
                                             pack(b, swarm_position::y(), n),
                                             pack(b, swarm_position::z(), n)};
                        coords.gcov_native(cX, gcov);
                        if (old_centering) {
                            make_camera_tetrad_old(gcov, cX, Econ, Ecov);
                        } else {
                            make_camera_tetrad(gcov, cX, Econ, Ecov);
                        }
                        N_to_tetrad(N_coord, Ecov, N_tetrad);
                        double S[4];
                        N_to_stokes(N_tetrad, S[0], S[1], S[2], S[3]);

                        // rotate Stokes Q, U if camera is rotated
                        if (camera.rotcam != 0) {
                            const double qu_angle = camera.rotcam * -2;
                            const double rot_Q = S[1] * m::cos(qu_angle) - S[2] * m::sin(qu_angle);
                            const double rot_U = S[1] * m::sin(qu_angle) + S[2] * m::cos(qu_angle);
                            S[1] = rot_Q;
                            S[2] = rot_U;
                        }
                        // Written Q/U generally use the opposite convention to transport:
                        // Transport is performed w/EVPA North of West, but parameters are
                        // written East of North.
                        if (camera.qu_conv == 0) {
                            S[1] *= -1;
                            S[2] *= -1;
                        }

                        const int &ipx = int_pack(b, rays::camera_i(), n);
                        const int &jpx = int_pack(b, rays::camera_j(), n);
                        for (int s = 0; s < 4; s++) {
                            stokes(ipx, jpx, s) = S[s] * freq3 * camera.scale;
                        }
                    }
                }
            );
        }
        Kokkos::fence();

        // Now fill the host-side arrays
        for (auto var : vars) {
            bool vector = (var == "stokes");
            if (vector) {
                // Get reducer
                HostArray3D<Real> *camera_reduce =
                    pkg->MutableParam<HostArray3D<Real>>(camname+"_reduce_"+var);
                // D->H copy.  Only zero-block since that's what we filled above!
                Kokkos::deep_copy(*camera_reduce,
                    Kokkos::subview(rc0->Get(camname+"_"+var).data,
                    0, 0, 0, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
            } else {
                HostArray2D<Real> *camera_reduce =
                    pkg->MutableParam<HostArray2D<Real>>(camname+"_reduce_"+var);
                Kokkos::deep_copy(*camera_reduce,
                    Kokkos::subview(rc0->Get(camname+"_"+var).data,
                    0, 0, 0, 0, Kokkos::ALL, Kokkos::ALL));
            }
        }
        std::cerr << "Filled camera " << camname << std::endl;
        Kokkos::fence();

        // New camera
        ncamera++;
    }

    // One more for good measure, this is run once
    Kokkos::fence();
    
    return TaskStatus::complete;
}

// 
TaskStatus Cameras::ReduceCameras(MeshData<Real>* md)
{
    PARTHENON_INSTRUMENT

    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    auto pkg = pmb0->packages.Get("Cameras");
    // const auto old_centering = pkg->Param<bool>("old_centering");
    // const auto polarized = pmb0->packages.Get("Model")->Param<bool>("polarized");
    // const Real x_max_mesh = pmb0->pmy_mesh->mesh_size.xmax(X1DIR);

    const auto camnames = pkg->Param<std::vector<std::string>>("camnames");
    // const auto cameras = pkg->Param<std::vector<Camera>>("cameras");
    const auto vars = pkg->Param<std::vector<std::string>>("vars");

    // TODO free these blocks from sequential dependencies someday
    // Start reductions, TODO store in vectors
    std::vector<Reduce<HostArray2D<Real>>> reducers_2d;
    std::vector<Reduce<HostArray3D<Real>>> reducers_3d;
    for (auto camname : camnames) {
        for (auto var : vars) {
            bool vector = (var == "stokes");
            if (vector) {
                // Get reducer
                auto camera_reduce = Reduce<HostArray3D<Real>>();
                auto camera_host =
                    *pkg->MutableParam<HostArray3D<Real>>(camname+"_reduce_"+var);
                camera_reduce.val = camera_host;
                // Start MPI reduce
                // TODO Parthenon's stuff only does sums probably, but we should be doing max()
                camera_reduce.StartReduce(0, MPI_SUM);
                // Wait
                while (camera_reduce.CheckReduce() == TaskStatus::incomplete);
                camera_host = camera_reduce.val;
            } else {
                // Get reducer
                auto camera_reduce = Reduce<HostArray2D<Real>>();
                auto camera_host =
                    *pkg->MutableParam<HostArray2D<Real>>(camname+"_reduce_"+var);
                camera_reduce.val = camera_host;
                // Start MPI reduce
                camera_reduce.StartReduce(0, MPI_SUM);
                // Wait
                while (camera_reduce.CheckReduce() == TaskStatus::incomplete);
                camera_host = camera_reduce.val;
            }
        }
    }

    // TODO now catch them all down here instead

    return TaskStatus::complete;
}

TaskStatus Cameras::WriteReducedCameras(MeshData<Real>* md)
{
    PARTHENON_INSTRUMENT

    if (!MPIRank0()) return TaskStatus::complete;

    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    auto pkg = pmb0->packages.Get("Cameras");
    const auto polarized = pmb0->packages.Get("Model")->Param<bool>("polarized");

    const auto camnames = pkg->Param<std::vector<std::string>>("camnames");
    const auto cameras = pkg->Param<std::vector<Camera>>("cameras");
    const auto vars = pkg->Param<std::vector<std::string>>("vars");

    int ncamera = 0;
    for (const Camera camera : cameras) {
        std::string camname = camnames[ncamera];

        // TODO This is stupid.  Write the reducers straight to file!!
        // (we're copying reduced values back into our block0's values, then to device...
        // just so Parthenon can pull them back to host and write them out)
        for (auto var : vars) {
            bool vector = (var == "stokes");
            if (vector) {
                // Get reducer
                HostArray3D<Real> camera_host =
                    *pkg->MutableParam<HostArray3D<Real>>(camname+"_reduce_"+var);
                // D->H copy.  Only zero-block since that's what we filled above!
                Kokkos::deep_copy(Kokkos::subview(
                    md->GetBlockData(0)->Get(camname+"_"+var).data, 0, 0, 0, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
                    camera_host);
            } else {
                HostArray2D<Real> camera_host =
                    *pkg->MutableParam<HostArray2D<Real>>(camname+"_reduce_"+var);
                Kokkos::deep_copy(Kokkos::subview(
                    md->GetBlockData(0)->Get(camname+"_"+var).data, 0, 0, 0, 0, Kokkos::ALL, Kokkos::ALL),
                    camera_host);
            }
        }

        if (pkg->Param<int>("verbose") > 0) {
            // Get reducer values
            HostArray2D<Real> image =
                *pkg->MutableParam<HostArray2D<Real>>(camname+"_reduce_unpol");
            // TODO any stats on nsteps etc etc

            HostArray3D<Real> stokes;
            if (polarized) {
                stokes = *pkg->MutableParam<HostArray3D<Real>>(camname+"_reduce_stokes");
            }

            double Ftot = 0.;
            double Ftot_unpol = 0.;
            double Imax = 0.0;
            double Iavg = 0.0;
            double Qtot = 0.;
            double Utot = 0.;
            double Vtot = 0.;
            size_t imax = 0;
            size_t jmax = 0;
            // TODO OpenMP at least?
            for (size_t i = 0; i < camera.nx; i++) {
                for (size_t j = 0; j < camera.ny; j++) {
                    // Reduction arrays are opposite order j, i, s
                    Ftot_unpol += image(i, j);

                    if (polarized) {
                        Ftot += stokes(i, j, 0);
                        Iavg += stokes(i, j, 0) / camera.scale;
                        Qtot += stokes(i, j, 1);
                        Utot += stokes(i, j, 2);
                        Vtot += stokes(i, j, 3);
                        if (stokes(i, j, 0) / camera.scale > Imax) {
                            imax = i;
                            jmax = j;
                            Imax = stokes(i, j, 0) / camera.scale;
                        }
                    }
                }
            }

            // output normal flux quantities
            fprintf(stderr, "\nscale = %e\n", camera.scale);
            fprintf(stderr, "imax=%ld jmax=%ld Imax=%g Iavg=%g\n", imax, jmax, Imax, Iavg / (camera.nx * camera.ny));
            fprintf(stderr, "freq: %g Ftot: %g Jy (%g Jy unpol xfer) scale=%g\n", camera.frequency, Ftot, Ftot_unpol, camera.scale);
            fprintf(stderr, "nuLnu = %g erg/s\n", 4.*M_PI*Ftot * camera.dsource * camera.dsource * JY * camera.frequency);
            // output polarized transport information
            double LPfrac = 100.*sqrt(Qtot*Qtot+Utot*Utot)/Ftot;
            double CPfrac = 100.*Vtot/Ftot;
            fprintf(stderr, "I,Q,U,V [Jy]: %g %g %g %g\n", Ftot, Qtot, Utot, Vtot);
            fprintf(stderr, "LP,CP [%%]: %g %g\n", LPfrac, CPfrac);
        }

        ncamera++;
    }

    return TaskStatus::complete;
}