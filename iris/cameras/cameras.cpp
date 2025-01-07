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

    std::vector<Camera> cameras;
    if (pin->DoesBlockExist("camera")) {
        cameras.push_back(Camera(coords, pin, "camera"));
        auto camera = cameras[0]; // the only
        if (verbose > 0) camera.print();
        // TODO can't make this float yet...
        Metadata image_meta = Metadata({Metadata::Real, Metadata::None, Metadata::Derived,
                                        Metadata::OneCopy}, std::vector<int>{camera.nx, camera.ny});
        pkg->AddField("camera_unpol", image_meta);
        if (polarized) {
            pkg->AddField("camera_I", image_meta);
            pkg->AddField("camera_Q", image_meta);
            pkg->AddField("camera_U", image_meta);
            pkg->AddField("camera_V", image_meta);
        }

        pkg->AddField("camera_nstep", image_meta);
        pkg->AddField("camera_pathlen", image_meta);
        pkg->AddField("camera_stopflag", image_meta);
        
        // TODO if any parameter is list, repeat & generate new blocks...
    } else {
        throw std::runtime_error("Multiple cameras not implemented!");
        // for block in camera blocks...
    }
    params.Add("cameras", cameras);

    return pkg;
}

TaskStatus Cameras::InitGeodesics(MeshBlock* pmb)
{
    PARTHENON_INSTRUMENT

    auto pkg = pmb->packages.Get("Cameras");
    auto swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get("rays");
    auto cameras = pkg->Param<std::vector<Camera>>("cameras");
    auto old_centering = pkg->Param<bool>("old_centering");

    // Meshblock geometry
    const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    auto &x0 = swarm->Get<Real>("t").Get();
    auto &x1 = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &x2 = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &x3 = swarm->Get<Real>(swarm_position::z::name()).Get();
    auto &k = swarm->Get<Real>("k").Get();

    auto &camera_id = swarm->Get<int>("camera_id").Get();
    auto &ipx = swarm->Get<int>("i").Get();
    auto &jpx = swarm->Get<int>("j").Get();

    auto swarm_d = swarm->GetDeviceContext();

    int ncamera = 0;
    for (Camera camera : cameras) {

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
                int i = new_n % camera.ny;
                int j = new_n / camera.ny;
                ipx(n) = i;
                jpx(n) = j;

                // Construct outgoing wavevector
                // xoff: allow arbitrary offset for e.g. ML training imgs
                // +0.5: project geodesics from px centers
                // xoff/yoff are separated to keep consistent behavior between refinement levels
                double dxoff = (i + 0.5 + camera.xoff - 0.01) / camera.nx - 0.5;
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

TaskStatus Cameras::WriteCameras(MeshBlock* pmb)
{
    PARTHENON_INSTRUMENT

    auto pkg = pmb->packages.Get("Cameras");
    auto swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get("rays");
    const auto cameras = pkg->Param<std::vector<Camera>>("cameras");
    const auto old_centering = pkg->Param<bool>("old_centering");

    const auto polarized = pmb->packages.Get("Model")->Param<bool>("polarized");

    // Meshblock geometry
    const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    auto &x0 = swarm->Get<Real>("t").Get();
    auto &x1 = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &x2 = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &x3 = swarm->Get<Real>(swarm_position::z::name()).Get();
    auto &k = swarm->Get<Real>("k").Get();
    // Tracing data
    auto &nstep = swarm->Get<int>("nstep_geo").Get();
    auto &pathlen = swarm->Get<Real>("path_len").Get();
    auto &stopflag = swarm->Get<int>("stop_flag").Get();
    // Radiation
    auto &I = swarm->Get<Real>("I").Get();
    auto &Nr = (polarized) ? swarm->Get<Real>("Nr").Get() : swarm->Get<Real>("I").Get();
    auto &Ni = (polarized) ? swarm->Get<Real>("Ni").Get() : swarm->Get<Real>("I").Get();

    auto &camera_id = swarm->Get<int>("camera_id").Get();
    auto &ipx = swarm->Get<int>("i").Get();
    auto &jpx = swarm->Get<int>("j").Get();

    int max_active_index = swarm->GetMaxActiveIndex();

    auto swarm_d = swarm->GetDeviceContext();

    const auto &coords = pmb->coords.coords;

    int ncamera = 0;
    for (const Camera camera : cameras) {
        const double freq3 = m::pow(camera.frequency, 3);

        // TODO can probably interleave writing all the cameras, but whatever.
        std::string camname;
        if (cameras.size() == 1) {
            camname = "camera";
        }
        auto &rc = pmb->meshblock_data.Get();

        auto camera_pathlen = rc->Get(camname+"_pathlen").data;
        auto camera_nstep = rc->Get(camname+"_nstep").data;
        auto camera_stopflag = rc->Get(camname+"_stopflag").data;

        auto camera_unpol = rc->Get(camname+"_unpol").data;
        auto camera_I = (polarized) ? rc->Get(camname+"_I").data : rc->Get(camname+"_unpol").data;
        auto camera_Q = (polarized) ? rc->Get(camname+"_Q").data : rc->Get(camname+"_unpol").data;
        auto camera_U = (polarized) ? rc->Get(camname+"_U").data : rc->Get(camname+"_unpol").data;
        auto camera_V = (polarized) ? rc->Get(camname+"_V").data : rc->Get(camname+"_unpol").data;

        pmb->par_for(PARTHENON_AUTO_LABEL, 0, max_active_index,
            KOKKOS_LAMBDA(const int &n) {
                if (camera_id(n) == ncamera) {
                    camera_pathlen(ipx(n), jpx(n)) = pathlen(n);
                    camera_nstep(ipx(n), jpx(n)) = nstep(n);
                    camera_stopflag(ipx(n), jpx(n)) = stopflag(n);
                    // Always write the unpolarized image
                    camera_unpol(ipx(n), jpx(n)) = I(n) * freq3 * camera.scale;
                    if (polarized) {
                        Kokkos::complex<double> N_coord[GR_DIM][GR_DIM], N_tetrad[GR_DIM][GR_DIM];
                        double SI, SQ, SU, SV;
                        read_N(Nr, Ni, n, N_coord);
                        double gcov[GR_DIM][GR_DIM], Ecov[GR_DIM][GR_DIM], Econ[GR_DIM][GR_DIM];
                        // To measure at the geodesic endpoint, if different from camera
                        double cX[GR_DIM] = {x0(n), x1(n), x2(n), x3(n)};
                        coords.gcov_native(cX, gcov);
                        if (old_centering) {
                            make_camera_tetrad_old(gcov, cX, Econ, Ecov);
                        } else {
                            make_camera_tetrad(gcov, cX, Econ, Ecov);
                        }
                        N_to_tetrad(N_coord, Ecov, N_tetrad);
                        N_to_stokes(N_tetrad, SI, SQ, SU, SV);

                        // rotate Stokes Q, U if camera is rotated
                        if (camera.rotcam != 0) {
                            double qu_angle = camera.rotcam * -2;
                            double rot_Q = SQ * m::cos(qu_angle) - SU * m::sin(qu_angle);
                            double rot_U = SQ * m::sin(qu_angle) + SU * m::cos(qu_angle);
                            SQ = rot_Q; SU = rot_U;
                        }
                        // Written Q/U generally use the opposite convention to transport:
                        // Transport is performed w/EVPA North of West, but parameters are
                        // written East of North.
                        if (camera.qu_conv == 0) {
                            SQ *= -1;
                            SU *= -1;
                        }

                        camera_I(ipx(n), jpx(n)) = SI * freq3 * camera.scale;
                        camera_Q(ipx(n), jpx(n)) = SQ * freq3 * camera.scale;
                        camera_U(ipx(n), jpx(n)) = SU * freq3 * camera.scale;
                        camera_V(ipx(n), jpx(n)) = SV * freq3 * camera.scale;
                    }
                }
            }
        );

        if (pkg->Param<int>("verbose") > 0) {
            // TODO fast device reductions
            // pmb->par_reduce(PARTHENON_AUTO_LABEL, 0, camera.nx-1, 0, camera.ny-1,
            //     KOKKOS_LAMBDA(const int &i, const int &j) {

            //     }
            // );
            Kokkos::fence();
            auto image = camera_unpol.GetHostMirrorAndCopy();
            auto imageI = camera_I.GetHostMirrorAndCopy();
            auto imageQ = camera_Q.GetHostMirrorAndCopy();
            auto imageU = camera_U.GetHostMirrorAndCopy();
            auto imageV = camera_V.GetHostMirrorAndCopy();

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
                    Ftot_unpol += image(i, j);

                    if (polarized) {
                        Ftot += imageI(i, j);
                        Iavg += imageI(i, j) / camera.scale;
                        Qtot += imageQ(i, j);
                        Utot += imageU(i, j);
                        Vtot += imageV(i, j);
                        if (imageI(i, j) / camera.scale > Imax) {
                            imax = i;
                            jmax = j;
                            Imax = imageI(i, j) / camera.scale;
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

    // TODO iterate over cameras and print summaries

    return TaskStatus::complete;
}