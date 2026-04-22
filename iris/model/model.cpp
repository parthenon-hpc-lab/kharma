/*
 *  File: model.cpp
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
#include "model.hpp"

// Iris headers
#include "rays.hpp"
#include "thin_disk.hpp"
#include "units.hpp"

// KHARMA headers
#include "coordinate_embedding.hpp"

#include <parthenon/parthenon.hpp>

std::shared_ptr<StateDescriptor> Model::Initialize(ParameterInput *pin)
{
    auto pkg = std::make_shared<StateDescriptor>("Model");
    Params &params = pkg->AllParams();

    // TODO allowable types etc
    std::string type = pin->GetString("model", "type");
    params.Add("type", type);
    bool polarized = pin->GetBoolean("model", "polarized");
    params.Add("polarized", polarized);
    bool radiative = pin->GetOrAddBoolean("model", "radiative", type != "thin_disk");
    params.Add("radiative", radiative);

    auto verbose = pin->GetInteger("debug", "verbose");
    params.Add("verbose", verbose); // TODO stash in separate package?

    // Units
    double MBH = pin->GetReal("model", "MBH") * MSUN; // Convert to CGS
    params.Add("MBH", MBH);
    double L_unit = GNEWT * MBH / (CL * CL);
    pin->SetReal("model", "L_unit", L_unit);
    params.Add("L_unit", L_unit);
    double T_unit = L_unit / CL;
    pin->SetReal("model", "T_unit", T_unit);
    params.Add("T_unit", T_unit);
    double Mdot_edd = 4. * M_PI * GNEWT * MBH * MP / CL / 0.1 / SIGMA_THOMSON;
    pin->SetReal("model", "Mdot_edd", Mdot_edd);
    params.Add("Mdot_edd", Mdot_edd);
    // Models don't require an M_unit if they set emission directly, e.g. thin disk
    double M_unit, Mdot;
    if (pin->DoesParameterExist("model", "Mdot")) {
        Mdot = pin->GetReal("model", "Mdot") * Mdot_edd;
        M_unit = Mdot;
    } else {
        M_unit = pin->GetReal("model", "M_unit");
    }
    params.Add("M_unit", M_unit);
    double RHO_unit = M_unit / pow(L_unit, 3);
    pin->SetReal("model", "RHO_unit", RHO_unit);
    params.Add("RHO_unit", RHO_unit);
    double U_unit = RHO_unit * CL * CL;
    pin->SetReal("model", "U_unit", U_unit);
    params.Add("U_unit", U_unit);
    double B_unit = CL * sqrt(4.*M_PI*RHO_unit);
    pin->SetReal("model", "B_unit", B_unit);
    params.Add("B_unit", B_unit);

    // Print units like ipole
    // technically thindisk problem doesn't print MBH but come on
    if (verbose > 0) {
        if (! (type == "thin_disk")) printf("MBH: %g [Msun]\n", MBH / MSUN);
        printf("L,T,M units: %g [cm] %g [s] %g [g]\n", L_unit, T_unit, M_unit);
        printf("rho,u,B units: %g [g cm^-3] %g [g cm^-1 s^-2] %g [G] \n", RHO_unit, U_unit, B_unit);
    }

    // TODO if analytic...
    // We haven't built the mesh yet, so just build a new CoordEmbed
    CoordinateEmbedding coords(pin);
    // Record for other package initializations
    params.Add("coords", coords);

    double Rout;
    if (type == "file") {
        // We reset these because it doesn't matter much, and they were "user" in KHARMA
        // TODO figure out merging meshblocks...
        pin->SetString("parthenon/mesh", "ix1_bc", "outflow");
        pin->SetString("parthenon/mesh", "ox1_bc", "outflow");
        pin->SetString("parthenon/mesh", "ix2_bc", "reflecting");
        pin->SetString("parthenon/mesh", "ox2_bc", "reflecting");
        // pin->SetString("parthenon/mesh", "ix3_bc", "periodic");
        // pin->SetString("parthenon/mesh", "ox3_bc", "periodic");

        // TODO maybe read from the mesh before setting these...
        pin->GetOrAddString("parthenon/swarm", "ix1_bc", "none");
        pin->GetOrAddString("parthenon/swarm", "ox1_bc", "none");
        pin->GetOrAddString("parthenon/swarm", "ix2_bc", "none");
        pin->GetOrAddString("parthenon/swarm", "ox2_bc", "none");
        pin->GetOrAddString("parthenon/swarm", "ix3_bc", "periodic");
        pin->GetOrAddString("parthenon/swarm", "ox3_bc", "periodic");

        // TODO probably shouldn't always match, allow override
        Rout = coords.X1_to_embed(pin->GetReal("parthenon/mesh", "x1max"));
    } else {
        // Fool Parthenon into building a mesh for us
        // Set any coordinate-determined boundaries...
        if (coords.startx(2) >= 0)
            pin->GetOrAddReal("parthenon/mesh", "x2min", coords.startx(2));
        if (coords.stopx(2) >= 0)
            pin->GetOrAddReal("parthenon/mesh", "x2max", coords.stopx(2));
        if (coords.startx(3) >= 0)
            pin->GetOrAddReal("parthenon/mesh", "x3min", coords.startx(3));
        if (coords.stopx(3) >= 0)
            pin->GetOrAddReal("parthenon/mesh", "x3max", coords.stopx(3));
        // And translate the radial boundaries
        Rout = pin->GetReal("coordinates", "r_out");
        pin->SetReal("parthenon/mesh", "x1min", coords.r_to_native(coords.get_horizon()));
        pin->SetReal("parthenon/mesh", "x1max", coords.r_to_native(Rout));

        // TODO Cartesian transport?

        // Analytic models need only a trivial mesh
        // But, allow overriding for testing
        pin->GetOrAddInteger("parthenon/mesh", "nx1", 8);
        pin->GetOrAddInteger("parthenon/mesh", "nx2", 1);
        pin->GetOrAddInteger("parthenon/mesh", "nx3", 1);
        pin->GetOrAddInteger("parthenon/meshblock", "nx1", 8);
        pin->GetOrAddInteger("parthenon/meshblock", "nx2", 1);
        pin->GetOrAddInteger("parthenon/meshblock", "nx3", 1);

        pin->GetOrAddString("parthenon/mesh", "ix1_bc", "outflow");
        pin->GetOrAddString("parthenon/mesh", "ox1_bc", "outflow");
        pin->GetOrAddString("parthenon/mesh", "ix2_bc", "reflecting");
        pin->GetOrAddString("parthenon/mesh", "ox2_bc", "reflecting");
        pin->GetOrAddString("parthenon/mesh", "ix3_bc", "periodic");
        pin->GetOrAddString("parthenon/mesh", "ox3_bc", "periodic");
        pin->GetOrAddString("parthenon/swarm", "ix1_bc", "none");
        pin->GetOrAddString("parthenon/swarm", "ox1_bc", "none");
        pin->GetOrAddString("parthenon/swarm", "ix2_bc", "none");
        pin->GetOrAddString("parthenon/swarm", "ox2_bc", "none");
        pin->GetOrAddString("parthenon/swarm", "ix3_bc", "periodic");
        pin->GetOrAddString("parthenon/swarm", "ox3_bc", "periodic");

        // printf("Mesh: %f %f %f to %f %f %f", pin->GetReal("parthenon/mesh", "x1min"),
        //         pin->GetReal("parthenon/mesh", "x2min"),
        //         pin->GetReal("parthenon/mesh", "x3min"),
        //         pin->GetReal("parthenon/mesh", "x1max"),
        //         pin->GetReal("parthenon/mesh", "x2max"),
        //         pin->GetReal("parthenon/mesh", "x3max"));

        // Probably don't need to set this?
        Globals::nghost = pin->GetOrAddInteger("driver", "nghost", 4);
        pin->SetInteger("parthenon/mesh", "nghost", Globals::nghost);

        // We need this tag to exist since we pack using it
        // The "File" package loads it if we're loading from file
        Metadata::AddUserFlag("Primitive");
    }

    if (verbose > 0) {
        if (type == "thin_disk") {
            // Some precomputation. Remember Mdot == M_unit in thin disk prob
            double T0 = m::pow(3.0 / 8.0 / M_PI * GNEWT * MBH * M_unit / m::pow(L_unit, 3) / SIG, 1. / 4.);
            params.Add("T0", T0);
            printf("Rh, Rin, Rout, r_isco, T0: %g %g %g %g %g\n", coords.get_horizon(), coords.get_horizon(), Rout, coords.get_isco(), T0);
            printf("Running thin disk:\nMBH: %g\nMdot: %g\na: %g\n\n", MBH, Mdot, coords.get_a());
        } else {
            printf("Rh, Rin, Rout, r_isco: %g %g %g %g\n", coords.get_horizon(), coords.get_horizon(), Rout, coords.get_isco());
        }
    }

    // Setup the stop condition
    params.Add("stop_condition", StopCondition(coords, pin));

    return pkg;
}

TaskStatus Model::SetStokesThindisk(MeshData<Real> *md)
{
    // Make sure to iterate through only the blocks in this MeshData
    for (int n = 0; n < md->NumBlocks(); n++) {
        auto &mbd = md->GetBlockData(n);
        SetStokesThindiskBlock(mbd->GetBlockPointer());
    }
    return TaskStatus::complete;
}

TaskStatus Model::SetStokesThindiskBlock(MeshBlock* pmb)
{
    PARTHENON_INSTRUMENT

    auto swarm = pmb->meshblock_data.Get("base")->GetSwarmData()->Get("rays");
    auto &model = pmb->packages.Get("Model")->AllParams();
    const auto polarized = model.Get<bool>("polarized");
    const auto verbose = model.Get<int>("verbose");

    // Thin disk precomputation
    const auto T0 = (model.Get<std::string>("type") == "thin_disk") ? model.Get<double>("T0") : 0.0;

    // For the emission plane loctation
    const auto stop_condition = model.Get<StopCondition>("stop_condition");

    int max_active_index = swarm->GetMaxActiveIndex();

    // Position
    auto &x0 = swarm->Get<Real>(rays::t::name()).Get();
    auto &x1 = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &x2 = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &x3 = swarm->Get<Real>(swarm_position::z::name()).Get();
    auto &k = swarm->Get<Real>(rays::k::name()).Get();
    // Reason geodesics were stopped, i.e. which hit the disk?
    auto &stopflag = swarm->Get<int>(rays::stop_flag::name()).Get();
    // Radiation
    auto &I = swarm->Get<Real>(rays::I::name()).Get();
    auto &Nr = (polarized) ? swarm->Get<Real>(rays::Nr::name()).Get() : swarm->Get<Real>(rays::I::name()).Get();
    auto &Ni = (polarized) ? swarm->Get<Real>(rays::Ni::name()).Get() : swarm->Get<Real>(rays::I::name()).Get();

    auto swarm_d = swarm->GetDeviceContext();

    const CoordinateEmbedding coords = pmb->coords.coords;

    pmb->par_for(PARTHENON_AUTO_LABEL, 0, max_active_index,
        KOKKOS_LAMBDA(const int &n) {
            if (swarm_d.IsActive(n)) {
                if (stopflag(n) == (int) StopFlag::thin_disk) {
                    // A thin disk problem emits nowhere but uses a boundary condition region defined by thindisk_region
                    // There we just get a starting value for intensity with get_model_i
                    double X[GR_DIM] = {x0(n), x1(n), x2(n), x3(n)};
                    double Kcon[GR_DIM] = {k(0, n), k(1, n), k(2, n), k(3, n)};
                    double SI, SQ, SU, SV;
                    ThinDisk::get_model_stokes(coords, X, Kcon, T0, SI, SQ, SU, SV);
                    I(n) = SI;

                    if (polarized) {
                        // For polarized emission, we get Stokes parameters in the fluid frame,
                        // then compute the transport tensor N and transform it to coordinate frame
                        // Make a tetrad
                        double gcov[GR_DIM][GR_DIM];
                        coords.gcov_native(X, gcov);
                        double Ucon[GR_DIM], Ucov[GR_DIM], Bcon[GR_DIM], Bcov[GR_DIM];
                        ThinDisk::get_model_fourv(coords, X, Kcon, T0, Ucon, Ucov, Bcon, Bcov);
                        double Ecov[GR_DIM][GR_DIM], Econ[GR_DIM][GR_DIM];
                        make_plasma_tetrad(Ucon, Kcon, Bcon, gcov, Econ, Ecov);

                        // make N_tetrad, and transform
                        Kokkos::complex<double> N_coord[GR_DIM][GR_DIM], N_tetrad[GR_DIM][GR_DIM] = {{0.}};
                        stokes_to_N(SI, SQ, SU, SV, N_tetrad);
                        N_to_coord(N_tetrad, Econ, N_coord);
                        write_N(N_coord, Nr, Ni, n);
                    }
                }
            }
        });


    if (polarized && verbose > 1) {
        pmb->par_for(PARTHENON_AUTO_LABEL, 0, max_active_index,
            KOKKOS_LAMBDA(const int &n) {
                if (swarm_d.IsActive(n)) {
                    // Can these temps be any smaller?
                    double X[GR_DIM] = {x0(n), x1(n), x2(n), x3(n)};
                    double Kcon[GR_DIM] = {k(0, n), k(1, n), k(2, n), k(3, n)};
                    double Ucon[GR_DIM], Ucov[GR_DIM], Bcon[GR_DIM], Bcov[GR_DIM];
                    double Ecov[GR_DIM][GR_DIM], Econ[GR_DIM][GR_DIM];
                    double gcov[GR_DIM][GR_DIM];
                    Kokkos::complex<double> N_coord[GR_DIM][GR_DIM], N_tetrad[GR_DIM][GR_DIM];

                    read_N(Nr, Ni, n, N_coord);
                    // old gcov
                    coords.gcov_native(X, gcov);
                    // old tetrad
                    ThinDisk::get_model_fourv(coords, X, Kcon, T0, Ucon, Ucov, Bcon, Bcov);
                    make_plasma_tetrad(Ucon, Kcon, Bcon, gcov, Econ, Ecov);
                    // N to tetrad frame
                    N_to_tetrad(N_coord, Ecov, N_tetrad);
                    double SI, SQ, SU, SV;
                    N_to_stokes(N_tetrad, SI, SQ, SU, SV);

                    Rays::geodesic_parallel_transport_step(coords, X, Kcon, N_coord, 0.01, 1e-7);

                    // New gcov
                    coords.gcov_native(X, gcov);
                    // New tetrad
                    ThinDisk::get_model_fourv(coords, X, Kcon, T0, Ucon, Ucov, Bcon, Bcov);
                    make_plasma_tetrad(Ucon, Kcon, Bcon, gcov, Econ, Ecov);
                    // N to tetrad frame
                    N_to_tetrad(N_coord, Ecov, N_tetrad);
                    double SInew, SQnew, SUnew, SVnew;
                    N_to_stokes(N_tetrad, SInew, SQnew, SUnew, SVnew);

                    // Print comparison
                    printf("One-step Stokes: (%g %g) (%g %g) (%g %g) (%g %g)\n", SI, SInew, SQ, SQnew, SU, SUnew, SV, SVnew);
                }
            }
        );
    }

    return TaskStatus::complete;
}
