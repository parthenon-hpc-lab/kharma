/*
 *  File: rays.hpp
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

#include "units.hpp"
#include "variables.hpp"

// KHARMA headers
#include "coordinate_embedding.hpp"
#include "decs.hpp"
#include "types.hpp"

using namespace parthenon;

#include "mesh_comms.hpp"

/**
 */
namespace Rays
{

std::shared_ptr<StateDescriptor> Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages);

// Solve just the geodesic equation backward, until defined stopping points
TaskStatus TraceGeodesics(MeshData<Real>* md);

// Trace in a vacuum. No-op to intensity, parallel transport polarization
TaskStatus TraceEmission(MeshData<Real>* md);

// Just to one block boundary
TaskStatus TraceGeodesicsBlock(MeshData<Real>* md);

template<bool store_paths>
TaskStatus TraceEmissionBlock(MeshData<Real>* md);

// Are there remaining rays at block boundaries?
template<bool emission>
int CountLiveParticles(MeshData<Real>* md);

template<bool emission, typename T>
inline TaskID AddTraceTasks(TaskID t_depends, TaskList& tl, T& TransportRays,
    MeshData<Real>* md_base, int max_iter)
{
    PARTHENON_INSTRUMENT

    TaskID t_none(0);

    auto t_start = tl.AddTask(t_depends,
        []
        {
            fprintf(stderr, "Starting trace\n");
            return TaskStatus::complete;
        });
    auto [itl, itl_id] = tl.AddSublist(t_start, {1, max_iter});

    // Wrapper to start comms
    auto t_reset_comms = itl.AddTask(t_none, TF(MeshResetCommunication), md_base);

    // Push rays (all blocks)
    auto t_transport =
        itl.AddTask(TaskQualifier::local_sync, t_reset_comms, TF(TransportRays), md_base);

    // Comms wrappers
    // Always all boundaries!
    auto t_send = itl.AddTask(t_transport, TF(MeshSend), md_base);
    auto t_receive = itl.AddTask(t_send, TF(MeshReceive), md_base);

    auto t_check =
        itl.AddTask(TaskQualifier::global_sync | TaskQualifier::completion, t_receive,
            [=]
            {
                return (CountLiveParticles<emission>(md_base) > 0) ? TaskStatus::iterate
                                                                   : TaskStatus::complete;
            });

    return itl_id;
}

KOKKOS_INLINE_FUNCTION Real phi_of(const Real& x3)
{
    Real phi = fmod(x3, M_2_PI);
    return phi < 0 ? phi + M_2_PI : phi;
}

// Strict check whether particle is in the physical space covered by (within or radially
// outside) a block. Should always be true of exactly one block -- blocks own left edges
// but not right
KOKKOS_INLINE_FUNCTION bool in_block_domain(
    const GRCoordinates& G, const double X[GR_DIM], const double& x_max_mesh)
{
    // Check just outside the faces of the "interior" of the domain, to let particles move
    // into ghost zones.
    Real phi = phi_of(X[3]);
    if (X[1] >= G.Xf<X1DIR>(G.ng) && (X[2] >= G.Xf<X2DIR>(G.ng) || G.n2 == 1) &&
        (phi >= G.Xf<X3DIR>(G.ng) || G.n3 == 1) &&
        /* ======OMITTED====== */ (X[2] < G.Xf<X2DIR>(G.n2 - G.ng) || G.n2 == 1) &&
        (phi < G.Xf<X3DIR>(G.n3 - G.ng) || G.n3 == 1)) {
        // If x is in the zone itelf, or we're the last zone in x1 on the mesh, the
        // particle is in our domain
        return (X[1] < G.Xf<X1DIR>(G.n1 - G.ng) ||
                (G.Xf<X1DIR>(G.n1 - G.ng) > 0.99 * x_max_mesh));
    }
    return false;
}
// Fuzzy block check allowing some movement into ghosts, for particle exchange
KOKKOS_INLINE_FUNCTION bool in_block(
    const GRCoordinates& G, const double X[GR_DIM], const double& x_max_mesh)
{
    // Check just outside the faces of the "interior" of the domain, to let particles move
    // into ghost zones.
    Real phi = phi_of(X[3]);
    if (X[1] > G.Xf<X1DIR>(G.ng - 1) && (X[2] >= G.Xf<X2DIR>(G.ng - 1) || G.n2 == 1) &&
        (phi >= G.Xf<X3DIR>(G.ng - 1) || G.n3 == 1) &&
        /* ======OMITTED====== */ (X[2] < G.Xf<X2DIR>(G.n2 - G.ng + 1) || G.n2 == 1) &&
        (phi < G.Xf<X3DIR>(G.n3 - G.ng + 1) || G.n3 == 1)) {
        // If x is in the zone itelf, or we're the last zone in x1 on the mesh, the
        // particle is in our domain
        return (X[1] < G.Xf<X1DIR>(G.n1 - G.ng + 1) ||
                (G.Xf<X1DIR>(G.n1 - G.ng) > 0.99 * x_max_mesh));
    }
    return false;
}

/**
 * Choose stepsize according to inverse Kcon, dramatically decreasing the step
 * toward the coordinate pole and EH.
 *
 * Use the sum of inverses by default; the strict minimum seems to occasionally
 * overstep even for small eps
 *
 * TODO this is the geometry step but dictates physics as well.
 * Optionally skip geometry steps near the pole for accuracy
 */
// TODO pick one or runtime
#define STEP_STRICT_MIN 0
KOKKOS_INLINE_FUNCTION double stepsize(const CoordinateEmbedding& coords,
    double X[GR_DIM], double Kcon[GR_DIM], const double& eps)
{
    if (coords.is_spherical()) {
        double r = coords.X1_to_embed(X[1]);
        double deh = fmin(fabs(r - coords.get_horizon()), 0.1);
        double dlx1 = eps * (10 * deh) / (fabs(Kcon[1]) + SMALL_NUM * SMALL_NUM);

        // Make the step cautious near the pole, improving accuracy of Stokes U
        double cut = 0.02;
        double lx2 = coords.stopx(2) - coords.startx(2);
        double dpole = fmin(fabs(X[2] / lx2), fabs((coords.stopx(2) - X[2]) / lx2));
        double d2fac = (dpole < cut) ? dpole / 3 : fmin(cut / 3 + (dpole - cut) * 10., 1);
        double dlx2 = eps * d2fac / (fabs(Kcon[2]) + SMALL_NUM * SMALL_NUM);

        double dlx3 = eps / (fabs(Kcon[3]) + SMALL_NUM * SMALL_NUM);

        if (STEP_STRICT_MIN) {
            return fmin(fmin(dlx1, dlx2), dlx3);
        } else {
            double idlx1 = 1. / (fabs(dlx1) + SMALL_NUM * SMALL_NUM);
            double idlx2 = 1. / (fabs(dlx2) + SMALL_NUM * SMALL_NUM);
            double idlx3 = 1. / (fabs(dlx3) + SMALL_NUM * SMALL_NUM);

            return 1. / (idlx1 + idlx2 + idlx3);
        }
    } else {
        return eps;
    }
}

KOKKOS_INLINE_FUNCTION void geodesic_substep(const double conn[GR_DIM][GR_DIM][GR_DIM],
    const double Xi[GR_DIM], const double Kconi[GR_DIM], const double Xh[GR_DIM],
    const double Kconh[GR_DIM], double Xf[GR_DIM], double Kconf[GR_DIM], const double& dl)
{
    // Single half- or full-step
    DLOOP1
        Xf[mu] = Xi[mu] + dl * Kconh[mu];

    double dKcon[GR_DIM] = {0., 0., 0., 0.};
    DLOOP3
        dKcon[mu] -= dl * conn[mu][nu][lam] * Kconh[nu] * Kconh[lam];
    DLOOP1
        Kconf[mu] = Kconi[mu] + dKcon[mu];
}

KOKKOS_INLINE_FUNCTION double geodesic_step(const CoordinateEmbedding& coords,
    double X[GR_DIM], double Kcon[GR_DIM], const double& eps, const double& conn_delta,
    const bool& back)
{
    // This stepsize function can be troublesome inside of R = 2M,
    // and should be used cautiously in this region.
    double dl = (back) ? -stepsize(coords, X, Kcon, eps) : stepsize(coords, X, Kcon, eps);

    // Basic predictor-corrector
    // TODO: higher order so we can trace back instead of store!!

    /** half-step **/
    double lconn[GR_DIM][GR_DIM][GR_DIM];
    coords.conn_native(X, conn_delta, lconn);
    double Xh[GR_DIM], Kconh[GR_DIM];
    geodesic_substep(lconn, X, Kcon, X, Kcon, Xh, Kconh, 0.5 * dl);

    /** full step **/
    coords.conn_native(Xh, conn_delta, lconn);
    geodesic_substep(lconn, X, Kcon, Xh, Kconh, X, Kcon, dl);

    return fabs(dl);
}

/*
 * parallel transport N over dl
 */
KOKKOS_INLINE_FUNCTION void parallel_transport_N(
    const double conn[GR_DIM][GR_DIM][GR_DIM], const double Kh[GR_DIM],
    const Kokkos::complex<double> Ni[GR_DIM][GR_DIM],
    const Kokkos::complex<double> Nh[GR_DIM][GR_DIM],
    Kokkos::complex<double> Nf[GR_DIM][GR_DIM], const double& dl)
{
    /* push N */
    DLOOP2
        Nf[mu][nu] = Ni[mu][nu];
    DLOOP4
        Nf[mu][nu] -= (conn[mu][lam][kap] * Nh[lam][nu] * Kh[kap] +
                          conn[nu][lam][kap] * Nh[mu][lam] * Kh[kap]) *
                      dl;
}

/*
 * parallel transport N over dl
 */
KOKKOS_INLINE_FUNCTION double geodesic_parallel_transport_step(
    const CoordinateEmbedding& coords, double X[GR_DIM], double Kcon[GR_DIM],
    Kokkos::complex<double> N[GR_DIM][GR_DIM], const double& eps,
    const double& conn_delta, const bool& back = false)
{
    // This stepsize function can be troublesome inside of R = 2M,
    // and should be used cautiously in this region.
    double dl = (back) ? -stepsize(coords, X, Kcon, eps) : stepsize(coords, X, Kcon, eps);

    // Basic predictor-corrector
    // TODO: higher order so we can trace back instead of store!!

    /** half-step **/
    double lconn[GR_DIM][GR_DIM][GR_DIM];
    coords.conn_native(X, conn_delta, lconn);
    double Xh[GR_DIM], Kconh[GR_DIM];
    geodesic_substep(lconn, X, Kcon, X, Kcon, Xh, Kconh, 0.5 * dl);
    Kokkos::complex<double> Nh[GR_DIM][GR_DIM];
    parallel_transport_N(lconn, Kcon, N, N, Nh, 0.5 * dl);

    /** full step **/
    coords.conn_native(Xh, conn_delta, lconn);
    geodesic_substep(lconn, X, Kcon, Xh, Kconh, X, Kcon, dl);
    parallel_transport_N(lconn, Kconh, N, Nh, N, dl);

    return fabs(dl);
}

/*
 * parallel transport N from stored trajectory
 */
KOKKOS_INLINE_FUNCTION void path_parallel_transport_step(
    const CoordinateEmbedding& coords, double Xi[GR_DIM], double Kconi[GR_DIM],
    double Xf[GR_DIM], double Kconf[GR_DIM], Kokkos::complex<double> N[GR_DIM][GR_DIM],
    const double& conn_delta, const double& dl)
{
    // Use stored path to transport N with RK2

    /** half-step **/
    double lconn[GR_DIM][GR_DIM][GR_DIM];
    coords.conn_native(Xi, conn_delta, lconn);
    Kokkos::complex<double> Nh[GR_DIM][GR_DIM];
    parallel_transport_N(lconn, Kconi, N, N, Nh, 0.5 * dl);

    // TODO do we lose accuracy using 'f' here/superstepping by 2?
    double Xh[GR_DIM], Kconh[GR_DIM];
    DLOOP1
        Xh[mu] = (Xi[mu] + Xf[mu]) / 2;
    DLOOP1
        Kconh[mu] = (Kconi[mu] + Kconf[mu]) / 2;

    /** full step **/
    coords.conn_native(Xh, conn_delta, lconn);
    parallel_transport_N(lconn, Kconh, N, Nh, N, dl);
}

}
