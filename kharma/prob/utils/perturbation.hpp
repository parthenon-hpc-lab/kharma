/*
 *  File: perturbation.hpp
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

#include "Kokkos_Random.hpp"
#include <random>

/**
 * Perturb the internal energy by a uniform random proportion per cell.
 * Resulting internal energies will be between u \pm u*u_jitter/2
 * i.e. u_jitter=0.1 -> \pm 5% randomization, 0.95u to 1.05u
 *
 * @param u_jitter see description
 * @param rng_seed is added to the MPI rank to seed the GSL RNG
 */
TaskStatus PerturbU(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();
    auto rho = rc->Get("prims.rho").data;
    auto u = rc->Get("prims.u").data;

    const Real u_jitter = pin->GetReal("perturbation", "u_jitter");
    // Don't jitter values set by floors, by default
    Real rho_min = pin->DoesParameterExist("floors", "rho_min_geom")
                       ? pin->GetReal("floors", "rho_min_geom")
                       : pin->GetReal("floors", "rho_min_const");
    const Real jitter_above_rho =
        pin->GetOrAddReal("perturbation", "jitter_above_rho", rho_min);
    // Note we add the MeshBlock gid to this value when seeding RNG,
    // to get a new sequence for every block
    const int rng_seed = pin->GetOrAddInteger("perturbation", "rng_seed", 31337);
    // Print real seed used for all blocks, to ensure they're different
    if (pmb->packages.Get("Globals")->Param<int>("verbose") > 1) {
        std::cout << "Seeding RNG in block " << pmb->gid << " with value "
                  << rng_seed + pmb->gid << std::endl;
    }
    const bool serial = pin->GetOrAddInteger("perturbation", "serial", false);

    // Should we jitter ghosts? If first boundary sync doesn't work it's marginally less
    // disruptive
    IndexDomain domain = IndexDomain::interior;
    const int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    const int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    const int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    // HYERIN (07/26/23) get fx1min and fx1max
    const Real fx1min = pin->GetOrAddReal("parthenon/mesh", "restart_x1min", -1);
    const Real fx1max = pin->GetOrAddReal("parthenon/mesh", "restart_x1max", -1);
    const bool rstf_exists = ((fx1min > 0) && (fx1max > 0)); // restart file exists?
    auto& G = pmb->coords;

    if (serial) {
        // Serial version
        // Probably guarantees better determinism, but CPU single-thread only
        std::mt19937 gen(rng_seed + pmb->gid);
        std::uniform_real_distribution<Real> dis(-u_jitter / 2, u_jitter / 2);

        auto u_host = u.GetHostMirrorAndCopy();
        for (int k = ks; k <= ke; k++)
            for (int j = js; j <= je; j++)
                for (int i = is; i <= ie; i++) u_host(k, j, i) *= 1. + dis(gen);
        u.DeepCopy(u_host);
    } else {
        // Kokkos version
        typedef typename Kokkos::Random_XorShift64_Pool<> RandPoolType;
        RandPoolType rand_pool(rng_seed + pmb->gid);
        typedef typename RandPoolType::generator_type gen_type;
        pmb->par_for("perturb_u", ks, ke, js, je, is, ie,
                     KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
            {
                GReal X[GR_DIM];
                G.coord(k, j, i, Loci::center, X);
                if ((!rstf_exists) || (X[1] < fx1min) || (X[1] > fx1max)) {
                    if (rho(k, j, i) > jitter_above_rho) {
                        gen_type rgen = rand_pool.get_state();
                        u(k, j, i) *= 1. + Kokkos::rand<gen_type, Real>::draw(
                                               rgen, -u_jitter / 2, u_jitter / 2);
                        rand_pool.free_state(rgen);
                    }
                }
            });
    }

    return TaskStatus::complete;
}
