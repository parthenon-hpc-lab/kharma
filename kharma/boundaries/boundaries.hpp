/*
 *  File: boundaries.hpp
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

#include "boundary_types.hpp"
#include "dirichlet.hpp"
#include "grmhd_functions.hpp"
#include "one_block_transmit.hpp"

/**
 * This package has any functions related to KHARMA's treatment of "domain" boundary
 * conditions: the exterior simulation edges, as opposed to internal meshblock boundaries.
 *
 * This package implements Parthenon's "user" boundary conditions in order to add some
 * features related to GRMHD.
 */
namespace KBoundaries
{

/**
 * Choose which boundary conditions will be used based on inputs,
 * declare any fields needed to store e.g. constant boundary conditions
 */
std::shared_ptr<KHARMAPackage> Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages);

/**
 * Generic KHARMA override function for Parthenon domain boundary conditions.
 * This is registered as the "user" boundary condition with Parthenon, and
 * wraps Parthenon's reflecting or outflow boundary conditions wherever those
 * would be applied.
 *
 */
void ApplyBoundary(
    std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse);
// Template version to conform to Parthenon's calling convention. See above.
template<IndexDomain domain>
inline void ApplyBoundaryTemplate(std::shared_ptr<MeshBlockData<Real>>& rc, bool coarse)
{
    ApplyBoundary(rc, domain, coarse);
}

/**
 * Fix fluxes on physical boundaries.
 * 1. Ensure no inflow of density onto the domain
 * 2. Ensure flux through the size-zero faces on poles is zero
 * OR
 * 2. Ensure that fluxes through & around the pole reflect a half-zone excision
 */
TaskStatus FixFlux(MeshData<Real>* rc);

/**
 * When fluxes are allowed across the pole, they are computed by assuming half-size zones
 * next to the pole, with the remaining space excised.  The flux divergence, however,
 * divides by the full zone volume (as with every other zone on the grid).
 * This doubles the flux-divergence term to reflect changes to the larger, real zones.
 * Since *only* the divergence term should be doubled, this must be run *first* after
 * FluxDivergence, which is ensured by Packages::AddSource.
 */
void AddSource(MeshData<Real>* md, MeshData<Real>* mdudt, IndexDomain domain);

// INTERNAL FUNCTIONS

/**
 * Check for inflowing material on an outflow boundary, and
 * reset the velocity of such material so it is no longer inflowing.
 */
void CheckInflow(
    std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse);

/**
 * Check for velocity toward the simulation domain in a zone, and eliminate it.
 */
KOKKOS_INLINE_FUNCTION void check_inflow(const GRCoordinates& G,
    const VariablePack<Real>& P, const IndexDomain domain, const int& index_u1,
    const int& k, const int& j, const int& i)
{
    // TODO fewer temporaries?
    Real uvec[NVEC], ucon[GR_DIM];
    VLOOP
        uvec[v] = P(index_u1 + v, k, j, i);
    GRMHD::calc_ucon(G, uvec, k, j, i, Loci::center, ucon);

    if (((ucon[1] > 0.) && (domain == IndexDomain::inner_x1)) ||
        ((ucon[1] < 0.) && (domain == IndexDomain::outer_x1))) {
        // Find gamma and remove it from primitive velocity
        double gamma = GRMHD::lorentz_calc(G, uvec, k, j, i, Loci::center);
        VLOOP
            uvec[v] /= gamma;

        // Reset radial velocity so radial 4-velocity is zero
        Real alpha = 1. / m::sqrt(-G.gcon(Loci::center, j, i, 0, 0));
        Real beta1 = G.gcon(Loci::center, j, i, 0, 1) * alpha * alpha;
        uvec[V1] = beta1 / alpha;

        // Now find new gamma and put it back in
        Real vsq = G.gcov(Loci::center, j, i, 1, 1) * uvec[V1] * uvec[V1] +
                   G.gcov(Loci::center, j, i, 2, 2) * uvec[V2] * uvec[V2] +
                   G.gcov(Loci::center, j, i, 3, 3) * uvec[V3] * uvec[V3] +
                   2. * (G.gcov(Loci::center, j, i, 1, 2) * uvec[V1] * uvec[V2] +
                            G.gcov(Loci::center, j, i, 1, 3) * uvec[V1] * uvec[V3] +
                            G.gcov(Loci::center, j, i, 2, 3) * uvec[V2] * uvec[V3]);

        vsq = clip(vsq, 1.e-13, 1. - 1. / (50. * 50.));

        gamma = 1. / m::sqrt(1. - vsq);

        VLOOP
            uvec[v] *= gamma;
        VLOOP
            P(index_u1 + v, k, j, i) = uvec[v];
    }
}

}
