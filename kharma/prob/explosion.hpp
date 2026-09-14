/*
 *  File: explosion.hpp
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

#include <complex>

#include "decs.hpp"

using namespace std::literals::complex_literals;
using namespace parthenon;

/**
 * Initialization of the strong cylindrical explosion of Komissarov 1999 section 7.3
 *
 * Note the problem setup assumes gamma=4/3
 *
 * Originally run on 2D Cartesian domain -6.0, 6.0 with a 200x200 grid, to tlim=4.0
 */
TaskStatus InitializeExplosion(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();

    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;

    const auto& G = pmb->coords;

    Real gam = pmb->packages.Get("eos")->Param<Real>("gm1") + 1.0;

    // All options are runtime options!
    const bool linear_ramp = pin->GetOrAddBoolean("explosion", "linear_ramp", false);
    const Real u_out = pin->GetOrAddReal("explosion", "u_out", 3.e-5 / (gam - 1));
    const Real rho_out = pin->GetOrAddReal("explosion", "rho_out", 1.e-4);
    const Real u_in = pin->GetOrAddReal("explosion", "u_in", 1.0 / (gam - 1));
    const Real rho_in = pin->GetOrAddReal("explosion", "rho_in", 1.e-2);

    // One buffer zone of linear decline, r_in -> r_out
    // Exponential decline inside here i.e. linear in logspace
    const Real r_in = pin->GetOrAddReal("explosion", "r_in", 0.8);
    const Real r_out = pin->GetOrAddReal("explosion", "r_out", 1.0);
    // Circle center
    const Real xoff = pin->GetOrAddReal("explosion", "xoff", 0.0);
    const Real yoff = pin->GetOrAddReal("explosion", "yoff", 0.0);

    IndexDomain domain = IndexDomain::interior;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
    pmb->par_for("explosion_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                 KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            Real X[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, X);
            const GReal rx = X[1] - xoff;
            const GReal ry = X[2] - yoff;
            const Real r = m::sqrt(rx * rx + ry * ry);

            if (r < r_in) {
                rho(k, j, i) = rho_in;
                u(k, j, i) = u_in;
            } else if (r >= r_in && r <= r_out) {
                const Real ramp = (r_out - r) / (r_out - r_in);

                if (linear_ramp) {
                    rho(k, j, i) = rho_out + ramp * (rho_in - rho_out);
                    u(k, j, i) = u_in + ramp * (u_in - u_out);
                } else {
                    const Real lrho_out = log(rho_out);
                    const Real lrho_in = log(rho_in);
                    const Real lu_out = log(u_out);
                    const Real lu_in = log(u_in);
                    rho(k, j, i) = m::exp(lrho_out + ramp * (lrho_in - lrho_out));
                    u(k, j, i) = m::exp(lu_out + ramp * (lu_in - lu_out));
                }
            } else {
                rho(k, j, i) = rho_out;
                u(k, j, i) = u_out;
            }
        });

    return TaskStatus::complete;
}
