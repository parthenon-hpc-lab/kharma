/*
 *  File: noh.hpp
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

using namespace parthenon;

/**
 * Noh shock tube test.
 */
TaskStatus InitializeNoh(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    const Real mach = pin->GetOrAddReal("noh", "mach", 49.);
    const Real rho0 = pin->GetOrAddReal("noh", "rho", 1.0);
    const Real v0 = pin->GetOrAddReal("noh", "v0", 1.e-3);
    bool zero_ug = pin->GetOrAddBoolean("noh", "zero_ug", false);
    bool centered = pin->GetOrAddBoolean("noh", "centered", true);
    bool set_tlim = pin->GetOrAddBoolean("noh", "set_tlim", false);

    const GReal x1min = pin->GetReal("parthenon/mesh", "x1min");
    const GReal x1max = pin->GetReal("parthenon/mesh", "x1max");
    const GReal center = (x1min + x1max) / 2.;

    // Given Mach and knowing that v = 1e-3 and rho = 1, we calculate u
    double cs2 = m::pow(v0, 2) / m::pow(mach, 2);
    double gamma = 1. / m::sqrt(1. - m::pow(v0, 2)); // Since we are in flat space
    const Real P = (zero_ug) ? 0. : rho0 * cs2 / (gam * (gam - 1) - cs2 * gam);

    if (set_tlim) {
        pin->SetReal("parthenon/time", "tlim", 0.6 * (x1max - x1min) / v0);
    }

    IndexDomain domain = IndexDomain::interior;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
    pmb->par_for("noh_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                 KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            rho(k, j, i) = rho0;
            u(k, j, i) = P / (gam - 1.);
            uvec(1, k, j, i) = 0.0;
            uvec(2, k, j, i) = 0.0;
        });
    const auto& G = pmb->coords;
    if (centered) {
        pmb->par_for("noh_cent", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                     KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
            {
                Real X[GR_DIM];
                G.coord_embed(k, j, i, Loci::center, X);
                const bool lhs = X[1] < center;
                uvec(0, k, j, i) = ((lhs) ? v0 : -v0) * gamma;
            });
    } else {
        pmb->par_for("noh_left", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                     KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
            {
                u(k, j, i) = P / (gam - 1.);
                uvec(0, k, j, i) = -v0 * gamma;
            });
    }

    return TaskStatus::complete;
}
