/*
 *  File: anisotropic_conduction.hpp
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

using namespace parthenon;

/**
 * Anisotropic heat conduction problem, see Chandra+ 2017
 */
TaskStatus InitializeAnisotropicConduction(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    // It is well and good this problem should cry if EMHD is disabled.
    GridVector q = rc->Get("prims.q").data;
    GridVector dP = rc->Get("prims.dP").data;

    const auto& G = pmb->coords;

    const Real A = pin->GetOrAddReal("anisotropic_conduction", "A", 0.2);
    const Real Rsq = pin->GetOrAddReal("anisotropic_conduction", "Rsq", 0.005);
    const Real B0 = pin->GetOrAddReal("anisotropic_conduction", "B0", 1e-4);
    const Real k0 = pin->GetOrAddReal("anisotropic_conduction", "k", 4.);

    const Real R = m::sqrt(Rsq);

    pin->GetOrAddString("b_field", "type", "wave");
    pin->GetOrAddReal("b_field", "phase", 0.);
    // Constant B1
    pin->GetOrAddReal("b_field", "B10", B0);
    // Amp & wavenumber of sin() for B2
    pin->GetOrAddReal("b_field", "amp2_B2", B0);
    pin->GetOrAddReal("b_field", "k1", 2 * M_PI * k0);

    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
    pmb->par_for("anisotropic_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                 KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            Real X[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, X);
            GReal r = m::sqrt(m::pow((X[1] - 0.5), 2) + m::pow((X[2] - 0.5), 2));

            // Initialize primitives
            rho(k, j, i) = 1 - (A * m::exp(-m::pow(r, 2) / m::pow(R, 2)));
            u(k, j, i) = 1.;
            uvec(0, k, j, i) = 0.;
            uvec(1, k, j, i) = 0.;
            uvec(2, k, j, i) = 0.;
            q(k, j, i) = 0.;
            dP(k, j, i) = 0.;
        });

    return TaskStatus::complete;
}
