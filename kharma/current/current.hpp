/*
 *  File: current.hpp
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

#include <parthenon/parthenon.hpp>

#include "decs.hpp"
#include "grmhd_functions.hpp"
#include "matrix.hpp"

namespace Current
{
/**
 * Initialize output field jcon
 */
std::shared_ptr<KHARMAPackage> Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages);

/**
 * Fill outputs, namely jcon.  Just calls CalculateCurrent below.
 */
void FillOutput(MeshBlock* pmb, ParameterInput* pin);

/**
 * Calculate the 4-current j^nu (jcon), given the MeshBlockData at the beginning and end
 * of a step, and the time between them.
 */
TaskStatus CalculateCurrent(
    MeshBlockData<Real>* rc0, MeshBlockData<Real>* rc1, const double& dt);

/**
 * Return mu, nu component of contravarient Maxwell tensor at grid zone i, j, k,
 * multiplied by the root negative metric determinant sqrt(-g). Easier to calculate than
 * the regular version as I needn't divide by gdet.
 */
KOKKOS_INLINE_FUNCTION double get_gdet_Fcon(const GRCoordinates& G, GridVector uvec,
    GridVector B_P, const int& mu, const int& nu, const int& k, const int& j,
    const int& i)
{
    if (mu == nu) return 0.;

    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, uvec, B_P, k, j, i, Loci::center, Dtmp);
    double Fcon = 0.;
    for (int kap = 0; kap < GR_DIM; kap++) {
        for (int lam = 0; lam < GR_DIM; lam++) {
            Fcon -= antisym(mu, nu, kap, lam) * Dtmp.ucov[kap] * Dtmp.bcov[lam];
        }
    }

    return Fcon;
}

} // namespace GRMHD
