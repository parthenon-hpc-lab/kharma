/*
 *  File: mhdmodes.hpp
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
 * Initialization for different analytic wave modes in magnetized plasma.
 * Note this assumes ideal EOS with gamma=4/3!
 *
 * @param nmode: type of linear wave, from:
 * 0. Entropy, static mode
 * 1. Slow mode
 * 2. Alfven wave
 * 3. Fast mode
 *
 * @param dir: direction of wave. 0 = components of each
 *
 * Note this SETS the stopping time corresponding to advection by 1 wavelength.
 * Generally this is what we want for tests (run by 1 cycle and compare).
 * Modify function or reset tlim after to override.
 */
TaskStatus InitializeMHDModes(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;

    const auto& G = pmb->coords;

    const int nmode = pin->GetOrAddInteger("mhdmodes", "nmode", 1);
    const bool one_period = pin->GetOrAddBoolean("mhdmodes", "one_period", nmode != 0);

    // Mean state
    const Real rho0 = pin->GetOrAddReal("mhdmodes", "rho0", 1.);
    const Real u0 = pin->GetOrAddReal("mhdmodes", "u0", 1.);
    const Real u10 = pin->GetOrAddReal("mhdmodes", "u10", 0.);
    const Real u20 = pin->GetOrAddReal("mhdmodes", "u20", 0.);
    const Real u30 = pin->GetOrAddReal("mhdmodes", "u30", 0.);

    // Wave parameters
    // dir sets "Faux-2D" plane orientation, good for asymmetry bugs
    // Set to 0 for "full" 3D wave.
    const int dir = pin->GetOrAddInteger("mhdmodes", "dir", 0);
    const Real amp = pin->GetOrAddReal("mhdmodes", "amp", 1.e-4);
    const Real phase = pin->GetOrAddReal("mhdmodes", "phase", 0.);

    // Note the modes below don't work right if you manually set these
    // TODO generate modes on the fly for any k values
    const Real k1 = pin->GetOrAddReal("mhdmodes", "k1", (dir == 1) ? 0. : 2. * M_PI);
    const Real k2 = pin->GetOrAddReal("mhdmodes", "k2", (dir == 2) ? 0. : 2. * M_PI);
    const Real k3 = pin->GetOrAddReal("mhdmodes", "k3", (dir == 3) ? 0. : 2. * M_PI);
    // Likewise
    const Real B10 =
        pin->GetOrAddReal("mhdmodes", "B10", (dir == 0 || dir == 3) ? 1.0 : 0.);
    const Real B20 = pin->GetOrAddReal("mhdmodes", "B20", (dir == 1) ? 1.0 : 0.);
    const Real B30 = pin->GetOrAddReal("mhdmodes", "B30", (dir == 2) ? 1.0 : 0.);

    std::complex<Real> omega;
    Real drho = 0, du = 0;
    Real du1 = 0, du2 = 0, du3 = 0;
    Real dB1 = 0, dB2 = 0, dB3 = 0;
    // Eigenmode definitions
    if (dir == 0) {
        // 3D (1,1,1) wave
        if (nmode == 0) { // Entropy
            drho = 1.;
        } else if (nmode == 1) { // Slow
            omega = 2.35896379113i;
            drho = 0.556500332363;
            du = 0.742000443151;
            du1 = -0.282334999306;
            du2 = 0.0367010491491;
            du3 = 0.0367010491491;
            dB1 = -0.195509141461;
            dB2 = 0.0977545707307;
            dB3 = 0.0977545707307;
        } else if (nmode == 2) { // Alfven
            omega = -3.44144232573i;
            du2 = -0.339683110243;
            du3 = 0.339683110243;
            dB2 = 0.620173672946;
            dB3 = -0.620173672946;
        } else { // Fast
            omega = 6.92915162882i;
            drho = 0.481846076323;
            du = 0.642461435098;
            du1 = -0.0832240462505;
            du2 = -0.224080007379;
            du3 = -0.224080007379;
            dB1 = 0.406380545676;
            dB2 = -0.203190272838;
            dB3 = -0.203190272838;
        }
    } else {
        // 2D (1,1,0), (1,0,1), (0,1,1) wave
        if (nmode == 0) { // Entropy
            drho = 1.;
        } else if (nmode == 1) { // Slow
            omega = 2.41024185339i;
            drho = 0.558104461559;
            du = 0.744139282078;
            if (dir == 1) {
                du2 = -0.277124827421;
                du3 = 0.0630348927707;
                dB2 = -0.164323721928;
                dB3 = 0.164323721928;
            } else if (dir == 2) {
                du3 = -0.277124827421;
                du1 = 0.0630348927707;
                dB3 = -0.164323721928;
                dB1 = 0.164323721928;
            } else if (dir == 3) {
                du1 = -0.277124827421;
                du2 = 0.0630348927707;
                dB1 = -0.164323721928;
                dB2 = 0.164323721928;
            }
        } else if (nmode == 2) { // Alfven
            omega = 3.44144232573i;
            if (dir == 1) {
                du1 = 0.480384461415;
                dB1 = 0.877058019307;
            } else if (dir == 2) {
                du2 = 0.480384461415;
                dB2 = 0.877058019307;
            } else if (dir == 3) {
                du3 = 0.480384461415;
                dB3 = 0.877058019307;
            }
        } else { // Fast
            omega = 5.53726217331i;
            drho = 0.476395427447;
            du = 0.635193903263;
            if (dir == 1) {
                du2 = -0.102965815319;
                du3 = -0.316873207561;
                dB2 = 0.359559114174;
                dB3 = -0.359559114174;
            } else if (dir == 2) {
                du3 = -0.102965815319;
                du1 = -0.316873207561;
                dB3 = 0.359559114174;
                dB1 = -0.359559114174;
            } else if (dir == 3) {
                du1 = -0.102965815319;
                du2 = -0.316873207561;
                dB1 = 0.359559114174;
                dB2 = -0.359559114174;
            }
        }
    }

    // Record the parameters we set via nmode
    // This might be useful to read when checking, too...
    pin->SetReal("mhdmodes", "omega_real", omega.real());
    pin->SetReal("mhdmodes", "omega_imag", omega.imag());
    pin->SetReal("mhdmodes", "drho", drho);
    pin->SetReal("mhdmodes", "du", du);
    pin->SetReal("mhdmodes", "du1", du1);
    pin->SetReal("mhdmodes", "du2", du2);
    pin->SetReal("mhdmodes", "du3", du3);
    pin->SetReal("mhdmodes", "dB1", dB1);
    pin->SetReal("mhdmodes", "dB2", dB2);
    pin->SetReal("mhdmodes", "dB3", dB3);

    // Set B field parameters for our mode
    pin->GetOrAddString("b_field", "type", "wave");
    pin->GetOrAddReal("b_field", "B10", B10);
    pin->GetOrAddReal("b_field", "B20", B20);
    pin->GetOrAddReal("b_field", "B30", B30);
    pin->GetOrAddReal("b_field", "amp_B1", amp * dB1);
    pin->GetOrAddReal("b_field", "amp_B2", amp * dB2);
    pin->GetOrAddReal("b_field", "amp_B3", amp * dB3);
    pin->GetOrAddReal("b_field", "k1", k1);
    pin->GetOrAddReal("b_field", "k2", k2);
    pin->GetOrAddReal("b_field", "k3", k3);
    pin->GetOrAddReal("b_field", "phase", phase);

    IndexDomain domain = IndexDomain::interior;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
    pmb->par_for("mhdmodes_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                 KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            Real X[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, X);
            Real mode = amp * m::cos(k1 * X[1] + k2 * X[2] + k3 * X[3]);
            rho(k, j, i) = rho0 + drho * mode;
            u(k, j, i) = u0 + du * mode;
            uvec(V1, k, j, i) = u10 + du1 * mode;
            uvec(V2, k, j, i) = u20 + du2 * mode;
            uvec(V3, k, j, i) = u30 + du3 * mode;
        });

    // Override end time to be exactly 1 period for moving modes, unless we set otherwise
    if (one_period) {
        pin->SetReal("parthenon/time", "tlim", 2. * M_PI / m::abs(omega.imag()));
    }

    return TaskStatus::complete;
}
