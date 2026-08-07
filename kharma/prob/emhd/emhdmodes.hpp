/*
 *  File: emhdmodes.hpp
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
 * Initialization of analytic wave modes in magnetized plasma w/viscosity and heat
 * conduction
 *
 * Note the end time is not set -- even after exactly 1 period, EMHD modes will
 * have lost amplitude due to having viscosity, which is kind of the point
 */
TaskStatus InitializeEMHDModes(
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

    const Real amp = pin->GetOrAddReal("emhdmodes", "amp", 1e-8);

    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb->packages);
    const auto& grmhd_pars = pmb->packages.Get("GRMHD")->AllParams();
    const Real& gam = grmhd_pars.Get<Real>("gamma");

    // TODO actually calculate the mode?  Figure something out
    const Real omega_real =
        pin->GetOrAddReal("emhdmodes", "omega_real", -0.5533585207638141);
    const Real omega_imag =
        pin->GetOrAddReal("emhdmodes", "omega_imag", -3.6262571286888425);

    // START POSSIBLE ARGS: take all these as parameters in pin?
    // Also note this is 2D only for now
    // Mean state
    const Real rho0 = 1.;
    const Real u0 = 2.;
    const Real u10 = 0.;
    const Real u20 = 0.;
    const Real u30 = 0.;
    const Real B10 = 0.1;
    const Real B20 = 0.3;
    const Real B30 = 0.;
    const Real q0 = 0.;
    const Real delta_p0 = 0.;

    // Wavevector
    const Real k1 = 2. * M_PI;
    const Real k2 = 4. * M_PI;
    // END POSSIBLE ARGS

    // Set magnetic field parameters for our field transport package
    pin->GetOrAddString("b_field", "type", "wave");
    pin->GetOrAddReal("b_field", "B10", B10);
    pin->GetOrAddReal("b_field", "B20", B20);
    pin->GetOrAddReal("b_field", "B30", B30);
    pin->GetOrAddReal("b_field", "k1", k1);
    pin->GetOrAddReal("b_field", "k2", k2);

    pin->GetOrAddReal("b_field", "amp_B1", amp * (-0.05973794979640743));
    pin->GetOrAddReal("b_field", "amp2_B1", amp * (0.03351707506150924));

    pin->GetOrAddReal("b_field", "amp_B2", amp * (0.02986897489820372));
    pin->GetOrAddReal("b_field", "amp2_B2", amp * (-0.016758537530754618));

    IndexDomain domain = IndexDomain::interior;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
    pmb->par_for("emhdmodes_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
        {
            Real X[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, X);
            const Real cos_phi = m::cos(k1 * X[1] + k2 * X[2]);
            const Real sin_phi = m::sin(k1 * X[1] + k2 * X[2]);

            // Perturbations: no higher-order terms
            const Real drho = amp * (((-0.518522524082246) * cos_phi) +
                                        ((0.1792647678001878) * sin_phi));
            const Real du = amp * ((0.5516170736393813) * cos_phi);
            const Real du1 = amp * (((0.008463122479547856) * cos_phi) +
                                       ((-0.011862022608466367) * sin_phi));
            const Real du2 = amp * (((-0.16175466371870734) * cos_phi) +
                                       ((0.034828080823603294) * sin_phi));
            const Real du3 = 0.;
            const Real dq = amp * (((0.5233486841539436) * cos_phi) -
                                      ((0.04767672501939603) * sin_phi));
            const Real ddelta_p = amp * (((0.2909106062057657) * cos_phi) -
                                            ((0.02159452055336572) * sin_phi));

            // Initialize primitives
            rho(k, j, i) = rho0 + drho;
            u(k, j, i) = u0 + du;
            uvec(V1, k, j, i) = u10 + du1;
            uvec(V2, k, j, i) = u20 + du2;
            uvec(V3, k, j, i) = u30 + du3;
            q(k, j, i) = q0 + dq;
            dP(k, j, i) = delta_p0 + ddelta_p;

            if (emhd_params.higher_order_terms) {
                Real tau, chi_e, nu_e;
                // Zeros are q, dP, and bsq, only needed for torus closure
                EMHD::set_parameters(G, rho(k, j, i), u(k, j, i), 0., 0., 0., emhd_params,
                    gam, j, i, tau, chi_e, nu_e);
                Real Theta = (gam - 1) * u(k, j, i) / rho(k, j, i);
                q(k, j, i) *= (chi_e != 0)
                                  ? m::sqrt(tau / (chi_e * rho(k, j, i) * Theta * Theta))
                                  : 0.;
                dP(k, j, i) *=
                    (nu_e != 0) ? m::sqrt(tau / (nu_e * rho(k, j, i) * Theta)) : 0.;
            }
        });

    Flux::BlockPtoU(rc.get(), IndexDomain::interior, false);

    return TaskStatus::complete;
}
