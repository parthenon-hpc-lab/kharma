/*
 *  File: radmhdmodes.hpp
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
#include "radM1_solvers.hpp"
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
TaskStatus InitializeRadMHDModes(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;

    GridScalar Erad = rc->Get("prims.u_rad").data;
    GridVector Frad = rc->Get("prims.uvec_rad").data;

    const bool use_radm1 = pin->GetOrAddBoolean("radM1", "on", false);

    const Real gam = pmb->packages.Get("eos")->Param<Real>("gm1") + 1.0;

    const auto& G = pmb->coords;

    const int nmode = pin->GetOrAddInteger("mhdmodes", "nmode", 1);
    const bool one_period = pin->GetOrAddBoolean("mhdmodes", "one_period", nmode != 0);

    // Plasma mean gas state taken from file
    const Real rho0 = pin->GetOrAddReal("mhdmodes", "rho0", 1.);
    const Real u0 = pin->GetOrAddReal("mhdmodes", "u0", 0.009137055837563452);
    const Real u10 = pin->GetOrAddReal("mhdmodes", "u10", 0.);
    const Real u20 = pin->GetOrAddReal("mhdmodes", "u20", 0.);
    const Real u30 = pin->GetOrAddReal("mhdmodes", "u30", 0.);

    const Real u10rad = pin->GetOrAddReal("mhdmodes", "u10rad", 0.);
    const Real u20rad = pin->GetOrAddReal("mhdmodes", "u20rad", 0.);
    const Real u30rad = pin->GetOrAddReal("mhdmodes", "u30rad", 0.);

    const std::string wavetype = pin->GetOrAddString("mhdmodes", "wavetype", "sonic");
    const std::string regime = pin->GetOrAddString("mhdmodes", "regime", "puremhd");

    const Real k1 = pin->GetOrAddReal("mhdmodes", "k1", 2. * M_PI);
    const Real k2 = pin->GetOrAddReal("mhdmodes", "k2", 0.);
    const Real k3 = pin->GetOrAddReal("mhdmodes", "k3", 0.);
    const Real phase = pin->GetOrAddReal("mhdmodes", "phase", 0.);

    Real B10 = pin->GetOrAddReal("mhdmodes", "B10", 0.10075854437197568);
    Real B20 = pin->GetOrAddReal("mhdmodes", "B20", 0.10075854437197568);
    Real B30 = pin->GetOrAddReal("mhdmodes", "B30", 0.);

    std::complex<Real> omega;
    std::complex<Real> drho = 0, du = 0;
    std::complex<Real> du1 = 0, du2 = 0, du3 = 0;
    std::complex<Real> dB1 = 0, dB2 = 0, dB3 = 0;
    std::complex<Real> dErad = 0, dF1rad = 0, dF2rad = 0, dF3rad = 0;

    Real P = 0.0;
    if (regime == "thin")
        P = 0.1;
    else if (regime == "thick")
        P = 10;

    if (use_radm1) {
        const Real T0 = (gam - 1.0) * u0;
        const Real sigma_rad = 3.0 * P * (gam - 1.0) * u0 / (4.0 * T0 * T0 * T0 * T0);
        auto& radm1_pkg = pmb->packages.Get("RadM1");
        radm1_pkg->UpdateParam<Real>("const_sigma", sigma_rad);
        radm1_pkg->UpdateParam<Real>("const_kappa_a", P);
        radm1_pkg->UpdateParam<Real>("const_kappa_sc", 0.0);
    }

    if (wavetype == "sonic") {
        B10 = 0.;
        B20 = 0.;
        B30 = 0.;
        if (regime == "puremhd") {
            drho = 1.e-6;
            du = 1.5228426395939093e-8;
            du1 = 1.0000000000000002e-7;
            du2 = 0.0;
            dB2 = 0.0;
            omega = 0.6283185307179587;
        } else if (regime == "thin") {
            drho = 1.e-6;
            du = 1.5155652908079845e-8 + 7.696929719530536e-10i;
            du1 = 9.97992249118626e-8 + 2.552072175928721e-9i;
            du2 = 0.0;
            dB2 = 0.0;
            dErad = 1.3314776991134588e-13 + 3.6001746512388956e-11i;
            dF1rad = -2.5247126486226934e-10 + 7.400407810152034e-11i;
            dF2rad = 0.0;
            omega = 0.627057023634126 + 0.016035142398657175i;
        } else if (regime == "thick") {
            drho = 1.e-6;
            du = 1.1706978034894262e-8 + 1.881532710186292e-9i;
            du1 = 2.662507979198814e-7 + 6.33514446524509e-8i;
            du2 = 0.0;
            dB2 = 0.0;
            dErad = 2.0541918444857084e-7 + 1.4985861843019722e-7i;
            dF1rad = -2.0730815318727886e-8 + 3.7755564579364684e-8i;
            dF2rad = 0.0;
            omega = 1.67290310151504 + 0.39804886622888025i;
        }
    } else if (wavetype == "fast") {
        if (regime == "puremhd") {
            drho = 1.e-6;
            du = 1.522842639593907e-8;
            du1 = 1.602940583015828e-7;
            du2 = -9.790871410382318e-8;
            dB2 = 1.6230255678865884e-7;
            omega = 1.007157271948693;
        } else if (regime == "thin") {
            drho = 1.e-6;
            du = 1.5198360895974991e-8 + 4.815752909936621e-10i;
            du1 = 1.6025131429328265e-7 + 7.238312005077197e-10i;
            du2 = -9.795442630848571e-8 + 9.836789501779977e-10i;
            dB2 = 1.6234366410161697e-7 - 8.96662164240542e-10i;
            dErad = 1.4842118188293356e-12 + 6.063223162955078e-11i;
            dF1rad = -3.9543271084234507e-10 + 8.51051304663626e-11i;
            dF2rad = 2.3667952154599258e-10 + 2.1118238693659835e-11i;
            omega = 1.0068887034237715 + 0.004547965563908265i;
        } else if (regime == "thick") {
            drho = 1.e-6;
            du = 1.1730539980454472e-8 + 1.7129010401392157e-9i;
            du1 = 2.784991109850316e-7 + 5.2380393168228656e-8i;
            du2 = -2.8109315354609682e-8 + 6.2558750019767086e-9i;
            dB2 = 1.1016964855189374e-7 - 4.033370850235303e-9i;
            dErad = 2.072941184085973e-7 + 1.3636362726103654e-7i;
            dF1rad = -1.833308481505651e-8 + 3.636638174654657e-8i;
            dF2rad = 2.6758144678721757e-10 + 1.2427179588260004e-9i;
            omega = 1.7498615222037273 + 0.3291157167389043i;
        }
    } else if (wavetype == "slow") {
        if (regime == "puremhd") {
            drho = 1.e-6;
            du = 1.522842639593909e-8;
            du1 = 6.177069527516586e-8;
            du2 = 1.0011836806552759e-7;
            dB2 = -6.25515978604029e-8;
            omega = 0.3881167249671897;
        } else if (regime == "thin") {
            drho = 1.e-6;
            du = 1.50174235106495e-8 + 1.22298943455801e-9i;
            du1 = 6.15332754996702e-8 + 1.83139801648519e-9i;
            du2 = 9.89772118301622e-8 + 6.54185791061938e-9i;
            dB2 = -6.14882091832378e-8 - 5.88315338295397e-9i;
            dErad = 1.9170283012363e-13 + 2.18721458210053e-11i;
            dF1rad = -1.65180532693438e-12 + 7.1752041819694e-11i;
            dF2rad = -2.23678888272661e-10 - 7.43141463935117e-11i;
            omega = 0.386624972522161 + 0.0115070131087776i;
        } else if (regime == "thick") {
            drho = 1.e-6;
            du = 9.461886706340277e-9 + 1.2137574760252086e-9i;
            du1 = 8.342687348874559e-8 + 1.2082877629545738e-8i;
            du2 = 1.1363325579507992e-7 + 2.7269723955801375e-7i;
            dB2 = -8.03822945939874e-8 - 3.0311425160368743e-7i;
            dErad = 2.5966624942266354e-8 + 9.678910913258438e-8i;
            dF1rad = -1.9826286725607242e-8 + 5.476096510182763e-9i;
            dF2rad = 3.6607454416002e-9 - 1.1474975240229893e-9i;
            omega = 0.5241865057284164 + 0.0759189591904107i;
        }
    }

    // Record the parameters we set via nmode
    // This might be useful to read when checking, too...
    pin->SetReal("mhdmodes", "omega_real", omega.real());
    pin->SetReal("mhdmodes", "omega_imag", omega.imag());
    pin->SetReal("mhdmodes", "drho_real", drho.real());
    pin->SetReal("mhdmodes", "drho_imag", drho.imag());
    pin->SetReal("mhdmodes", "du_real", du.real());
    pin->SetReal("mhdmodes", "du_imag", du.imag());
    pin->SetReal("mhdmodes", "du1_real", du1.real());
    pin->SetReal("mhdmodes", "du1_imag", du1.imag());
    pin->SetReal("mhdmodes", "du2_real", du2.real());
    pin->SetReal("mhdmodes", "du2_imag", du2.imag());
    pin->SetReal("mhdmodes", "du3_real", du3.real());
    pin->SetReal("mhdmodes", "du3_imag", du3.imag());
    pin->SetReal("mhdmodes", "dB1_real", dB1.real());
    pin->SetReal("mhdmodes", "dB1_imag", dB1.imag());
    pin->SetReal("mhdmodes", "dB2_real", dB2.real());
    pin->SetReal("mhdmodes", "dB2_imag", dB2.imag());
    pin->SetReal("mhdmodes", "dB3_real", dB3.real());
    pin->SetReal("mhdmodes", "dB3_imag", dB3.imag());

    // Set B field parameters for our mode
    pin->GetOrAddString("b_field", "type", "wave");
    pin->GetOrAddReal("b_field", "B10", B10);
    pin->GetOrAddReal("b_field", "B20", B20);
    pin->GetOrAddReal("b_field", "B30", B30);
    pin->GetOrAddReal("b_field", "amp_B1", dB1.real());
    pin->GetOrAddReal("b_field", "amp_B2", dB2.real());
    pin->GetOrAddReal("b_field", "amp_B3", dB3.real());
    pin->GetOrAddReal("b_field", "amp2_B1", dB1.imag());
    pin->GetOrAddReal("b_field", "amp2_B2", dB2.imag());
    pin->GetOrAddReal("b_field", "amp2_B3", dB3.imag());
    pin->GetOrAddReal("b_field", "k1", k1);
    pin->GetOrAddReal("b_field", "k2", k2);
    pin->GetOrAddReal("b_field", "k3", k3);
    pin->GetOrAddReal("b_field", "phase", phase);

    Real Erad0 = 3 * P * (gam - 1.0) * u0;
    IndexDomain domain = IndexDomain::interior;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
    pmb->par_for("radmhdmodes_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            Real X[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, X);
            Real phase = k1 * X[1] + k2 * X[2] + k3 * X[3];
            m::complex<Real> emode = m::exp(m::complex<Real>(0, -phase));

            rho(k, j, i) = rho0 + (drho * emode).real();
            u(k, j, i) = u0 + (du * emode).real();
            uvec(V1, k, j, i) = u10 + (du1 * emode).real();
            uvec(V2, k, j, i) = u20 + (du2 * emode).real();
            uvec(V3, k, j, i) = u30 + (du3 * emode).real();

            if (use_radm1) {
                // This is in fluid frame, not in the frame used for our primitives (M1
                // rest frame). So, we need to change it
                Real E_hat = Erad0 + (dErad * emode).real();
                Real F_hat[GR_DIM] = {0., (dF1rad * emode).real(),
                    (dF2rad * emode).real(), (dF3rad * emode).real()};

                // I think this fails if non cartesian metric. Be careful! In cartesian
                // minkowski coordinate basis IS the orthonormal tetrad.
                Real uvec_gas[NVEC] = {u10 + (du1 * emode).real(),
                    u20 + (du2 * emode).real(), u30 + (du3 * emode).real()};
                Real ucon_gas[GR_DIM];
                GRMHD::calc_ucon(G, uvec_gas, k, j, i, Loci::center, ucon_gas);

                Real R_con_t[GR_DIM];
                for (int nu = 0; nu < 4; ++nu) {
                    R_con_t[nu] = (4. / 3.) * E_hat * ucon_gas[0] * ucon_gas[nu] +
                                  (1. / 3.) * E_hat * G.gcon(Loci::center, j, i, 0, nu) +
                                  ucon_gas[0] * F_hat[nu] + F_hat[0] * ucon_gas[nu];
                }
                Real R_t_mu[GR_DIM];
                G.lower(R_con_t, R_t_mu, k, j, i, Loci::center);

                const Real gdet = G.gdet(Loci::center, j, i);
                Real U_rad_init[GR_DIM] = {gdet * R_t_mu[0], gdet * R_t_mu[1],
                    gdet * R_t_mu[2], gdet * R_t_mu[3]};

                Real P_rad_init[GR_DIM];
                RadM1::u_to_p_rad(G, U_rad_init, P_rad_init, k, j, i);

                Erad(k, j, i) = P_rad_init[0];
                Frad(V1, k, j, i) = P_rad_init[1];
                Frad(V2, k, j, i) = P_rad_init[2];
                Frad(V3, k, j, i) = P_rad_init[3];
            }
        });

    // Override end time to be exactly 1 period for moving modes, unless we set otherwise
    if (one_period) {
        pin->SetReal("parthenon/time", "tlim", 2. * M_PI / m::abs(omega.real()));
    }

    return TaskStatus::complete;
}
