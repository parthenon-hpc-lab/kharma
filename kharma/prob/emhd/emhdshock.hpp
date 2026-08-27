/*
 *  File: emhdshock.hpp
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
#include "emhd.hpp"

using namespace std::literals::complex_literals;
using namespace parthenon;

#define STRLEN 2048

/**
 * Initialization of the EMHD shock test in magnetized plasma w/viscosity and heat
 * conduction
 *
 * The BVP solution (kharma/prob/emhd/shock_soln_${RES}_default) is the input to the code.
 * Since the BVP solution is a steady-state, time-independent solution of the EMHD
 * equations, the code should maintain the solution.
 *
 * An alternate option is to initialize with the ideal MHD Rankine-Hugoniot jump
 * condition. If higher order terms have been implemented correctly, the primitives should
 * relax to the steady state solution. However, they may differ by a translation to the
 * BVP solution.
 *
 * Therefore, to quantitatively check the EMHD implementation, we prefer the BVP solution
 * as the input.
 */
TaskStatus InitializeEMHDShock(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();

    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    GridVector q = rc->Get("prims.q").data;
    GridVector dP = rc->Get("prims.dP").data;

    const auto& G = pmb->coords;

    // Type of input to the problem
    const std::string input = pin->GetOrAddString("emhdshock", "input", "BVP");

    // Both shocks do not have a B jump condition, just set B10
    // TODO take magnetization?
    pin->GetOrAddString("b_field", "type", "constant");
    pin->GetOrAddReal("b_field", "B10", 1.e-5);

    // Obtain EMHD params
    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb->packages);
    // Obtain GRMHD params
    const auto& grmhd_pars = pmb->packages.Get("GRMHD")->AllParams();
    const Real& gam = grmhd_pars.Get<Real>("gamma");

    // Bounds of the domain
    IndexDomain domain = IndexDomain::interior;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);

    if (input == "BVP") {

        // Load file names into strings
        char fbvp_rho[STRLEN], fbvp_u[STRLEN], fbvp_u1[STRLEN], fbvp_q[STRLEN],
            fbvp_dP[STRLEN];
        sprintf(fbvp_rho, "shock_soln_rho.txt");
        sprintf(fbvp_u, "shock_soln_u.txt");
        sprintf(fbvp_u1, "shock_soln_u1.txt");
        sprintf(fbvp_q, "shock_soln_q.txt");
        sprintf(fbvp_dP, "shock_soln_dP.txt");

        // Assign file pointers
        FILE *fp_rho, *fp_u, *fp_u1, *fp_q, *fp_dP;
        fp_rho = fopen(fbvp_rho, "r");
        fp_u = fopen(fbvp_u, "r");
        fp_u1 = fopen(fbvp_u1, "r");
        fp_q = fopen(fbvp_q, "r");
        fp_dP = fopen(fbvp_dP, "r");

        if (fp_rho == NULL || fp_u == NULL || fp_u1 == NULL || fp_q == NULL ||
            fp_dP == NULL) {
            std::cout << std::endl; // flush messages in output buffer before we error
            throw std::runtime_error("Could not open conducting atmosphere solution!");
        }

        auto rho_host = rho.GetHostMirror();
        auto u_host = u.GetHostMirror();
        auto uvec_host = uvec.GetHostMirror();
        auto q_host = q.GetHostMirror();
        auto dP_host = dP.GetHostMirror();

        for (int k = kb.s; k <= kb.e; k++) {
            for (int j = jb.s; j <= jb.e; j++) {
                for (int i = ib.s; i <= ib.e; i++) {

                    Real X[GR_DIM];
                    G.coord_embed(k, j, i, Loci::center, X);

                    // First initialize primitives that are read from .txt files
                    fscanf(fp_rho, "%lf", &(rho_host(k, j, i)));
                    fscanf(fp_u, "%lf", &(u_host(k, j, i)));
                    fscanf(fp_u1, "%lf", &(uvec_host(0, k, j, i)));
                    fscanf(fp_q, "%lf", &(q_host(k, j, i)));
                    fscanf(fp_dP, "%lf", &(dP_host(k, j, i)));

                    // Now the remaining primitives
                    uvec_host(1, k, j, i) = 0.;
                    uvec_host(2, k, j, i) = 0.;

                    if (emhd_params.higher_order_terms) {

                        // Initialize local variables (for improved readability)
                        const Real rho_temp = rho_host(k, j, i);
                        const Real u_temp = u_host(k, j, i);
                        const Real Theta = (gam - 1.) * u_temp / rho_temp;

                        // Set EMHD parameters
                        Real tau, chi_e, nu_e;
                        // Zeros are q, dP, and bsq, only needed for torus closure
                        EMHD::set_parameters(G, rho_temp, u_temp, 0., 0., 0., emhd_params,
                            gam, j, i, tau, chi_e, nu_e);

                        // Update q and dP (which now are q_tilde and dP_tilde)
                        Real q_tilde = q_host(k, j, i);
                        Real dP_tilde = dP_host(k, j, i);
                        if (emhd_params.higher_order_terms) {
                            q_tilde *=
                                (chi_e != 0)
                                    ? m::sqrt(tau / (chi_e * rho_temp * Theta * Theta))
                                    : 0.;
                            dP_tilde *= (nu_e != 0)
                                            ? m::sqrt(tau / (nu_e * rho_temp * Theta))
                                            : 0.;
                        }
                        q_host(k, j, i) = q_tilde;
                        dP_host(k, j, i) = dP_tilde;
                    }
                }
            }
        }

        // disassociate file pointer
        fclose(fp_rho);
        fclose(fp_u);
        fclose(fp_u1);
        fclose(fp_q);
        fclose(fp_dP);

        // Deep copy to device
        rho.DeepCopy(rho_host);
        u.DeepCopy(u_host);
        uvec.DeepCopy(uvec_host);
        q.DeepCopy(q_host);
        dP.DeepCopy(dP_host);
        Kokkos::fence();

    } else { // Any other input corresponds to ideal MHD shock initial conditions

        // Need the limits of the problem size to determine center
        const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
        const Real x1max = pin->GetReal("parthenon/mesh", "x1max");

        // Left and right states
        double rhoL = 1., rhoR = 3.08312999;
        double uL = 1., uR = 4.94577705;
        double u1L = 1., u1R = 0.32434571;
        double u2L = 0., u2R = 0.;
        double u3L = 0., u3R = 0.;
        const GReal x1_center = (x1min + x1max) / 2.;

        pmb->par_for("emhdshock_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
            {
                Real X[GR_DIM];
                G.coord_embed(k, j, i, Loci::center, X);

                bool lhs = X[1] < x1_center;

                // Initialize primitives
                rho(k, j, i) = (lhs) ? rhoL : rhoR;
                u(k, j, i) = (lhs) ? uL : uR;
                uvec(V1, k, j, i) = (lhs) ? u1L : u1R;
                uvec(V2, k, j, i) = (lhs) ? u2L : u2R;
                uvec(V3, k, j, i) = (lhs) ? u3L : u3R;
                q(k, j, i) = 0.;
                dP(k, j, i) = 0.;
            }

        );
    }

    return TaskStatus::complete;
}
