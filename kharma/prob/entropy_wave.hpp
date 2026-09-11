/*
 *  File: entropy_wave.hpp
 *
 *  BSD 3-Clause License
 *
 *  Copyright (c) 2020, AFD Group at UIUC
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
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
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

#include "decs.hpp"
#include "domain.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

/*
 * Advected entropy wave: a test for the Entropy package's Ktot_adv.
 *
 * A uniform 4-velocity, a uniform gas pressure, and an arbitrary density profile is an
 * *exact nonlinear* solution of the relativistic Euler equations in flat space. With
 * u^mu constant and p constant, the stress-energy divergence reduces to
 * u^nu (u^mu d_mu w), and the enthalpy w = rho + gam/(gam-1) p differs from rho by a
 * constant, so both that and continuity are satisfied by any rho advected along the
 * flow. The profile therefore simply translates at the coordinate velocity v^i, with
 * no steepening and no sound waves, for arbitrarily large amplitude.
 *
 * Since p is uniform and rho is not, the specific entropy
 *     K = (gam-1) u / rho^gam = p / rho^gam
 * is strongly *non-uniform*, and is conserved along fluid worldlines. That is exactly
 * the property Ktot_adv is supposed to have: it is advected as a passive scalar and
 * never reset, so it must reproduce the translated profile. This matters because the
 * other analytic entropy tests available (Noh, Hubble) all have a spatially
 * *uniform* entropy, and a uniform passive scalar is advected exactly by any scheme
 * whose scalar flux is proportional to the mass flux -- so they cannot catch an error
 * in the Ktot_adv flux term itself. This problem can.
 *
 * Run for one full period (the default) and the analytic solution at t=tlim is
 * identically the initial condition, so the error needs no separate analytic
 * evaluation -- see tests/entropy_wave/check.py.
 */
TaskStatus InitializeEntropyWave(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;

    const Real rho0 = pin->GetOrAddReal("entropy_wave", "rho0", 1.0);
    // Amplitude as a fraction of rho0.  The solution is exact for any amp < 1, so this
    // is deliberately large: we want the entropy to vary by a big factor, not by an
    // amount that could hide in truncation error.
    const Real amp = pin->GetOrAddReal("entropy_wave", "amp", 0.5);
    // Gas pressure, uniform.  This is what makes K = p/rho^gam vary with rho.
    const Real pgas = pin->GetOrAddReal("entropy_wave", "pgas", 1.0);

    // Coordinate 3-velocity of the drift, v^i = u^i/u^t.  Must be subluminal.
    const Real v1 = pin->GetOrAddReal("entropy_wave", "v1", 0.5);
    const Real v2 = pin->GetOrAddReal("entropy_wave", "v2", 0.0);
    const Real v3 = pin->GetOrAddReal("entropy_wave", "v3", 0.0);
    // Integer wavenumbers: the profile is sin(2 pi (n1 x1/L1 + n2 x2/L2 + n3 x3/L3))
    const int n1 = pin->GetOrAddInteger("entropy_wave", "n1", 1);
    const int n2 = pin->GetOrAddInteger("entropy_wave", "n2", 0);
    const int n3 = pin->GetOrAddInteger("entropy_wave", "n3", 0);

    const Real vsq = v1 * v1 + v2 * v2 + v3 * v3;
    if (vsq >= 1.) {
        std::cout << std::endl;
        throw std::invalid_argument("Entropy wave drift velocity must be subluminal!");
    }

    // KHARMA's velocity primitive is the relative 4-velocity utilde^i, with
    // Gamma = sqrt(1 + gamma_ij utilde^i utilde^j) (see GRMHD::lorentz_calc).  In flat
    // Cartesian coordinates (lapse 1, zero shift, gamma_ij = delta_ij) that means
    // u^t = Gamma, u^i = utilde^i, so v^i = utilde^i/Gamma and utilde^i = Gamma v^i.
    const Real lorentz = 1. / m::sqrt(1. - vsq);
    const Real ut1 = lorentz * v1, ut2 = lorentz * v2, ut3 = lorentz * v3;

    const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
    const Real x1max = pin->GetReal("parthenon/mesh", "x1max");
    const Real x2min = pin->GetReal("parthenon/mesh", "x2min");
    const Real x2max = pin->GetReal("parthenon/mesh", "x2max");
    const Real x3min = pin->GetReal("parthenon/mesh", "x3min");
    const Real x3max = pin->GetReal("parthenon/mesh", "x3max");
    const Real k1 = 2. * M_PI * n1 / (x1max - x1min);
    const Real k2 = 2. * M_PI * n2 / (x2max - x2min);
    const Real k3 = 2. * M_PI * n3 / (x3max - x3min);

    // The pattern moves as sin(k.x - (k.v) t), so it returns to its initial state after
    // one period 2 pi/(k.v).  Ending there lets the check compare against the t=0 dump
    // rather than re-deriving the analytic solution.
    const Real kdotv = k1 * v1 + k2 * v2 + k3 * v3;
    if (pin->GetOrAddBoolean("entropy_wave", "one_period", true)) {
        if (kdotv == 0.) {
            std::cout << std::endl;
            throw std::invalid_argument(
                "Entropy wave has no drift along its wavevector: k.v = 0, so there is no "
                "period to run for.  Set <entropy_wave>/one_period = false.");
        }
        pin->SetReal("parthenon/time", "tlim", 2. * M_PI / m::abs(kdotv));
    }

    const auto& G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    IndexDomain domain = IndexDomain::entire;
    IndexRange3 b = KDomain::GetRange(rc, domain, 0, 0);
    pmb->par_for("entropy_wave_init", b.ks, b.ke, b.js, b.je, b.is, b.ie,
                 KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            GReal X[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, X);

            rho(k, j, i) = rho0 * (1. + amp * m::sin(k1 * X[1] + k2 * X[2] + k3 * X[3]));
            // Uniform pressure: this is the whole point, it makes K = p/rho^gam vary
            u(k, j, i) = pgas / (gam - 1.);
            uvec(0, k, j, i) = ut1;
            uvec(1, k, j, i) = ut2;
            uvec(2, k, j, i) = ut3;
        });

    return TaskStatus::complete;
}
