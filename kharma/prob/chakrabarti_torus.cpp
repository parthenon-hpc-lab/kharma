/*
 *  File: chakrabarti_torus.cpp
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

#include "chakrabarti_torus.hpp"

#include "coordinate_utils.hpp"
#include "floors.hpp"
#include "types.hpp"

TaskStatus InitializeChakrabartiTorus(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;

    const GReal rin = pin->GetOrAddReal("torus", "rin", 6.0);
    const GReal rmax = pin->GetOrAddReal("torus", "rmax", 12.0);
    const GReal rho_max = pin->GetOrAddReal("torus", "rho_max", 1.0);
    const GReal tilt_deg = pin->GetOrAddReal("torus", "tilt", 0.0);
    const GReal tilt = tilt_deg / 180. * M_PI;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real gm1 = gam - 1.0;

    IndexDomain domain = IndexDomain::interior;
    const int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    const int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    const int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    // Get coordinate systems
    // G clearly holds a reference to an existing system G.coords.base,
    // but we don't know if it's KS or BL coordinates
    // Since we can't create a system and assign later, we just
    // rebuild copies of both based on the BH spin "a"
    const auto& G = pmb->coords;
    const GReal a = G.coords.get_a();

    // Chakrabarti torus parameters
    GReal cc, nn;
    cn_calc(a, rin, rmax, &cc, &nn);
    const Real l_max = l_calc(a, rmax, 1.0, cc, nn);

    // Thermodynamic quantities at rin, rmax
    const Real lnh_in = lnh_calc(a, rin, rin, 1.0, cc, nn);
    const Real lnh_peak = lnh_calc(a, rin, rmax, 1.0, cc, nn) - lnh_in;
    const Real pgas_over_rho_peak = gm1 / gam * (m::exp(lnh_peak) - 1.0);
    const Real rho_peak = m::pow(pgas_over_rho_peak, 1.0 / gm1) / rho_max;

    pmb->par_for("chakrabarti_torus_init", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
        {
            GReal Xnative[GR_DIM], Xembed[GR_DIM], Xmidplane[GR_DIM];
            G.coord(k, j, i, Loci::center, Xnative);
            G.coord_embed(k, j, i, Loci::center, Xembed);
            // What are our corresponding "midplane" values for evaluating the function?
            rotate_polar(Xembed, tilt, Xmidplane);

            GReal r = Xmidplane[1], th = Xmidplane[2];
            GReal sth = sin(th);
            GReal cth = cos(th);

            // Boyer-Lindquist metric components
            Real r2 = r * r;
            Real a2 = a * a;
            Real DD = r2 - 2. * r + a2;
            Real AA = m::pow(r2 + a2, 2) - DD * a2 * sth * sth;
            Real SS = r2 + a2 * cth * cth;

            Real gcov_tt = -(1.0 - (2.0 * r / SS));
            Real gcov_tphi = -2.0 * a * r / SS * sth * sth;
            Real gcov_phiphi = (SS + (1.0 + (2.0 * r / SS)) * a2 * sth * sth) * sth * sth;
            Real gcon_tt = -AA / (DD * SS);
            Real gcon_tphi = -2.0 * a * r / (DD * SS);

            // Compute logarithm of enthalpy and check if we're inside the torus
            Real lnh;
            bool in_torus = false;
            if (r >= rin) {
                lnh = lnh_calc(a, rin, r, sth, cc, nn) - lnh_in;
                if (lnh >= 0.0) {
                    in_torus = true;
                }
            }

            // Region inside magnetized torus
            // Initialize fluid primitives
            // Everything outside is left 0 to be added by the floors
            if (in_torus) {

                // Calculate rho and u
                Real pg_over_rho = gm1 / gam * (m::exp(lnh) - 1.0);
                Real rho_l = m::pow(pg_over_rho, 1. / gm1) / rho_peak;
                Real u_l = (pg_over_rho * rho_l) / gm1;

                // Calculate four velocity in Boyer-Lindquist coordinates
                Real l = l_calc(a, r, sth, cc, nn);
                Real u_t = u_t_calc(a, r, sth, l);
                Real omega =
                    -(gcov_tphi + (l * gcov_tt)) / (gcov_phiphi + (l * gcov_tphi));
                Real ucon_t = (gcon_tt - (l * gcon_tphi)) * u_t;
                Real ucon_phi = omega * u_t;

                const Real ucon_tilt[GR_DIM] = {ucon_t, 0., 0., ucon_phi};
                Real ucon_bl[GR_DIM];
                rotate_polar_vec(Xmidplane, ucon_tilt, -tilt, Xembed, ucon_bl);

                // Then set u^t and transform the 4-vector to KS if necessary,
                // and then to native coordinates
                Real ucon_native[GR_DIM];
                G.coords.bl_fourvel_to_native(Xnative, ucon_bl, ucon_native);

                // Convert native 4-vector to primitive u-twiddle, see Gammie '04
                Real gcon[GR_DIM][GR_DIM], u_prim[NVEC];
                G.gcon(Loci::center, j, i, gcon);
                fourvel_to_prim(gcon, ucon_native, u_prim);

                rho(k, j, i) = rho_l;
                u(k, j, i) = u_l;
                uvec(0, k, j, i) = u_prim[0];
                uvec(1, k, j, i) = u_prim[1];
                uvec(2, k, j, i) = u_prim[2];
            }
        });

    // Apply floors to initialize the rest of the domain (regardless of the
    // 'disable_floors' param) Since the conserved vars U are not initialized, this is
    // done in *fluid frame*, even if NOF frame is chosen (iharm3d does the same iirc)
    // This is probably not a huge issue, just good to state explicitly
    Floors::ApplyInitialFloors(pin, rc.get(), IndexDomain::interior);

    return TaskStatus::complete;
}
