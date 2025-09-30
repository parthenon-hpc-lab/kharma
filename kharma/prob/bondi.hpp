/* 
 *  File: bondi.hpp
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

#include "gr_coordinates.hpp"
#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"
#include "coordinate_utils.hpp"
#include "types.hpp"
#include "hdf5_utils.h"

#include <parthenon/parthenon.hpp>

/**
 * Initialize a Bondi problem over the domain
 */
TaskStatus InitializeBondi(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin);

/**
 * Record parameters Bondi problem (or boundaries!) will need throughout the run
 * Currently uses "GRMHD" package as convenient proxy (TODO fix that with problem-packages)
 */
void AddBondiParameters(ParameterInput *pin, Packages_t &packages);

/**
 * Set all values on a given domain to the Bondi inflow analytic steady-state solution.
 * Use the template version when possible, which just calls through
 */
TaskStatus SetBondiImpl(std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse);

template<IndexDomain domain>
TaskStatus SetBondi(std::shared_ptr<MeshBlockData<Real>>& rc, bool coarse=false) {
    return SetBondiImpl(rc, domain, coarse);
}

/**
 * Supporting functions for Bondi flow calculations
 * 
 * Adapted from M. Chandra
 * Modified by Hyerin Cho and Ramesh Narayan
 */
KOKKOS_INLINE_FUNCTION Real get_Tfunc(const Real T, const GReal r, const Real C1, const Real C2, const Real n)
{
    const Real A = 1. + (1. + n) * T;
    const Real B = C1 / (r * r * m::pow(T, n));
    return A * A * (1. - 2. / r + B * B) - C2;
}

KOKKOS_INLINE_FUNCTION Real get_T(const GReal r, const Real C1, const Real C2, const Real n, const Real rs)
{
    Real rtol = 1.e-12;
    Real ftol = 1.e-14;
    Real Tinf = (m::sqrt(C2) - 1.) / (n + 1); // temperature at infinity
    Real Tnear = m::pow(C1 * m::sqrt(2. / (r*r*r)), 1. / n); // temperature near the BH

    // There are two branches of solutions (see Michel et al. 1971) and the two branches cross at rs.
    // These bounds are set to select the inflowing solution only.
    Real Tmin = (r < rs) ? Tinf  : m::max(Tnear,Tinf);
    Real Tmax = (r < rs) ? Tnear : 1.0;

    Real f0, f1, fh;
    Real T0, T1, Th;
    T0 = Tmin;
    f0 = get_Tfunc(T0, r, C1, C2, n);
    T1 = Tmax;
    f1 = get_Tfunc(T1, r, C1, C2, n);
    // TODO(BSP) where does this trigger an error?  Can we make it clearer?
    if (f0 * f1 > 0) return -1.;

    Th = (T0 + T1) / 2.; // a simple bisection method which is stable and fast
    fh = get_Tfunc(Th, r, C1, C2, n);
    Real epsT = rtol * (Tmin + Tmax);
    while (m::abs(Th - T0) > epsT && m::abs(Th - T1) > epsT && m::abs(fh) > ftol)
    {
        if (fh * f0 > 0.) {
            T0 = Th;
            f0 = fh;
        } else {
            T1 = Th;
            f1 = fh;
        }

        Th = (T0 + T1) / 2.; 
        fh = get_Tfunc(Th, r, C1, C2, n);
    }

    return Th;
}

KOKKOS_INLINE_FUNCTION void get_bondi_soln(const Real &r, const Real &rs, const Real &mdot, const Real &gam,
                                            Real &rho, Real &u, Real &ur)
{
    // Solution constants
    // These don't depend on which zone we're calculating
    const Real n = 1. / (gam - 1.);
    const Real uc = m::sqrt(1. / (2. * rs));
    const Real Vc = m::sqrt(uc * uc / (1. - 3. * uc * uc));
    const Real Tc = -n * Vc * Vc / ((n + 1.) * (n * Vc * Vc - 1.));
    const Real C1 = uc * rs * rs * m::pow(Tc, n);
    const Real A = 1. + (1. + n) * Tc;
    const Real C2 = A * A * (1. - 2. / rs + uc * uc);
    const Real K  = m::pow(4 * M_PI * C1 / mdot, 1/n);
    const Real Kn = m::pow(K, n);

    const Real T = get_T(r, C1, C2, n, rs);
    const Real Tn = m::pow(T, n);

    rho = Tn / Kn;
    u = rho * T * n;
    ur = -C1 / (Tn * r * r);
}

KOKKOS_INLINE_FUNCTION void get_prim_bondi(const GRCoordinates& G, const bool diffinit, const Real &rs, const Real &mdot, const Real &gam,
                                            const Real ur_frac, const Real uphi, const Real rin_bondi, const Real bondi_clear_angle,
                                            const bool fill_interior, Real &rho, Real &u, Real u_prim[NVEC], 
                                            const int& k, const int& j, const int& i)
{
    // Get primitive values initialized
    GReal Xnative[GR_DIM], Xembed[GR_DIM];
    G.coord(k, j, i, Loci::center, Xnative);
    G.coord_embed(k, j, i, Loci::center, Xembed);
    GReal r = Xembed[1];
    GReal th = Xembed[2];

    // Allow cutting out areas by angle or radius to be replaced by floors
    if ((th < bondi_clear_angle) || (th > M_PI - bondi_clear_angle)) {
        rho = 0.;
        u = 0.;
        u_prim[0] = 0.;
        u_prim[1] = 0.;
        u_prim[2] = 0.;
        return;
    } else if (r < rin_bondi) {
        // Optionally fill the interior region with the innermost analytically computed value
        if (fill_interior) {
            // just match at the rin_bondi value
            r = rin_bondi;
            // TODO(BSP) previous impl could also do values at inf, restore that?
        } else {
            rho = 0.;
            u = 0.;
            u_prim[0] = 0.;
            u_prim[1] = 0.;
            u_prim[2] = 0.;
            return;
        }
    }

    Real rho_tmp, u_tmp, T_tmp, ur_tmp;
    Real n = 1. / (gam - 1.);
    get_bondi_soln(r, rs, mdot, gam, rho_tmp, u_tmp, ur_tmp);
    T_tmp = u_tmp / (rho_tmp * n);

    Real rb; // Bondi radius
    Real rho0, u0, ur0;
    if (diffinit) {
        // Get r^{-1} density initialization instead
    
        // values at infinity (obtained by putting r = 100 rb)
        if (m::abs(n - 1.5) < 0.01) rb = rs * rs * 80. / (27. * gam);
        else rb = (4 * (n + 1)) / (2 * (n + 3) - 9) * rs;
        get_bondi_soln(100 * rb, rs, mdot, gam, rho0, u0, ur0);

        // interpolation between inner and outer regimes
        rho = rho0 * (r + rb) / r;
        u = rho * T_tmp * n;
    } else {
        // Normal bondi initialization
        rho = rho_tmp;
        u = u_tmp;
    }
    Real ur = ur_tmp; // Bondi radial velocity solution


    // Get the native-coordinate 4-vector corresponding to ur
    const Real ucon_bl[GR_DIM] = {0, ur * ur_frac, 0, uphi * m::pow(r,-3./2.)};
    Real ucon_native[GR_DIM];
    G.coords.bl_fourvel_to_native(Xnative, ucon_bl, ucon_native);

    // Convert native 4-vector to primitive u-twiddle, see Gammie '04
    Real gcon[GR_DIM][GR_DIM];
    G.gcon(Loci::center, j, i, gcon);
    fourvel_to_prim(gcon, ucon_native, u_prim);
}
