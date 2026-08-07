/*
 *  File: fixup.cpp
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

#include "inverter.hpp"

#include "domain.hpp"
#include "floors.hpp"
#include "floors_functions.hpp"
#include "flux_functions.hpp"
#include "pack.hpp"

// Version of "PLOOP" guaranteeing specifically the 5 GRMHD fixup-amenable primitive vars
#define NPRIM 5
#define PRIMLOOP for (int p = 0; p < NPRIM; ++p)

TaskStatus Inverter::FixUtoP(MeshBlockData<Real>* rc)
{
    // We expect primitives all the way out to 3 ghost zones on all sides.
    // But we can only fix primitives with their neighbors.
    // This may actually mean we require the 4 ghost zones Parthenon "wants" us to have,
    // if we need to use only fixed zones.
    auto pmb = rc->GetBlockPointer();
    // Bail if we're not enabled
    const bool fix_average =
        pmb->packages.Get("Inverter")->Param<bool>("fix_average_neighbors");
    const bool fix_atmo = pmb->packages.Get("Inverter")->Param<bool>("fix_atmosphere");
    const bool velrecover = pmb->packages.Get("Inverter")->Param<bool>("vel_recovery");
    // Velocity recovery replaces other fixups (TODO(BSP) should it always?)
    if (velrecover) return Inverter::VelRecover(rc);
    if (!fix_average && !fix_atmo) return TaskStatus::complete;

    Flag("Inverter::FixUtoP");
    // Only fixup the core 5 prims TODO build by flag, HD + anything implicit
    auto P = GRMHD::PackHDPrims(rc);

    GridScalar pflag = rc->Get("pflag").data;

    const auto& pars = pmb->packages.Get("GRMHD")->AllParams();
    const Real gam = pars.Get<Real>("gamma");

    // Only yell about neighbors on extreme verbosity.
    const int flag_verbose = pmb->packages.Get("Globals")->Param<int>("flag_verbose");

    // UtoP is applied and fixed over all "Physical" zones -- anything in the domain,
    // OR in an MPI boundary.  This is because it is applied *after* the MPI sync,
    // but before physical boundary zones are computed (which it should never use anyway)

    const IndexRange3 b = KDomain::GetPhysicalRange(rc);

    const auto& G = pmb->coords;

    pmb->par_for("fix_U_to_P", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
        {
            if (failed(pflag(k, j, i))) {
                double wsum = 0.;
                double sum[NPRIM] = {0.};
                if (fix_average) {
                    // Luckily fixups are rare, so we don't have to worry about optimizing
                    // this *too* much For all neighboring cells...
                    for (int n = -1; n <= 1; n++) {
                        for (int m = -1; m <= 1; m++) {
                            for (int l = -1; l <= 1; l++) {
                                int ii = i + l, jj = j + m, kk = k + n;
                                // If we haven't overstepped array bounds...
                                if (KDomain::inside(kk, jj, ii, b)) {
                                    // Count only the good cells (not failed AND not
                                    // corner), if we can Note interpolated "fixed" cells
                                    // stay flagged
                                    if (!failed(pflag(kk, jj, ii))) {
                                        // Weight by distance
                                        double w =
                                            1. / (m::abs(l) + m::abs(m) + m::abs(n) + 1);
                                        wsum += w;
                                        PRIMLOOP sum[p] += w * P(p, kk, jj, ii);
                                    }
                                }
                            }
                        }
                    }
                }

                // Set to atmosphere/floors, zero velocity
                // Fallback fix if we're averaging, only fix if not
                if (wsum < 1.e-10) {
                    // We fill this with floor values below
                    PRIMLOOP P(p, k, j, i) = 0.;
                } else {
                    PRIMLOOP P(p, k, j, i) = sum[p] / wsum;
                }
            }
        });

    // Use values from floors package if it's enabled, otherwise any we've been asked to
    // apply
    const Floors::Prescription floors =
        pmb->packages.AllPackages().count("Floors")
            ? pmb->packages.Get("Floors")->Param<Floors::Prescription>("prescription")
            : pmb->packages.Get("Inverter")
                  ->Param<Floors::Prescription>("inverter_prescription");
    const Floors::Prescription floors_inner =
        pmb->packages.AllPackages().count("Floors")
            ? pmb->packages.Get("Floors")->Param<Floors::Prescription>(
                  "prescription_inner")
            : pmb->packages.Get("Inverter")
                  ->Param<Floors::Prescription>("inverter_prescription");

    // We need the full packs of prims/cons for p_to_u
    // Pack new variables
    PackIndexMap prims_map, cons_map;
    auto U = GRMHD::PackMHDCons(rc, cons_map);
    P = GRMHD::PackMHDPrims(rc, prims_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    // Get new sizes
    const int nvar = P.GetDim(4);

    // Get floor flag
    GridScalar fflag = rc->Get("fflag").data;

    pmb->par_for("fix_U_to_P_floors", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
        {
            if (failed(pflag(k, j, i))) {
                // Make sure all fixed values still abide by floors
                // TODO Full floors instead of just geo?
                int fflagl = fflag(0, k, j, i);
                fflagl |= Floors::apply_geo_floors(
                    G, P, m_p, gam, k, j, i, floors, floors_inner);
                fflag(0, k, j, i) = fflagl;

                // Make sure to keep lockstep
                // This will only be run for GRMHD, so we can call its p_to_u
                GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u);
            }
        });

    EndFlag();
    return TaskStatus::complete;
}

TaskStatus Inverter::VelRecover(MeshBlockData<Real>* rc)
{
    Flag("Inverter::VelRecover");
    auto pmb = rc->GetBlockPointer();

    auto& pars = pmb->packages.Get("Inverter")->AllParams();
    const Real tol = pars.Get<Real>("err_tol");

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    // Use values from floors package if it's enabled, otherwise any we've been asked to
    // apply
    const Floors::Prescription floors =
        pmb->packages.AllPackages().count("Floors")
            ? pmb->packages.Get("Floors")->Param<Floors::Prescription>("prescription")
            : pmb->packages.Get("Inverter")
                  ->Param<Floors::Prescription>("inverter_prescription");
    const Floors::Prescription floors_inner =
        pmb->packages.AllPackages().count("Floors")
            ? pmb->packages.Get("Floors")->Param<Floors::Prescription>(
                  "prescription_inner")
            : pmb->packages.Get("Inverter")
                  ->Param<Floors::Prescription>("inverter_prescription");

    // Get floor flag
    GridScalar fflag = rc->Get("fflag").data;

    // We need the full packs of prims/cons in order to fix internal energy
    // Pack new variables
    PackIndexMap prims_map, cons_map;
    auto U = GRMHD::PackMHDCons(rc, cons_map);
    auto P = GRMHD::PackMHDPrims(rc, prims_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    // Get new sizes
    const int nvar = P.GetDim(4);

    const IndexRange3 b = KDomain::GetPhysicalRange(rc);

    const auto& G = pmb->coords;

    // Minimum internal energy to set here. Can be anything small
    // const Real umin = floors.u_min_const;
    const Real umin = 1e-15;

    const Real bad_vel_tolerance = 200 * tol;

    // If after the first round of floors, we still reconstructed
    pmb->par_for("fix_U_to_P_energy", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
        {
            // If we reconstructed a negative or zero internal energy (even after floors!)
            // const Real rho = P(m_p.RHO, k, j, i);
            // const Real u = P(m_p.UU, k, j, i);
            const Real uvec[NVEC] = {
                P(m_p.U1, k, j, i), P(m_p.U2, k, j, i), P(m_p.U3, k, j, i)};
            const Real B_P[NVEC] = {
                P(m_p.B1, k, j, i), P(m_p.B2, k, j, i), P(m_p.B3, k, j, i)};
            Real rho_ut, T[GR_DIM];
            // GRMHD::p_to_u_mhd(G, rho, u, uvec, B_P, gam, k, j, i, rho_ut, T);
            if (P(m_p.UU, k, j, i) < umin) {
                // ((m::abs((T[1] - U(m_u.U1, k, j, i)) / U(m_u.U1, k, j, i)) >
                // bad_vel_tolerance) &&
                //  (m::abs(U(m_u.U1, k, j, i)) > bad_vel_tolerance)) ||
                // ((m::abs((T[2] - U(m_u.U2, k, j, i)) / U(m_u.U2, k, j, i)) >
                // bad_vel_tolerance) &&
                //  (m::abs(U(m_u.U2, k, j, i) > bad_vel_tolerance))) ||
                // ((m::abs((T[3] - U(m_u.U3, k, j, i)) / U(m_u.U3, k, j, i)) >
                // bad_vel_tolerance) &&
                //  (m::abs(U(m_u.U3, k, j, i) > bad_vel_tolerance)))) {
                // Add to existing floor flags
                int fflagl = fflag(0, k, j, i);

                // Calculate P->U on the inverted values
                const Real D =
                    U(m_u.RHO, k, j, i) / (m::sqrt(-G.gcon(Loci::center, j, i, 0, 0)) *
                                              G.gdet(Loci::center, j, i));
                const Real W = GRMHD::lorentz_calc(G, uvec, k, j, i, Loci::center);

                // Calculate the total energy of the fluid at rest
                const Real uvec0[NVEC] = {0.};
                GRMHD::p_to_u_mhd(G, D, umin, uvec0, B_P, gam, k, j, i, rho_ut, T);
                // If we're below the at-rest energy, just bump it to that and kill all
                // kinetic energy
                if (m::abs(T[0]) >= m::abs(U(m_u.UU, k, j, i))) {
                    // W = 1
                    P(m_p.RHO, k, j, i) = D;
                    P(m_p.UU, k, j, i) = umin;
                    P(m_p.U1, k, j, i) = 0.;
                    P(m_p.U2, k, j, i) = 0.;
                    P(m_p.U3, k, j, i) = 0.;
                    fflagl |= Floors::FFlag::FIXUP_ENERGY;
                } else {
                    // Otherwise, solve for the Lorentz factor which makes the internal
                    // energy at least the minimum (basically, pulling from the kinetic
                    // energy) This really should be a guaranteed solve!
                    auto f = [&](Real iW)
                    {
                        // Rescale velocity
                        Real gamma_fac = m::sqrt((SQR(1. / iW) - 1.) / (SQR(W) - 1.));
                        const Real uv[NVEC] = {gamma_fac * uvec[0], gamma_fac * uvec[1],
                            gamma_fac * uvec[2]};
                        // Rescale rest mass
                        const Real rho = D * iW;

                        // Calculate tensor (we only need T0)
                        GRMHD::p_to_u_mhd(G, rho, umin, uv, B_P, gam, k, j, i, rho_ut, T);
                        // Check that it matches
                        return (T[0] - U(m_u.UU, k, j, i)) / U(m_u.UU, k, j, i);
                    };

                    // Rootfind for iW that would have produced the current u
                    bool e_solve_failed = false;
                    Real iWm = 1 / 51., iWp = 1.;
                    Real iW;
                    if (f(iWp) * f(iWm) > 0.) {
                        e_solve_failed = true;
                    } else {
                        Real iWc = 0.75;
                        while (1) {
                            Real resv = m::abs(f(iWc));
                            if ((resv < tol) || (m::abs((iWp - iWm) / 2) < tol)) {
                                iW = iWc;
                                e_solve_failed = (resv > tol);
                                break;
                            }
                            // Same sign as right side -> center now right side
                            if (f(iWc) * f(iWp) > 0.)
                                iWp = iWc;
                            else // default to shifting window up -> slower
                                iWm = iWc;
                            iWc = (iWm + iWp) / 2.;
                        }
                    }

                    // Compute what we really need
                    Real gamma_fac = m::sqrt((SQR(1. / iW) - 1.) / (SQR(W) - 1.));
                    if (!e_solve_failed && gamma_fac < 1) {
                        // Rescale just the density & velocities with new iW
                        P(m_p.RHO, k, j, i) = D * iW;
                        P(m_p.UU, k, j, i) = umin;
                        P(m_p.U1, k, j, i) *= gamma_fac;
                        P(m_p.U2, k, j, i) *= gamma_fac;
                        P(m_p.U3, k, j, i) *= gamma_fac;
                        fflagl |= Floors::FFlag::FIXUP_MOMENTUM;
                    } else {
                        // The only reason this *should* fail is if we missed
                        // somehow that we really do lack the rest energy
                        P(m_p.RHO, k, j, i) = D;
                        P(m_p.UU, k, j, i) = umin;
                        P(m_p.U1, k, j, i) = 0.;
                        P(m_p.U2, k, j, i) = 0.;
                        P(m_p.U3, k, j, i) = 0.;
                        fflagl |= Floors::FFlag::FIXUP_ENERGY;
                    }
                }
                fflag(0, k, j, i) = fflagl;

                // Reapply ceilings
                apply_ceilings(G, P, m_p, gam, k, j, i, floors, floors_inner, U, m_u);
                // Set remaining floors the dumb way if they're still low
                // TODO(BSP) record
                P(m_p.RHO, k, j, i) = m::max(P(m_p.RHO, k, j, i), floors.rho_min_const);
                P(m_p.UU, k, j, i) = m::max(P(m_p.UU, k, j, i), floors.u_min_const);

                GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u);
            }
        });
    EndFlag();
    return TaskStatus::complete;
}
