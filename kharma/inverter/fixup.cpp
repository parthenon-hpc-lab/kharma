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
    const bool backstop = pmb->packages.Get("Inverter")->Param<bool>("backstop");
    // Velocity recovery replaces other fixups (TODO(CEP) should it always?)
    if (backstop) return Inverter::Backstop(rc);
    if (!fix_average && !fix_atmo) return TaskStatus::complete;

    Flag("Inverter::FixUtoP");
    // Only fixup the core 5 prims TODO build by flag, HD + anything implicit
    auto P = GRMHD::PackHDPrims(rc);

    GridScalar pflag = rc->Get("flags.inverter").data;

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
    GridScalar fflag = rc->Get("flags.floors").data;

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

TaskStatus Inverter::Backstop(MeshBlockData<Real>* rc)
{
    Flag("Inverter::Backstop");
    auto pmb = rc->GetBlockPointer();

    auto& pars = pmb->packages.Get("Inverter")->AllParams();
    const Real tol = pars.Get<Real>("err_tol");
    const bool backstop_recover_vel = pars.Get<bool>("backstop_recover_vel");
    const bool backstop_recover_u = pars.Get<bool>("backstop_recover_u");

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

    // Get flags
    GridScalar fflag = rc->Get("flags.floors").data;
    GridScalar pflag = rc->Get("flags.inverter").data;

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
    // const Real umin = 1e-15;

    // If after the first round of floors, we still reconstructed
    pmb->par_for("fix_U_to_P_energy", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
        {
            // If the solve failed, because we reconstructed a
            // negative or zero internal energy (even after floors!)
            Real rhomin_geom, umin_geom;
            determine_geo_floors(
                G, P, m_p, gam, k, j, i, floors, floors_inner, rhomin_geom, umin_geom);
            const Real umin = umin_geom;
            if (failed(pflag(k, j, i)) && (P(m_p.UU, k, j, i) < umin)) {
                // const Real rho = P(m_p.RHO, k, j, i);
                // const Real u = P(m_p.UU, k, j, i);
                const Real uvec[NVEC] = {
                    P(m_p.U1, k, j, i), P(m_p.U2, k, j, i), P(m_p.U3, k, j, i)};
                Real B_P[NVEC] = {0.};
                if (m_p.B1 >= 0) {
                    B_P[V1] = P(m_p.B1, k, j, i);
                    B_P[V2] = P(m_p.B2, k, j, i);
                    B_P[V3] = P(m_p.B3, k, j, i);
                }
                // Real rho_ut, T[GR_DIM];
                // GRMHD::p_to_u_mhd(G, rho, u, uvec, B_P, gam, k, j, i, rho_ut, T);

                int fflagl = fflag(0, k, j, i);

                // Calculate P->U on the inverted values
                const Real D =
                    U(m_u.RHO, k, j, i) / (m::sqrt(-G.gcon(Loci::center, j, i, 0, 0)) *
                                              G.gdet(Loci::center, j, i));
                const Real W = GRMHD::lorentz_calc(G, uvec, k, j, i, Loci::center);

                // Calculate the total energy of the fluid at rest
                const Real uvec0[NVEC] = {0.};
                Real rho_ut = 0.;
                Real Trest[GR_DIM] = {0.};
                GRMHD::p_to_u_mhd(G, D, umin, uvec0, B_P, gam, k, j, i, rho_ut, Trest);
                // If we're below the at-rest energy (within tolerance),
                // just bump it to that and kill all kinetic energy
                if ((Trest[0] - U(m_u.UU, k, j, i)) / U(m_u.UU, k, j, i) > -tol ||
                    (!backstop_recover_vel && !backstop_recover_u)) {
                    // W = 1
                    P(m_p.RHO, k, j, i) = D;
                    P(m_p.UU, k, j, i) = umin;
                    P(m_p.U1, k, j, i) = 0.;
                    P(m_p.U2, k, j, i) = 0.;
                    P(m_p.U3, k, j, i) = 0.;
                    fflagl |= Floors::FFlag::FIXUP_ENERGY;
                } else if (backstop_recover_u) {
                    // If the user really wants it, add the extra energy to the internal
                    // energy, preserving T00 by increasing u above umin
                    const Real uvec0[NVEC] = {0.};
                    auto f = [&](Real u)
                    {
                        // Calculate tensor (we only need T0)
                        Real rho_ut, T[GR_DIM];
                        GRMHD::p_to_u_mhd(G, D, u, uvec0, B_P, gam, k, j, i, rho_ut, T);
                        // Check that it matches
                        return (T[0] - U(m_u.UU, k, j, i)) / U(m_u.UU, k, j, i);
                    };

                    // Rootfind for u that gives us T00
                    bool e_solve_failed = false;
                    Real um = umin, up = 1.;
                    Real uu;
                    if (f(up) * f(um) > 0.) {
                        e_solve_failed = true;
                        fflagl |= Floors::FFlag::FIXUP_U_RANGE;
                    } else {
                        Real uc = (up + um) / 2.;
                        while (1) {
                            Real resv = m::abs(f(uc));
                            if ((resv < tol) || (m::abs((up - um) / 2) < tol / 10)) {
                                uu = uc;
                                e_solve_failed = (resv > tol);
                                break;
                            }
                            // Same sign as right side -> center now right side
                            if (f(uc) * f(up) > 0.)
                                up = uc;
                            else // default to shifting window up -> slower
                                um = uc;
                            uc = (um + up) / 2.;
                        }
                    }

                    if (!e_solve_failed) {
                        // Set zero velocity with new UU
                        P(m_p.RHO, k, j, i) = D;
                        P(m_p.UU, k, j, i) = uu;
                        P(m_p.U1, k, j, i) = 0.;
                        P(m_p.U2, k, j, i) = 0.;
                        P(m_p.U3, k, j, i) = 0.;
                        fflagl |= Floors::FFlag::FIXUP_U;
                    } else {
                        // The only reason this *should* fail is if we missed
                        // somehow (i.e., round-off) that we really do lack the rest
                        // energy
                        P(m_p.RHO, k, j, i) = D;
                        P(m_p.UU, k, j, i) = umin;
                        P(m_p.U1, k, j, i) = 0.;
                        P(m_p.U2, k, j, i) = 0.;
                        P(m_p.U3, k, j, i) = 0.;
                        fflagl |= Floors::FFlag::FIXUP_U_FAILED;
                    }
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
                        // Calculate tensor (we only need T0)
                        Real rho_ut, T[GR_DIM];
                        GRMHD::p_to_u_mhd(
                            G, D * iW, umin, uv, B_P, gam, k, j, i, rho_ut, T);
                        // Check that it matches
                        return (T[0] - U(m_u.UU, k, j, i)) / U(m_u.UU, k, j, i);
                    };

                    // Rootfind for iW that would have produced the current u
                    bool e_solve_failed = false;
                    Real iWm = 1 / m::min(W, 50.), iWp = 1.;
                    Real iW;
                    if (f(iWp) * f(iWm) > 0.) {
                        e_solve_failed = true;
                        fflagl |= Floors::FFlag::FIXUP_VEL_RANGE;
                    } else {
                        Real iWc = (iWp + iWm) / 2.;
                        while (1) {
                            Real resv = m::abs(f(iWc));
                            if ((resv < tol) || (m::abs((iWp - iWm) / 2) < tol / 10)) {
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
                    if (gamma_fac > 1 && !e_solve_failed) {
                        fflagl |= Floors::FFlag::FIXUP_VEL_GAMMA;
                        e_solve_failed = true;
                        // TO PRINT (for verifying this only happens via round-off error)
                        // Refresh initial T, we used it in solve
                        // Real T[GR_DIM];
                        // GRMHD::p_to_u_mhd(G, P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i),
                        //                     uvec, B_P, gam, k, j, i, rho_ut, T);
                        // Compute T from solved vals to compare
                        // Real Tsolved[GR_DIM] = {0.};
                        // const Real uvecr[NDIM] = {uvec[0]*gamma_fac, uvec[1]*gamma_fac,
                        // uvec[2]*gamma_fac}; GRMHD::p_to_u_mhd(G, D * iW, umin, uvecr,
                        // B_P, gam, k, j, i, rho_ut, Tsolved); printf("bad gamma
                        // recovery: total deficit %g rest deficit %g remaining deficit
                        // %g\nfinal gamma_fac: %g rho %g u %g uvec %g %g %g B %g %g
                        // %g\n",
                        //         (T[0] - U(m_u.UU, k, j, i)) / U(m_u.UU, k, j, i),
                        //         (Trest[0] - U(m_u.UU, k, j, i)) / U(m_u.UU, k, j, i),
                        //         (Tsolved[0] - U(m_u.UU, k, j, i)) / U(m_u.UU, k, j, i),
                        //         gamma_fac, D * iW, umin, uvecr[0], uvecr[1], uvecr[2],
                        //         B_P[0], B_P[1], B_P[2]);
                    }
                    if (!e_solve_failed) {
                        // Rescale just the density & velocities with new iW
                        // TODO(CEP) still limit iW >= etc
                        P(m_p.RHO, k, j, i) = D * iW;
                        P(m_p.UU, k, j, i) = umin;
                        P(m_p.U1, k, j, i) *= gamma_fac;
                        P(m_p.U2, k, j, i) *= gamma_fac;
                        P(m_p.U3, k, j, i) *= gamma_fac;
                        fflagl |= Floors::FFlag::FIXUP_VEL;
                    } else {
                        P(m_p.RHO, k, j, i) = D;
                        P(m_p.UU, k, j, i) = umin;
                        P(m_p.U1, k, j, i) = 0.;
                        P(m_p.U2, k, j, i) = 0.;
                        P(m_p.U3, k, j, i) = 0.;
                        fflagl |= Floors::FFlag::FIXUP_VEL_FAILED;
                    }
                }

                // Set remaining floors the dumb way if they're *still* low
                // Verified this basically never happens. Not sure if it's possible
                // TODO should this just be up to SMALL and not the constant floor?
                if (P(m_p.RHO, k, j, i) < floors.rho_min_const) {
                    P(m_p.RHO, k, j, i) = floors.rho_min_const;
                    fflagl |= Floors::FFlag::FIXUP_RHO_DIRECT;
                }
                if (P(m_p.UU, k, j, i) < floors.u_min_const) {
                    P(m_p.UU, k, j, i) = floors.u_min_const;
                    fflagl |= Floors::FFlag::FIXUP_U_DIRECT;
                }
                fflag(0, k, j, i) = fflagl;

                // We definitely fixed a zone that definitely needed fixing.  Respect
                // primitive vars
                GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u);
            }
        });
    EndFlag();
    return TaskStatus::complete;
}
