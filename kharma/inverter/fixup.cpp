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

    // Re-apply floors to fixed zones
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
                Floors::apply_geo_floors(G, P, m_p, gam, k, j, i, floors, floors_inner);

                // Make sure to keep lockstep
                // This will only be run for GRMHD, so we can call its p_to_u
                GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u);
            }
        });

    EndFlag();
    return TaskStatus::complete;
}
