/*
 *  File: geometry.hpp
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

/**
 * Per-point geometry computation shared by the GRCoordinates caches and the KGeometry
 * package fields.  Both call these same functions, so both produce identical values.
 *
 * Source: any object with coord(k, j, i, loc, X) and .coords (a CoordinateEmbedding);
 *         for now, the block's GRCoordinates.
 * Out:    any object with Real& accessors gcov(loc,j,i,mu,nu), gcon(loc,j,i,mu,nu),
 *         gdet(loc,j,i), conn(j,i,mu,nu,lam), gdet_conn(j,i,mu,nu,lam).
 */
namespace GeomFill
{

// Finite-difference step for the connection, formerly DELTA in gr_coordinates.cpp
constexpr GReal CONN_DELTA = 1.e-8;

template<typename Source, typename Out>
KOKKOS_INLINE_FUNCTION void fill_point(const Source& G, const int connection_average_points,
    const int n1, const int n2, const int j, const int i, const Out& out)
{
    // Zero this point first: the cache is fresh, but package fields are refilled after
    // every remesh, and the averages below accumulate with +=
    for (int iloc = 0; iloc < NLOC; iloc++) {
        const Loci loc = (Loci)iloc;
        out.gdet(loc, j, i) = 0.;
        DLOOP2 {
            out.gcov(loc, j, i, mu, nu) = 0.;
            out.gcon(loc, j, i, mu, nu) = 0.;
        }
    }
    if (i < n1 && j < n2) {
        DLOOP3 {
            out.conn(j, i, mu, nu, lam) = 0.;
            out.gdet_conn(j, i, mu, nu, lam) = 0.;
        }
    }

    // Iterate through locations. This could be done in fancy ways, but
    // this highlights what's actually going on.
    for (int iloc = 0; iloc < NLOC; iloc++) {
        Loci loc = (Loci)iloc;
        // radius of points to sample, floor(npoints/2)
        const int radius = connection_average_points / 2;
        const int diameter = connection_average_points;
        const int square = connection_average_points * connection_average_points;
        if (loc == Loci::center || loc == Loci::face3) {
            // This prevents overstepping conn's bounds by halting in the last zone
            if (i >= n1 || j >= n2) continue;
            // Get a square of points evenly across each cell,
            // over both nontrivial geometry directions 1,2
            // Note this never hits/passes the pole
            for (int k = -radius; k <= radius; k++) {
                for (int l = -radius; l <= radius; l++) {
                    GReal X[GR_DIM];
                    G.coord(0, j, i, loc, X);
                    GReal Xn1[GR_DIM], Xn2[GR_DIM];
                    G.coord(0, j, i + 1, loc, Xn1);
                    G.coord(0, j + 1, i, loc, Xn2);
                    X[1] += (Xn1[1] - X[1]) / connection_average_points * k;
                    X[2] += (Xn2[2] - X[2]) / connection_average_points * l;
                    // Get geometry at points
                    GReal gcov_loc[GR_DIM][GR_DIM], gcon_loc[GR_DIM][GR_DIM];
                    G.coords.gcov_native(X, gcov_loc);
                    const GReal gdet = G.coords.gcon_from_gcov(gcov_loc, gcon_loc);
                    // Add to running averages
                    out.gdet(loc, j, i) += gdet / square;
                    DLOOP2 {
                        out.gcov(loc, j, i, mu, nu) += gcov_loc[mu][nu] / square;
                        out.gcon(loc, j, i, mu, nu) += gcon_loc[mu][nu] / square;
                    }
                    if (loc == Loci::center) {
                        // In the center, get the connection and gdet*connection
                        Real conn_loc[GR_DIM][GR_DIM][GR_DIM];
                        G.coords.conn_native(X, CONN_DELTA, conn_loc);
                        DLOOP3 {
                            out.conn(j, i, mu, nu, lam) += conn_loc[mu][nu][lam] / square;
                            out.gdet_conn(j, i, mu, nu, lam) +=
                                gdet * conn_loc[mu][nu][lam] / square;
                        }
                    }
                }
            }
        } else if (loc == Loci::face1 || loc == Loci::face2) {
            for (int k = -radius; k <= radius; k++) {
                // Like the above, but only average over a particular face (line
                // for 2D geometry)
                GReal X[GR_DIM];
                G.coord(0, j, i, loc, X);
                GReal Xn1[GR_DIM];
                // Step in the nontrivial direction perpendicular to the normal
                const int avg_dir = (loc == Loci::face1) ? X2DIR : X1DIR;
                // Get the direction/distance
                G.coord(0, j + (avg_dir == X2DIR), i + (avg_dir == X1DIR), loc, Xn1);
                X[avg_dir] += (Xn1[avg_dir] - X[avg_dir]) / diameter * k;
                // Get geometry at the point
                GReal gcov_loc[GR_DIM][GR_DIM], gcon_loc[GR_DIM][GR_DIM];
                G.coords.gcov_native(X, gcov_loc);
                const GReal gdet = G.coords.gcon_from_gcov(gcov_loc, gcon_loc);
                // Add to running averages
                out.gdet(loc, j, i) += gdet / diameter;
                DLOOP2 {
                    out.gcov(loc, j, i, mu, nu) += gcov_loc[mu][nu] / diameter;
                    out.gcon(loc, j, i, mu, nu) += gcon_loc[mu][nu] / diameter;
                }
            }
        } else { // corner or exotic locations, no averaging
            // Just one point
            GReal X[GR_DIM];
            G.coord(0, j, i, loc, X);
            // Get geometry
            GReal gcov_loc[GR_DIM][GR_DIM], gcon_loc[GR_DIM][GR_DIM];
            G.coords.gcov_native(X, gcov_loc);
            const GReal gdet = G.coords.gcon_from_gcov(gcov_loc, gcon_loc);
            // Set geometry
            out.gdet(loc, j, i) = gdet;
            DLOOP2 {
                out.gcov(loc, j, i, mu, nu) = gcov_loc[mu][nu];
                out.gcon(loc, j, i, mu, nu) = gcon_loc[mu][nu];
            }
        }
    }
}

// McKinney et al. (2012) connection correction:
// https://ui.adsabs.harvard.edu/abs/2012MNRAS.423.3083M/abstract
// Reads face gdet at (j+1, i+1), so run it as a separate kernel after fill_point
template<typename Source, typename Out>
KOKKOS_INLINE_FUNCTION void correct_connections_point(
    const Source& G, const int j, const int i, const Out& out)
{
    GReal X[GR_DIM];
    G.coord(0, j, i, Loci::center, X);
    for (int lam = 1; lam < GR_DIM; lam++) {
        const Loci loc = loc_of(lam);
        // Get gdet values at faces we calculated above
        GReal Xfm[GR_DIM], Xfp[GR_DIM];
        G.coord(0, j, i, loc, Xfm);
        G.coord(0, j + (lam == X2DIR), i + (lam == X1DIR), loc, Xfp);
        double gdetfm = out.gdet(loc, j, i);
        double gdetfp = out.gdet(loc, j + (lam == X2DIR), i + (lam == X1DIR));

        // Get cell center metric determinant
        double gdet_c = out.gdet(Loci::center, j, i);

        GReal D_k = (gdetfp - gdetfm) / ((Xfp[lam] - Xfm[lam] + SMALL_NUM) * gdet_c);

        // Then sum the coefficients and record nonzero ones for modification
        GReal C_k = 0;
        GReal S_k = 1e-300;
        DLOOP1 {
            GReal Gam_local = out.gdet_conn(j, i, mu, mu, lam) / gdet_c;

            C_k += Gam_local;
            S_k += m::abs(Gam_local);
        }

        GReal dS_k = D_k - C_k;

        DLOOP1 {
            GReal Gam_val = out.gdet_conn(j, i, mu, mu, lam) / gdet_c;
            GReal W_kmu = m::abs(Gam_val) / S_k;
            GReal Delta_Gamma = dS_k * W_kmu;
            GReal correction_term = Delta_Gamma * gdet_c;
            out.gdet_conn(j, i, mu, mu, lam) += correction_term;
            // Enforce symmetry of Christoffel symbols: Gamma^mu_lam_nu =
            // Gamma^mu_nu_lam We correct Gamma^mu_mu_lam and Gamma^mu_lam_mu identicaly
            out.gdet_conn(j, i, mu, lam, mu) = out.gdet_conn(j, i, mu, mu, lam);
        }
    }
}

// Out adapter writing into the GRCoordinates caches
struct CacheOut {
    GeomTensor2 gcov_, gcon_;
    GeomScalar gdet_;
    GeomTensor3 conn_, gdet_conn_;
    KOKKOS_FORCEINLINE_FUNCTION Real& gcov(const Loci loc, const int j, const int i,
        const int mu, const int nu) const { return gcov_(loc, j, i, mu, nu); }
    KOKKOS_FORCEINLINE_FUNCTION Real& gcon(const Loci loc, const int j, const int i,
        const int mu, const int nu) const { return gcon_(loc, j, i, mu, nu); }
    KOKKOS_FORCEINLINE_FUNCTION Real& gdet(const Loci loc, const int j, const int i) const
    { return gdet_(loc, j, i); }
    KOKKOS_FORCEINLINE_FUNCTION Real& conn(const int j, const int i, const int mu,
        const int nu, const int lam) const { return conn_(j, i, mu, nu, lam); }
    KOKKOS_FORCEINLINE_FUNCTION Real& gdet_conn(const int j, const int i, const int mu,
        const int nu, const int lam) const { return gdet_conn_(j, i, mu, nu, lam); }
};

} // namespace GeomFill
