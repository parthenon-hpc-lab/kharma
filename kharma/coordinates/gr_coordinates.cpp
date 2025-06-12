/* 
 *  File: gr_coordinates.cpp
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

/*
 * Coordinate functions for GR 
 */

#include "gr_coordinates.hpp"

// This needs to be included only here -- it requires full-formed Parthenon
// types, which are not available when importing this file's header
#include "types.hpp"

using Kokkos::MDRangePolicy;
using Kokkos::Rank;

// Stepsize for numerical derivatives of the metric
#define DELTA 1.e-8

#if FAST_CARTESIAN
/**
 * Fast Cartesian GRCoordinates objects just use the underlying UniformCartesian object for everything
 */
GRCoordinates::GRCoordinates(const RegionSize &rs, ParameterInput *pin): UniformCartesian(rs, pin) {}
GRCoordinates::GRCoordinates(const GRCoordinates &src, int coarsen): UniformCartesian(src, coarsen) {}
#else
// Internal function for initializing cache
void init_GRCoordinates(GRCoordinates& G);

/**
 * Construct a GRCoordinates object with a transformation according to preferences set in the package
 */
GRCoordinates::GRCoordinates(const RegionSize &rs, ParameterInput *pin): UniformCartesian(rs, pin),
    coords(pin)
{
    // TODO use new .symmetric?
    n1 = rs.nx(X1DIR) + 2*Globals::nghost;
    n2 = rs.nx(X2DIR) > 1 ? rs.nx(X2DIR) + 2*Globals::nghost : 1;
    n3 = rs.nx(X3DIR) > 1 ? rs.nx(X3DIR) + 2*Globals::nghost : 1;
    //cout << "Initialized coordinates with nghost " << Globals::nghost << std::endl;

    connection_average_points = pin->GetOrAddInteger("coordinates", "connection_average_points", 1);
    correct_connections = pin->GetOrAddBoolean("coordinates", "correct_connections", false);

    init_GRCoordinates(*this);
}

GRCoordinates::GRCoordinates(const GRCoordinates &src, int coarsen): UniformCartesian(src, coarsen),
    coords(src.coords), n1(src.n1/coarsen), n2(src.n2/coarsen), n3(src.n3/coarsen),
    connection_average_points(src.connection_average_points),
    correct_connections(src.correct_connections)
{
    //std::cerr << "Calling coarsen constructor" << std::endl;
    init_GRCoordinates(*this);
}

/**
 * Initialize any cached geometry that GRCoordinates will need to return. While
 * GRCoordinates objects will be moved device-side, this can be run only on the
 * host.
 *
 * This needs to be defined *outside* of the GRCoordinates object, because of some
 * fun issues with C++ Lambda capture, which Kokkos brings to the fore
 */
void init_GRCoordinates(GRCoordinates& G) {
    const int n1 = G.n1;
    const int n2 = G.n2;
    //const int n3 = G.n3;
    const bool correct_connections = G.correct_connections;
    const int connection_average_points = G.connection_average_points;

    //cerr << "Creating GRCoordinate cache size " << n1 << " " << n2 << std::endl;
    // Cache geometry.  May be faster than re-computing. May not be.
    G.gcon_direct = GeomTensor2("gcon", NLOC, n2+1, n1+1, GR_DIM, GR_DIM);
    G.gcov_direct = GeomTensor2("gcov", NLOC, n2+1, n1+1, GR_DIM, GR_DIM);
    G.gdet_direct = GeomScalar("gdet", NLOC, n2+1, n1+1);
    G.conn_direct = GeomTensor3("conn", n2, n1, GR_DIM, GR_DIM, GR_DIM);
    G.gdet_conn_direct = GeomTensor3("conn", n2, n1, GR_DIM, GR_DIM, GR_DIM);

    // Member variables have an implicit this->
    // C++ Lambdas (and therefore Kokkos Lambdas) capture pointers to objects, not full objects
    // Hence, you *CANNOT* use this->, or members, from inside kernels
    auto gcon_local = G.gcon_direct;
    auto gcov_local = G.gcov_direct;
    auto gdet_local = G.gdet_direct;
    auto conn_local = G.conn_direct;
    auto gdet_conn_local = G.gdet_conn_direct;

    Kokkos::parallel_for("init_geom", MDRangePolicy<Rank<2>>({0,0}, {n2+1, n1+1}),
        KOKKOS_LAMBDA (const int& j, const int& i) {
            // Iterate through locations. This could be done in fancy ways, but
            // this highlights what's actually going on.
            for (int iloc =0; iloc < NLOC; iloc++) {
                Loci loc = (Loci) iloc;
                // radius of points to sample, floor(npoints/2)
                const int radius = connection_average_points / 2;
                const int diameter = connection_average_points;
                const int square = connection_average_points*connection_average_points;
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
                            G.coord(0, j, i+1, loc, Xn1);
                            G.coord(0, j+1, i, loc, Xn2);
                            X[1] += (Xn1[1] - X[1])/connection_average_points * k;
                            X[2] += (Xn2[2] - X[2])/connection_average_points * l;
                            // Get geometry at points
                            GReal gcov_loc[GR_DIM][GR_DIM], gcon_loc[GR_DIM][GR_DIM];
                            G.coords.gcov_native(X, gcov_loc);
                            const GReal gdet = G.coords.gcon_from_gcov(gcov_loc, gcon_loc);
                            // Add to running averages
                            gdet_local(loc, j, i) += gdet / square;
                            DLOOP2 {
                                gcov_local(loc, j, i, mu, nu) += gcov_loc[mu][nu] / square;
                                gcon_local(loc, j, i, mu, nu) += gcon_loc[mu][nu] / square;
                            }
                            if (loc == Loci::center) {
                                // In the center, get the connection and gdet*connection
                                Real conn_loc[GR_DIM][GR_DIM][GR_DIM];
                                G.coords.conn_native(X, DELTA, conn_loc);
                                DLOOP3 {
                                    conn_local(j, i, mu, nu, lam) += conn_loc[mu][nu][lam] / square;
                                    gdet_conn_local(j, i, mu, nu, lam) += gdet*conn_loc[mu][nu][lam] / square;
                                }
                            }
                        }
                    }
                } else if (loc == Loci::face1 || loc == Loci::face2) {
                    for (int k=-radius; k <= radius; k++) {
                        // Like the above, but only average over a particular face (line for 2D geometry)
                        GReal X[GR_DIM];
                        G.coord(0, j, i, loc, X);
                        GReal Xn1[GR_DIM];
                        // Step in the nontrivial direction perpendicular to the normal
                        const int avg_dir = (loc == Loci::face1) ? X2DIR : X1DIR;
                        // Get the direction/distance
                        G.coord(0, j + (avg_dir == X2DIR), i + (avg_dir == X1DIR), loc, Xn1);
                        X[avg_dir] += (Xn1[avg_dir] - X[avg_dir])/diameter * k;
                        // Get geometry at the point
                        GReal gcov_loc[GR_DIM][GR_DIM], gcon_loc[GR_DIM][GR_DIM];
                        G.coords.gcov_native(X, gcov_loc);
                        const GReal gdet = G.coords.gcon_from_gcov(gcov_loc, gcon_loc);
                        // Add to running averages
                        gdet_local(loc, j, i) += gdet / diameter;
                        DLOOP2 {
                            gcov_local(loc, j, i, mu, nu) += gcov_loc[mu][nu] / diameter;
                            gcon_local(loc, j, i, mu, nu) += gcon_loc[mu][nu] / diameter;
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
                    gdet_local(loc, j, i) = gdet;
                    DLOOP2 {
                        gcov_local(loc, j, i, mu, nu) = gcov_loc[mu][nu];
                        gcon_local(loc, j, i, mu, nu) = gcon_loc[mu][nu];
                    }
                }
            }
        }
    );
    if (correct_connections) {
        Kokkos::parallel_for("geom_corrections", MDRangePolicy<Rank<2>>({0,0}, {n2, n1}),
            KOKKOS_LAMBDA (const int& j, const int& i) {
                // In the two directions the grid changes, make sure that we *exactly*
                // satisfy the req't gdet*conn^mu_mu_nu = d_nu gdet, when evaluated on faces
                // This will make the source term exactly balance the flux differences,
                // crucial near the poles
                GReal X[GR_DIM];
                G.coord(0, j, i, Loci::center, X);
                if (1) { //(m::abs(X[2] - 0) < 0.08 || m::abs(X[2] - 1.0) < 0.08)) {
                    for (int lam=1; lam < GR_DIM; lam++) {
                        const Loci loc = loc_of(lam);
                        // Get gdet values at faces we calculated above
                        GReal Xfm[GR_DIM], Xfp[GR_DIM];
                        G.coord(0, j, i, loc, Xfm);
                        G.coord(0, j + (lam == X2DIR), i + (lam == X1DIR), loc, Xfp);
                        double gdetfm = gdet_local(loc, j, i);
                        double gdetfp = gdet_local(loc, j + (lam == X2DIR), i + (lam == X1DIR));
                        GReal target = (gdetfp - gdetfm) / (Xfp[lam] - Xfm[lam] + SMALL_NUM);

                        // Then sum the coefficients and record nonzero ones for modification
                        GReal test_sum = 0;
                        GReal sum_portions, portions[GR_DIM] = {0};
                        DLOOP1 {
                            test_sum += gdet_conn_local(j, i, mu, mu, lam);
                            portions[mu] = m::abs(gdet_conn_local(j, i, mu, mu, lam));
                            sum_portions += portions[mu];
                        }
                        DLOOP1 portions[mu] /= sum_portions;
                        //printf("Zone %d %d target: %.3g test_sum: %.3g correction: %.3g\n", i, j, target, test_sum, diff);

                        // Add the difference among components equally
                        const GReal diff = test_sum - target;
                        DLOOP1 gdet_conn_local(j, i, mu, mu, lam) = gdet_conn_local(j, i, mu, mu, lam) - diff*portions[mu];

                        // This is separated and set equal, as there will be one self-assignment
                        DLOOP1 gdet_conn_local(j, i, mu, lam, mu) = gdet_conn_local(j, i, mu, mu, lam);
                    }
                }
            }
        );
    }
}
#endif // FAST_CARTESIAN
