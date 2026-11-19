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
#include "geometry_fill.hpp"

using Kokkos::MDRangePolicy;
using Kokkos::Rank;

// Stepsize for numerical derivatives of the metric
#define DELTA 1.e-8

#if FAST_CARTESIAN
/**
 * Fast Cartesian GRCoordinates objects just use the underlying UniformCartesian object
 * for everything
 */
GRCoordinates::GRCoordinates(const RegionSize& rs, ParameterInput* pin)
    : UniformCartesian(rs, pin)
{}
GRCoordinates::GRCoordinates(const GRCoordinates& src, int coarsen)
    : UniformCartesian(src, coarsen)
{}
#else
// Internal function for initializing cache
void init_GRCoordinates(GRCoordinates& G);

/**
 * Construct a GRCoordinates object with a transformation according to preferences set in
 * the package
 */
GRCoordinates::GRCoordinates(const RegionSize& rs, ParameterInput* pin)
    : UniformCartesian(rs, pin)
    , coords(pin)
{
    // TODO use new .symmetric?
    n1 = rs.nx(X1DIR) > 1 ? rs.nx(X1DIR) + 2 * Globals::nghost : 1;
    n2 = rs.nx(X2DIR) > 1 ? rs.nx(X2DIR) + 2 * Globals::nghost : 1;
    n3 = rs.nx(X3DIR) > 1 ? rs.nx(X3DIR) + 2 * Globals::nghost : 1;
    // cout << "Initialized coordinates with nghost " << Globals::nghost << std::endl;

    connection_average_points =
        pin->GetOrAddInteger("coordinates", "connection_average_points", 1);
    if (connection_average_points % 2 == 0 || connection_average_points < 1)
        throw std::invalid_argument(
            "connection_average_points must be positive and odd!");
    correct_connections =
        pin->GetOrAddBoolean("coordinates", "correct_connections", false);

    init_GRCoordinates(*this);
}

GRCoordinates::GRCoordinates(const GRCoordinates& src, int coarsen)
    : UniformCartesian(src, coarsen)
    , coords(src.coords)
    , n1(src.n1 > 1 ? (src.n1 - 2 * Globals::nghost) / coarsen + 2 * Globals::nghost : 1)
    , n2(src.n2 > 1 ? (src.n2 - 2 * Globals::nghost) / coarsen + 2 * Globals::nghost : 1)
    , n3(src.n3 > 1 ? (src.n3 - 2 * Globals::nghost) / coarsen + 2 * Globals::nghost : 1)
    , connection_average_points(src.connection_average_points)
    , correct_connections(src.correct_connections)
{
    // std::cerr << "Calling coarsen constructor" << std::endl;
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
void init_GRCoordinates(GRCoordinates& G)
{
    const int n1 = G.n1;
    const int n2 = G.n2;
    // const int n3 = G.n3;
    const bool correct_connections = G.correct_connections;
    const int connection_average_points = G.connection_average_points;

    // cerr << "Creating GRCoordinate cache size " << n1 << " " << n2 << std::endl;
    //  Cache geometry.  May be faster than re-computing. May not be.
    G.gcon_direct = GeomTensor2("gcon", NLOC, n2 + 1, n1 + 1, GR_DIM, GR_DIM);
    G.gcov_direct = GeomTensor2("gcov", NLOC, n2 + 1, n1 + 1, GR_DIM, GR_DIM);
    G.gdet_direct = GeomScalar("gdet", NLOC, n2 + 1, n1 + 1);
    G.conn_direct = GeomTensor3("conn", n2, n1, GR_DIM, GR_DIM, GR_DIM);
    G.gdet_conn_direct = GeomTensor3("conn", n2, n1, GR_DIM, GR_DIM, GR_DIM);

    const GeomFill::CacheOut out{G.gcov_direct, G.gcon_direct, G.gdet_direct,
        G.conn_direct, G.gdet_conn_direct};

    Kokkos::parallel_for("init_geom", MDRangePolicy<Rank<2>>({0, 0}, {n2 + 1, n1 + 1}),
        KOKKOS_LAMBDA(const int& j, const int& i)
        {
            GeomFill::fill_point(G, connection_average_points, n1, n2, j, i, out);
        });
    if (correct_connections) {
        Kokkos::parallel_for("geom_corrections", MDRangePolicy<Rank<2>>({0, 0}, {n2, n1}),
            KOKKOS_LAMBDA(const int& j, const int& i)
            {
                GeomFill::correct_connections_point(G, j, i, out);
            });
    }

}
#endif // FAST_CARTESIAN
