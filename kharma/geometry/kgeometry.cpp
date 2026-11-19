/*
 *  File: kgeometry.cpp
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
#include "kgeometry.hpp"
#include "geometry_fill.hpp"

std::shared_ptr<KHARMAPackage> KGeometry::Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("KGeometry");
    Params& params = pkg->AllParams();

    // Same options GRCoordinates reads, so both paths agree
    const int connection_average_points =
        pin->GetOrAddInteger("coordinates", "connection_average_points", 1);
    if (connection_average_points % 2 == 0 || connection_average_points < 1)
        throw std::invalid_argument("connection_average_points must be positive and odd!");
    params.Add("connection_average_points", connection_average_points);
    params.Add("correct_connections",
        pin->GetOrAddBoolean("coordinates", "correct_connections", false));

    // Compare fields against GRCoordinates caches after every fill
    params.Add("verify", pin->GetOrAddBoolean("kgeometry", "verify", false));

#if !FAST_CARTESIAN
    // Geometry cache size per block, computed exactly as in the GRCoordinates
    // constructor: total zones including ghosts in nontrivial directions, else 1.
    // There are no blocks yet, so read the block size from the input deck
    // (Parthenon uses the whole mesh as one block if <parthenon/meshblock> is absent).
    auto block_nx = [pin](const std::string& dir) {
        const int mesh_nx = pin->GetInteger("parthenon/mesh", dir);
        return pin->DoesParameterExist("parthenon/meshblock", dir)
                   ? pin->GetInteger("parthenon/meshblock", dir)
                   : mesh_nx;
    };
    const int nx1 = block_nx("nx1"), nx2 = block_nx("nx2");
    const int n1 = nx1 > 1 ? nx1 + 2 * Globals::nghost : 1;
    const int n2 = nx2 > 1 ? nx2 + 2 * Globals::nghost : 1;
    params.Add("n1", n1);
    params.Add("n2", n2);

    // Not Cell: these have no k index, and must not be picked up by flag-based packs.
    // Not Restart: they are refilled at every structural event.
    const std::vector<MetadataFlag> flags = {Metadata::Derived, Metadata::OneCopy};
    // Shapes are {i, j, k, components}, i fastest.  Faces/corners need the extra +1.
    const int nmetric = NLOC * GR_DIM * GR_DIM;
    const int nconn = GR_DIM * GR_DIM * GR_DIM;
    pkg->AddField("kg.gcov", Metadata(flags, std::vector<int>{n1 + 1, n2 + 1, 1, nmetric}));
    pkg->AddField("kg.gcon", Metadata(flags, std::vector<int>{n1 + 1, n2 + 1, 1, nmetric}));
    pkg->AddField("kg.gdet", Metadata(flags, std::vector<int>{n1 + 1, n2 + 1, 1, NLOC}));
    // Connections exist only at cell centers, as in GRCoordinates
    pkg->AddField("kg.conn", Metadata(flags, std::vector<int>{n1, n2, 1, nconn}));
    pkg->AddField("kg.gdet_conn", Metadata(flags, std::vector<int>{n1, n2, 1, nconn}));
#endif

    return pkg;
}

// Pack this block's geometry fields by name (flag-based packs would skip them)
static KGeometry::FieldOut GetFieldOut(MeshBlock* pmb)
{
    auto rc = pmb->meshblock_data.Get();
    PackIndexMap imap;
    auto pack = rc->PackVariables(std::vector<std::string>{"kg.gcov", "kg.gcon",
        "kg.gdet", "kg.conn", "kg.gdet_conn"}, imap);
    return KGeometry::FieldOut{pack, imap["kg.gcov"].first, imap["kg.gcon"].first,
        imap["kg.gdet"].first, imap["kg.conn"].first, imap["kg.gdet_conn"].first};
}

void KGeometry::SetGeometryBlock(MeshBlock* pmb, ParameterInput* pin)
{
#if !FAST_CARTESIAN
    const auto& params = pmb->packages.Get("KGeometry")->AllParams();
    const int n1 = params.Get<int>("n1");
    const int n2 = params.Get<int>("n2");
    const int connection_average_points = params.Get<int>("connection_average_points");

    // Metric source.  For now, the block's GRCoordinates: its coord() for positions and
    // its CoordinateEmbedding for the metric.  This is the swap point for later sources.
    const GRCoordinates G = pmb->coords;
    PARTHENON_REQUIRE(G.n1 == n1 && G.n2 == n2,
        "KGeometry field size does not match GRCoordinates cache size!");

    const FieldOut out = GetFieldOut(pmb);
    pmb->par_for("kgeometry_fill", 0, n2, 0, n1,
        KOKKOS_LAMBDA(const int& j, const int& i)
        {
            GeomFill::fill_point(G, connection_average_points, n1, n2, j, i, out);
        });
    if (params.Get<bool>("correct_connections")) {
        pmb->par_for("kgeometry_correct_connections", 0, n2 - 1, 0, n1 - 1,
            KOKKOS_LAMBDA(const int& j, const int& i)
            {
                GeomFill::correct_connections_point(G, j, i, out);
            });
    }

    if (params.Get<bool>("verify")) VerifyBlock(pmb);
#endif
}

// Equal, counting NaN == NaN (both paths compute the same thing, NaNs included)
KOKKOS_FORCEINLINE_FUNCTION bool same(const Real a, const Real b)
{
    return a == b || (a != a && b != b);
}

void KGeometry::VerifyBlock(MeshBlock* pmb)
{
#if !FAST_CARTESIAN
    const auto& params = pmb->packages.Get("KGeometry")->AllParams();
    const int n1 = params.Get<int>("n1");
    const int n2 = params.Get<int>("n2");
    const GRCoordinates G = pmb->coords;
    const FieldOut f = GetFieldOut(pmb);

    int nbad = 0;
    parthenon::par_reduce(parthenon::loop_pattern_mdrange_tag, "kgeometry_verify",
        DevExecSpace(), 0, n2, 0, n1,
        KOKKOS_LAMBDA(const int j, const int i, int& lbad)
        {
            for (int iloc = 0; iloc < NLOC; iloc++) {
                const Loci loc = (Loci)iloc;
                if (!same(f.gdet(loc, j, i), G.gdet_direct(loc, j, i))) lbad++;
                DLOOP2 {
                    if (!same(f.gcov(loc, j, i, mu, nu), G.gcov_direct(loc, j, i, mu, nu)))
                        lbad++;
                    if (!same(f.gcon(loc, j, i, mu, nu), G.gcon_direct(loc, j, i, mu, nu)))
                        lbad++;
                }
            }
            if (i < n1 && j < n2) {
                DLOOP3 {
                    if (!same(f.conn(j, i, mu, nu, lam), G.conn_direct(j, i, mu, nu, lam)))
                        lbad++;
                    if (!same(f.gdet_conn(j, i, mu, nu, lam),
                            G.gdet_conn_direct(j, i, mu, nu, lam)))
                        lbad++;
                }
            }
        },
        nbad);

    std::cout << "KGeometry verify: block " << pmb->gid << " (" << n1 << "x" << n2
              << "): " << nbad << " mismatches" << std::endl;
    if (nbad > 0) throw std::runtime_error("KGeometry fields differ from GRCoordinates!");
#endif
}

