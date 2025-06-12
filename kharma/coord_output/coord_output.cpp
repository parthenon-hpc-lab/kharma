/*
 *  File: coord_output.cpp
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
#include "coord_output.hpp"

#include "domain.hpp"

std::shared_ptr<KHARMAPackage> CoordinateOutput::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("CoordinateOutput");
    Params &params = pkg->AllParams();

    // Any options? Which fields to output is determined in outputs

    // Fields: cell-center values for geometry only
    // TODO add values elsewhere e.g. faces, edges?
    // Note these are all marked with the "CoordinateOutput" flag
    std::vector<int> s_4vector({GR_DIM});
    std::vector<int> s_4tensor({GR_DIM, GR_DIM});
    std::vector<int> s_4conn({GR_DIM, GR_DIM, GR_DIM});
    std::vector<MetadataFlag> flags_geom = {Metadata::Real, Metadata::Cell, Metadata::Derived,
                                            Metadata::OneCopy};
    auto m0 = Metadata(flags_geom);
    auto m1 = Metadata(flags_geom, s_4vector);
    auto m2 = Metadata(flags_geom, s_4tensor);
    auto m3 = Metadata(flags_geom, s_4conn);

    // Native coordinates, t/X1/X2/X3
    pkg->AddField("coords.Xnative", m1);
    pkg->AddField("coords.X1", m0);
    pkg->AddField("coords.X2", m0);
    pkg->AddField("coords.X3", m0);
    // Cartesian (or cartesianized KS) coordinates
    pkg->AddField("coords.Xcart", m1);
    pkg->AddField("coords.x", m0);
    pkg->AddField("coords.y", m0);
    pkg->AddField("coords.z", m0);
    // Spherical KS coordinates
    pkg->AddField("coords.Xks", m1);
    pkg->AddField("coords.r", m0);
    pkg->AddField("coords.th", m0);
    pkg->AddField("coords.phi", m0);

    // Metric
    pkg->AddField("coords.gcon", m2);
    pkg->AddField("coords.gcov", m2);
    pkg->AddField("coords.gdet", m0);
    pkg->AddField("coords.lapse", m0);
    pkg->AddField("coords.conn", m3);

    // Metric, embedding (i.e., KS) coordinates
    pkg->AddField("coords.gcon_embed", m2);
    pkg->AddField("coords.gcov_embed", m2);
    pkg->AddField("coords.gdet_embed", m0);
    // TODO?
    //pkg->AddField("coords.lapse_embed", m0);
    //pkg->AddField("coords.conn_embed", m3);

    // Register our output.  This will be called before *any* output,
    // but we will only fill the fields before the first.
    // This is all that's needed unless:
    // 1. Someone wants geometry in an AMR sim with remeshing
    // 2. Parthenon decides to include a way to delete fields, which we would want to do here
    pkg->BlockUserWorkBeforeOutput = CoordinateOutput::BlockUserWorkBeforeOutput;

    return pkg;
}

TaskStatus CoordinateOutput::BlockUserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin)
{
    auto& globals = pmb->packages.Get("Globals")->AllParams();
    if (!globals.Get<bool>("in_loop")) {
        auto rc = pmb->meshblock_data.Get("base");

        PackIndexMap geom_map;
        auto Geom = rc->PackVariables({Metadata::GetUserFlag("CoordinateOutput")}, geom_map);

        const auto& G = pmb->coords;

        const int mXnative = geom_map["coords.Xnative"].first;
        const int mX1 = geom_map["coords.X1"].first;
        const int mX2 = geom_map["coords.X2"].first;
        const int mX3 = geom_map["coords.X3"].first;

        const int mXcart = geom_map["coords.Xcart"].first;
        const int mx = geom_map["coords.x"].first;
        const int my = geom_map["coords.y"].first;
        const int mz = geom_map["coords.z"].first;

        const int mXsph = geom_map["coords.Xsph"].first;
        const int mr = geom_map["coords.r"].first;
        const int mth = geom_map["coords.th"].first;
        const int mphi = geom_map["coords.phi"].first;

        const int mgcov = geom_map["coords.gcov"].first;
        const int mgcon = geom_map["coords.gcon"].first;
        const int mgdet = geom_map["coords.gdet"].first;
        const int mlapse = geom_map["coords.lapse"].first;
        const int mconn = geom_map["coords.conn"].first;

        const int mgcov_embed = geom_map["coords.gcov_embed"].first;
        const int mgcon_embed = geom_map["coords.gcon_embed"].first;
        const int mgdet_embed = geom_map["coords.gdet_embed"].first;

        IndexRange3 b = KDomain::GetRange(rc, IndexDomain::entire);
        pmb->par_for("set_geometry", b.ks, b.ke, b.js, b.je, b.is, b.ie,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                // Native
                GReal Xnative[GR_DIM];
                G.coord(k, j, i, Loci::center, Xnative);
                Geom(mXnative+1, k, j, i) = Geom(mX1, k, j, i) = Xnative[1];
                Geom(mXnative+2, k, j, i) = Geom(mX2, k, j, i) = Xnative[2];
                Geom(mXnative+3, k, j, i) = Geom(mX3, k, j, i) = Xnative[3];
                // Cartesian
                Geom(mXcart+1, k, j, i) = Geom(mx, k, j, i) = G.x(k, j, i);
                Geom(mXcart+2, k, j, i) = Geom(my, k, j, i) = G.y(k, j, i);
                Geom(mXcart+3, k, j, i) = Geom(mz, k, j, i) = G.z(k, j, i);
                // Spherical
                Geom(mXsph+1, k, j, i) = Geom(mr, k, j, i) = G.r(k, j, i);
                Geom(mXsph+2, k, j, i) = Geom(mth, k, j, i) = G.th(k, j, i);
                Geom(mXsph+3, k, j, i) = Geom(mphi, k, j, i) = G.phi(k, j, i);

                // Metric, native
                DLOOP2 Geom(mgcov+GR_DIM*mu+nu, k, j, i) = G.gcov(Loci::center, j, i, mu, nu);
                DLOOP2 Geom(mgcon+GR_DIM*mu+nu, k, j, i) = G.gcon(Loci::center, j, i, mu, nu);
                Geom(mgdet, k, j, i) = G.gdet(Loci::center, j, i);
                Geom(mlapse, k, j, i) = 1. / m::sqrt(-G.gcon(Loci::center, j, i, 0, 0));
                // shift? = G.gcon(Loci::center, j, i, 0, 1) * alpha * alpha;
                // Connection
                DLOOP3 Geom(mconn+GR_DIM*GR_DIM*mu+GR_DIM*nu+lam, k, j, i) = G.conn(j, i, mu, nu, lam);

                // Metric, embedding
                GReal Xembed[GR_DIM], gcov_embed[GR_DIM][GR_DIM], gcon_embed[GR_DIM][GR_DIM];
                G.coord_embed(k, j, i, Loci::center, Xembed);
                G.coords.gcov_embed(Xembed, gcov_embed);
                GReal gdet = G.coords.gcon_embed(Xembed, gcon_embed); // Save a tiny bit of time
                DLOOP2 Geom(mgcov_embed+GR_DIM*mu+nu, k, j, i) = gcov_embed[mu][nu];
                DLOOP2 Geom(mgcon_embed+GR_DIM*mu+nu, k, j, i) = gcon_embed[mu][nu];
                Geom(mgdet_embed, k, j, i) = gdet;
            }
        );
    }

    return TaskStatus::complete;
}
