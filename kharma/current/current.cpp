/*
 *  File: current.cpp
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

#include "current.hpp"

std::shared_ptr<KHARMAPackage> Current::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Current");
    Params &params = pkg->AllParams();

    // 4-current jcon. Calculated only for output
    std::vector<int> s_fourvector({GR_DIM});
    auto m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_fourvector);
    pkg->AddField("jcon", m);

    // Temporaries
    std::vector<int> s_vector({NVEC});
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_vector);
    pkg->AddField("Current.uvec_c", m);
    pkg->AddField("Current.B_P_c", m);

    pkg->BlockUserWorkBeforeOutput = Current::FillOutput;

    return pkg;
}

TaskStatus Current::CalculateCurrent(MeshBlockData<Real> *rc0, MeshBlockData<Real> *rc1, const double& dt)
{
    auto pmb = rc0->GetBlockPointer();
    GridVector uvec_old = rc0->Get("prims.uvec").data;
    GridVector B_P_old = rc0->Get("prims.B").data;
    GridVector uvec_new = rc1->Get("prims.uvec").data;
    GridVector B_P_new = rc1->Get("prims.B").data;
    GridVector jcon = rc1->Get("jcon").data;

    GridVector uvec_c = rc1->Get("Current.uvec_c").data;
    GridVector B_P_c = rc1->Get("Current.B_P_c").data;

    const auto& G = pmb->coords;
    const int ndim = pmb->pmy_mesh->ndim;

    // Calculate time-centered primitives
    // We could pack, but we just need the vectors, U1,2,3 and B1,2,3
    // Apply over the whole grid, as calculating j requires neighbors
    const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
    const IndexRange nv = IndexRange{0, NVEC-1};
    pmb->par_for("get_center", nv.s, nv.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &p, const int &k, const int &j, const int &i) {
            uvec_c(p, k, j, i) = 0.5*(uvec_old(p, k, j, i) + uvec_new(p, k, j, i));
            B_P_c(p, k, j, i) = 0.5*(B_P_old(p, k, j, i) + B_P_new(p, k, j, i));
        }
    );

    // Calculate j^{\mu} using centered differences for active zones
    const IndexRange ib_i = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    const IndexRange jb_i = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange kb_i = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    const IndexRange n4v = IndexRange{0, GR_DIM-1};
    pmb->par_for("jcon_calc", n4v.s, n4v.e, kb_i.s, kb_i.e, jb_i.s, jb_i.e, ib_i.s, ib_i.e,
        KOKKOS_LAMBDA (const int &mu, const int &k, const int &j, const int &i) {
            // Get sqrt{-g}*F^{mu nu} at neighboring points
            // TODO(BSP) this recalculates Fcon a lot...
            const Real gF0p = get_gdet_Fcon(G, uvec_new, B_P_new, 0, mu, k, j, i);
            const Real gF0m = get_gdet_Fcon(G, uvec_old, B_P_old, 0, mu, k, j, i);
            const Real gF1p = get_gdet_Fcon(G, uvec_c, B_P_c, 1, mu, k, j, i+1);
            const Real gF1m = get_gdet_Fcon(G, uvec_c, B_P_c, 1, mu, k, j, i-1);
            const Real gF2p = (ndim > 1) ? get_gdet_Fcon(G, uvec_c, B_P_c, 2, mu, k, j+1, i) : 0.;
            const Real gF2m = (ndim > 1) ? get_gdet_Fcon(G, uvec_c, B_P_c, 2, mu, k, j-1, i) : 0.;
            const Real gF3p = (ndim > 2) ? get_gdet_Fcon(G, uvec_c, B_P_c, 3, mu, k+1, j, i) : 0.;
            const Real gF3m = (ndim > 2) ? get_gdet_Fcon(G, uvec_c, B_P_c, 3, mu, k-1, j, i) : 0.;

            // Difference: D_mu F^{mu nu} = 4 \pi j^nu
            jcon(mu, k, j, i) = 1. / (m::sqrt(4. * M_PI) * G.gdet(Loci::center, j, i)) *
                                ((gF0p - gF0m) / dt +
                                (gF1p - gF1m) / (2 * G.Dxc<1>(i)) +
                                (gF2p - gF2m) / (2 * G.Dxc<2>(j)) +
                                (gF3p - gF3m) / (2 * G.Dxc<3>(k)));
        }
    );

    return TaskStatus::complete;
}

void Current::FillOutput(MeshBlock *pmb, ParameterInput *pin)
{
    // The "preserve" container will only exist after we've taken a step,
    // catch that situation
    auto& rc1 = pmb->meshblock_data.Get("base");
    auto rc0 = rc1; // Avoid writing rc0's type when initializing. Still light.
    try {
        // Get the state at beginning of the step
        rc0 = pmb->meshblock_data.Get("preserve");
    } catch (const std::runtime_error& e) {
        // We expect this to happen the first step
        // We just don't need to fill jcon the first time around
        //std::cerr << "This should only happen once: " << e.what() << std::endl;
        return;
    }

    // Get the duration of the last timestep from the "Globals" package
    // (see kharma.cpp)
    Real dt_last = pmb->packages.Get("Globals")->Param<Real>("dt_last");

    Current::CalculateCurrent(rc0.get(), rc1.get(), dt_last);
}
