/*
 *  File: floors_impl.hpp
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

#include "floors.hpp"
// phoebus includes
#include "microphysics/eos_kharma/eos_kharma.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "phoebus_utils/variables.hpp"
#include "domain.hpp"

namespace Floors
{

/**
 * Template to call through to templated apply_floors
 */
template<InjectionFrame frame>
TaskStatus ApplyFloorsInFrame(MeshData<Real>* md, IndexDomain domain)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    PackIndexMap prims_map, cons_map;
    auto P = md->PackVariables(
        std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prims_map);
    auto U = md->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    auto fflag = md->PackVariables(std::vector<std::string>{"fflag"});
    auto pflag = md->PackVariables(std::vector<std::string>{"pflag"});
    PackIndexMap floors_map;
    auto floor_vals = md->PackVariables(
        std::vector<std::string>{"Floors.rho_floor", "Floors.u_floor"}, floors_map);
    const int rhofi = floors_map["Floors.rho_floor"].first;
    const int ufi = floors_map["Floors.u_floor"].first;

    const Real gam = pmb0->packages.Get("GRMHD")->Param<Real>("gamma");

    const auto& eos_params = pmb0->packages.Get("eos")->AllParams();
    auto eos = eos_params.Get<Microphysics::EOS::EOS>("d.EOS");

    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb0->packages);
    const Real switch_r =
        (frame == InjectionFrame::mixed_fluid_normal)
            ? pmb0->packages.Get("Floors")->Param<Real>("frame_switch_r")
            : 0;
    const Real switch_beta =
        (frame == InjectionFrame::mixed_normal_drift)
            ? pmb0->packages.Get("Floors")->Param<Real>("frame_switch_beta")
            : 0;
    // Still needed for ceilings and determining floors
    const Floors::Prescription floors =
        pmb0->packages.Get("Floors")->Param<Floors::Prescription>("prescription");
    const Floors::Prescription floors_inner =
        pmb0->packages.Get("Floors")->Param<Floors::Prescription>("prescription_inner");

    const IndexRange3 b = KDomain::GetRange(md, domain);
    const IndexRange block = IndexRange{0, P.GetDim(5) - 1};
    pmb0->par_for("apply_floors", block.s, block.e, b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i)
        {
            if (static_cast<int>(fflag(b, 0, k, j, i)) ||
                static_cast<int>(pflag(b, 0, k, j, i))) {
                const auto& G = P.GetCoords(b);
                // apply_floors can involve another U_to_P call, capture that flag
                // this is the default return for "no inversion"
                int pflag_l = -1;
                // These would be constexpr except Nvidia doesn't like lambda-capture in
                // constexpr ifs
                if (frame == InjectionFrame::mixed_fluid_normal) {
                    if (G.r(k, j, i) > switch_r) {
                        pflag_l = apply_floors<InjectionFrame::fluid>(G, P(b), m_p, gam,
                            k, j, i, floor_vals(b, rhofi, k, j, i),
                            floor_vals(b, ufi, k, j, i), U(b), m_u);
                    } else {
                        // TODO should mixed frames respect Kastaun vs 1Dw?
                        // Since no prior simulations use mixed frames thus requiring
                        // back-compat, I said no
                        pflag_l = apply_floors<InjectionFrame::normal_kastaun>(G, P(b),
                            m_p, gam, k, j, i, floor_vals(b, rhofi, k, j, i),
                            floor_vals(b, ufi, k, j, i), U(b), m_u);
                    }
                } else if (frame == InjectionFrame::mixed_normal_drift) {
                    FourVectors Dtmp;
                    GRMHD::calc_4vecs(G, P(b), m_p, k, j, i, Loci::center, Dtmp);
                    Real mag_switch =
                        m::min(P(b, m_p.RHO, k, j, i), P(b, m_p.UU, k, j, i)) /
                        dot(Dtmp.bcon, Dtmp.bcov);
                    if (mag_switch < switch_beta) {
                        pflag_l = apply_floors<InjectionFrame::drift>(G, P(b), m_p, gam,
                            k, j, i, floor_vals(b, rhofi, k, j, i),
                            floor_vals(b, ufi, k, j, i), U(b), m_u);
                    } else {
                        pflag_l = apply_floors<InjectionFrame::normal_kastaun>(G, P(b),
                            m_p, gam, k, j, i, floor_vals(b, rhofi, k, j, i),
                            floor_vals(b, ufi, k, j, i), U(b), m_u);
                    }
                } else {
                    pflag_l = apply_floors<frame>(G, P(b), m_p, gam, k, j, i,
                        floor_vals(b, rhofi, k, j, i), floor_vals(b, ufi, k, j, i), U(b),
                        m_u);
                }

                // Record the pflag if we applied normal floors -- successful or not
                if (pflag_l >= 0) pflag(b, 0, k, j, i) = pflag_l;

                // Apply ceilings *after* floors, to make the temperature ceiling
                // better-behaved
                apply_ceilings(
                    G, P(b), m_p, gam, k, j, i, floors, floors_inner, U(b), m_u);

                // P->U if we inverted *correctly* (or didn't invert)
                if (pflag_l <= 0)
                    Flux::p_to_u_mhd(
                        G, P(b), m_p, emhd_params, gam, k, j, i, U(b), m_u, Loci::center);
            }
        });

    return TaskStatus::complete;
}

} // namespace Floors
