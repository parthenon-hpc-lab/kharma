/*
 *  File: fix_solve.cpp
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

#include "implicit.hpp"

#include "domain.hpp"
#include "floors.hpp"
#include "flux_functions.hpp"

#define NFVAR_MAX 10

#if DISABLE_EMHD

TaskStatus Implicit::FixSolve(MeshBlockData<Real>* mbd) { return TaskStatus::complete; }

#else

// TODO(CEP) should merge this with FixUtoP by generalizing that
TaskStatus Implicit::FixSolve(MeshBlockData<Real>* mbd)
{

    Flag("FixSolve");
    // Get MeshBlock pointer and obtain flag for primitives
    auto pmb = mbd->GetBlockPointer();

    // Get number of implicit variables
    PackIndexMap implicit_prims_map;
    auto implicit_vars =
        Implicit::GetOrderedNames(mbd, Metadata::GetUserFlag("Primitive"), true);
    auto& P = mbd->PackVariables(implicit_vars, implicit_prims_map);
    const int nfvar = P.GetDim(4);

    // Since we're after sync, we run over the entire domain
    const IndexRange3 b = KDomain::GetRange(mbd, IndexDomain::entire);
    const auto& G = pmb->coords;

    GridScalar solve_fail = mbd->Get("solve_fail").data;

    const int flag_verbose = pmb->packages.Get("Globals")->Param<int>("flag_verbose");

    pmb->par_for("fix_solver_failures", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            // Fix only bad zones
            // Remember "failed" here has a different implementation
            if (failed(solve_fail(k, j, i))) {
                // printf("Fixing zone %d %d %d!\n", i, j, k);
                double wsum = 0., wsum_x = 0.;
                double sum[NFVAR_MAX] = {0.}, sum_x[NFVAR_MAX] = {0.};
                // For all neighboring cells...
                for (int n = -1; n <= 1; n++) {
                    for (int m = -1; m <= 1; m++) {
                        for (int l = -1; l <= 1; l++) {
                            int ii = i + l, jj = j + m, kk = k + n;
                            // If we haven't overstepped array bounds...
                            if (KDomain::inside(kk, jj, ii, b)) {
                                // Weight by distance
                                // TODO abs(l) == l*l always?
                                double w = 1. / (m::abs(l) + m::abs(m) + m::abs(n) + 1);

                                // Count only the good cells, if we can
                                if (!failed(solve_fail(kk, jj, ii))) {
                                    // Weight by distance.  Note interpolated "fixed"
                                    // cells stay flagged
                                    wsum += w;
                                    FLOOP sum[ip] += w * P(ip, kk, jj, ii);
                                }
                                // Just in case, keep a sum of even the bad ones
                                wsum_x += w;
                                FLOOP sum_x[ip] += w * P(ip, kk, jj, ii);
                            }
                        }
                    }
                }

                if (wsum < 1.e-10) {
                // TODO probably should crash here.
#ifndef KOKKOS_ENABLE_SYCL
                    if (flag_verbose >=
                        3) // && KDomain::inside(k, j, i, kb_b, jb_b, ib_b))
                           // // If an interior zone...
                        printf("No neighbors were available at %d %d %d!\n", i, j, k);
#endif
                    FLOOP P(ip, k, j, i) = sum_x[ip] / wsum_x;
                } else {
                    FLOOP P(ip, k, j, i) = sum[ip] / wsum;
                }
            }
        });

    // Since floors were applied earlier, we assume the zones obtained by averaging the
    // neighbors also respect the floors. Compute new conserved variables
    // TODO encapsulate version from FixUtoP & try calling whole thing here?
    PackIndexMap prims_map, cons_map;
    auto& P_all = mbd->PackVariables(
        std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prims_map);
    auto& U_all =
        mbd->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    const auto& eos_params = pmb->packages.Get("eos")->AllParams();
    auto eos = eos_params.Get<Microphysics::EOS::EOS>("d.EOS");

    // Need emhd_params object
    const EMHD::EMHD_parameters emhd_params = EMHD::GetEMHDParameters(pmb->packages);

    pmb->par_for("fix_solver_failures_PtoU", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            if (failed(solve_fail(k, j, i)))
                Flux::p_to_u(G, P_all, m_p, emhd_params, eos, k, j, i, U_all, m_u);
        });

    EndFlag();
    return TaskStatus::complete;
}

#endif
