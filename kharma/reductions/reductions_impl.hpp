/*
 *  File: reductions_variables.hpp
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

// This is a weird header. It's included at the end of reductions.hpp, in order to provide
// the source of the below templates to other files to instantiate.
// Otherwise, it operates like a normal cpp (NOT hpp) file.

// Satisfy IDE parsers who aren't wise to our schemes
#include "reductions.hpp"

// phoebus includes
#include "microphysics/eos_kharma/eos_kharma.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "phoebus_utils/variables.hpp"

template<typename T, bool all_reduce>
inline std::string GetPoolName()
{
    if constexpr (all_reduce) {
        if constexpr (std::is_same<T, Real>::value) return "allreduce_pool";
        if constexpr (std::is_same<T, int>::value) return "int_allreduce_pool";
        if constexpr (std::is_same<T, std::vector<Real>>::value)
            return "vector_allreduce_pool";
        if constexpr (std::is_same<T, std::vector<int>>::value)
            return "vector_int_allreduce_pool";
    } else {
        if constexpr (std::is_same<T, Real>::value) return "reduce_pool";
        if constexpr (std::is_same<T, int>::value) return "int_reduce_pool";
        if constexpr (std::is_same<T, std::vector<Real>>::value)
            return "vector_reduce_pool";
        if constexpr (std::is_same<T, std::vector<int>>::value)
            return "vector_int_reduce_pool";
    }
}

// MPI reduction starts
template<typename T>
void Reductions::Start(MeshData<Real>* md, int channel, T val, MPI_Op op)
{
    // Get the relevant reducer
    const std::string pool_name = GetPoolName<T, false>();
    auto& pars = md->GetMeshPointer()->packages.Get("Reductions")->AllParams();
    auto* reduce_pool = pars.GetMutable<std::vector<Reduce<T>>>(pool_name);
    while (reduce_pool->size() <= channel) reduce_pool->push_back(Reduce<T>());
    auto& reduce = (*reduce_pool)[channel];
    // Fill with flags
    reduce.val = val;
    reduce.StartReduce(0, op);
}
template<typename T>
void Reductions::StartToAll(MeshData<Real>* md, int channel, T val, MPI_Op op)
{
    // Get the relevant reducer
    const std::string pool_name = GetPoolName<T, true>();
    auto& pars = md->GetMeshPointer()->packages.Get("Reductions")->AllParams();
    auto* allreduce_pool = pars.GetMutable<std::vector<AllReduce<T>>>(pool_name);
    while (allreduce_pool->size() <= channel) allreduce_pool->push_back(AllReduce<T>());
    auto& reduce = (*allreduce_pool)[channel];
    // Fill with flags
    reduce.val = val;
    reduce.StartReduce(op);
}

// MPI reduction checks
template<typename T>
T Reductions::Check(MeshData<Real>* md, int channel)
{
    // Get the relevant reducer and result
    const std::string pool_name = GetPoolName<T, false>();
    auto& pars = md->GetMeshPointer()->packages.Get("Reductions")->AllParams();
    auto* reduce_pool = pars.GetMutable<std::vector<Reduce<T>>>(pool_name);
    auto& reducer = (*reduce_pool)[channel];

    while (reducer.CheckReduce() == TaskStatus::incomplete);
    return reducer.val;
}
template<typename T>
T Reductions::CheckOnAll(MeshData<Real>* md, int channel)
{
    // Get the relevant reducer and result
    const std::string pool_name = GetPoolName<T, true>();
    auto& pars = md->GetMeshPointer()->packages.Get("Reductions")->AllParams();
    auto* reduce_pool = pars.GetMutable<std::vector<AllReduce<T>>>(pool_name);
    auto& reducer = (*reduce_pool)[channel];

    while (reducer.CheckReduce() == TaskStatus::incomplete);
    return reducer.val;
}

#define INSIDE                                                                           \
    (x[1] >= startx1 && x[2] >= startx2 && x[3] >= startx3) &&                           \
        (trivial1 ? xin[1] < startx1 : x[1] <= stopx1) &&                                \
        (trivial2 ? xin[2] < startx2 : x[2] <= stopx2) &&                                \
        (trivial3 ? xin[3] < startx3 : x[3] <= stopx3)

// TODO additionally template on return type to avoid counting flags with Reals
template<Reductions::Var var, UserHistoryOperation op, typename T>
T Reductions::DomainReduction(MeshData<Real>* md, const GReal startx[3],
    const GReal stopx[3], int channel, bool plane_outward)
{
    Flag("DomainReduction");
    auto pmesh = md->GetMeshPointer();

    const auto& pars = pmesh->packages.Get("GRMHD")->AllParams();
    const auto& emhd_params = EMHD::GetEMHDParameters(pmesh->packages);
       
    const auto& eos_params = pmesh->packages.Get("eos")->AllParams();
    auto eos = eos_params.Get<Microphysics::EOS::EOS>("d.EOS");

    // Just pass in everything we might want. Probably slow?
    PackIndexMap prims_map, cons_map;
    const auto& P = md->PackVariables(
        std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive"), Metadata::Cell},
        prims_map);
    const auto& U = md->PackVariablesAndFluxes(
        std::vector<MetadataFlag>{Metadata::Conserved, Metadata::Cell}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    const auto& cmax = md->PackVariables(std::vector<std::string>{"Flux.cmax"});
    const auto& cmin = md->PackVariables(std::vector<std::string>{"Flux.cmin"});

    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    IndexRange ib = pmb0->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb0->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb0->cellbounds.GetBoundsK(IndexDomain::interior);
    IndexRange block = IndexRange{0, U.GetDim(5) - 1};

    bool trivial_tmp[3] = {false, false, false};
    VLOOP
        trivial_tmp[v] = (startx[v] == stopx[v]);

    // Pull values to pass to device, because passing views is cumbersome
    const bool trivial1 = trivial_tmp[0];
    const bool trivial2 = trivial_tmp[1];
    const bool trivial3 = trivial_tmp[2];
    const GReal startx1 = startx[0];
    const GReal startx2 = startx[1];
    const GReal startx3 = startx[2];
    const GReal stopx1 = stopx[0];
    const GReal stopx2 = stopx[1];
    const GReal stopx3 = stopx[2];

    // TODO is 'if constexpr' faster now op is compile-time?
    T result = 0.;
    MPI_Op mop;
    switch (op) {
        case UserHistoryOperation::sum: {
            Kokkos::Sum<T> sum_reducer(result);
            // TODO these should be nested parallel loops to skip blocks fully outside
            // domain
            pmb0->par_reduce(
                "domain_sum", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                KOKKOS_LAMBDA(const int& b, const int& k, const int& j, const int& i,
                    T& local_result)
                {
                    const auto& G = U.GetCoords(b);
                    GReal x[GR_DIM], xin[GR_DIM];
                    G.coord_embed(k, j, i, Loci::center, x);
                    if (trivial1 || trivial2 || trivial3) {
                        if (plane_outward) {
                            G.coord_embed(k, j, i, Loci::center, xin);
                            G.coord_embed(k + trivial3, j + trivial2, i + trivial1,
                                Loci::center, x);
                        } else {
                            G.coord_embed(k - trivial3, j - trivial2, i - trivial1,
                                Loci::center, xin);
                        }
                    }
                    if (INSIDE) {
                        local_result += reduction_var<var>(REDUCE_FUNCTION_CALL) *
                                        ((trivial3) ? 1. : G.Dxc<3>(k)) *
                                        ((trivial2) ? 1. : G.Dxc<2>(j)) *
                                        ((trivial1) ? 1. : G.Dxc<1>(i));
                    }
                },
                sum_reducer);
            mop = MPI_SUM;
            break;
        }
        case UserHistoryOperation::max: {
            Kokkos::Max<T> max_reducer(result);
            pmb0->par_reduce(
                "domain_max", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                KOKKOS_LAMBDA(const int& b, const int& k, const int& j, const int& i,
                    T& local_result)
                {
                    const auto& G = U.GetCoords(b);
                    GReal x[GR_DIM], xin[GR_DIM];
                    G.coord_embed(k, j, i, Loci::center, x);
                    if (trivial1 || trivial2 || trivial3) {
                        if (plane_outward) {
                            G.coord_embed(k, j, i, Loci::center, xin);
                            G.coord_embed(k + trivial3, j + trivial2, i + trivial1,
                                Loci::center, x);
                        } else {
                            G.coord_embed(k - trivial3, j - trivial2, i - trivial1,
                                Loci::center, xin);
                        }
                    }
                    if (INSIDE) {
                        const Real val = reduction_var<var>(REDUCE_FUNCTION_CALL) *
                                         ((trivial3) ? 1. : G.Dxc<3>(k)) *
                                         ((trivial2) ? 1. : G.Dxc<2>(j)) *
                                         ((trivial1) ? 1. : G.Dxc<1>(i));
                        if (val > local_result) local_result = val;
                    }
                },
                max_reducer);
            mop = MPI_MAX;
            break;
        }
        case UserHistoryOperation::min: {
            Kokkos::Min<T> min_reducer(result);
            pmb0->par_reduce(
                "domain_min", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                KOKKOS_LAMBDA(const int& b, const int& k, const int& j, const int& i,
                    T& local_result)
                {
                    const auto& G = U.GetCoords(b);
                    GReal x[GR_DIM], xin[GR_DIM];
                    G.coord_embed(k, j, i, Loci::center, x);
                    if (trivial1 || trivial2 || trivial3) {
                        if (plane_outward) {
                            G.coord_embed(k, j, i, Loci::center, xin);
                            G.coord_embed(k + trivial3, j + trivial2, i + trivial1,
                                Loci::center, x);
                        } else {
                            G.coord_embed(k - trivial3, j - trivial2, i - trivial1,
                                Loci::center, xin);
                        }
                    }
                    if (INSIDE) {
                        const Real val = reduction_var<var>(REDUCE_FUNCTION_CALL) *
                                         ((trivial3) ? 1. : G.Dxc<3>(k)) *
                                         ((trivial2) ? 1. : G.Dxc<2>(j)) *
                                         ((trivial1) ? 1. : G.Dxc<1>(i));
                        if (val < local_result) local_result = val;
                    }
                },
                min_reducer);
            mop = MPI_MIN;
            break;
        }
    }

    // Optionally start an MPI reducer w/given index, so the mesh-wide result is ready
    // when we want it
    if (channel >= 0) {
        Start<T>(md, channel, result, mop);
    }

    // fprintf(stderr, "r: %f trivial: %d %d %d result: %f\n", startx1, trivial1,
    // trivial2, trivial3, result);

    EndFlag();
    return result;
}

#undef INSIDE
#undef REDUCE_FUNCTION_CALL
#undef REDUCE_FUNCTION_ARGS
