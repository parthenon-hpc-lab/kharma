/* 
 *  File: reductions.cpp
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

#include "reductions.hpp"

#include <parthenon/parthenon.hpp>

// TODO none of this machinery preserves zone locations,
// which we pretty often would like...

std::shared_ptr<KHARMAPackage> Reductions::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Reductions");
    Params &params = pkg->AllParams();

    // These pools are vectors of Reducers which operate on vectors (or scalars)
    // They exist to allow several reductions to be in-flight at once to hide latency
    // (even reductions over vectors, as with the different flags)
    std::vector<Reduce<std::vector<int>>> vector_int_reduce_pool;
    params.Add("vector_int_reduce_pool", vector_int_reduce_pool, true);
    std::vector<Reduce<std::vector<Real>>> vector_reduce_pool;
    params.Add("vector_reduce_pool", vector_reduce_pool, true);
    std::vector<Reduce<int>> int_reduce_pool;
    params.Add("int_reduce_pool", int_reduce_pool, true);
    std::vector<Reduce<Real>> reduce_pool;
    params.Add("reduce_pool", reduce_pool, true);

    std::vector<AllReduce<std::vector<int>>> vector_int_allreduce_pool;
    params.Add("vector_int_allreduce_pool", vector_int_allreduce_pool, true);
    std::vector<AllReduce<std::vector<Real>>> vector_allreduce_pool;
    params.Add("vector_allreduce_pool", vector_allreduce_pool, true);
    std::vector<AllReduce<int>> int_allreduce_pool;
    params.Add("int_allreduce_pool", int_allreduce_pool, true);
    std::vector<AllReduce<Real>> allreduce_pool;
    params.Add("allreduce_pool", allreduce_pool, true);

    // Reductions sometimes need global elements of the simulation we don't otherwise keep
    if (pin->GetReal("coordinates", "spherical")) {
        params.Add("domain_r_in", (GReal) pin->GetReal("coordinates", "r_in"));
        params.Add("domain_r_eh", (GReal) pin->GetReal("coordinates", "r_eh"));
        params.Add("domain_r_out", (GReal) pin->GetReal("coordinates", "r_out"));
    } else {
        params.Add("domain_r_in", 0.);
        params.Add("domain_r_eh", 0.);
        params.Add("domain_r_out", 0.);
    }

    return pkg;
}

// Flag reductions: local
int Reductions::CountFlag(MeshData<Real> *md, std::string field_name, const int& flag_val, IndexDomain domain, bool is_bitflag)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Pack variables
    std::vector<std::string> flag_vec = {field_name};
    auto& flag = md->PackVariables(flag_vec);

    // Get sizes
    IndexRange ib = md->GetBoundsI(domain);
    IndexRange jb = md->GetBoundsJ(domain);
    IndexRange kb = md->GetBoundsK(domain);
    IndexRange block = IndexRange{0, flag.GetDim(5) - 1};

    int n_flag;
    Kokkos::Sum<int> flag_ct(n_flag);
    pmb0->par_reduce("count_flag", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, int &local_result) {
            if ((is_bitflag && static_cast<int>(flag(b, 0, k, j, i)) & flag_val) ||
                (!is_bitflag && static_cast<int>(flag(b, 0, k, j, i)) == flag_val))
                ++local_result;
        }
    , flag_ct);
    return n_flag;
}

#define MAX_NFLAGS 20

std::vector<int> Reductions::CountFlags(MeshData<Real> *md, std::string field_name, const std::map<int, std::string> &flag_values, IndexDomain domain, bool is_bitflag)
{
    Flag("CountFlags_"+field_name);
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // Pack variables
    std::vector<std::string> flag_vec = {field_name};
    auto& flag = md->PackVariables(flag_vec);

    // Get sizes
    IndexRange ib = md->GetBoundsI(domain);
    IndexRange jb = md->GetBoundsJ(domain);
    IndexRange kb = md->GetBoundsK(domain);
    IndexRange block = IndexRange{0, flag.GetDim(5) - 1};

    // Man, moving arrays is clunky.  Oh well.
    const int n_of_flags = flag_values.size();
    ParArray1D<int> flag_val_list("flag_values", MAX_NFLAGS);
    auto flag_val_list_h = flag_val_list.GetHostMirror();
    int f=1;
    for (auto &flag : flag_values) {
        flag_val_list_h[f] = flag.first;
        f++;
    }
    flag_val_list.DeepCopy(flag_val_list_h);
    Kokkos::fence();

    // Count all nonzero (technically, >0) values,
    // and all values which match each flag.
    // This works for pflags or fflags, so long as they're separate
    // We don't count negative pflags as they denote zones that shouldn't be fixed
    Reductions::array_type<int, MAX_NFLAGS> flag_reducer;
    pmb0->par_reduce("count_flags", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, 
                       Reductions::array_type<int, MAX_NFLAGS> &local_result) {
            const int flag_int = static_cast<int>(flag(b, 0, k, j, i));
            // First element is total count
            if (flag_int > 0) ++local_result.my_array[0];
            // The rest of the list is individual flags
            for (int f=1; f <= n_of_flags; f++)
                if ((is_bitflag && flag_int & flag_val_list(f)) ||
                    (!is_bitflag && flag_int == flag_val_list(f)))
                    ++local_result.my_array[f];
        }
    , Reductions::ArraySum<int, HostExecSpace, MAX_NFLAGS>(flag_reducer));

    std::vector<int> n_each_flag;
    for (int f=0; f < n_of_flags+1; f++)
        n_each_flag.push_back(flag_reducer.my_array[f]);
    
    EndFlag();
    return n_each_flag;
}

// Flag reductions: global
void Reductions::StartFlagReduce(MeshData<Real> *md, std::string field_name, const std::map<int, std::string> &flag_values, IndexDomain domain, bool is_bitflag, int channel)
{
    Start<std::vector<int>>(md, channel, CountFlags(md, field_name, flag_values, domain, is_bitflag), MPI_SUM);
}

std::vector<int> Reductions::CheckFlagReduceAndPrintHits(MeshData<Real> *md, std::string field_name, const std::map<int, std::string> &flag_values,
                                                     IndexDomain domain, bool is_bitflag, int channel)
{
    Flag("CheckFlagReduce");
    const auto& pmesh = md->GetMeshPointer();
    const auto& verbose = pmesh->packages.Get("Globals")->Param<int>("flag_verbose");

    // Get the relevant reducer and result
    auto& pars = md->GetMeshPointer()->packages.Get("Reductions")->AllParams();
    auto *vector_int_reduce_pool = pars.GetMutable<std::vector<Reduce<std::vector<int>>>>("vector_int_reduce_pool");
    auto& vector_int_reduce = (*vector_int_reduce_pool)[channel];

    while (vector_int_reduce.CheckReduce() == TaskStatus::incomplete);
    const std::vector<int> &total_flag_counts = vector_int_reduce.val;

    // Print flags 
    if (total_flag_counts[0] > 0 && verbose > 0) {
        if (MPIRank0()) {
            // Always our domain size times total number of blocks
            IndexRange ib = md->GetBoundsI(domain);
            IndexRange jb = md->GetBoundsJ(domain);
            IndexRange kb = md->GetBoundsK(domain);
            int n_cells = pmesh->nbtotal * (kb.e - kb.s + 1) * (jb.e - jb.s + 1) * (ib.e - ib.s + 1);

            int nflags = total_flag_counts[0];
            std::cout << field_name << ": " << nflags << " (" << (int)(((double) nflags )/n_cells * 100) << "% of all cells)" << std::endl;
            if (verbose > 1) {
                // Print nonzero vector contents against flag names in order
                int i = 1;
                for (auto& status : flag_values) {
                    if (total_flag_counts[i] > 0) std::cout << status.second << ": " << total_flag_counts[i] << std::endl;
                    ++i;
                }
                std::cout << std::endl;
            }
        }
    }

    EndFlag();
    return total_flag_counts;
}
