/* 
 *  File: one_block_transmit.cpp
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

#include "one_block_transmit.hpp"

#include "domain.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

using namespace parthenon;

void KBoundaries::TransmitImpl(MeshBlockData<Real> *rc, BoundaryFace bface, bool coarse)
{
    // Get all cell-centered ghosts, minus anything just used at startup
    using FC = Metadata::FlagCollection;
    FC ghost_vars = FC({Metadata::FillGhost, Metadata::Cell})
                  - FC({Metadata::GetUserFlag("StartupOnly")});
    PackIndexMap bounds_map;
    auto q = rc->PackVariables(ghost_vars, bounds_map, coarse);
    TransmitSetTE(rc, q, bface, bounds_map, coarse, false);

    FC ghost_vars_f = FC({Metadata::FillGhost, Metadata::Face})
                    - FC({Metadata::GetUserFlag("StartupOnly")});
    auto q_f = rc->PackVariables(ghost_vars_f, bounds_map, coarse);
    TransmitSetTE(rc, q_f, bface, bounds_map, coarse, true);
}

void KBoundaries::TransmitSetTE(MeshBlockData<Real> *rc, VariablePack<Real> &q, BoundaryFace bface,
                                PackIndexMap &bounds_map, bool coarse, bool do_face)
{
    // We're sometimes called without any variables to sync (e.g. syncing flags, EMFs), just return
    if (q.GetDim(4) == 0) return;

    // Pull boundary properties
    auto pmb = rc->GetBlockPointer();
    const bool binner = BoundaryIsInner(bface);
    const int bdir = BoundaryDirection(bface);
    const auto domain = BoundaryDomain(bface);
    const auto bname = BoundaryName(bface);

    if (bdir != 2)
        throw std::runtime_error("Transmitting polar conditions only defined for X2!");

    std::vector<TopologicalElement> el_list;
    if (do_face) {
        el_list = {F1, F2, F3};
    } else {
        el_list = {CC};
    }
    for (TopologicalElement &el : el_list) {
        // This automatically includes zones on domain faces, which we set to 0 below
        const IndexRange3 b = KDomain::GetBoundaryRange(rc, domain, el, coarse);
        const IndexRange3 bi = KDomain::GetRange(rc, IndexDomain::interior, CC, coarse);
        // Whether we're on e.g. X2 face for X2 boundary
        const bool corresponding_face = (el == FaceOf(bdir));
        const int reflect_offset = corresponding_face ? 0 : (binner ? 1 : -1);

        // TODO SPECIFIC TO X2
        // Total physical/interior *zones* in dir 3.  Note first/last face will share an opposite,
        // i.e. both F3(0,x,x) and F3(N3,x,x) -> F3(N3/2,x,x), but in reverse only the former
        const int Nk3p = (bi.ke - bi.ks + 1);
        const int Nk3p2 = Nk3p/2;
        const int ksp = bi.ks;
        // Calculate j just like for reflecting conditions
        const int jpivot = (binner) ? b.je : b.js;
        // B3 component on X3 face should be inverted even if not marked "vector"
        // TODO honor SplitVector and vector components here rather than hard-coding
        const bool do_face_invert = (el == F3 || el == F2);
        // TODO figure out fixing '.vector_component' in Parthenon
        // Subtracting from 'second' ensures -1 returns remain negative
        const int x2_index_1 = bounds_map["cons.uvec"].second-1;
        const int x2_index_2 = bounds_map["prims.uvec"].second-1;
        const int x2_index_3 = bounds_map["cons.B"].second-1;
        const int x2_index_4 = bounds_map["prims.B"].second-1;
        const int x3_index_1 = bounds_map["cons.uvec"].second;
        const int x3_index_2 = bounds_map["prims.uvec"].second;
        const int x3_index_3 = bounds_map["cons.B"].second;
        const int x3_index_4 = bounds_map["prims.B"].second;

        //std::cerr << "Transmitting boundary. el=" << static_cast<int>(el) << " do_invert=" << do_face_invert << " corr_face=" << corresponding_face << std::endl;
        //std::cerr << "Elements: " << q.GetDim(4) << " vector components: ";
        //pmb->par_for(
        //    "print_" + bname, 0, q.GetDim(4)-1,
        //    KOKKOS_LAMBDA (const int &v) {
        //        printf("%d ", q(el, v).vector_component);
        //    }
        //);
        //std::cerr << std::endl;

        pmb->par_for(
            "transmitting_polar_boundary_" + bname, 0, q.GetDim(4)-1, b.ks, b.ke, b.js, b.je, b.is, b.ie,
            KOKKOS_LAMBDA (const int &v, const int &k, const int &j, const int &i) {
                const int ki = ((k - ksp + Nk3p2) % Nk3p) + ksp;
                const int ji = jpivot + reflect_offset + (jpivot - j);
                const int ii = i;
                const Real invert = (do_face_invert ||
                                    v == x2_index_1 || v == x2_index_2 ||
                                    v == x2_index_3 || v == x2_index_4 ||
                                    v == x3_index_1 || v == x3_index_2 ||
                                    v == x3_index_3 || v == x3_index_4 ||
                                    q(el, v).vector_component == X2DIR ||
                                    q(el, v).vector_component == X3DIR) ? -1. : 1.;
                // if (i == 10 && j == 3 && k == 10)
                //    printf("Set el %d v %d zone %d %d %d from %d %d %d invert: %f\n", (int) el, v, k, j, i, ki, ji, ii, invert);
                q(el, v, k, j, i) = (corresponding_face && j == jpivot) ? 0. : invert * q(el, v, ki, ji, ii);
            }
        );
    }
}
