/*
 *  File: file.cpp
 *
 *  BSD 3-Clause License
 *
 *  Copyright (c) 2025, Iris contributors
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
#include "file.hpp"

// Iris headers
#include "rays.hpp"
#include "thin_disk.hpp"
#include "units.hpp"
// KHARMA headers
#include "coordinate_embedding.hpp"
#include "domain.hpp"
// Parthenon headers
#include <parthenon/parthenon.hpp>

std::shared_ptr<StateDescriptor> File::Initialize(ParameterInput *pin)
{
    auto pkg = std::make_shared<StateDescriptor>("File");
    Params &params = pkg->AllParams();

    // TODO separate file loading, so we can regrid/slow light/etc
    std::string fname = pin->GetString("model", "file");
    params.Add("fname", fname);

    // Read prims directly: mark them "Restart" to read them when passing -r to Parthenon
    Metadata::AddUserFlag("Primitive");
    auto m = Metadata({Metadata::Cell, Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::GetUserFlag("Primitive"), Metadata::Restart, Metadata::FillGhost});
    pkg->AddField("prims.rho", m);
    pkg->AddField("prims.u", m);
    std::vector<int> s_vector({NVEC});
    m = Metadata({Metadata::Cell, Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Vector, Metadata::GetUserFlag("Primitive"), Metadata::Restart, Metadata::FillGhost}, s_vector);
    pkg->AddField("prims.uvec", m);

    // Try to read either cell- or face-centered fields
    m = Metadata({Metadata::Cell, Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Vector, Metadata::Conserved, Metadata::Restart}, s_vector);
    pkg->AddField("cons.B", m);
    m = Metadata({Metadata::Face, Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Conserved, Metadata::Restart});
    pkg->AddField("cons.fB", m);
    // We'll fill this with UtoP when initializing.  Note we only need/sync prims.B
    m = Metadata({Metadata::Cell, Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Vector, Metadata::GetUserFlag("Primitive"), Metadata::FillGhost}, s_vector);
    pkg->AddField("prims.B", m);

    return pkg;
}

TaskStatus File::InitMeshBlock(MeshBlock* pmb)
{
    auto rc = pmb->meshblock_data.Get("base");
    const int ndim = pmb->pmy_mesh->ndim;
    auto B_Uf = rc->PackVariables(std::vector<std::string>{"cons.fB"});
    auto B_U = rc->PackVariables(std::vector<std::string>{"cons.B"});
    auto B_P = rc->PackVariables(std::vector<std::string>{"prims.B"});

    const IndexRange3 bc = KDomain::GetRange(rc, IndexDomain::entire);
    const auto& G = pmb->coords;

    // This could be made faster but it's initialization code
    double maxfB;
    pmb->par_reduce("Max_Bf", bc.ks, bc.ke, bc.js, bc.je, bc.is, bc.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i, Real &local_result) {
            if (B_Uf(F1, 0, k, j, i) > local_result) local_result = B_Uf(F1, 0, k, j, i);
            if (B_Uf(F2, 0, k, j, i) > local_result) local_result = B_Uf(F2, 0, k, j, i);
            if (B_Uf(F3, 0, k, j, i) > local_result) local_result = B_Uf(F3, 0, k, j, i);
        }
    , Kokkos::Max<Real>(maxfB));

    if (maxfB > 0.) {
        // Average the primitive vals to zone centers
        pmb->par_for("UtoP_B_center", bc.ks, bc.ke, bc.js, bc.je, bc.is, bc.ie,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                B_P(V1, k, j, i) = (B_Uf(F1, 0, k, j, i) / G.gdet(Loci::face1, j, i)
                                + B_Uf(F1, 0, k, j, i + 1) / G.gdet(Loci::face1, j, i + 1)) / 2;
                B_P(V2, k, j, i) = (ndim > 1) ? (B_Uf(F2, 0, k, j, i) / G.gdet(Loci::face2, j, i)
                                            + B_Uf(F2, 0, k, j + 1, i) / G.gdet(Loci::face2, j + 1, i)) / 2
                                            : B_Uf(F2, 0, k, j, i) / G.gdet(Loci::face2, j, i);
                B_P(V3, k, j, i) = (ndim > 2) ? (B_Uf(F3, 0, k, j, i) / G.gdet(Loci::face3, j, i)
                                            + B_Uf(F3, 0, k + 1, j, i) / G.gdet(Loci::face3, j, i)) / 2
                                            : B_Uf(F3, 0, k, j, i) / G.gdet(Loci::face3, j, i);
            }
        );
        // Recover conserved B at centers
        pmb->par_for("UtoP_B_centerPtoU", 0, NVEC-1, bc.ks, bc.ke, bc.js, bc.je, bc.is, bc.ie,
            KOKKOS_LAMBDA (const int &v, const int &k, const int &j, const int &i) {
                B_U(v, k, j, i) = B_P(v, k, j, i) * G.gdet(Loci::center, j, i);
            }
        );
    } else {
        // Recover primitive B at centers
        pmb->par_for("UtoP_B_centerPtoU", 0, NVEC-1, bc.ks, bc.ke, bc.js, bc.je, bc.is, bc.ie,
            KOKKOS_LAMBDA (const int &v, const int &k, const int &j, const int &i) {
                B_P(v, k, j, i) = B_U(v, k, j, i) / G.gdet(Loci::center, j, i);
            }
        );
    }

    return TaskStatus::complete;
}
