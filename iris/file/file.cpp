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
// Parthenon headers
#include <parthenon/parthenon.hpp>

std::shared_ptr<StateDescriptor> File::Initialize(ParameterInput *pin)
{
    auto pkg = std::make_shared<StateDescriptor>("File");
    Params &params = pkg->AllParams();

    // TODO separate file loading, so we can regrid/slow light/etc
    std::string fname = pin->GetString("model", "file");
    params.Add("fname", fname);

    // Read prims directly
    Metadata::AddUserFlag("Primitive");
    auto m = Metadata({Metadata::Cell, Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::GetUserFlag("Primitive"), Metadata::Restart});
    pkg->AddField("prims.rho", m);
    pkg->AddField("prims.u", m);
    std::vector<int> s_vector({NVEC});
    m = Metadata({Metadata::Cell, Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Vector, Metadata::GetUserFlag("Primitive"), Metadata::Restart}, s_vector);
    pkg->AddField("prims.uvec", m);

    // Try to read either cell- or face-centered fields
    m = Metadata({Metadata::Cell, Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Vector, Metadata::Conserved, Metadata::Restart}, s_vector);
    pkg->AddField("cons.B", m);
    m = Metadata({Metadata::Face, Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Conserved, Metadata::Restart});
    pkg->AddField("cons.fB", m);
    // We'll fill this with UtoP
    m = Metadata({Metadata::Cell, Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Vector, Metadata::GetUserFlag("Primitive")}, s_vector);
    pkg->AddField("prims.B", m);

    return pkg;
}

TaskStatus File::InitMeshBlock(MeshBlock *pmb)
{
    auto rc = pmb->meshblock_data.Get();
    auto B_U = rc->PackVariables(std::vector<std::string>{"cons.B"});
    auto B_P = rc->PackVariables(std::vector<std::string>{"prims.B"});

    const auto& G = pmb->coords;

    auto bounds = pmb->cellbounds;
    const IndexRange ib = bounds.GetBoundsI(IndexDomain::entire);
    const IndexRange jb = bounds.GetBoundsJ(IndexDomain::entire);
    const IndexRange kb = bounds.GetBoundsK(IndexDomain::entire);
    const IndexRange vec = IndexRange({0, B_U.GetDim(4)-1});

    pmb->par_for("UtoP_B_FluxCT_Block", vec.s, vec.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &mu, const int &k, const int &j, const int &i) {
            // Update the primitive B-fields
            B_P(mu, k, j, i) = B_U(mu, k, j, i) / G.gdet(Loci::center, j, i);
        }
    );

    return TaskStatus::complete;
}
