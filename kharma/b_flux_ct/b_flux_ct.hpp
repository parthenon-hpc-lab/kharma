/*
 *  File: b_flux_ct.hpp
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

#include "b_flux_ct_functions.hpp"
#include "decs.hpp"
#include "grmhd_functions.hpp"
#include "reductions.hpp"
#include "types.hpp"

#include <memory>

/**
 *
 * This physics package implements B field transport with Flux-CT (Toth 2000)
 *
 * This requires only the magnetic field value at cell centers
 *
 * This implementation includes conversion from "primitive" to "conserved" B and back
 */
namespace B_FluxCT
{
/**
 * Declare fields, initialize parameters
 */
std::shared_ptr<KHARMAPackage> Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages);

/**
 * Get the primitive variables, which in Parthenon's nomenclature are "derived".
 * Also applies floors to the calculated primitives, and fixes up any inversion errors
 *
 * Defaults to entire domain, as the KHARMA algorithm relies on applying UtoP over ghost
 * zones.
 *
 * input: Conserved B = sqrt(-gdet) * B^i
 * output: Primitive B = B^i
 */
TaskStatus BlockUtoP(MeshBlockData<Real>* md, IndexDomain domain, bool coarse = false);
TaskStatus MeshUtoP(MeshData<Real>* md, IndexDomain domain, bool coarse = false);

/**
 * Reverse of the above.  Only used alone during initialization.
 * Generally, use Flux::BlockPtoU/Flux::MeshPtoU
 */
void BlockPtoU(MeshBlockData<Real>* md, IndexDomain domain, bool coarse = false);
void MeshPtoU(MeshData<Real>* md, IndexDomain domain, bool coarse = false);

/**
 * Apply all flux corrections required by this package to ensure small divB
 */
void FixFlux(MeshData<Real>* md);

/**
 * Modify the B field fluxes to take a constrained-transport step as in Toth (2000)
 */
void FluxCT(MeshData<Real>* md);

/**
 * Modify the B field fluxes just beyond the polar (or radial) boundary so as to
 * ensure no flux through the boundary after applying FluxCT
 */
void ZeroBoundaryFlux(MeshData<Real>* md, IndexDomain domain, bool coarse);

/**
 * Modify the B field fluxes just beyond the radial (or polar) boundary so as to
 * ensure the magnetic divergence is zero, even
 */
void Bflux0(MeshData<Real>* md, IndexDomain domain, bool coarse);

/**
 * Alternate B field fix for X1 boundary, keeps zero divergence while permitting flux
 * through the boundary, at the cost of a short non-local solve.
 */
TaskStatus FixX1Flux(MeshData<Real>* md);

/**
 * Calculate maximum corner-centered divergence of magnetic field,
 * to check it is being preserved ~=0
 * Used as a Parthenon History function, so must take exactly the
 * listed arguments
 */
double MaxDivB(MeshData<Real>* md);

/**
 * Returns the global maximum value, rather than the maximum over this rank's MeshData
 *
 * By default, only returns the correct value on rank 0 for printing
 */
double GlobalMaxDivB(MeshData<Real>* md, bool all_reduce = false);

/**
 * Diagnostics printed/computed after each step
 * Currently just max divB
 */
TaskStatus PrintGlobalMaxDivB(MeshData<Real>* md, bool kill_on_large_divb = false);

/**
 * Diagnostics function should print divB, and optionally stop execution if it's large
 */
inline TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real>* md)
{
    auto& params = md->GetMeshPointer()->packages.Get("B_FluxCT")->AllParams();
    return PrintGlobalMaxDivB(md, params.Get<bool>("kill_on_large_divb"));
}

/**
 * Fill fields which are calculated only for output to file, i.e., divB
 */
void FillOutput(MeshBlock* pmb, ParameterInput* pin);
/**
 * Fill the field 'divb_field_name' with divB
 */
void CalcDivB(MeshData<Real>* md, std::string divb_field_name = "divB");

}
