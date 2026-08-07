/*
 *  File: flux.hpp
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

#include "decs.hpp"

#include <parthenon/parthenon.hpp>

#include "floors.hpp"
#include "flux_functions.hpp"
#include "pack.hpp"
#include "reconstruction.hpp"
#include "types.hpp"

namespace Flux
{

std::shared_ptr<KHARMAPackage> Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages);

TaskStatus CheckCtop(MeshData<Real>* md);

TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real>* md);

TaskStatus MarkFOFC(MeshData<Real>* md);

/**
 * Given a "guess" in which the fflag reflects zones which would fail with current fluxes,
 * replace fluxes surrounding flagged zones with donor-cell/first-order versions
 */
TaskStatus FOFC(MeshData<Real>* md, MeshData<Real>* guess);

/**
 * Add the geometric source term present in the covariant derivative of the stress-energy
 * tensor, S_nu = sqrt(-g) T^kap_lam Gamma^lam_nu_kap This is defined in Flux:: rather
 * than GRMHD:: because the stress-energy tensor may contain (E)GR(R)(M)HD terms.
 */
void AddGeoSource(MeshData<Real>* md, MeshData<Real>* mdudt, IndexDomain domain);
// Version returning TaskStatus, for calling alone in FOFC "update"
// TODO(BSP) switch KHARMAPackage (and Parthenon packages?) to expect all TaskStatus
inline TaskStatus AddGeoSourceTask(
    MeshData<Real>* md, MeshData<Real>* mdudt, IndexDomain domain)
{
    AddGeoSource(md, mdudt, domain);
    return TaskStatus::complete;
}

/**
 * Likewise, the conversion P->U, even for just the GRMHD variables, requires (consists
 * of) the stress-energy tensor.
 */
TaskStatus BlockPtoUMHD(MeshBlockData<Real>* rc, IndexDomain domain, bool coarse = false);

/**
 * When calculating fluxes, we use Flux::prim_to_flux, which must generate conserved
 * variables and fluxes for all loaded packages correctly. These calls just run that
 * function over the grid.
 */
TaskStatus BlockPtoU(MeshBlockData<Real>* rc, IndexDomain domain, bool coarse = false);
TaskStatus MeshPtoU(MeshData<Real>* md, IndexDomain domain, bool coarse = false);

/**
 * As above, except that IndexDomains of ghost cells are taken to cover
 * cells *sent* (that is, part of the domain) rather than received
 *
 * UNUSED
 */
TaskStatus BlockPtoU_Send(MeshBlockData<Real>* rc, IndexDomain domain, bool coarse);

/**
 * Count how many zones forced first-order fluxes (*not* the number of fluxes reduced)
 */
int CountFOFCFlags(MeshData<Real>* md);

// Fluxes a.k.a. "Approximate Riemann Solvers"
// More complex solvers require speed estimates not calculable completely from
// invariants, necessitating frame transformations and related madness.
// These have identical signatures, so that we could runtime relink w/variant like
// coordinate_embedding

// Local Lax-Friedrichs flux (usual, more stable)
KOKKOS_INLINE_FUNCTION Real llf(const Real& fluxL, const Real& fluxR, const Real& cmax,
    const Real& cmin, const Real& Ul, const Real& Ur)
{
    Real ctop = m::max(cmax, cmin);
    return 0.5 * (fluxL + fluxR - ctop * (Ur - Ul));
}
// Harten, Lax, van Leer, & Einfeldt flux (early problems but not extensively studied
// since)
KOKKOS_INLINE_FUNCTION Real hlle(const Real& fluxL, const Real& fluxR, const Real& cmax,
    const Real& cmin, const Real& Ul, const Real& Ur)
{
    return (cmax * fluxL + cmin * fluxR - cmax * cmin * (Ur - Ul)) / (cmax + cmin);
}

}
