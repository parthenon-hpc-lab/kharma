/* 
 *  File: b_ct.hpp
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

#include "b_ct_functions.hpp"
#include "decs.hpp"
#include "kharma_driver.hpp"
#include "reductions.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

/**
 * This physics package implements Constrained Transport of a split face-centered B field.
 * Any CT implementations should probably go here.
 */
namespace B_CT {
/**
 * Declare fields, initialize (few) parameters
 */
std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

/**
 * Get the primitive variables, which in Parthenon's nomenclature are "derived"
 * 
 * Defaults to entire domain, as the KHARMA algorithm relies on applying UtoP over ghost zones.
 * 
 * input: Conserved B = sqrt(-g) * B^i
 * output: Primitive B = B^i
 */
TaskStatus BlockUtoP(MeshBlockData<Real> *mbd, IndexDomain domain, bool coarse=false);
TaskStatus MeshUtoP(MeshData<Real> *md, IndexDomain domain, bool coarse=false);

/**
 * Interpolate primitive B to faces, then multiply by gdet
 * DANGEROUS as it replaces conserved field state. Only used for resizing runs,
 * and only with B_Cleanup::CleanupDivergence directly afterward!
 */
TaskStatus DangerousPtoU(MeshData<Real> *md, IndexDomain domain, bool coarse=false);

/**
 * Calculate the EMF around edges of faces caused by the flux of B field
 * through each face.
 */
TaskStatus CalculateEMF(MeshData<Real> *md);

/**
 * Calculate the change in magnetic field on faces for this step,
 * from the EMFs at edges.
 */
TaskStatus AddSource(MeshData<Real> *md, MeshData<Real> *mdudt, IndexDomain domain);

/**
 * Derefine face-centered B at poles without introducing divergence
 */
TaskStatus DerefinePoles(MeshData<Real> *md);

/**
 * Calculate maximum corner-centered divergence of magnetic field,
 * to check it is being preserved ~=0
 * Used as a Parthenon History function, so must take exactly the
 * listed arguments
 */
double MaxDivB(MeshData<Real> *md);
double BlockMaxDivB(MeshBlockData<Real> *rc);

/**
 * Returns the global maximum value, rather than the maximum over this rank's MeshData
 */
double GlobalMaxDivB(MeshData<Real> *md, bool all_reduce=false);

/**
 * Diagnostics printed/computed after each step
 * Currently just max divB
 */
TaskStatus PrintGlobalMaxDivB(MeshData<Real> *md, bool kill_on_large_divb=false);

/**
 * Diagnostics function should print divB, and optionally stop execution if it's large
 */
inline TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md)
{
    auto& params = md->GetMeshPointer()->block_list[0]->packages.Get("B_CT")->AllParams();
    return PrintGlobalMaxDivB(md, params.Get<bool>("kill_on_large_divb"));
}

/**
 * Fill fields which are calculated only for output to file, i.e., divB
 */
void FillOutput(MeshBlock *pmb, ParameterInput *pin);
/**
 * Fill field "name" with divB
 */
void CalcDivB(MeshData<Real> *md, std::string divb_field_name="divB");

// BOUNDARY FUNCTIONS
// Maintaining zero divergence for face fields on boundaries takes some extra work

/**
 * Don't allow EMF inside of a boundary, effectively making it a superconducting surface*
 * Used for Dirichlet and reflecting conditions.
 * 
 * *mostly. I think
 */
void ZeroBoundaryEMF(MeshBlockData<Real> *rc, IndexDomain domain, const VariablePack<Real> &emfpack, bool coarse);

/**
 * Average all EMFs corresponding to the coordinate pole location, e.g. usually all E1 on X2 faces
 */
void AverageBoundaryEMF(MeshBlockData<Real> *rc, IndexDomain domain, const VariablePack<Real> &emfpack, bool coarse);

/**
 * Subtract the average B3 from each face, as if a loop reconnected across the polar boundary.
 * Preserves divB, since differences across cells remain the same after subtracting a constant.
 */
void ReconnectBoundaryB3(MeshBlockData<Real> *rc, IndexDomain domain, const VariablePack<Real> &emfpack, bool coarse);

/**
 * Reset an outflow condition to have no divergence, even if a field line exits the domain.
 * Could maybe be used on other boundaries, but resets the perpendicular face so use with caution.
 */
void DestructiveBoundaryClean(MeshBlockData<Real> *rc, IndexDomain domain, const VariablePack<Real> &fpack, bool coarse);

/**
 * Take the curl over the whole domain. Defined in-header since it's templated on face and NDIM
 */
template<TE el, int NDIM>
inline void EdgeCurl(MeshBlockData<Real> *rc, const GridVector& A,
                                     const VariablePack<Real>& B_U, IndexDomain domain)
{
    auto pmb = rc->GetBlockPointer();
    const auto &G = pmb->coords;
    IndexRange3 bB = KDomain::GetRange(rc, domain, el);
    pmb->par_for(
        "EdgeCurl", bB.ks, bB.ke, bB.js, bB.je, bB.is, bB.ie,
        KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
            B_CT::edge_curl<el, NDIM>(G, A, B_U, k, j, i);
        }
    );
}

}
