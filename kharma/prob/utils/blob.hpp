/*
 *  File: blob.hpp
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

// Insert blobs (hotspots) of various types
// Spherical coordinates only

#include "decs.hpp"

#include "grmhd_functions.hpp"
#include "pack.hpp"

#include <parthenon/parthenon.hpp>

/**
 * Insert a blob of material at a given temperature,
 * by scaling the density by a factor, then setting the temperature
 * constant.
 *
 * TODO MeshData
 */
void InsertBlob(MeshBlockData<Real>* rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();
    PackIndexMap prims_map;
    auto P = GRMHD::PackMHDPrims(rc, prims_map);
    const VarMap m_p(prims_map, false);

    GRCoordinates G = pmb->coords;
    Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    Real desired_sigma = pin->GetOrAddReal("blob", "desired_sigma", 1.1);
    Real u_over_rho = pin->GetOrAddReal("blob", "u_over_rho", 50.0);

    // One buffer zonkharma-re of linear decline, r_in -> r_out
    // Exponential decline inside here i.e. linear in logspace
    GReal sz_in = pin->GetOrAddReal("blob", "sz_in", 4.5);
    GReal sz_out = pin->GetOrAddReal("blob", "sz_out", 5.0);
    // Blob center
    GReal blob_r = pin->GetOrAddReal("blob", "r", 15.0);
    GReal blob_th = pin->GetOrAddReal("blob", "th", M_PI / 8);
    GReal blob_phi = pin->GetOrAddReal("blob", "phi", 0.0);

    IndexDomain domain = IndexDomain::interior;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
    pmb->par_for("insert_blob", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
        {
            Real X[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, X);
            Real d = m::sqrt(blob_r * blob_r + X[1] * X[1] -
                             2 * blob_r * X[1] *
                                 (sin(blob_th) * sin(X[2]) * cos(blob_phi - X[3]) +
                                     cos(blob_th) * cos(X[2])));

            if (d <= sz_out) {
                FourVectors Dtmp;
                GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
                Real bsq = dot(Dtmp.bcon, Dtmp.bcov);
                Real rho_factor = bsq / desired_sigma;

                if (d < sz_in) {
                    P(m_p.RHO, k, j, i) *= rho_factor;
                    P(m_p.UU, k, j, i) = u_over_rho * P(m_p.RHO, k, j, i);
                } else if (d >= sz_in) {
                    Real ramp = (sz_out - d) / (sz_out - sz_in);

                    // P(m_p.RHO, k, j, i) = rho_out + ramp * (rho_in - rho_out);
                    Real lrho_factor_in = log(rho_factor);
                    P(m_p.RHO, k, j, i) *= m::exp(ramp * lrho_factor_in);

                    P(m_p.UU, k, j, i) = u_over_rho * P(m_p.RHO, k, j, i);
                }
            }
        });
}
