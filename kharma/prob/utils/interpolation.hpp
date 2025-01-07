/* 
 *  File: interpolation.hpp
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

/**
 * Routines for interpolating on a grid, using values given in a flattened array.
 * Mostly used in resize_restart.cpp, which must interpolate from old simulation
 * data.
 * 
 * Note that resizing or resampling of magnetic fields usually requires
 * fixing a resulting divergence -- see b_cleanup/ for details.
 */
namespace Interpolation {

/**
 * Finds the closest grid zone index (i,j,k) with a center left of the given point.
 * Additionally returns the point's proportional distance measured from the left
 * zone center to the right (e.g., to (i+1, j, k) in X1) 
 * 
 * This proportion is useful in interpolation, since linear interpolation corresponds to
 * del*var[i+1] + (1. - del)*var[i]
 */
KOKKOS_INLINE_FUNCTION void Xtoijk(const GReal X[GR_DIM],
                                   const GReal startx[GR_DIM],
                                   const GReal dx[GR_DIM],
                                   int& i, int& j, int& k, GReal del[GR_DIM])
{
    // Normal operation
    // get provisional zone index. see note above function for details. note we
    // shift to zone centers because that's where variables are most exact.
    i = (int) ((X[1] - startx[1]) / dx[1] - 0.5 + 1000) - 1000;
    j = (int) ((X[2] - startx[2]) / dx[2] - 0.5 + 1000) - 1000;
    k = (int) ((X[3] - startx[3]) / dx[3] - 0.5 + 1000) - 1000;

    // Distance from closest zone center on the left
    // i.e., portion of left zone to use vs right when interpolating
    del[1] = (X[1] - ((i + 0.5) * dx[1] + startx[1])) / dx[1];
    del[2] = (X[2] - ((j + 0.5) * dx[2] + startx[2])) / dx[2];
    del[3] = (X[3] - ((k + 0.5) * dx[3] + startx[3])) / dx[3];
}

/**
 * Return the grid zone index (i,j,k) corresponding which contains the point X.
 * Note this is different from the above!
 */
KOKKOS_INLINE_FUNCTION void Xtoijk_nearest(const GReal X[GR_DIM],
                                   const GReal startx[GR_DIM],
                                   const GReal dx[GR_DIM],
                                   int& i, int& j, int& k)
{
    // Get the index of the zone this point falls into.
    // i.e., are we >= the left corner?
    i = (int) ((X[1] - startx[1]) / dx[1] + 1000) - 1000;
    j = (int) ((X[2] - startx[2]) / dx[2] + 1000) - 1000;
    k = (int) ((X[3] - startx[3]) / dx[3] + 1000) - 1000;
}

// For using the ipole routines in a recognizable form on a 1D array
#define ind(i, j, k) ( (k) * n2 * n1 + (j) * n1 + (i))

/**
 * Dumb linear interpolation: no special cases for boundaries.
 * Takes indices i,j,k and a block size n1, n2, n3,
 * as well as a flat array var.
 * 
 * TODO version(s) with View(s) for real device-side operation
 */
KOKKOS_INLINE_FUNCTION Real linear(const int& i, const int& j, const int& k,
                                   const int& n1, const int& n2, const int& n3,
                                   const double del[4], const double *var)
{
    // Interpolate in 1D at a time to avoid reading zones we don't have
    Real interp = var[ind(i    , j    , k)]*(1. - del[1]) +
                  var[ind(i + 1, j    , k)]*del[1];
    if (n2 > 1) {
        interp = (1. - del[2])*interp +
                 del[2]*(var[ind(i    , j + 1, k)]*(1. - del[1]) +
                         var[ind(i + 1, j + 1, k)]*del[1]);
    }
    if (n3 > 1) {
        interp = (1. - del[3])*interp +
                 del[3]*(var[ind(i    , j    , k + 1)]*(1. - del[1])*(1. - del[2]) +
                         var[ind(i + 1, j    , k + 1)]*del[1]*(1. - del[2]) +
                         var[ind(i    , j + 1, k + 1)]*(1. - del[1])*del[2] +
                         var[ind(i + 1, j + 1, k + 1)]*del[1]*del[2]);
    }
    return interp;
}

/**
 * Dumb linear interpolation: no special cases for boundaries.
 * Takes indices i,j,k and a block size n1, n2, n3,
 * as well as a flat array var.
 * 
 * TODO version(s) with View(s) for real device-side operation
 */
KOKKOS_INLINE_FUNCTION Real linear(const int& i, const int& j, const int& k,
                                   const double del[4], const int& p,
                                   const VariablePack<Real> &var)
{
    // Interpolate in 1D at a time to avoid reading zones we don't have
    Real interp = var(p, k, j, i    ) * (1. - del[1]) +
                  var(p, k, j, i + 1) * del[1];
    if (var.GetDim(2) > 1) {
        interp = (1. - del[2]) * interp +
                 del[2]*(var(p, k, j + 1, i    ) * (1. - del[1]) +
                         var(p, k, j + 1, i + 1) * del[1]);
    }
    if (var.GetDim(3) > 1) {
        interp = (1. - del[3]) * interp +
                 del[3]*(var(p, k + 1, j    , i    ) * (1. - del[1]) * (1. - del[2]) +
                         var(p, k + 1, j    , i + 1) * del[1] * (1. - del[2]) +
                         var(p, k + 1, j + 1, i    ) * (1. - del[1]) * del[2] +
                         var(p, k + 1, j + 1, i + 1) * del[1] * del[2]);
    }
    return interp;
}

} // Interpolation
