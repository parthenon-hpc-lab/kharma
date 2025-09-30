/*
 *  File: chandra_table.hpp
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
#pragma once

// KHARMA headers
#include "decs.hpp"

KOKKOS_INLINE_FUNCTION double get_weight(const double *xx, double x, int *jlo)
{
    // Get the value _before_ x in the table
    while (xx[*jlo] < x) {
        ++*jlo;
    }
    --*jlo;

    // Return weight for table values jlo, jlo+1
    return (x - xx[*jlo]) / m::max(xx[*jlo + 1] - xx[*jlo], SMALL_NUM);
}

KOKKOS_INLINE_FUNCTION void interp_chandra(double mu, double *i, double *del)
{
    static constexpr double ch_mu[21] = {0.000000, 0.050000, 0.100000, 0.150000,
        0.200000, 0.250000, 0.300000, 0.350000, 0.400000, 0.450000, 0.500000,
        0.550000, 0.600000, 0.650000, 0.700000, 0.750000, 0.800000, 0.850000,
        0.900000, 0.950000, 1.000000};

    static constexpr double ch_I[21] = {0.414410, 0.474900, 0.523970, 0.570010,
        0.614390, 0.657700, 0.700290, 0.742340, 0.783980, 0.825300, 0.866370,
        0.907220, 0.947890, 0.988420, 1.02882,  1.06911,  1.10931,  1.14943,
        1.18947, 1.22945, 1.26938};

    static constexpr double ch_delta[21] = {0.117130, 0.0897900, 0.0744800, 0.0631100,
        0.0541000, 0.0466700, 0.0404100, 0.0350200, 0.03033000, 0.02619000, 0.0225200,
        0.0192300, 0.0162700, 0.0135800, 0.0111230, 0.00888000, 0.00681800, 0.00491900,
        0.00315500, 0.00152200, 0.00000000};

    int indx = 0;
    double weight = get_weight(ch_mu, mu, &indx);
    *i = (1. - weight) * ch_I[indx] + weight * ch_I[indx + 1];
    *del = (1. - weight) * ch_delta[indx] + weight * ch_delta[indx + 1];
}