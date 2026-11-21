/*
 *  File: root_find.hpp
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
#include "roots.hpp"

#define ROOTFIND_TOL 1.e-9

/**
 * Root finder macros for X[2] (ROOT_FIND) and X[1] (ROOT_FIND_1), since they are
 * sometimes not analytically invertible. ROOT_FIND uses bisect_illinois. ROOT_FIND_1 uses
 * bisect, as r(X1) spans too many orders of magnitude over [0, 100] for false position
 * Written with a common interface for doing 2D solves, if those are ever required
 *
 * TODO ASSUMES Xnative bounds are [0,1] and Xembed bounds are [0,M_PI]!!!!!
 * TODO I cannot find a remotely simple way to do this with templates or function
 * pointers: if this is taken out of the enclosing scope we need to resolve
 * coord_to_embed, which is haaaard
 *
 * Takes names Xembed (filled, not modified), Xnative (filled, modifies X[2]), and calls
 * function coord_to_embed
 */
#define ROOT_FIND                                                                        \
    const GReal th = Xembed[2];                                                          \
    GReal Xc[GR_DIM], Xtmp[GR_DIM];                                                      \
    Xc[0] = Xnative[0];                                                                  \
    Xc[1] = Xnative[1];                                                                  \
    Xc[3] = Xnative[3];                                                                  \
    auto resid = [&](const GReal x2)                                                     \
    {                                                                                    \
        Xc[2] = x2;                                                                      \
        coord_to_embed(Xc, Xtmp);                                                        \
        return Xtmp[2] - th;                                                             \
    };                                                                                   \
    /* Search the half of X[2] holding th */                                             \
    const GReal x2_lo = (th < M_PI_2) ? 0. : 0.5 - 1.e-14;                               \
    const GReal x2_hi = (th < M_PI_2) ? 0.5 + 1.e-14 : 1.;                               \
    if (Roots::bisect_illinois<Roots::RootEval::Any>(Xnative[2], resid, x2_lo, x2_hi,    \
            (GReal)ROOTFIND_TOL, (GReal)0., 1000) == Roots::RootStatus::NotBracketed) {  \
        /* th out of range by round-off: take the closer end */                          \
        Xnative[2] = (m::abs(resid(x2_lo)) < m::abs(resid(x2_hi))) ? x2_lo : x2_hi;      \
    }

#define ROOT_FIND_1                                                                      \
    const GReal r = Xembed[1];                                                           \
    GReal Xc[GR_DIM], Xtmp[GR_DIM];                                                      \
    Xc[0] = Xnative[0];                                                                  \
    Xc[2] = Xnative[2];                                                                  \
    Xc[3] = Xnative[3];                                                                  \
    auto resid = [&](const GReal x1)                                                     \
    {                                                                                    \
        Xc[1] = x1;                                                                      \
        coord_to_embed(Xc, Xtmp);                                                        \
        return Xtmp[1] - r;                                                              \
    };                                                                                   \
    const GReal x1_lo = 0., x1_hi = 100.;                                                \
    if (Roots::bisect<Roots::RootEval::Any>(Xnative[1], resid, x1_lo, x1_hi,             \
            (GReal)ROOTFIND_TOL, (GReal)0., 1000) == Roots::RootStatus::NotBracketed) {  \
        /* r out of range: take the closer end */                                        \
        Xnative[1] = (m::abs(resid(x1_lo)) < m::abs(resid(x1_hi))) ? x1_lo : x1_hi;      \
    }
