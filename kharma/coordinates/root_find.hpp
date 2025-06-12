/* 
 *  File: coordinate_systems.hpp
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

#define ROOTFIND_TOL 1.e-9

/**
 * Root finder macro for X[2] since it is sometimes not analytically invertible
 * Written with a common interface for doing 2D solves, if those are ever required
 * 
 * TODO ASSUMES Xnative bounds are [0,1] and Xembed bounds are [0,M_PI]!!!!!
 * TODO I cannot find a remotely simple way to do this with templates or function pointers:
 *      if this is taken out of the enclosing scope we need to resolve coord_to_embed, which is haaaard
 * 
 * Takes names Xembed (filled, not modified), Xnative (filled, modifies X[2]), and calls function coord_to_embed
 */
#define ROOT_FIND \
    double th = Xembed[2];\
    double tha, thb, thc;\
\
    double Xa[GR_DIM], Xb[GR_DIM], Xc[GR_DIM], Xtmp[GR_DIM];\
    Xa[1] = Xnative[1];\
    Xa[3] = Xnative[3];\
\
    Xb[1] = Xa[1];\
    Xb[3] = Xa[3];\
    Xc[1] = Xa[1];\
    Xc[3] = Xa[3];\
\
    if (Xembed[2] < M_PI / 2.) {\
        Xa[2] = 0.;\
        Xb[2] = 0.5 + 1.e-14;\
    } else {\
        Xa[2] = 0.5 - 1.e-14;\
        Xb[2] = 1.;\
    }\
\
    coord_to_embed(Xa, Xtmp); tha = Xtmp[2];\
    coord_to_embed(Xb, Xtmp); thb = Xtmp[2];\
\
    if (m::abs(tha-th) < ROOTFIND_TOL) {\
        Xnative[2] = Xa[2]; return;\
    } else if (m::abs(thb-th) < ROOTFIND_TOL) {\
        Xnative[2] = Xb[2]; return;\
    }\
    for (int i = 0; i < 1000; i++) {\
        Xc[2] = 0.5 * (Xa[2] + Xb[2]);\
        coord_to_embed(Xc, Xtmp); thc = Xtmp[2];\
\
        if (m::abs(thc - th) < ROOTFIND_TOL) break;\
        else if ((thc - th) * (thb - th) < 0.) Xa[2] = Xc[2];\
        else Xb[2] = Xc[2];\
    }\
    Xnative[2] = Xc[2];

#define ROOT_FIND_1 \
    double r = Xembed[1];\
    double ra, rb, rc;\
\
    double Xa[GR_DIM], Xb[GR_DIM], Xc[GR_DIM], Xtmp[GR_DIM];\
    Xa[2] = Xnative[2];\
    Xa[3] = Xnative[3];\
\
    Xb[2] = Xa[2];\
    Xb[3] = Xa[3];\
    Xc[2] = Xa[2];\
    Xc[3] = Xa[3];\
\
    Xa[1] = 0.;\
    Xb[1] = 100.;\
\
    coord_to_embed(Xa, Xtmp); ra = Xtmp[1];\
    coord_to_embed(Xb, Xtmp); rb = Xtmp[1];\
\
    if (m::abs(ra-r) < ROOTFIND_TOL) {\
        Xnative[1] = Xa[1]; return;\
    } else if (m::abs(rb-r) < ROOTFIND_TOL) {\
        Xnative[1] = Xb[1]; return;\
    }\
    for (int i = 0; i < 1000; i++) {\
        Xc[1] = 0.5 * (Xa[1] + Xb[1]);\
        coord_to_embed(Xc, Xtmp); rc = Xtmp[1];\
\
        if (m::abs(rc - r) < ROOTFIND_TOL) break;\
        else if ((rc - r) * (rb - r) < 0.) Xa[1] = Xc[1];\
        else Xb[1] = Xc[1];\
    }\
    Xnative[1] = Xc[1];
