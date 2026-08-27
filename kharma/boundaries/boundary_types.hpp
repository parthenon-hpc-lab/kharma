/*
 *  File: boundary_types.hpp
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

#include <mesh/meshblock.hpp>

using namespace parthenon;

namespace KBoundaries
{

inline bool BoundaryIsInner(const IndexDomain domain)
{
    return domain == IndexDomain::inner_x1 || domain == IndexDomain::inner_x2 ||
           domain == IndexDomain::inner_x3;
}

inline bool BoundaryIsInner(const BoundaryFace bface)
{
    return bface == BoundaryFace::inner_x1 || bface == BoundaryFace::inner_x2 ||
           bface == BoundaryFace::inner_x3;
}

inline int BoundaryDirection(const IndexDomain domain)
{
    switch (domain) {
        case IndexDomain::inner_x1:
        case IndexDomain::outer_x1:
            return X1DIR;
        case IndexDomain::inner_x2:
        case IndexDomain::outer_x2:
            return X2DIR;
        case IndexDomain::inner_x3:
        case IndexDomain::outer_x3:
            return X3DIR;
        default:
            return 0;
    }
}

inline int BoundaryDirection(const BoundaryFace face)
{
    switch (face) {
        case BoundaryFace::inner_x1:
        case BoundaryFace::outer_x1:
            return X1DIR;
        case BoundaryFace::inner_x2:
        case BoundaryFace::outer_x2:
            return X2DIR;
        case BoundaryFace::inner_x3:
        case BoundaryFace::outer_x3:
            return X3DIR;
        default:
            return 0;
    }
}

inline std::string BoundaryName(const BoundaryFace face)
{
    switch (face) {
        case BoundaryFace::inner_x1:
            return "inner_x1";
        case BoundaryFace::outer_x1:
            return "outer_x1";
        case BoundaryFace::inner_x2:
            return "inner_x2";
        case BoundaryFace::outer_x2:
            return "outer_x2";
        case BoundaryFace::inner_x3:
            return "inner_x3";
        case BoundaryFace::outer_x3:
            return "outer_x3";
        default:
            return "unknown";
    }
}

inline std::string DomainName(const IndexDomain domain)
{
    switch (domain) {
        case IndexDomain::inner_x1:
            return "inner_x1";
        case IndexDomain::outer_x1:
            return "outer_x1";
        case IndexDomain::inner_x2:
            return "inner_x2";
        case IndexDomain::outer_x2:
            return "outer_x2";
        case IndexDomain::inner_x3:
            return "inner_x3";
        case IndexDomain::outer_x3:
            return "outer_x3";
        case IndexDomain::interior:
            return "interior";
        case IndexDomain::entire:
            return "entire";
        default:
            return "unknown";
    }
}

inline IndexDomain BoundaryDomain(const BoundaryFace face)
{
    switch (face) {
        case BoundaryFace::inner_x1:
            return IndexDomain::inner_x1;
        case BoundaryFace::outer_x1:
            return IndexDomain::outer_x1;
        case BoundaryFace::inner_x2:
            return IndexDomain::inner_x2;
        case BoundaryFace::outer_x2:
            return IndexDomain::outer_x2;
        case BoundaryFace::inner_x3:
            return IndexDomain::inner_x3;
        case BoundaryFace::outer_x3:
            return IndexDomain::outer_x3;
        case BoundaryFace::undef:
        default:
            std::cout << std::endl; // flush messages in output buffer before we error
            throw std::runtime_error("Undefined boundary face has no domain!");
    }
}

inline BoundaryFace BoundaryFaceOf(const IndexDomain domain)
{
    switch (domain) {
        case IndexDomain::inner_x1:
            return BoundaryFace::inner_x1;
        case IndexDomain::outer_x1:
            return BoundaryFace::outer_x1;
        case IndexDomain::inner_x2:
            return BoundaryFace::inner_x2;
        case IndexDomain::outer_x2:
            return BoundaryFace::outer_x2;
        case IndexDomain::inner_x3:
            return BoundaryFace::inner_x3;
        case IndexDomain::outer_x3:
            return BoundaryFace::outer_x3;
        case IndexDomain::interior:
        case IndexDomain::entire:
        default:
            return BoundaryFace::undef;
    }
}

/**
 * Function for checking boundary flags: is this a domain or internal bound?
 */
template<typename T>
inline bool IsPhysicalBoundary(T pmb, const BoundaryFace face)
{
    return !(pmb->boundary_flag[face] == BoundaryFlag::block ||
             pmb->boundary_flag[face] == BoundaryFlag::periodic);
}

}
