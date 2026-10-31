/*
 *  File: b_double.hpp
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

#include <cfloat>
#include <cstdint>
#include <string>
#include <vector>

#include <Kokkos_BitManipulation.hpp>
#include <interface/metadata.hpp>

namespace BDouble
{

inline std::string Lo(const std::string& name) { return (SPLIT_B) ? "lo." + name : name; }

inline std::vector<std::string> CellNames() { return {"prims.B", "cons.B"}; }
inline std::vector<std::string> FaceNames() { return {"cons.fB"}; }

KOKKOS_FORCEINLINE_FUNCTION double ScaleOf(const float hi)
{
    const uint64_t bits = Kokkos::bit_cast<uint64_t>(static_cast<double>(hi));
    const uint64_t e = (bits >> 52) & 0x7FF;
    return (e > 53) ? Kokkos::bit_cast<double>((e - 53) << 52) : 0.;
}

KOKKOS_FORCEINLINE_FUNCTION double JoinFloat(const float hi, const float lo)
{
    const uint32_t w = Kokkos::bit_cast<uint32_t>(lo);
    if (w == 0) return static_cast<double>(hi);
    const uint32_t u = (w & 0x7FFFFFu) | (((((w >> 23) & 0xFFu) - 64u) & 0x7Fu) << 23) |
                       ((w >> 31) << 30);
    const int32_t q = static_cast<int32_t>(u) - (1 << 30);
    return static_cast<double>(hi) + static_cast<double>(q) * ScaleOf(hi);
}

KOKKOS_FORCEINLINE_FUNCTION void SplitFloat(const double val, float& hi, float& lo)
{
    const float h = static_cast<float>(val);
    hi = h;
    int64_t q = 0;
    if (m::abs(val) >= FLT_MIN && m::isfinite(h))
        q = static_cast<int64_t>((val - static_cast<double>(h)) / ScaleOf(h));
    if (q == 0) {
        lo = 0.f;
    } else {
        const uint32_t u = static_cast<uint32_t>(q + (1 << 30));
        const uint32_t w = (u & 0x7FFFFFu) | ((((u >> 23) & 0x7Fu) + 64u) << 23) |
                           (((u >> 30) & 1u) << 31);
        lo = Kokkos::bit_cast<float>(w);
    }
}

KOKKOS_FORCEINLINE_FUNCTION BReal Join(const Real& hi, const Real& lo)
{
#if SPLIT_B
    return JoinFloat(hi, lo);
#else
    return hi;
#endif
}

KOKKOS_FORCEINLINE_FUNCTION void Split(const BReal& val, Real& hi, Real& lo)
{
#if SPLIT_B
    SplitFloat(val, hi, lo);
#else
    hi = val;
#endif
}

template<typename P>
struct Pair {
    P hi, lo;
    bool has_lo = true;

    template<typename... Args>
    KOKKOS_FORCEINLINE_FUNCTION BReal operator()(const Args&... args) const
    {
        return (has_lo) ? Join(hi(args...), lo(args...)) : static_cast<BReal>(hi(args...));
    }

    template<typename... Args>
    KOKKOS_FORCEINLINE_FUNCTION void Set(const BReal& val, const Args&... args) const
    {
#if SPLIT_B
        if (has_lo) {
            SplitFloat(val, hi(args...), lo(args...));
            return;
        }
#endif
        hi(args...) = val;
    }

    KOKKOS_FORCEINLINE_FUNCTION auto Block(const int& b) const
    {
        return Pair<std::remove_cv_t<std::remove_reference_t<decltype(hi(b))>>>{
            hi(b), (has_lo) ? lo(b) : hi(b), has_lo};
    }

    KOKKOS_FORCEINLINE_FUNCTION int GetDim(const int& i) const { return hi.GetDim(i); }
};

template<typename P, typename... Args>
KOKKOS_FORCEINLINE_FUNCTION void Store(const P& p, const BReal& val, const Args&... args)
{
    p(args...) = val;
}
template<typename P, typename... Args>
KOKKOS_FORCEINLINE_FUNCTION void Store(
    const Pair<P>& p, const BReal& val, const Args&... args)
{
    p.Set(val, args...);
}

inline parthenon::Metadata LoMetadata(const std::vector<parthenon::MetadataFlag>& flags,
    const std::vector<int>& shape = {})
{
    using parthenon::Metadata;
    std::vector<parthenon::MetadataFlag> lo_flags = {
        Metadata::Real, Metadata::Derived, Metadata::GetOrAddFlag("BLo")};
    for (const auto& f : flags)
        if (f == Metadata::Cell || f == Metadata::Face || f == Metadata::FillGhost ||
            f == Metadata::Restart)
            lo_flags.push_back(f);
    return (shape.empty()) ? Metadata(lo_flags) : Metadata(lo_flags, shape);
}

template<typename T>
inline auto Pack(T* data, const std::string& name)
{
    using PackT = std::remove_cv_t<std::remove_reference_t<decltype(data->PackVariables(
        std::vector<std::string>{name}))>>;
    PackT hi = data->PackVariables(std::vector<std::string>{name});
    PackT lo = data->PackVariables(std::vector<std::string>{Lo(name)});
    const bool has_lo = SPLIT_B && hi.GetDim(4) > 0 && lo.GetDim(4) == hi.GetDim(4);
    return Pair<PackT>{hi, (has_lo) ? lo : hi, has_lo};
}

}
