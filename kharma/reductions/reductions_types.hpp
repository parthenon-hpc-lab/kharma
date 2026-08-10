/*
 *  File: reductions_variables.hpp
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

// This file is included with types.hpp,
// so that all files have access to the extra Kokkos reduction machinery

#include "decs.hpp"

namespace Reductions
{
// Array type for reducing arbitrary numbers of reals or ints
template<class ScalarType, int N>
struct array_type
{
    ScalarType my_array[N];

    KOKKOS_INLINE_FUNCTION
    array_type() { init(); }

    KOKKOS_INLINE_FUNCTION
    array_type(const array_type& rhs)
    {
        for (int i = 0; i < N; i++) {
            my_array[i] = rhs.my_array[i];
        }
    }

    KOKKOS_INLINE_FUNCTION void init()
    {
        for (int i = 0; i < N; i++) {
            my_array[i] = 0;
        }
    }

    // The kokkos example defines both of these,
    // but we clearly can't. Guess.
    KOKKOS_INLINE_FUNCTION
    array_type& operator+=(const array_type& src)
    {
        for (int i = 0; i < N; i++) {
            my_array[i] += src.my_array[i];
        }
        return *this;
    }

    // KOKKOS_INLINE_FUNCTION
    // void operator+=(const array_type& src) {
    //     for (int i = 0; i < N; i++) {
    //         my_array[i] += src.my_array[i];
    //     }
    // }
};

template<class T, class Space, int N>
struct ArraySum
{
  public:
    // Required
    typedef ArraySum reducer;
    typedef array_type<T, N> value_type;
    typedef Kokkos::View<value_type*, Space, Kokkos::MemoryUnmanaged> result_view_type;

  private:
    value_type& value;

  public:
    KOKKOS_INLINE_FUNCTION
    ArraySum(value_type& value_)
        : value(value_)
    {}

    // Required
    KOKKOS_INLINE_FUNCTION
    void join(value_type& dest, const value_type& src) const { dest += src; }

    KOKKOS_INLINE_FUNCTION
    void init(value_type& val) const { val.init(); }

    KOKKOS_INLINE_FUNCTION
    value_type& reference() const { return value; }

    KOKKOS_INLINE_FUNCTION
    result_view_type view() const { return result_view_type(&value, 1); }

    KOKKOS_INLINE_FUNCTION
    bool references_scalar() const { return true; }
};
}
