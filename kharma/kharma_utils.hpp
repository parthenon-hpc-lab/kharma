/*
 *  File: kharma_utils.hpp
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

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

/*
 * General C/C++ convenience functions, anything not specific to KHARMA's datatypes
 */

/**
 * String formatting in errors.
 * Courtesy https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
 */
template<typename... Args>
std::string string_format(const std::string& format, Args... args)
{
    size_t size =
        snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
    if (size <= 0) {
        std::cout << std::endl; // flush messages in output buffer before we error
        throw std::runtime_error("Error during formatting.");
    }
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}
// If we need to disable it, set this version instead
// template<typename ... Args>
// std::string string_format( const std::string& format, Args ... args )
// { return std::string(""); }

/**
 * Formatted printing functions for looking at vectors, tensors (in future, array areas?)
 * Optionally kill the program if a NaN value is encountered.
 */
KOKKOS_INLINE_FUNCTION void print_matrix(
    const std::string name, const double g[GR_DIM][GR_DIM], bool kill_on_nan = false)
{
    // Print a name and a matrix
    printf("%s:\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n",
        name.c_str(), g[0][0], g[0][1], g[0][2], g[0][3], g[1][0], g[1][1], g[1][2],
        g[1][3], g[2][0], g[2][1], g[2][2], g[2][3], g[3][0], g[3][1], g[3][2], g[3][3]);

    if (kill_on_nan) {
        // Additionally kill things if/when we hit NaNs
        DLOOP2
            if (m::isnan(g[mu][nu])) exit(-1);
    }
}
KOKKOS_INLINE_FUNCTION void print_vector(
    const std::string name, const double v[GR_DIM], bool kill_on_nan = false)
{
    printf("%s: %g\t%g\t%g\t%g\n", name.c_str(), v[0], v[1], v[2], v[3]);

    if (kill_on_nan) {
        DLOOP2
            if (m::isnan(v[nu])) exit(-1);
    }
}

/**
 * This takes a number n and clips it to lie on the real line between 'lower' and 'upper'
 * If n is NaN, it returns the *lower* bound, unless this is also NaN, in which case it
 * returns the upper. Note that you can disable a bound by passing NaN. If lower > upper,
 * clip() will return the bound marked 'upper', that is, the lower number.
 *
 * Lightly edited from
 * https://stackoverflow.com/questions/9323903/most-efficient-elegant-way-to-clip-a-number
 * Note that in C++17+ this can likely be a straight std::clamp call
 */
template<typename T>
KOKKOS_INLINE_FUNCTION T clip(const T& n, const T& lower, const T& upper)
{
#if TRACE
    // This isn't so useful without context
    // if (m::isnan(n)) printf("Clipping a NaN value!\n");
    // if (n > upper) printf("Clip %g to %g\n", n, upper);
    // if (n < lower) printf("Clip %g to %g\n", n, lower);
#endif
    return m::min(m::max(lower, n), upper);
}
// Version which "bounces" any excess over the bounds, useful for the polar coordinate
template<typename T>
KOKKOS_INLINE_FUNCTION T bounce(const T& n, const T& lower, const T& upper)
{
    return (n < lower) ? 2 * lower - n : ((n > upper) ? 2 * upper - n : n);
}
// Version which "excises" anything within a range
template<typename T>
KOKKOS_INLINE_FUNCTION T excise(const T& n, const T& center, const T& range)
{
    return (m::abs(n - center) > range)
               ? n
               : ((n > center) ? center + range : center - range);
}

/**
 * Every physics code ends up with something like this
 */
template<typename T>
KOKKOS_INLINE_FUNCTION T close_to(
    const T& x, const T& y, const Real& rel_tol = 1e-8, const Real& abs_tol = 1e-8)
{
    return ((m::abs(x - y) / y) < rel_tol) ||
           (m::abs(x) < abs_tol && m::abs(y) < abs_tol);
}

// Quickly zero n elements of an array
// Types can fail to resolve if gzeroN() calls zeroN(),
// so we duplicate code a bit
// TODO forceinline
template<typename T>
KOKKOS_INLINE_FUNCTION void zero(T* a, const int& n)
{
    memset(a, 0, n * sizeof(T));
    // for(int i = 0; i < n; i++) a[i] = 0.;
}
template<typename T>
KOKKOS_INLINE_FUNCTION void gzero(T a[GR_DIM])
{
    memset(a, 0, GR_DIM * sizeof(T));
    // for(int i = 0; i < GR_DIM; i++) a[i] = 0.;
}
template<typename T>
KOKKOS_INLINE_FUNCTION void zero2(T* a[], const int& n)
{
    memset(&(a[0][0]), 0, n * sizeof(T));
    // for(int i = 0; i < n; i++) (&(a[0][0]))[i] = 0.;
}
template<typename T>
KOKKOS_INLINE_FUNCTION void gzero2(T a[GR_DIM][GR_DIM])
{
    memset(&(a[0][0]), 0, GR_DIM * GR_DIM * sizeof(T));
    // for(int i = 0; i < GR_DIM*GR_DIM; i++) (&(a[0][0]))[i] = 0.;
}
