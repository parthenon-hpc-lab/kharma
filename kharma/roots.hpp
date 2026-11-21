/*
 *  File: roots.hpp
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
#include <cstdint>

// Bracketed root finders for functions of one variable, usable inside kernels

namespace Roots
{

// Returned after a bisection algorithm
enum class RootStatus : uint8_t {
    Success,      // Converged within the iteration budget
    MaxIter,      // Root detected within an interval but ran out of iterations
    NotBracketed, // No root found (for bisection algorithms)
    Failure,      // Did not find a root (general)
};

/*
In some applications, bisection is used to bracket a function's root.
Due to convergence tolerances or machine precision, the computed root
may not coincide exactly with the true root. Some algorithms therefore
prefer the returned approximation to lie on a specific side of the root,
such that the function value is slightly positive or negative.
*/
enum class RootEval : uint8_t {
    Positive, // f(root) = 0+
    Negative, // f(root) = 0-
    Any,      // No preference
};

template<RootEval RP, typename T, typename Callable>
KOKKOS_INLINE_FUNCTION RootStatus bisect_illinois(T& out, Callable&& f, const T low,
    const T high, const T ftol, const T xtol, const int max_iter)
{
    // Uses false position method with Illinois algorithm for acceleration.
    /**
    @tparam RP: Side of the root to return: f(out) >= 0, f(out) <= 0, or Any
    @param f: A function takes a T and returns a T, with a root between low and high
    @param low: Lower bound of root
    @param high: Upper bound of root
    @param ftol: Absolute tolerance of f for convergence (0. turns it off)
    @param xtol: Absolute tolerance of the bracket width for convergence (0. turns it off)
    @param max_iter: maximum number of iterations
    @note With ftol and xtol both 0., the function will return success if machine
    precision causes the brackets to collapse before max_iter is reached.

    */
    // The ends swap as the iterations go, so they can be in either order:
    // a is the end kept from earlier iterations, b is the latest iterate
    T a = low;
    T b = high;
    T fa = f(a);
    T fb = f(b);
    T z = 0.5 * (a + b);

    // The bracket end on the requested side of the root, or z for Any
    auto side_end = [&]() -> T
    {
        if constexpr (RP == RootEval::Positive) {
            return fa > 0 ? a : b;
        } else if constexpr (RP == RootEval::Negative) {
            return fa > 0 ? b : a;
        } else {
            // Round-off can put z just outside
            return m::min(m::max(z, m::min(a, b)), m::max(a, b));
        }
    };

    // Initial checks before entering the main loop
    if (fa * fb > 0) {
        return RootStatus::NotBracketed;
    } else if (fa == 0) {
        out = a;
        return RootStatus::Success;
    } else if (fb == 0) {
        out = b;
        return RootStatus::Success;
    } else if (m::abs(b - a) <= xtol) {
        out = side_end();
        return RootStatus::Success;
    }

    // Loop until convergence is met
    // or brackets collapse to machine precision
    for (int iter = 0; iter < max_iter; iter++) {
        z = (a * fb - b * fa) / (fb - fa); // linear interpolation to point f(z)=0
        if constexpr (RP != RootEval::Any) {
            // Round-off put z on the bracket: one end is at the root, but the end on
            // the requested side may not be. Bisect instead, so that end converges too
            if ((z - a) * (z - b) >= 0) {
                z = 0.5 * (a + b);
            }
        }
        if ((z - a) * (z - b) >= 0) {
            // Bracket collapsed to machine precision.
            out = side_end();
            return RootStatus::Success;
        }
        T fz = f(z);

        // Depending on RP, this branch will be compiled away
        // so no need to bloat the code with trivial constexpr branches
        if ((m::abs(fz) <= ftol) &&
            (RP == RootEval::Any || (RP == RootEval::Positive ? fz >= 0 : fz <= 0))) {
            // Converged on the requested side of the root
            out = z;
            return RootStatus::Success;
        } else if (fz * fb < 0.0) {
            // Root bracketed by z and b: b becomes the kept end
            a = b;
            fa = fb;
        } else {
            // Root bracketed by a and z: a is kept again
            // Illinois: halve its f to accelerate convergence
            fa = 0.5 * fa;
        }
        b = z;
        fb = fz;
        // Converged in x. Tested after the update, so the last iteration still counts
        if (m::abs(b - a) <= xtol) {
            out = side_end();
            return RootStatus::Success;
        }
    }

    out = side_end(); // Best guess respecting the requested root position
    return RootStatus::MaxIter;
}

template<RootEval RP, typename T, typename Callable>
KOKKOS_INLINE_FUNCTION RootStatus bisect(T& out, Callable&& f, const T low, const T high,
    const T ftol, const T xtol, const int max_iter)
{
    // Plain bisection. Slower than bisect_illinois, but each step halves the bracket
    // whatever the size of f, so it also suits an f that spans many orders of magnitude
    // across the bracket, or overflows at one end.
    /**
    Same arguments and return values as bisect_illinois, except that for RP==Any, a
    stop on xtol or max_iter returns the midpoint of the final bracket.
    */
    T a = low;
    T b = high;
    T fa = f(a);
    T fb = f(b);

    // The bracket end on the requested side of the root, or the midpoint for Any
    auto side_end = [&]() -> T
    {
        if constexpr (RP == RootEval::Positive) {
            return fa > 0 ? a : b;
        } else if constexpr (RP == RootEval::Negative) {
            return fa > 0 ? b : a;
        } else {
            return 0.5 * (a + b);
        }
    };

    // Initial checks before entering the main loop
    if (fa * fb > 0) {
        return RootStatus::NotBracketed;
    } else if (fa == 0) {
        out = a;
        return RootStatus::Success;
    } else if (fb == 0) {
        out = b;
        return RootStatus::Success;
    } else if (m::abs(b - a) <= xtol) {
        out = side_end();
        return RootStatus::Success;
    }

    for (int iter = 0; iter < max_iter; iter++) {
        const T z = 0.5 * (a + b);
        if ((z - a) * (z - b) >= 0) {
            // Bracket collapsed to machine precision
            out = side_end();
            return RootStatus::Success;
        }
        const T fz = f(z);

        if ((m::abs(fz) <= ftol) &&
            (RP == RootEval::Any || (RP == RootEval::Positive ? fz >= 0 : fz <= 0))) {
            // Converged on the requested side of the root
            out = z;
            return RootStatus::Success;
        } else if (fz * fb < 0.0) {
            // Root bracketed by z and b
            a = z;
            fa = fz;
        } else {
            // Root bracketed by a and z
            b = z;
            fb = fz;
        }
        // Converged in x. Tested after the update, so the last iteration still counts
        if (m::abs(b - a) <= xtol) {
            out = side_end();
            return RootStatus::Success;
        }
    }

    out = side_end(); // Best guess respecting the requested root position
    return RootStatus::MaxIter;
}

} // namespace Roots
