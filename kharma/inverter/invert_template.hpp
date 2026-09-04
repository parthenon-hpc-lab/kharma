/*
 *  File: invert_template.hpp
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

// This houses only the template for u_to_p.
// It is included by each implementation, and implementations
// are *then* included by inverter.hpp,
// which is the only header which should be imported outside this package.

#include "decs.hpp"
#include "types.hpp"

#include "floors.hpp"

namespace Inverter
{

// Denote inverter types. Currently just one
enum class Type { none = 0, onedw, kastaun };

// Denote inversion failures (pflags)
// This enum should grow to cover any inversion algorithm
enum class Status {
    success = 0,
    neg_input,
    max_iter,
    bad_ut,
    bad_gamma,
    neg_rho,
    neg_u,
    neg_rhou,
    floor,
    bad_velocity
};

static const std::map<int, std::string> status_names = {
    {(int)Status::neg_input, "Negative input"}, {(int)Status::max_iter, "Hit max iter"},
    {(int)Status::bad_ut, "Velocity invalid"}, {(int)Status::bad_gamma, "Gamma invalid"},
    {(int)Status::neg_rho, "Negative rho"}, {(int)Status::neg_u, "Negative U"},
    {(int)Status::neg_rhou, "Negative rho & U"},
    {(int)Status::floor, "Floor during inversion"},
    {(int)Status::bad_velocity, "No velocity solution"}};

template<typename T>
KOKKOS_INLINE_FUNCTION bool failed(T status_flag)
{
    // Return only values >0, among the failure flags
    return static_cast<int>(status_flag) > static_cast<int>(Status::success);
}

template<typename T>
KOKKOS_INLINE_FUNCTION bool valid(T status_flag)
{
    // Return only values >0, among the failure flags
    return static_cast<int>(status_flag) == static_cast<int>(Status::success);
}

/**
 * Recover local primitive variables, with a one-dimensional Newton-Raphson iterative
 * solver. Iteration starts from the current primitive values, and otherwse may *fail to
 * converge*
 *
 * Returns a code indicating whether the solver converged (success), failed (max_iter), or
 * indicating that the converged solution was unphysical (bad_ut, neg_rhou, neg_rho,
 * neg_u)
 *
 * On error, will not write replacement values, leaving the previous step's values in
 * place These are fixed later, in FixUtoP
 *
 * This is the function template: implementations are filled in in their own headers.
 * Be VERY CAREFUL to define any specializations by including those headers,
 * BEFORE you instantiate the template.
 */
// TODO macro the arguments. Also can we set recover_velocity default false?
template<Type inverter>
KOKKOS_INLINE_FUNCTION int u_to_p(const GRCoordinates& G, const VariablePack<Real>& U,
    const VarMap& m_u, const Microphysics::EOS::EOS& eos, const int& k, const int& j, const int& i,
    const VariablePack<Real>& P, const VarMap& m_p, const Loci& loc,
    const int& max_iterations, const Real& tol);
} // namespace Inverter
