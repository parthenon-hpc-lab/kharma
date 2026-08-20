//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
#pragma once

//! \file plm.hpp
//  \brief implements piecewise linear reconstruction

#include "coordinates/coordinates.hpp"
#include "mesh/mesh.hpp"

namespace parthenon
{

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX1()
//  \brief
template<typename T>
KOKKOS_INLINE_FUNCTION void PiecewiseLinearX1(parthenon::team_mbr_t const& member,
    const int k, const int j, const int il, const int iu, const T& q,
    ScratchPad2D<Real>& ql, ScratchPad2D<Real>& qr, ScratchPad2D<Real>& qc,
    ScratchPad2D<Real>& dql, ScratchPad2D<Real>& dqr, ScratchPad2D<Real>& dqm)
{
    const int nu = q.GetDim(4) - 1;

    // compute L/R slopes for each variable
    for (int n = 0; n <= nu; ++n) {
        parthenon::par_for_inner(member, il, iu,
            [&](const int i)
            {
                // renamed dw* -> dq* from plm.cpp
                dql(n, i) = (q(n, k, j, i) - q(n, k, j, i - 1));
                dqr(n, i) = (q(n, k, j, i + 1) - q(n, k, j, i));
                qc(n, i) = q(n, k, j, i);
            });
    }
    member.team_barrier();

    // Apply simplified van Leer (VL) limiter expression for a Cartesian-like coordinate
    // with uniform mesh spacing
    for (int n = 0; n <= nu; ++n) {
        parthenon::par_for_inner(member, il, iu,
            [&](const int i)
            {
                Real dq2 = dql(n, i) * dqr(n, i);
                dqm(n, i) = 2.0 * dq2 / (dql(n, i) + dqr(n, i));
                if (dq2 <= 0.0) dqm(n, i) = 0.0;
            });
    }
    member.team_barrier();

    // compute ql_(i+1/2) and qr_(i-1/2) using limited slopes
    for (int n = 0; n <= nu; ++n) {
        parthenon::par_for_inner(member, il, iu,
            [&](const int i)
            {
                // Mignone equation 30
                ql(n, i + 1) = qc(n, i) + 0.5 * dqm(n, i);
                qr(n, i) = qc(n, i) - 0.5 * dqm(n, i);
            });
    }
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX2()
//  \brief
template<typename T>
KOKKOS_INLINE_FUNCTION void PiecewiseLinearX2(parthenon::team_mbr_t const& member,
    const int k, const int j, const int il, const int iu, const T& q,
    ScratchPad2D<Real>& ql, ScratchPad2D<Real>& qr, ScratchPad2D<Real>& qc,
    ScratchPad2D<Real>& dql, ScratchPad2D<Real>& dqr, ScratchPad2D<Real>& dqm)
{
    const int nu = q.GetDim(4) - 1;

    // compute L/R slopes for each variable
    for (int n = 0; n <= nu; ++n) {
        parthenon::par_for_inner(member, il, iu,
            [&](const int i)
            {
                // renamed dw* -> dq* from plm.cpp
                dql(n, i) = (q(n, k, j, i) - q(n, k, j - 1, i));
                dqr(n, i) = (q(n, k, j + 1, i) - q(n, k, j, i));
                qc(n, i) = q(n, k, j, i);
            });
    }
    member.team_barrier();

    // Apply simplified van Leer (VL) limiter expression for a Cartesian-like coordinate
    // with uniform mesh spacing
    for (int n = 0; n <= nu; ++n) {
        parthenon::par_for_inner(member, il, iu,
            [&](const int i)
            {
                Real dq2 = dql(n, i) * dqr(n, i);
                dqm(n, i) = 2.0 * dq2 / (dql(n, i) + dqr(n, i));
                if (dq2 <= 0.0) dqm(n, i) = 0.0;
            });
    }
    member.team_barrier();

    // compute ql_(j+1/2) and qr_(j-1/2) using limited slopes
    for (int n = 0; n <= nu; ++n) {
        parthenon::par_for_inner(member, il, iu,
            [&](const int i)
            {
                ql(n, i) = qc(n, i) + 0.5 * dqm(n, i);
                qr(n, i) = qc(n, i) - 0.5 * dqm(n, i);
            });
    }
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX3()
//  \brief
template<typename T>
KOKKOS_INLINE_FUNCTION void PiecewiseLinearX3(parthenon::team_mbr_t const& member,
    const int k, const int j, const int il, const int iu, const T& q,
    ScratchPad2D<Real>& ql, ScratchPad2D<Real>& qr, ScratchPad2D<Real>& qc,
    ScratchPad2D<Real>& dql, ScratchPad2D<Real>& dqr, ScratchPad2D<Real>& dqm)
{
    const int nu = q.GetDim(4) - 1;

    // compute L/R slopes for each variable
    for (int n = 0; n <= nu; ++n) {
        parthenon::par_for_inner(member, il, iu,
            [&](const int i)
            {
                // renamed dw* -> dq* from plm.cpp
                dql(n, i) = (q(n, k, j, i) - q(n, k - 1, j, i));
                dqr(n, i) = (q(n, k + 1, j, i) - q(n, k, j, i));
                qc(n, i) = q(n, k, j, i);
            });
    }
    member.team_barrier();

    // Apply simplified van Leer (VL) limiter expression for a Cartesian-like coordinate
    // with uniform mesh spacing
    for (int n = 0; n <= nu; ++n) {
        parthenon::par_for_inner(member, il, iu,
            [&](const int i)
            {
                Real dq2 = dql(n, i) * dqr(n, i);
                dqm(n, i) = 2.0 * dq2 / (dql(n, i) + dqr(n, i));
                if (dq2 <= 0.0) dqm(n, i) = 0.0;
            });
    }
    member.team_barrier();

    // compute ql_(k+1/2) and qr_(k-1/2) using limited slopes
    for (int n = 0; n <= nu; ++n) {
        parthenon::par_for_inner(member, il, iu,
            [&](const int i)
            {
                ql(n, i) = qc(n, i) + 0.5 * dqm(n, i);
                qr(n, i) = qc(n, i) - 0.5 * dqm(n, i);
            });
    }
}

} // namespace parthenon
