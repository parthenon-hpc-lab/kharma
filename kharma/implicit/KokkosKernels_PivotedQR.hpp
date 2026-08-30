#ifndef __KOKKOSBATCHED_APPLY_PIVOT_DECL_HPP__
#define __KOKKOSBATCHED_APPLY_PIVOT_DECL_HPP__

#include "KokkosBatched_ApplyHouseholder_Serial_Internal.hpp"
#include "KokkosBatched_ApplyPivot_Internal.hpp"
#include "KokkosBatched_Dot.hpp"
#include "KokkosBatched_Dot_Internal.hpp"
#include "KokkosBatched_FindAmax_Internal.hpp"
#include "KokkosBatched_Householder_Serial_Internal.hpp"
#include "KokkosBatched_Util.hpp"

namespace KokkosBatched
{

// *** PIVOTS ***

template<typename ArgSide, typename ArgDirect>
struct SerialApplyPivot
{
    template<typename AViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(const int piv, const AViewType& A);

    template<typename PivViewType, typename AViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(const PivViewType piv, const AViewType& A);
};

///
/// Forward
///
struct SerialApplyPivotVectorForwardInternal
{
    template<typename ValueType>
    KOKKOS_INLINE_FUNCTION static int invoke(const int piv,
        /* */ ValueType* KOKKOS_RESTRICT A, const int as0)
    {
        if (piv != 0) {
            const int idx_p = piv * as0;
            const ValueType tmp = A[0];
            A[0] = A[idx_p];
            A[idx_p] = tmp;
        }
        return 0;
    }
};
/// Pivot a row
struct SerialApplyPivotMatrixForwardInternal
{
    template<typename ValueType>
    KOKKOS_INLINE_FUNCTION static int invoke(const int n, const int piv,
        /* */ ValueType* KOKKOS_RESTRICT A, const int as0, const int as1)
    {
        if (piv != 0) {
            for (int j = 0; j < n; j++) {
                ValueType* KOKKOS_RESTRICT A_at_j = A + j * as1;
                const int idx_p = piv * as0;
                const ValueType tmp = A_at_j[0];
                A_at_j[0] = A_at_j[idx_p];
                A_at_j[idx_p] = tmp;
            }
        }
        return 0;
    }
};

///
/// Backward
///
struct SerialApplyPivotVectorBackwardInternal
{
    template<typename ValueType>
    KOKKOS_INLINE_FUNCTION static int invoke(const int piv,
        /* */ ValueType* KOKKOS_RESTRICT A, const int as0)
    {
        if (piv != 0) {
            const int idx_p = piv * as0;
            const ValueType tmp = A[0];
            A[0] = A[idx_p];
            A[idx_p] = tmp;
        }
        return 0;
    }

    template<typename IntType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int invoke(const int plen,
        const IntType* KOKKOS_RESTRICT p, const int ps0,
        /* */ ValueType* KOKKOS_RESTRICT A, const int as0)
    {
        for (int i = (plen - 1); i >= 0; --i) {
            const int piv = p[i * ps0];
            if (piv != 0) {
                const int idx_i = i * as0, idx_p = (i + piv) * as0;
                const ValueType tmp = A[idx_i];
                A[idx_i] = A[idx_p];
                A[idx_p] = tmp;
            }
        }
        return 0;
    }
};
/// Pivot a row
struct SerialApplyPivotMatrixBackwardInternal
{
    template<typename ValueType>
    KOKKOS_INLINE_FUNCTION static int invoke(const int n, const int piv,
        /* */ ValueType* KOKKOS_RESTRICT A, const int as0, const int as1)
    {
        if (piv != 0) {
            for (int j = 0; j < n; ++j) {
                ValueType* KOKKOS_RESTRICT A_at_j = A + j * as1;
                const int idx_p = piv * as0;
                const ValueType tmp = A_at_j[0];
                A_at_j[0] = A_at_j[idx_p];
                A_at_j[idx_p] = tmp;
            }
        }
        return 0;
    }

    template<typename IntType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int invoke(const int n, const int plen,
        const IntType* KOKKOS_RESTRICT p, const int ps0,
        /* */ ValueType* KOKKOS_RESTRICT A, const int as0, const int as1)
    {
        for (int j = 0; j < n; ++j) {
            ValueType* KOKKOS_RESTRICT A_at_j = A + j * as1;
            for (int i = (plen - 1); i >= 0; --i) {
                const int piv = p[i * ps0];
                if (piv != 0) {
                    const int idx_i = i * as0, idx_p = (i + piv) * as0;
                    const ValueType tmp = A_at_j[idx_i];
                    A_at_j[idx_i] = A_at_j[idx_p];
                    A_at_j[idx_p] = tmp;
                }
            }
        }
        return 0;
    }
};

///
/// Backward pivot apply
///

/// row swap
template<>
struct SerialApplyPivot<Side::Left, Direct::Backward>
{
    template<typename AViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(const int piv, const AViewType& A)
    {
        if (AViewType::rank == 1) {
            const int as0 = A.stride(0);
            SerialApplyPivotVectorBackwardInternal::invoke(piv, A.data(), as0);
        } else if (AViewType::rank == 2) {
            const int n = A.extent(1), as0 = A.stride(0), as1 = A.stride(1);
            SerialApplyPivotMatrixBackwardInternal::invoke(n, piv, A.data(), as0, as1);
        }
        return 0;
    }

    template<typename PivViewType, typename AViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(const PivViewType piv, const AViewType& A)
    {
        if (AViewType::rank == 1) {
            const int plen = piv.extent(0), ps0 = piv.stride(0), as0 = A.stride(0);
            SerialApplyPivotVectorBackwardInternal::invoke(
                plen, piv.data(), ps0, A.data(), as0);
        } else if (AViewType::rank == 2) {
            // row permutation
            const int plen = piv.extent(0), ps0 = piv.stride(0), n = A.extent(1),
                      as0 = A.stride(0), as1 = A.stride(1);
            SerialApplyPivotMatrixBackwardInternal::invoke(
                n, plen, piv.data(), ps0, A.data(), as0, as1);
        }
        return 0;
    }
};

// *** SERIAL QR ***

///
/// Serial Internal Impl
/// ====================
///
/// this impl follows the flame interface of householder transformation
///
struct SerialUpdateColumnNormsInternal
{
    template<typename ValueType>
    KOKKOS_INLINE_FUNCTION static int invoke(const int n,
        const ValueType* KOKKOS_RESTRICT a, const int as0,
        /* */ ValueType* KOKKOS_RESTRICT norm, const int ns0)
    {
        using ats = KokkosKernels::ArithTraits<ValueType>;
        for (int j = 0; j < n; ++j) {
            const int idx_a = j * as0, idx_n = j * ns0;
            norm[idx_n] -= ats::conj(a[idx_a]) * a[idx_a];
        }
        return 0;
    }
};

struct SerialPivotedQR_Internal
{
    template<typename ValueType, typename IntType>
    KOKKOS_INLINE_FUNCTION static int invoke(const int m, // m = NumRows(A)
        const int n,                                      // n = NumCols(A)
        /* */ ValueType* A, const int as0, const int as1,
        /* */ ValueType* t, const int ts0,
        /* */ IntType* p, const int ps0,
        /* */ ValueType* w)
    {
        using value_type = ValueType;
        using int_type = IntType;
        using ats = KokkosKernels::ArithTraits<value_type>;

        /// Given a matrix A, it computes QR decomposition of the matrix
        ///  - t is to store tau and w is for workspace

        // partitions used for loop iteration
        Partition2x2<value_type> A_part2x2(as0, as1);
        Partition3x3<value_type> A_part3x3(as0, as1);

        Partition2x1<value_type> t_part2x1(ts0);
        Partition3x1<value_type> t_part3x1(ts0);

        // row vector for norm and p (size of n)
        Partition1x2<int_type> p_part1x2(ps0);
        Partition1x3<int_type> p_part1x3(ps0);

        Partition1x2<value_type> norm_part1x2(1);
        Partition1x3<value_type> norm_part1x3(1);

        // loop size
        const int min_mn = m < n ? m : n;

        // workspace (norm and householder application, 2*max(m,n) is needed)
        value_type* norm = w;
        w += n;

        // initial partition of A where ATL has a zero dimension
        A_part2x2.partWithATL(A, m, n, 0, 0);
        t_part2x1.partWithAT(t, min_mn, 0);

        p_part1x2.partWithAL(p, n, 0);
        norm_part1x2.partWithAL(norm, n, 0);

        // compute initial column norms (replaced by dot product)
        // Had previously:
        SerialDotInternal::invoke(m, n, A, as0, as1, A, as0, as1, norm, 1);
        // In TeamVector this is:
        //Impl::TeamVectorDotInternal::invoke(member, KokkosBlas::Impl::OpConj(), m, n, A, as0, as1, A, as0, as1, norm, 1);
        // So:
        //SerialDot<Trans::ConjTranspose, 1>::invoke(A, A, norm);
        // But this seems to require A to be a View

        int matrix_rank = min_mn;
        value_type max_diag(0);
        for (int m_atl = 0; m_atl < min_mn; ++m_atl) {
            const int n_AR = n - m_atl;

            // part 2x2 into 3x3
            A_part3x3.partWithABR(A_part2x2, 1, 1);
            const int m_A22 = m - m_atl - 1;
            const int n_A22 = n - m_atl - 1;

            t_part3x1.partWithAB(t_part2x1, 1);
            value_type* tau = t_part3x1.A1;

            p_part1x3.partWithAR(p_part1x2, 1);
            int_type* pividx = p_part1x3.A1;

            norm_part1x3.partWithAR(norm_part1x2, 1);

            /// -----------------------------------------------------
            // find max location
            SerialFindAmaxInternal::invoke(n_AR, norm_part1x2.AR, 1, pividx);

            // apply pivot
            SerialApplyPivotVectorForwardInternal::invoke(*pividx, norm_part1x2.AR, 1);
            SerialApplyPivotMatrixForwardInternal::invoke(
                m, *pividx, A_part2x2.ATR, as1, as0);

            // perform householder transformation
            SerialLeftHouseholderInternal::invoke(
                m_A22, A_part3x3.A11, A_part3x3.A21, as0, tau);

            // left apply householder to A22
            SerialApplyLeftHouseholderInternal<Trans::Transpose>::invoke(m_A22, n_A22,
                tau, A_part3x3.A21, as0, A_part3x3.A12, as1, A_part3x3.A22, as0, as1, w);

            // break condition
            if (matrix_rank == min_mn) {
                if (m_atl == 0) max_diag = ats::abs(A[0]);
                const value_type val_diag = ats::abs(A_part3x3.A11[0]),
                                 threshold(10 * max_diag * ats::epsilon());
                if (val_diag < threshold) {
                    matrix_rank = m_atl;
                    // Triggered if matrix_rank passed in is -1. Probably this is allowed?
                    if (true) break;
                }
            }

            // norm update
            SerialUpdateColumnNormsInternal::invoke(
                n_A22, A_part3x3.A12, as1, norm_part1x3.A2, 1);

            /// -----------------------------------------------------
            A_part2x2.mergeToATL(A_part3x3);
            t_part2x1.mergeToAT(t_part3x1);
            p_part1x2.mergeToAL(p_part1x3);
            norm_part1x2.mergeToAL(norm_part1x3);
        }

        return 0;
    }
};

///
/// Serial QR
///

template<typename ArgAlgo>
struct SerialPivotedQR
{
    template<typename AViewType, typename tViewType, typename pViewType,
        typename wViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(
        const AViewType& A, const tViewType& t, const pViewType& p, const wViewType& w);
};

///
/// Serial Impl
/// ===========

template<>
template<typename AViewType, typename tViewType, typename pViewType, typename wViewType>
KOKKOS_INLINE_FUNCTION int SerialPivotedQR<Algo::QR::Unblocked>::invoke(
    const AViewType& A, const tViewType& t, const pViewType& p, const wViewType& w)
{
    return SerialPivotedQR_Internal::invoke(A.extent(0), A.extent(1), A.data(),
        A.stride(0), A.stride(1), t.data(), t.stride(0), p.data(), p.stride(0), w.data());
}

}

#endif
