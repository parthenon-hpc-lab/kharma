//========================================================================================
// (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
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

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

#include "b_cleanup_gmg.hpp"
#include "domain.hpp"

namespace B_CleanupGMG
{

constexpr parthenon::TopologicalElement te = parthenon::TopologicalElement::CC;

// This class implement methods for calculating A.x = y and returning the diagonal of A,
// where A is the the matrix representing the discretized Poisson equation on the grid.
// Here we implement the Laplace operator in terms of a flux divergence to (potentially)
// consistently deal with coarse fine boundaries on the grid. Only the routines Ax and
// SetDiagonal need to be defined for interfacing this with solvers. The other methods
// are internal, but can't be marked private or protected because they launch kernels
// on device.
template<class var_t>
class PoissonEquation
{
  public:
    bool do_flux_cor = true;
    bool set_flux_boundary = true;
    bool include_flux_dx = false;

    using IndependentVars = parthenon::TypeList<var_t>;

    PoissonEquation(parthenon::ParameterInput* pin, const std::string& label)
    {
        do_flux_cor = pin->GetOrAddBoolean(label, "flux_correct", true);
        set_flux_boundary = pin->GetOrAddBoolean(label, "set_flux_boundary", true);
        include_flux_dx =
            (pin->GetOrAddString(label, "boundary_prolongation", "Linear") == "Constant");
    }

    // Add tasks to calculate the result of the matrix A (which is implicitly defined by
    // this class) being applied to x_t and store it in field out_t
    parthenon::TaskID Ax(parthenon::TaskList& tl, parthenon::TaskID depends_on,
        std::shared_ptr<parthenon::MeshData<Real>>& md_mat,
        std::shared_ptr<parthenon::MeshData<Real>>& md_in,
        std::shared_ptr<parthenon::MeshData<Real>>& md_out)
    {
        auto flux_res = tl.AddTask(depends_on, CalculateFluxes, md_mat, md_in);
        if (set_flux_boundary) {
            flux_res = tl.AddTask(flux_res, SetFluxBoundariesZero, md_in);
        }
        // if (do_flux_cor && !(md_mat->grid.type ==
        // parthenon::GridType::two_level_composite)) {
        //   auto start_flxcor =
        //       tl.AddTask(flux_res, parthenon::StartReceiveFluxCorrections, md_in);
        //   auto send_flxcor =
        //       tl.AddTask(flux_res, parthenon::LoadAndSendFluxCorrections, md_in);
        //   auto recv_flxcor =
        //       tl.AddTask(start_flxcor, parthenon::ReceiveFluxCorrections, md_in);
        //   flux_res = tl.AddTask(recv_flxcor, parthenon::SetFluxCorrections, md_in);
        // }
        if (do_flux_cor &&
            !(md_mat->grid.type == parthenon::GridType::two_level_composite)) {
            flux_res = parthenon::AddBoundaryExchangeTasks(
                flux_res, tl, md_in, md_in->GetMeshPointer()->multilevel);
        }
        if (set_flux_boundary) {
            flux_res = tl.AddTask(flux_res, SetFluxBoundariesZero, md_in);
        }
        return tl.AddTask(flux_res, FluxMultiplyMatrix, md_in, md_out);
    }

    // Calculate an approximation to the diagonal of the matrix A and store it in diag_t.
    // For a uniform grid or when flux correction is ignored, this diagonal calculation
    // is exact. Exactness is (probably) not required since it is just used in Jacobi
    // iterations.
    parthenon::TaskStatus SetDiagonal(std::shared_ptr<parthenon::MeshData<Real>>& md_mat,
        std::shared_ptr<parthenon::MeshData<Real>>& md_diag)
    {
        using namespace parthenon;
        const int ndim = md_diag->GetMeshPointer()->ndim;
        IndexRange ib = md_diag->GetBoundsI(IndexDomain::interior, te);
        IndexRange jb = md_diag->GetBoundsJ(IndexDomain::interior, te);
        IndexRange kb = md_diag->GetBoundsK(IndexDomain::interior, te);

        auto pkg = md_diag->GetMeshPointer()->packages.Get("B_CleanupGMG");
        const auto alpha = pkg->Param<Real>("diagonal_alpha");

        int nblocks = md_diag->NumBlocks();
        std::vector<bool> include_block(nblocks, true);

        auto desc_diag = parthenon::MakePackDescriptor<var_t>(md_diag.get());
        auto pack_diag = desc_diag.GetPack(md_diag.get(), include_block);
        using TE = parthenon::TopologicalElement;
        parthenon::par_for("StoreDiagonal", 0, pack_diag.GetNBlocks() - 1, kb.s, kb.e,
            jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i)
            {
                const auto& coords = pack_diag.GetCoordinates(b);
                // Build the unigrid diagonal of the matrix
                Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
                Real diag_elem = -2 / (dx1 * dx1) - alpha;
                if (ndim > 1) {
                    Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
                    diag_elem -= 2 / (dx2 * dx2) - alpha;
                }
                if (ndim > 2) {
                    Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
                    diag_elem -= 2 / (dx3 * dx3) - alpha;
                }
                pack_diag(b, te, var_t(), k, j, i) = diag_elem;
            });
        return TaskStatus::complete;
    }

    static parthenon::TaskStatus CalculateFluxes(
        std::shared_ptr<parthenon::MeshData<Real>>& md_mat,
        std::shared_ptr<parthenon::MeshData<Real>>& md)
    {
        using namespace parthenon;
        const int ndim = md->GetMeshPointer()->ndim;
        IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
        IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
        IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

        using TE = parthenon::TopologicalElement;

        int nblocks = md->NumBlocks();
        std::vector<bool> include_block(nblocks, true);

        auto desc =
            parthenon::MakePackDescriptor<var_t>(md.get(), {}, {PDOpt::WithFluxes});
        auto pack = desc.GetPack(md.get(), include_block);
        parthenon::par_for("CaclulateFluxes", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s,
            jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i)
            {
                const auto& coords = pack.GetCoordinates(b);
                Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
                pack.flux(b, X1DIR, var_t(), k, j, i) =
                    1. / dx1 *
                    (pack(b, te, var_t(), k, j, i - 1) - pack(b, te, var_t(), k, j, i));
                if (i == ib.e)
                    pack.flux(b, X1DIR, var_t(), k, j, i + 1) =
                        1. / dx1 *
                        (pack(b, te, var_t(), k, j, i) -
                            pack(b, te, var_t(), k, j, i + 1));

                if (ndim > 1) {
                    Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
                    pack.flux(b, X2DIR, var_t(), k, j, i) =
                        (pack(b, te, var_t(), k, j - 1, i) -
                            pack(b, te, var_t(), k, j, i)) /
                        dx2;
                    if (j == jb.e)
                        pack.flux(b, X2DIR, var_t(), k, j + 1, i) =
                            (pack(b, te, var_t(), k, j, i) -
                                pack(b, te, var_t(), k, j + 1, i)) /
                            dx2;
                }

                if (ndim > 2) {
                    Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
                    pack.flux(b, X3DIR, var_t(), k, j, i) =
                        (pack(b, te, var_t(), k - 1, j, i) -
                            pack(b, te, var_t(), k, j, i)) /
                        dx3;
                    if (k == kb.e)
                        pack.flux(b, X2DIR, var_t(), k + 1, j, i) =
                            (pack(b, te, var_t(), k, j, i) -
                                pack(b, te, var_t(), k + 1, j, i)) /
                            dx3;
                }
            });
        return TaskStatus::complete;
    }

    static parthenon::TaskStatus SetFluxBoundariesZero(
        std::shared_ptr<parthenon::MeshData<Real>>& md)
    {
        using namespace parthenon;
        const int ndim = md->GetMeshPointer()->ndim;
        IndexRange ib = md->GetBoundsI(IndexDomain::interior);
        IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
        IndexRange kb = md->GetBoundsK(IndexDomain::interior);

        using TE = parthenon::TopologicalElement;

        int nblocks = md->NumBlocks();
        std::vector<bool> include_block(nblocks, true);

        auto desc =
            parthenon::MakePackDescriptor<var_t>(md.get(), {}, {PDOpt::WithFluxes});
        auto pack = desc.GetPack(md.get(), include_block);
        const std::size_t scratch_size_in_bytes = 0;
        const std::size_t scratch_level = 1;

        const parthenon::Indexer3D idxers[6]{parthenon::Indexer3D(kb, jb, {ib.s, ib.s}),
            parthenon::Indexer3D(kb, jb, {ib.e + 1, ib.e + 1}),
            parthenon::Indexer3D(kb, {jb.s, jb.s}, ib),
            parthenon::Indexer3D(kb, {jb.e + 1, jb.e + 1}, ib),
            parthenon::Indexer3D({kb.s, kb.s}, jb, ib),
            parthenon::Indexer3D({kb.e + 1, kb.e + 1}, jb, ib)};
        constexpr int x1off[6]{-1, 1, 0, 0, 0, 0};
        constexpr int x2off[6]{0, 0, -1, 1, 0, 0};
        constexpr int x3off[6]{0, 0, 0, 0, -1, 1};
        constexpr int dirs[6]{X1DIR, X1DIR, X2DIR, X2DIR, X3DIR, X3DIR};
        constexpr bool do_side[6]{false, true, true, true, false, false};
        parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "SetFluxBoundaries",
            DevExecSpace(), scratch_size_in_bytes, scratch_level, 0,
            pack.GetNBlocks() - 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b)
            {
                const auto& coords = pack.GetCoordinates(b);
                for (int face = 0; face < ndim * 2; ++face) {
                    const auto& idxer = idxers[face];
                    const auto dir = dirs[face];
                    // Impose the zero Dirichlet boundary condition at the actual boundary
                    if (pack.IsPhysicalBoundary(
                            b, x3off[face], x2off[face], x1off[face]) &&
                        do_side[face]) {
                        parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, 0,
                            idxer.size() - 1,
                            [&](const int idx)
                            {
                                const auto [k, j, i] = idxer(idx);
                                pack.flux(b, dir, var_t(), k, j, i) = 0.;
                            });
                    }
                }
            });

        return TaskStatus::complete;
    }

    // Calculate A in_t = out_t (in the region covered by md) for a given set of fluxes
    // calculated with in_t (which have possibly been corrected at coarse fine boundaries)
    static parthenon::TaskStatus FluxMultiplyMatrix(
        std::shared_ptr<parthenon::MeshData<Real>>& md,
        std::shared_ptr<parthenon::MeshData<Real>>& md_out)
    {
        using namespace parthenon;
        const int ndim = md->GetMeshPointer()->ndim;
        IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
        IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
        IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

        auto pkg = md->GetMeshPointer()->packages.Get("B_CleanupGMG");
        const auto alpha = pkg->Param<Real>("diagonal_alpha");

        int nblocks = md->NumBlocks();
        std::vector<bool> include_block(nblocks, true);

        static auto desc =
            parthenon::MakePackDescriptor<var_t>(md.get(), {}, {PDOpt::WithFluxes});
        static auto desc_out = parthenon::MakePackDescriptor<var_t>(md_out.get());
        auto pack = desc.GetPack(md.get(), include_block);
        auto pack_out = desc_out.GetPack(md_out.get(), include_block);
        double norm = 0.;
        parthenon::par_reduce(
            "FluxMultiplyMatrix", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
            ib.e,
            KOKKOS_LAMBDA(
                const int b, const int k, const int j, const int i, double& local_result)
            {
                const auto& coords = pack.GetCoordinates(b);
                Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
                pack_out(b, te, var_t(), k, j, i) =
                    -alpha * pack(b, te, var_t(), k, j, i);
                pack_out(b, te, var_t(), k, j, i) +=
                    (pack.flux(b, X1DIR, var_t(), k, j, i) -
                        pack.flux(b, X1DIR, var_t(), k, j, i + 1)) /
                    dx1;

                if (ndim > 1) {
                    Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
                    pack_out(b, te, var_t(), k, j, i) +=
                        (pack.flux(b, X2DIR, var_t(), k, j, i) -
                            pack.flux(b, X2DIR, var_t(), k, j + 1, i)) /
                        dx2;
                }

                if (ndim > 2) {
                    Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
                    pack_out(b, te, var_t(), k, j, i) +=
                        (pack.flux(b, X3DIR, var_t(), k, j, i) -
                            pack.flux(b, X3DIR, var_t(), k + 1, j, i)) /
                        dx3;
                }

                GReal Xnative[GR_DIM];
                coords.coord(k, j, i, Loci::corner, Xnative);
                if (m::abs(Xnative[1]) < coords.coords.startx(1) + 1.e-6 &&
                    m::abs(Xnative[2]) < 1.e-6 && m::abs(Xnative[3]) < 1.e-6)
                    pack_out(b, te, var_t(), k, j, i) = 0;

                // local_result += pack_out(b, te, var_t(), k, j, i);
            },
            Kokkos::Sum<double>(norm));
        // norm /= pack.GetNBlocks() * (kb.e-kb.s+1) * (jb.e-jb.s+1) * (ib.e-ib.s+1);
        // parthenon::par_for(
        //     "MatrixRenorm", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        //     ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        //       pack_out(b, te, var_t(), k, j, i) -= norm;
        //     });

        return TaskStatus::complete;
    }
};

} // namespace B_CleanupGMG
