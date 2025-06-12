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
#include <string>
#include <utility>
#include <vector>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

#include "b_cleanup_gmg.hpp"
#include "domain.hpp"

namespace B_CleanupGMG {

// This class implement methods for calculating A.x = y and returning the diagonal of A,
// where A is the the matrix representing the discretized Poisson equation on the grid.
// Here we implement the Laplace operator in terms of a flux divergence to (potentially)
// consistently deal with coarse fine boundaries on the grid. Only the routines Ax and
// SetDiagonal need to be defined for interfacing this with solvers. The other methods
// are internal, but can't be marked private or protected because they launch kernels
// on device.
class PoissonEquation {
  public:
    bool do_flux_cor = false;

    // Add tasks to calculate the result of the matrix A (which is implicitly defined by
    // this class) being applied to x_t and store it in field out_t
    template <class x_t, class out_t, class TL_t>
    parthenon::TaskID Ax(TL_t &tl, parthenon::TaskID depends_on,
                        std::shared_ptr<parthenon::MeshData<Real>> &md)
    {
        auto flux_res = tl.AddTask(depends_on, CalculateFluxes<x_t>, md);
        if (do_flux_cor && !(md->grid.type == parthenon::GridType::two_level_composite)) {
            auto start_flxcor =
                tl.AddTask(flux_res, parthenon::StartReceiveFluxCorrections, md);
            auto send_flxcor = tl.AddTask(flux_res, parthenon::LoadAndSendFluxCorrections, md);
            auto recv_flxcor = tl.AddTask(start_flxcor, parthenon::ReceiveFluxCorrections, md);
            flux_res = tl.AddTask(recv_flxcor, parthenon::SetFluxCorrections, md);
        }
        return tl.AddTask(flux_res, FluxMultiplyMatrix<x_t, out_t>, md);
    }

    // Calculate an approximation to the diagonal of the matrix A and store it in diag_t.
    // For a uniform grid or when flux correction is ignored, this diagonal calculation
    // is exact. Exactness is (probably) not required since it is just used in Jacobi
    // iterations.
    template <class diag_t>
    parthenon::TaskStatus SetDiagonal(std::shared_ptr<parthenon::MeshData<Real>> &md)
    {
        using namespace parthenon;
        const int ndim = md->GetMeshPointer()->ndim;
        IndexRange3 bd = KDomain::GetRange(md, IndexDomain::interior);

        using TE = parthenon::TopologicalElement;
        constexpr TE te = TE::CC;

        auto pkg = md->GetMeshPointer()->packages.Get("B_CleanupGMG");
        const auto alpha = pkg->Param<Real>("diagonal_alpha");

        int nblocks = md->NumBlocks();
        std::vector<bool> include_block(nblocks, true);

        auto desc = parthenon::MakePackDescriptor<diag_t>(md.get());
        auto pack = desc.GetPack(md.get(), include_block);
        parthenon::par_for(
            "StoreDiagonal", 0, pack.GetNBlocks() - 1, bd.ks, bd.ke, bd.js, bd.je, bd.is, bd.ie,
            KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
                const auto &coords = pack.GetCoordinates(b);
                // Build the unigrid diagonal of the matrix
                Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
                Real diag_elem = - 2 / (dx1 * dx1) - alpha;
                if (ndim > 1) {
                    Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
                    diag_elem -= 2 / (dx2 * dx2);
                }
                if (ndim > 2) {
                    Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
                    diag_elem -= 2 / (dx3 * dx3);
                }
                pack(b, te, diag_t(), k, j, i) = diag_elem;
            }
        );
        return TaskStatus::complete;
    }

    template <class var_t>
    static parthenon::TaskStatus CalculateFluxes(std::shared_ptr<parthenon::MeshData<Real>> &md)
    {
        using namespace parthenon;
        auto pmesh = md->GetMeshPointer();
        const int ndim = pmesh->ndim;
        IndexRange3 bd = KDomain::GetRange(md, IndexDomain::interior, 0, 1);

        using TE = parthenon::TopologicalElement;
        constexpr TE te = TE::CC;

        int nblocks = md->NumBlocks();
        std::vector<bool> include_block(nblocks, true);

        auto desc =
            parthenon::MakePackDescriptor<var_t>(md.get(), {}, {PDOpt::WithFluxes});
        auto pack = desc.GetPack(md.get(), include_block);
        parthenon::par_for(
            "CaclulateFluxes", 0, pack.GetNBlocks() - 1, bd.ks, bd.ke, bd.js, bd.je, bd.is, bd.ie,
            KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
                const auto &coords = pack.GetCoordinates(b);
                const Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
                pack.flux(b, X1DIR, var_t(), k, j, i) =
                    (pack(b, te, var_t(), k, j, i - 1) - pack(b, te, var_t(), k, j, i)) / dx1;

                if (ndim > 1) {
                    const Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
                    pack.flux(b, X2DIR, var_t(), k, j, i) =
                        (pack(b, te, var_t(), k, j - 1, i) - pack(b, te, var_t(), k, j, i)) / dx2;
                }

                if (ndim > 2) {
                    const Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
                    pack.flux(b, X3DIR, var_t(), k, j, i) =
                        (pack(b, te, var_t(), k - 1, j, i) - pack(b, te, var_t(), k, j, i)) / dx3;
                }
            }
        );

        // Make sure B on poles is zero
        auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
        if (pmb0->coords.coords.is_spherical() && ndim > 1) {
            for (auto rc : md->GetAllBlockData()) {
                auto pmb = rc->GetBlockPointer();
                auto desc =
                    parthenon::MakePackDescriptor<var_t>(rc.get(), {}, {PDOpt::WithFluxes});
                auto pack = desc.GetPack(rc.get());
                const IndexRange3 bc = KDomain::GetRange(md, IndexDomain::entire, F2, 1, 0);
                const IndexRange3 bi2 = KDomain::GetRange(md, IndexDomain::interior, F2);
                if (pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::user) {
                    pmb->par_for("dB_boundary", bc.ks, bc.ke, bi2.js, bi2.js, bc.is, bc.ie,
                        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                            const auto &coords = pack.GetCoordinates(0);
                            const Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
                            const Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
                            pack.flux(0, X2DIR, var_t(), k, j, i) = (j == bi2.js) ? 0. :
                                (pack(0, te, var_t(), k, j - 1, i) - pack(0, te, var_t(), k, j, i)) / dx2;
                            pack.flux(0, X1DIR, var_t(), k, j, i) =
                                (pack(0, te, var_t(), k, j, i - 1) - pack(0, te, var_t(), k, j, i)) / dx1;
                            if (ndim > 2) {
                                const Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
                                pack.flux(0, X3DIR, var_t(), k, j, i) =
                                    (pack(0, te, var_t(), k - 1, j, i) - pack(0, te, var_t(), k, j, i)) / dx3;
                            }
                            
                        }
                    );
                }
                if (pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::user) {
                    pmb->par_for("dB_boundary", bc.ks, bc.ke, bi2.je, bi2.je, bc.is, bc.ie,
                        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                            const auto &coords = pack.GetCoordinates(0);
                            const Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
                            const Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
                            pack.flux(0, X2DIR, var_t(), k, j, i) = (j == bi2.je) ? 0. :
                                (pack(0, te, var_t(), k, j - 1, i) - pack(0, te, var_t(), k, j, i)) / dx2;
                            pack.flux(0, X1DIR, var_t(), k, j, i) =
                                (pack(0, te, var_t(), k, j, i - 1) - pack(0, te, var_t(), k, j, i)) / dx1;
                            if (ndim > 2) {
                                const Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
                                pack.flux(0, X3DIR, var_t(), k, j, i) =
                                    (pack(0, te, var_t(), k - 1, j, i) - pack(0, te, var_t(), k, j, i)) / dx3;
                            }
                        }
                    );
                }
            }
        }

        return TaskStatus::complete;
    }

    // Calculate A in_t = out_t (in the region covered by md) for a given set of fluxes
    // calculated with in_t (which have possibly been corrected at coarse fine boundaries)
    template <class in_t, class out_t>
    static parthenon::TaskStatus FluxMultiplyMatrix(std::shared_ptr<parthenon::MeshData<Real>> &md)
    {
        using namespace parthenon;
        auto pmesh = md->GetMeshPointer();
        const int ndim = pmesh->ndim;
        IndexRange3 bd = KDomain::GetRange(md, IndexDomain::interior);

        using TE = parthenon::TopologicalElement;
        constexpr TE te = TE::CC;

        auto pkg = pmesh->packages.Get("B_CleanupGMG");
        const auto alpha = pkg->Param<Real>("diagonal_alpha");

        int nblocks = md->NumBlocks();
        std::vector<bool> include_block(nblocks, true);

        auto desc =
            parthenon::MakePackDescriptor<in_t, out_t>(md.get(), {}, {PDOpt::WithFluxes});
        auto pack = desc.GetPack(md.get(), include_block);
        parthenon::par_for(
            "FluxMultiplyMatrix", 0, pack.GetNBlocks() - 1, bd.ks, bd.ke, bd.js, bd.je, bd.is,
            bd.ie, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
                const auto &coords = pack.GetCoordinates(b);
                Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
                pack(b, te, out_t(), k, j, i) = -alpha * pack(b, te, in_t(), k, j, i);
                pack(b, te, out_t(), k, j, i) += (pack.flux(b, X1DIR, in_t(), k, j, i) -
                                                  pack.flux(b, X1DIR, in_t(), k, j, i + 1)) /
                                                  dx1;

                if (ndim > 1) {
                    Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
                    pack(b, te, out_t(), k, j, i) += (pack.flux(b, X2DIR, in_t(), k, j, i) -
                                                      pack.flux(b, X2DIR, in_t(), k, j + 1, i)) /
                                                      dx2;
                }

                if (ndim > 2) {
                    Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
                    pack(b, te, out_t(), k, j, i) += (pack.flux(b, X3DIR, in_t(), k, j, i) -
                                                      pack.flux(b, X3DIR, in_t(), k + 1, j, i)) /
                                                      dx3;
                }
            }
        );

        // Make sure divB on outflows is 0
        // Our outflow conditions guarantee divergence-free last zones, so we shouldn't clean for them
        // TODO only covers x1 for now!
        const IndexRange3 bc = KDomain::GetRange(md, IndexDomain::entire, te);
        const IndexRange3 bic = KDomain::GetRange(md, IndexDomain::interior, te);
        if (pmesh->packages.Get("Boundaries")->Param<std::string>("inner_x1") == "outflow") {
            for (auto rc : md->GetAllBlockData()) {
                auto pmb = rc->GetBlockPointer();
                auto desc =
                    parthenon::MakePackDescriptor<out_t>(rc.get());
                auto pack = desc.GetPack(rc.get());
                if (pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user) {
                    pmb->par_for("lap_boundary", bc.ks, bc.ke, bc.js, bc.je, bc.is, bic.is,
                        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                            pack(0, te, out_t(), k, j, i) = 0.;
                        }
                    );
                }
            }
        }
        if (pmesh->packages.Get("Boundaries")->Param<std::string>("outer_x1") == "outflow") {
            for (auto rc : md->GetAllBlockData()) {
                auto pmb = rc->GetBlockPointer();
                auto desc =
                    parthenon::MakePackDescriptor<out_t>(rc.get());
                auto pack = desc.GetPack(rc.get());
                if (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::user) {
                    pmb->par_for("lap_boundary", bc.ks, bc.ke, bc.js, bc.je, bic.ie, bc.ie,
                        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                            pack(0, te, out_t(), k, j, i) = 0.;
                        }
                    );
                }
            }
        }

        return TaskStatus::complete;
    }
};

} // namespace B_CleanupGMG
