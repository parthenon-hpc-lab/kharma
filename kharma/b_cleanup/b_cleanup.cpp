/*
 *  File: b_cleanup.cpp
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
#include "b_cleanup.hpp"

#include "b_ct.hpp"
#include "b_ct_functions.hpp"
#include "boundaries.hpp"
#include "decs.hpp"
#include "domain.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"
#include "kharma_driver.hpp"
#include "one_block_transmit.hpp"
#include "types.hpp"

#include "poisson_equation.hpp"

#include <solvers/bicgstab_solver.hpp>
#include <solvers/cg_solver.hpp>
#include <solvers/mg_solver.hpp>
#include <solvers/solver_utils.hpp>
#include <solvers/tridiag_solver.hpp>

#if DISABLE_CLEANUP

// The package should never be loaded if there is not a global solve to be done.
// Therefore we yell at load time rather than waiting for the first solve
std::shared_ptr<KHARMAPackage> B_Cleanup::Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages)
{
    throw std::runtime_error(
        "KHARMA was compiled without global solvers!  Cannot clean B Field!");
}
TaskStatus B_Cleanup::CleanupDivergence(std::shared_ptr<MeshData<Real>>& md)
{
    throw std::runtime_error(
        "KHARMA was compiled without global solvers!  Cannot clean B Field!");
}

#else

std::shared_ptr<KHARMAPackage> B_Cleanup::Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("B_Cleanup");
    Params& params = pkg->AllParams();

    // Set boundary conditions for Poisson variables
    using BF = parthenon::BoundaryFace;
    for (int i = 0; i < BOUNDARY_NFACES; i++) {
        const auto bface = (BF)i;
        const auto bname = KBoundaries::BoundaryName(bface);
        const auto btype = pin->GetString("boundaries", bname);
        if (btype != "periodic") {
            if (btype == "reflecting") {
                switch (bface) {
                    case BoundaryFace::inner_x1:
                        pkg->UserBoundaryFunctions[BF::inner_x1].push_back(
                            GetBCReflecting<X1DIR, BCSide::Inner>());
                        break;
                    case BoundaryFace::outer_x1:
                        pkg->UserBoundaryFunctions[BF::outer_x1].push_back(
                            GetBCReflecting<X1DIR, BCSide::Outer>());
                        break;
                    case BoundaryFace::inner_x2:
                        pkg->UserBoundaryFunctions[BF::inner_x2].push_back(
                            GetBCReflecting<X2DIR, BCSide::Inner>());
                        break;
                    case BoundaryFace::outer_x2:
                        pkg->UserBoundaryFunctions[BF::outer_x2].push_back(
                            GetBCReflecting<X2DIR, BCSide::Outer>());
                        break;
                    case BoundaryFace::inner_x3:
                        pkg->UserBoundaryFunctions[BF::inner_x3].push_back(
                            GetBCReflecting<X3DIR, BCSide::Inner>());
                        break;
                    case BoundaryFace::outer_x3:
                        pkg->UserBoundaryFunctions[BF::outer_x3].push_back(
                            GetBCReflecting<X3DIR, BCSide::Outer>());
                        break;
                }
            } else {
                // Outflow & dirichlet.  TODO not outflow?
                switch (bface) {
                    case BoundaryFace::inner_x1:
                        pkg->UserBoundaryFunctions[BF::inner_x1].push_back(
                            GetBCDirichlet<X1DIR, BCSide::Inner>());
                        break;
                    case BoundaryFace::outer_x1:
                        pkg->UserBoundaryFunctions[BF::outer_x1].push_back(
                            GetBCDirichlet<X1DIR, BCSide::Outer>());
                        break;
                    case BoundaryFace::inner_x2:
                        pkg->UserBoundaryFunctions[BF::inner_x2].push_back(
                            GetBCDirichlet<X2DIR, BCSide::Inner>());
                        break;
                    case BoundaryFace::outer_x2:
                        pkg->UserBoundaryFunctions[BF::outer_x2].push_back(
                            GetBCDirichlet<X2DIR, BCSide::Outer>());
                        break;
                    case BoundaryFace::inner_x3:
                        pkg->UserBoundaryFunctions[BF::inner_x3].push_back(
                            GetBCDirichlet<X3DIR, BCSide::Inner>());
                        break;
                    case BoundaryFace::outer_x3:
                        pkg->UserBoundaryFunctions[BF::outer_x3].push_back(
                            GetBCDirichlet<X3DIR, BCSide::Outer>());
                        break;
                }
            }
        }
    }

    // For skipping cleaning when it might have been triggered accidentally
    // e.g. on subsequent restarts after a clean
    double init_tolerance = pin->GetOrAddReal("b_cleanup", "no_clean_below", 1.e-8);
    pkg->AddParam<>("init_tolerance", init_tolerance);
    bool use_normalized_divb =
        pin->GetOrAddBoolean("b_cleanup", "use_normalized_divb", false);
    params.Add("use_normalized_divb", use_normalized_divb);

    Real diagonal_alpha = pin->GetOrAddReal("b_cleanup", "diagonal_alpha", 0.0);
    pkg->AddParam<>("diagonal_alpha", diagonal_alpha);

    std::string solver = pin->GetOrAddString("b_cleanup", "solver", "BiCGSTAB");
    pkg->AddParam<>("solver", solver);

    double tolerance = pin->GetOrAddReal("b_cleanup", "tolerance", 1.e-12);
    pkg->AddParam<>("tolerance", tolerance);
    pin->SetReal("b_cleanup", "residual_tolerance", tolerance);

    std::string prolong =
        pin->GetOrAddString("b_cleanup", "boundary_prolongation", "Linear");

    using PoissEq = B_Cleanup::PoissonEquation<u, D>;
    // PoissEq eq(pin, "b_cleanup");
    // pkg->AddParam<>("poisson_equation", eq, parthenon::Params::Mutability::Mutable);

    std::shared_ptr<parthenon::solvers::SolverBase> psolver;
    using prolongator_t = parthenon::solvers::ProlongationBlockInteriorZeroDirichlet;
    using restrictor_t = parthenon::solvers::RestrictionCombined;
    using preconditioner_t =
        parthenon::solvers::MGSolver<PoissEq, prolongator_t, restrictor_t>;
    if (solver == "MG") {
        psolver = std::make_shared<parthenon::solvers::MGSolver<PoissEq, prolongator_t>>(
            "base", "u", "rhs", pin, "b_cleanup", PoissEq(pin, "b_cleanup"));
    } else if (solver == "CG") {
        psolver =
            std::make_shared<parthenon::solvers::CGSolver<PoissEq, preconditioner_t>>(
                "base", "u", "rhs", pin, "b_cleanup", PoissEq(pin, "b_cleanup"));
    } else if (solver == "BiCGSTAB") {
        psolver = std::make_shared<
            parthenon::solvers::BiCGSTABSolver<PoissEq, preconditioner_t>>(
            "base", "u", "rhs", pin, "b_cleanup", PoissEq(pin, "b_cleanup"));
    } else if (solver == "Tridiag") {
        psolver = std::make_shared<parthenon::solvers::TridiagSolver<PoissEq>>(
            "base", "u", "rhs", pin, "b_cleanup", PoissEq(pin, "b_cleanup"));
    } else {
        PARTHENON_FAIL("Unknown solver type " + solver + ".");
    }
    pkg->AddParam<>("solver_pointer", psolver);

    using namespace parthenon::refinement_ops;
    auto mD = Metadata({Metadata::Independent, Metadata::OneCopy, Metadata::Face,
        Metadata::GMGRestrict});
    mD.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();

    // Holds the discretized version of D in \nabla \cdot D(\vec{x}) \nabla u = rhs. D = 1
    // for the standard Poisson equation.
    pkg->AddField(D::name(), mD);

    std::vector<MetadataFlag> flags{Metadata::Cell, Metadata::Independent,
        Metadata::FillGhost, Metadata::WithFluxes, Metadata::GMGRestrict,
        Metadata::GMGProlongate, Metadata::CommunicateOne};
    auto mflux_comm = Metadata(flags);
    if (prolong == "Linear") {
        mflux_comm.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
    } else if (prolong == "Constant") {
        mflux_comm.RegisterRefinementOps<ProlongatePiecewiseConstant, RestrictAverage>();
    } else {
        PARTHENON_FAIL("Unknown prolongation method for Poisson boundaries.");
    }
    // u is the solution vector that starts with an initial guess and then gets updated
    // by the solver
    pkg->AddField(u::name(), mflux_comm);

    auto m_no_ghost = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    // rhs is the field that contains the desired rhs side
    pkg->AddField(rhs::name(), m_no_ghost);

    return pkg;
}

void InitializeD(MeshData<Real>* md)
{
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    auto desc = parthenon::MakePackDescriptor<B_Cleanup::D>(md);
    auto pack = desc.GetPack(md);

    constexpr auto te = B_Cleanup::te;
    using TE = parthenon::TopologicalElement;
    auto& cellbounds = pmb->cellbounds;
    auto ib = cellbounds.GetBoundsI(IndexDomain::entire, te);
    auto jb = cellbounds.GetBoundsJ(IndexDomain::entire, te);
    auto kb = cellbounds.GetBoundsK(IndexDomain::entire, te);
    pmb->par_for("initialize_D", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i)
        {
            pack(b, TE::F1, B_Cleanup::D(), k, j, i) = 1.;
            pack(b, TE::F2, B_Cleanup::D(), k, j, i) = 1.;
            pack(b, TE::F3, B_Cleanup::D(), k, j, i) = 1.;
        });
}

TaskStatus B_Cleanup::CleanupDivergence(std::shared_ptr<MeshData<Real>>& md)
{
    auto pmesh = md->GetMeshPointer();
    auto pkg = pmesh->packages.Get<KHARMAPackage>("B_Cleanup");
    auto init_tolerance = pkg->Param<double>("init_tolerance");
    auto tolerance = pkg->Param<double>("tolerance");
    auto use_normalized = pkg->Param<bool>("use_normalized_divb");
    auto solver = pkg->Param<std::string>("solver");

    auto verbose = pmesh->packages.Get("Globals")->Param<int>("verbose");

    if (!pmesh->multigrid)
        throw std::runtime_error("Cannot clean w/GMG if Mesh not marked "
                                 "multigrid!  Set parthenon/mesh/multigrid=true!");

    if (MPIRank0() && verbose > 0) {
        std::cout << "Cleaning divB to tolerance " << tolerance << " using solver "
                  << solver << std::endl;
    }

    // Calculate/print inital max divB exactly as we would during run
    double divb_start;
    divb_start = B_CT::GlobalMaxDivB(md.get());
    if (divb_start < init_tolerance) {
        // If divB is "pretty good" and we allow not solving...
        if (MPIRank0())
            std::cout << "Magnetic field divergence of " << divb_start
                      << " is below tolerance. Skipping B field cleanup." << std::endl;
        return TaskStatus::complete;
    } else {
        if (MPIRank0())
            std::cout << "Starting magnetic field divergence: " << divb_start
                      << std::endl;
    }

    // make sure B is sync'd before computing RHS
    KHARMADriver::SyncAllBounds(md);

    // Initialize the divB variable, which we'll be solving against.
    // This includes ghosts
    B_CT::CalcDivB(md.get(), rhs::name());
    if (use_normalized) {
        // Normalize divB by local metric determinant for fairer weighting of errors
        // Note that laplacian operator will also have to be normalized ofc
        auto divb_rhs = md->PackVariables(std::vector<std::string>{rhs::name()});
        auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
        const IndexRange ib = md->GetBoundsI(IndexDomain::entire);
        const IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
        const IndexRange kb = md->GetBoundsK(IndexDomain::entire);
        pmb0->par_for("normalize_divB", 0, divb_rhs.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e,
            ib.s, ib.e,
                      KOKKOS_LAMBDA(const int& b, const int& k, const int& j,
                                    const int& i)
            {
                const auto& G = divb_rhs.GetCoords(b);
                divb_rhs(b, CC, 0, k, j, i) /= G.gdet(Loci::center, j, i);
            });
    }

    // Set D=1.  TODO remove D altogether?
    InitializeD(md.get());

    // Execute the solve
    MakeTaskCollection(pmesh).Execute();

    // Recalculate divB max for post-solve check
    double divb_post = B_CT::GlobalMaxDivB(md.get());
    // TODO fail if not converged!
    if (MPIRank0()) {
        std::cout << "Magnetic field after cleanup/before sync: " << divb_post
                  << std::endl;
    }

    // Synchronize to update cons.B's ghost zones
    KHARMADriver::SyncAllBounds(md);
    // Make sure prims.B reflects solution
    B_CT::MeshUtoP(md.get(), IndexDomain::entire, false);

    // Recalculate divB max for one last check
    double divb_end = B_CT::GlobalMaxDivB(md.get());
    if (MPIRank0()) {
        std::cout << "Magnetic field divergence after sync: " << divb_end << std::endl;
    }

    // TODO actually fail if divb_end is high -- needn't make sure it's within `tol`
    // as that's relative but like, make sure it's less than would immediately crash

    return TaskStatus::complete;
}

TaskStatus B_Cleanup::ApplySolution(MeshData<Real>* msolve, MeshData<Real>* md)
{
    Flag("ApplySolution");
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    auto P = msolve->PackVariables(std::vector<std::string>{u::name()});
    auto B = md->PackVariables(std::vector<std::string>{"cons.fB"});

    const int ndim = P.GetNdim();

    // dB = grad(p), defined at cell centers, subtract to make field divergence-free
    // Apply on all physical faces, we'll be syncing/updating ghosts
    const IndexRange3 b = KDomain::GetRange(msolve, IndexDomain::entire, 1, 0);
    pmb0->par_for("gradient_P", 0, P.GetDim(5) - 1, b.ks, b.ke, b.js, b.je, b.is, b.ie,
                  KOKKOS_LAMBDA(const int& b, const int& k, const int& j, const int& i)
        {
            const auto& G = P.GetCoords(b);
            B(b, F1, 0, k, j, i) -= B_CT::face_grad<X1DIR>(G, P(b), k, j, i);
            if (ndim > 1)
                B(b, F2, 0, k, j, i) -= B_CT::face_grad<X2DIR>(G, P(b), k, j, i);
            if (ndim > 2)
                B(b, F3, 0, k, j, i) -= B_CT::face_grad<X3DIR>(G, P(b), k, j, i);
        });

    EndFlag();
    return TaskStatus::complete;
}

#endif // DISABLE_CLEANUP
