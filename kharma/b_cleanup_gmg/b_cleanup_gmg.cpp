/*
 *  File: b_cleanup_gmg.cpp
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
#include "b_cleanup_gmg.hpp"

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

#if DISABLE_GMG_CLEANUP

// The package should never be loaded if there is not a global solve to be done.
// Therefore we yell at load time rather than waiting for the first solve
std::shared_ptr<KHARMAPackage> B_CleanupGMG::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{throw std::runtime_error("KHARMA was compiled without global solvers!  Cannot clean B Field!");}
// We still need some stubs to compile
// TODO throw to ensure these aren't called if package isn't loaded?
TaskStatus B_CleanupGMG::CleanupDivergence(std::shared_ptr<MeshData<Real>>& md)
{
    return TaskStatus::complete;
}

#else

std::shared_ptr<KHARMAPackage> B_CleanupGMG::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("B_CleanupGMG");
    Params &params = pkg->AllParams();

    // Set boundary conditions for Poisson variables
    using BF = parthenon::BoundaryFace;
    //pkg->UserBoundaryFunctions[BF::inner_x1].push_back(GetBCReflecting<X1DIR, BCSide::Inner>());
    pkg->UserBoundaryFunctions[BF::outer_x1].push_back(GetBCReflecting<X1DIR, BCSide::Outer>());
    pkg->UserBoundaryFunctions[BF::inner_x2].push_back(GetBCReflecting<X2DIR, BCSide::Inner>());
    pkg->UserBoundaryFunctions[BF::outer_x2].push_back(GetBCReflecting<X2DIR, BCSide::Outer>());
    // pkg->UserBoundaryFunctions[BF::inner_x3].push_back(GetBCReflecting<X3DIR, BCSide::Inner>());
    // pkg->UserBoundaryFunctions[BF::outer_x3].push_back(GetBCReflecting<X3DIR, BCSide::Outer>());

    pkg->UserBoundaryFunctions[BF::inner_x1].push_back(GetBCDirichlet<X1DIR, BCSide::Inner>());
    // pkg->UserBoundaryFunctions[BF::outer_x1].push_back(GetBCDirichlet<X1DIR, BCSide::Outer>());
    // pkg->UserBoundaryFunctions[BF::inner_x2].push_back(GetBCDirichlet<X2DIR, BCSide::Inner>());
    // pkg->UserBoundaryFunctions[BF::outer_x2].push_back(GetBCDirichlet<X2DIR, BCSide::Outer>());
    // pkg->UserBoundaryFunctions[BF::inner_x3].push_back(GetBCDirichlet<X3DIR, BCSide::Inner>());
    // pkg->UserBoundaryFunctions[BF::outer_x3].push_back(GetBCDirichlet<X3DIR, BCSide::Outer>());

    double init_tolerance = pin->GetOrAddReal("b_cleanup", "no_clean_below", 1.e-10);
    pkg->AddParam<>("init_tolerance", init_tolerance);
    bool use_normalized_divb = pin->GetOrAddBoolean("b_cleanup", "use_normalized_divb", false);
    params.Add("use_normalized_divb", use_normalized_divb);

    Real diagonal_alpha = pin->GetOrAddReal("b_cleanup", "diagonal_alpha", 0.0);
    pkg->AddParam<>("diagonal_alpha", diagonal_alpha);

    std::string solver = pin->GetOrAddString("b_cleanup", "solver", "BiCGSTAB");
    pkg->AddParam<>("solver", solver);

    double tolerance = pin->GetOrAddReal("b_cleanup", "tolerance", 1.e-8);
    pkg->AddParam<>("tolerance", tolerance);
    pin->SetReal("b_cleanup/solver_params", "residual_tolerance", tolerance);

    std::string prolong = pin->GetOrAddString("b_cleanup", "boundary_prolongation", "Linear");

    using PoissEq = PoissonEquation<p>;
    using prolongator_t = parthenon::solvers::ProlongationBlockInteriorZeroDirichlet;
    using preconditioner_t =
        parthenon::solvers::MGSolver<PoissEq, prolongator_t>;

    std::shared_ptr<parthenon::solvers::SolverBase> psolver;
    PoissEq poisson = PoissEq(pin, "b_cleanup");
    //   if (solver == "MG") {
    //     parthenon::solvers::MGParams params(pin, "b_cleanup/solver_params");
    //     psolver = std::make_shared<parthenon::solvers::MGSolver<p, rhs, PoissonEquation>>(
    //         pkg.get(), params, eq);
    //   } else
    if (solver == "CG") {
        psolver = std::make_shared<
            parthenon::solvers::CGSolver<PoissEq, preconditioner_t>>(
            "base", "p", "rhs", pin, "b_cleanup/solver_params", poisson);
    } else if (solver == "BiCGSTAB") {
        psolver = std::make_shared<
            parthenon::solvers::BiCGSTABSolver<PoissEq, preconditioner_t>>(
            "base", "p", "rhs", pin, "b_cleanup/solver_params", poisson);
    } else {
        PARTHENON_FAIL("Unknown solver type.");
    }
    pkg->AddParam<>("solver_pointer", psolver);
    pkg->AddParam<>("poisson_eq", poisson);

    // Setup flags for the solve variable "p"
    using namespace parthenon::refinement_ops;
    std::vector<MetadataFlag> flags{Metadata::Cell, Metadata::Independent,
                                    Metadata::FillGhost, Metadata::WithFluxes,
                                    Metadata::GMGRestrict, Metadata::GetUserFlag("StartupOnly")};
    if (solver == "CG" || solver == "BiCGSTAB") {
        flags.push_back(Metadata::GMGProlongate);
    }
    auto mflux_comm = Metadata(flags);
    if (prolong == "Linear") {
        mflux_comm.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
    } else if (prolong == "Constant") {
        mflux_comm.RegisterRefinementOps<ProlongatePiecewiseConstant, RestrictAverage>();
    } else {
        PARTHENON_FAIL("Unknown prolongation method for Poisson boundaries.");
    }
    mflux_comm.SetFluxName("custom_flux::"+p::name());

    // Setup flux of the solve variable manually, as we need to sync its ghosts always,
    // not just for flux corrections
    std::vector<MetadataFlag> flux_flags{Metadata::Face, Metadata::Derived, Metadata::OneCopy,
                                    //Metadata::FillGhost, //Metadata::Flux,
                                    Metadata::GetUserFlag("StartupOnly")};
    auto mflux = Metadata(flux_flags);

    // Declare them
    pkg->AddField(p::name(), mflux_comm);
    pkg->AddField("custom_flux::"+p::name(), mflux);

    // rhs is the field that contains the desired rhs side
    auto m_rhs = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy,
                           Metadata::GetUserFlag("StartupOnly")});
    pkg->AddField(rhs::name(), m_rhs);
//#endif
    return pkg;
}

TaskStatus B_CleanupGMG::CleanupDivergence(std::shared_ptr<MeshData<Real>>& md)
{
    auto pmesh = md->GetMeshPointer();
    auto pkg = pmesh->packages.Get<KHARMAPackage>("B_CleanupGMG");
    auto init_tolerance = pkg->Param<double>("init_tolerance");
    auto tolerance = pkg->Param<double>("tolerance");
    auto use_normalized = pkg->Param<bool>("use_normalized_divb");

    auto verbose = pmesh->packages.Get("Globals")->Param<int>("verbose");

    //if (!pmesh->multigrid) throw std::runtime_error("Cannot clean w/GMG if Mesh not marked multigrid!  Set parthenon/mesh/multigrid=true!");

    // auto fail_flag = pkg->Param<bool>("fail_without_convergence");
    // auto warn_flag = pkg->Param<bool>("warn_without_convergence");
    if (MPIRank0() && verbose > 0) {
        std::cout << "Cleaning divB to tolerance " << tolerance << std::endl;
        // if (warn_flag) std::cout << "Convergence failure will produce a warning." << std::endl;
        // if (fail_flag) std::cout << "Convergence failure will produce an error." << std::endl;
    }

    // Calculate/print inital max divB exactly as we would during run
    double divb_start;
    divb_start = B_CT::GlobalMaxDivB(md.get());
    if (divb_start < init_tolerance) {
        // If divB is "pretty good" and we allow not solving...
        if (MPIRank0())
            std::cout << "Magnetic field divergence of " << divb_start << " is below tolerance. Skipping B field cleanup." << std::endl;
        return TaskStatus::complete;
    } else {
        if(MPIRank0())
            std::cout << "Starting magnetic field divergence: " << divb_start << std::endl;
    }

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
        pmb0->par_for("normalize_divB", 0, divb_rhs.GetDim(5)-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
                const auto& G = divb_rhs.GetCoords(b);
                divb_rhs(b, CC, 0, k, j, i) /= G.gdet(Loci::center, j, i);
            }
        );
    }

    // make sure RHS is sync'd
    //KHARMADriver::SyncAllBounds(md);

    // Pull a switcheroo: avoid calling KHARMA's boundaries during this solve, ever.
    auto bound_pkg = pmesh->packages.Get<KHARMAPackage>("Boundaries");
    for (int i_bnd = 0; i_bnd < BOUNDARY_NFACES; i_bnd++) {
        auto bface = (BoundaryFace) i_bnd;
        pkg->KBoundaries[bface] = bound_pkg->KBoundaries[bface];
        bound_pkg->KBoundaries[bface] = nullptr;
    }

    std::cerr << "STARTING SOLVE" << std::endl;
    // Execute the solve
    // Solver only syncs what it needs, so we don't need the container trick from B_Cleanup
    MakeSolverTaskCollection(pmesh).Execute();

    // Recalculate divB max for post-solve check
    double divb_post = B_CT::GlobalMaxDivB(md.get());
    // TODO fail if not converged!
    if (MPIRank0()) {
        std::cout << "Magnetic field according to cleanup: " << divb_post << std::endl;
    }

    for (int i_bnd = 0; i_bnd < BOUNDARY_NFACES; i_bnd++) {
        auto bface = (BoundaryFace) i_bnd;
        bound_pkg->KBoundaries[bface] = pkg->KBoundaries[bface];
    }

    // Synchronize to update cons.B's ghost zones
    KHARMADriver::SyncAllBounds(md);
    // Make sure prims.B reflects solution
    B_CT::MeshUtoP(md.get(), IndexDomain::entire, false);
    // Recalculate divB max for one last check
    double divb_end = B_CT::GlobalMaxDivB(md.get());

    // TODO fail if not converged!
    if (MPIRank0()) {
        std::cout << "Magnetic field divergence after sync: " << divb_end << std::endl;
    }

    return TaskStatus::complete;
}

TaskStatus B_CleanupGMG::ApplyPFace(MeshData<Real> *msolve, MeshData<Real> *md)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    auto P = msolve->PackVariablesAndFluxes(std::vector<std::string>{p::name()});
    auto B = md->PackVariables(std::vector<std::string>{"cons.fB"});

    const int ndim = P.GetNdim();

    // dB = grad(p), defined at cell centers, subtract to make field divergence-free
    // Apply on all physical faces, we'll be syncing/updating ghosts
    const IndexRange3 b = KDomain::GetRange(msolve, IndexDomain::interior, 0, 1);
    pmb0->par_for("gradient_P", 0, P.GetDim(5) - 1, b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
            const auto& G = P.GetCoords(b);
            B(b, F1, 0, k, j, i) += P(b).flux(X1DIR, 0, k, j, i);
            B(b, F2, 0, k, j, i) += P(b).flux(X2DIR, 0, k, j, i);
            B(b, F3, 0, k, j, i) += P(b).flux(X3DIR, 0, k, j, i);
        }
    );

    return TaskStatus::complete;
}

#endif // DISABLE_GMG_CLEANUP