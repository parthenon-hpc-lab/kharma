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

#include <bvals/boundary_conditions_generic.hpp>

#include "b_ct.hpp"
#include "boundaries.hpp"
#include "decs.hpp"
#include "domain.hpp"
#include "kharma.hpp"
#include "kharma_driver.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"
#include "types.hpp"

#include "poisson_equation.hpp"

using namespace parthenon;
using parthenon::refinement_ops::ProlongateSharedLinear;
using parthenon::refinement_ops::RestrictAverage;
using parthenon::BoundaryFunction::BCSide;

// Build type that selects only variables within our namespace. Internal solver
// variables have the namespace of input variables prepended, so they will also be
// selected by this type.
struct any_bclean : public parthenon::variable_names::base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION any_bclean(Ts &&...args)
      : base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return "b_clean_gmg[.].*"; }
};

template <CoordinateDirection DIR, BCSide SIDE>
auto GetBC() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    using namespace parthenon;
    using namespace parthenon::BoundaryFunction;
    GenericBC<DIR, SIDE, BCType::FixedFace, any_bclean>(rc, coarse, 0.0);
  };
}

std::shared_ptr<KHARMAPackage> B_CleanupGMG::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("B_CleanupGMG");
    Params &params = pkg->AllParams();

  // Set boundary conditions on the pole to 
  using BF = parthenon::BoundaryFace;
  pkg->UserBoundaryFunctions[BF::inner_x2].push_back(GetBC<X2DIR, BCSide::Inner>());
  pkg->UserBoundaryFunctions[BF::outer_x2].push_back(GetBC<X2DIR, BCSide::Outer>());

    // TODO implement
    // bool fail_without_convergence = pin->GetOrAddBoolean("b_cleanup", "fail_without_convergence", true);
    // params.Add("fail_without_convergence", fail_without_convergence);
    // bool warn_without_convergence = pin->GetOrAddBoolean("b_cleanup", "warn_without_convergence", false);
    // params.Add("warn_without_convergence", warn_without_convergence);

    double init_tolerance = pin->GetOrAddReal("b_cleanup", "no_clean_below", 1.e-10);
    pkg->AddParam<>("init_tolerance", init_tolerance);
    bool use_normalized_divb = pin->GetOrAddBoolean("b_cleanup", "use_normalized_divb", false);
    params.Add("use_normalized_divb", use_normalized_divb);

    int max_iter = pin->GetOrAddInteger("b_cleanup", "max_iterations", 10000);
    pkg->AddParam<>("max_iterations", max_iter);

    Real diagonal_alpha = pin->GetOrAddReal("b_cleanup", "diagonal_alpha", 0.0);
    pkg->AddParam<>("diagonal_alpha", diagonal_alpha);

    std::string solver = pin->GetOrAddString("b_cleanup", "solver", "MG");
    pkg->AddParam<>("solver", solver);

    bool flux_correct = pin->GetOrAddBoolean("b_cleanup", "flux_correct", true);
    pkg->AddParam<>("flux_correct", flux_correct);

    double tolerance = pin->GetOrAddReal("b_cleanup", "tolerance", 1.e-8);
    pkg->AddParam<>("tolerance", tolerance);
    pin->SetReal("b_cleanup", "residual_tolerance", tolerance);

    PoissonEquation eq;
    eq.do_flux_cor = flux_correct;

    parthenon::solvers::MGParams mg_params(pin, "b_cleanup");
    mg_params.residual_tolerance = tolerance;
    parthenon::solvers::MGSolver<p, rhs, PoissonEquation> mg_solver(pkg.get(), mg_params,
                                                                    eq);
    pkg->AddParam<>("MGsolver", mg_solver, parthenon::Params::Mutability::Mutable);

    parthenon::solvers::BiCGSTABParams bicgstab_params(pin, "b_cleanup");

    parthenon::solvers::BiCGSTABSolver<p, rhs, PoissonEquation> bicg_solver(
        pkg.get(), bicgstab_params, eq);
    pkg->AddParam<>("MGBiCGSTABsolver", bicg_solver,
                    parthenon::Params::Mutability::Mutable);

    auto mflux_comm = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                                Metadata::WithFluxes, Metadata::GMGRestrict, Metadata::GetUserFlag("StartupOnly")});
    mflux_comm.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
    // The solution vector that starts with an initial guess and then gets updated
    // by the solver
    pkg->AddField(p::name(), mflux_comm);

    auto m_no_ghost = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::GetUserFlag("StartupOnly")});
    // rhs is the field that contains the desired rhs side
    pkg->AddField(rhs::name(), m_no_ghost);

    return pkg;
}

TaskStatus B_CleanupGMG::CleanupDivergence(std::shared_ptr<MeshData<Real>>& md)
{
    auto pmesh = md->GetMeshPointer();
    auto pkg = pmesh->packages.Get("B_CleanupGMG");
    auto init_tolerance = pkg->Param<double>("init_tolerance");
    auto tolerance = pkg->Param<double>("tolerance");
    auto use_normalized = pkg->Param<bool>("use_normalized_divb");

    auto verbose = pmesh->packages.Get("Globals")->Param<int>("verbose");

    if (!pmesh->multigrid) throw std::runtime_error("Cannot clean w/GMG if Mesh not marked multigrid!  Set parthenon/mesh/multigrid=true!");

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

    // make sure divB_RHS is sync'd
    KHARMADriver::SyncAllBounds(md);

    // Execute the solve
    // Solver only syncs what it needs, so we don't need the container trick from B_Cleanup
    MakeSolverTaskCollection(md).Execute();

    // Make sure solution's ghost zones are sync'd
    KHARMADriver::SyncAllBounds(md);

    // Apply the result
    if (MPIRank0() && verbose > 0) {
        std::cout << "Applying magnetic field correction" << std::endl;
    }
    // Update the (conserved) magnetic field on physical zones using our solution
    B_CleanupGMG::ApplyPFace(md.get(), md.get());

    // Synchronize to update cons.B's ghost zones
    KHARMADriver::SyncAllBounds(md);
    // Make sure prims.B reflects solution
    B_CT::MeshUtoP(md.get(), IndexDomain::entire, false);
    // Recalculate divB max for one last check
    double divb_end = B_CT::GlobalMaxDivB(md.get());

    if (MPIRank0()) {
        std::cout << "Magnetic field divergence after cleanup: " << divb_end << std::endl;
    }

    return TaskStatus::complete;
}

TaskStatus B_CleanupGMG::ApplyPFace(MeshData<Real> *msolve, MeshData<Real> *md)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    auto P = msolve->PackVariables(std::vector<std::string>{p::name()});
    auto B = md->PackVariables(std::vector<std::string>{"cons.fB"});

    const int ndim = P.GetNdim();

    // dB = grad(p), defined at cell centers, subtract to make field divergence-free
    // Apply on all physical faces, we'll be syncing/updating ghosts
    const IndexRange3 b = KDomain::GetRange(msolve, IndexDomain::interior, 0, 1);
    pmb0->par_for("gradient_P", 0, P.GetDim(5) - 1, b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
            const auto& G = P.GetCoords(b);
            B(b, F1, 0, k, j, i) -= B_CT::face_grad<X1DIR>(G, P(b), k, j, i);
            B(b, F2, 0, k, j, i) -= B_CT::face_grad<X2DIR>(G, P(b), k, j, i);
            B(b, F3, 0, k, j, i) -= B_CT::face_grad<X3DIR>(G, P(b), k, j, i);
        }
    );

    return TaskStatus::complete;
}
