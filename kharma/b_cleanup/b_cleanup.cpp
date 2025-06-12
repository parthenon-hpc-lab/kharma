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

// For a bunch of utility functions
#include "b_flux_ct.hpp"
#include "b_ct.hpp"

#include "boundaries.hpp"
#include "decs.hpp"
#include "domain.hpp"
#include "kharma.hpp"
#include "kharma_driver.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"
#include "types.hpp"

#if DISABLE_CLEANUP

// The package should never be loaded if there is not a global solve to be done.
// Therefore we yell at load time rather than waiting for the first solve
std::shared_ptr<KHARMAPackage> B_Cleanup::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{throw std::runtime_error("KHARMA was compiled without global solvers!  Cannot clean B Field!");}
// We still need some stubs to compile
// TODO throw to ensure these aren't called if package isn't loaded?
TaskStatus B_Cleanup::CleanupDivergence(std::shared_ptr<MeshData<Real>>& md)
{
    return TaskStatus::complete;
}
bool B_Cleanup::CleanupThisStep(Mesh* pmesh, int nstep)
{
    return false;
}

#else

#include <parthenon/parthenon.hpp>
// This is now part of KHARMA, but builds on some stuff not in all Parthenon versions
#include "bicgstab_solver.hpp"

using namespace parthenon;
using namespace parthenon::solvers;

std::shared_ptr<KHARMAPackage> B_Cleanup::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("B_Cleanup");
    Params &params = pkg->AllParams();

    // If face centered fields...
    const bool use_b_ct = packages->AllPackages().count("B_CT");

    // Solver options
    // Allow setting tolerance relative to starting value
    // Parthenon's BiCGStab solver stops on abs || rel, so this disables rel
    Real rel_tolerance = pin->GetOrAddReal("b_cleanup", "rel_tolerance", 1e-20);
    params.Add("rel_tolerance", rel_tolerance);
    Real abs_tolerance = pin->GetOrAddReal("b_cleanup", "abs_tolerance", 1e-9);
    params.Add("abs_tolerance", abs_tolerance);
    int max_iterations = pin->GetOrAddInteger("b_cleanup", "max_iterations", 1e8);
    params.Add("max_iterations", max_iterations);
    int check_interval = pin->GetOrAddInteger("b_cleanup", "check_interval", 20);
    params.Add("check_interval", check_interval);
    bool fail_without_convergence = pin->GetOrAddBoolean("b_cleanup", "fail_without_convergence", true);
    params.Add("fail_without_convergence", fail_without_convergence);
    bool warn_without_convergence = pin->GetOrAddBoolean("b_cleanup", "warn_without_convergence", false);
    params.Add("warn_without_convergence", warn_without_convergence);
    bool always_solve = pin->GetOrAddBoolean("b_cleanup", "always_solve", false);
    params.Add("always_solve", always_solve);
    bool use_normalized_divb = pin->GetOrAddBoolean("b_cleanup", "use_normalized_divb", false);
    params.Add("use_normalized_divb", use_normalized_divb);

    // Initialize the solver
    // Translate parameters
    params.Add("bicgstab_max_iterations", max_iterations);
    params.Add("bicgstab_check_interval", check_interval);
    params.Add("bicgstab_abort_on_fail", fail_without_convergence);
    params.Add("bicgstab_warn_on_fail", warn_without_convergence);
    params.Add("bicgstab_print_checks", true);

    // Sparse matrix.  Never built, we leave it blank
    pkg->AddParam<std::string>("spm_name", "");
    // Solution
    pkg->AddParam<std::string>("sol_name", "p");
    // RHS.  Must not just be "divB" as that field does not sync boundaries
    pkg->AddParam<std::string>("rhs_name", "RHS_divB");
    // Construct a solver. We don't need the template parameter, so we use 'int'.
    // The flag "StartupOnly" marks solver variables not to be sync'd later,
    // even though they're also marked FillGhost
    BiCGStabSolver<int> solver(pkg.get(), rel_tolerance, abs_tolerance,
                                SparseMatrixAccessor(), {}, {Metadata::GetUserFlag("StartupOnly")});
    // Set callback
    if (use_b_ct) {
        solver.user_MatVec = B_Cleanup::CenterLaplacian;
    } else {
        solver.user_MatVec = B_Cleanup::CornerLaplacian;
    }

    params.Add("solver", solver);

    // FIELDS
    std::vector<int> s_vector({NVEC});
    std::vector<MetadataFlag> cleanup_flags({Metadata::Real, Metadata::Derived, Metadata::OneCopy,
                                             Metadata::GetUserFlag("StartupOnly")});
    if (use_b_ct) {
        auto cleanup_flags_cell = cleanup_flags;
        cleanup_flags_cell.push_back(Metadata::FillGhost);
        cleanup_flags_cell.push_back(Metadata::Cell);
        auto cleanup_flags_face = cleanup_flags;
        cleanup_flags_face.push_back(Metadata::Face);
        //cleanup_flags_face.push_back(Metadata::FillGhost);
        //cleanup_flags_face.push_back(Metadata::GetUserFlag("SplitVector"));
        // Scalar potential, solution to del^2 p = div B
        pkg->AddField("p", Metadata(cleanup_flags_cell));
        // Gradient of potential; temporary for gradient calc
        pkg->AddField("dB", Metadata(cleanup_flags_face));
        // Field divergence as RHS, i.e. including boundary sync
        pkg->AddField("RHS_divB", Metadata(cleanup_flags_cell));
    } else {
        auto cleanup_flags_node = cleanup_flags;
        cleanup_flags_node.push_back(Metadata::FillGhost);
        cleanup_flags_node.push_back(Metadata::Node);
        auto cleanup_flags_cell = cleanup_flags;
        cleanup_flags_cell.push_back(Metadata::Cell);
        // Scalar potential, solution to del^2 p = div B
        pkg->AddField("p", Metadata(cleanup_flags_node));
        // Gradient of potential; temporary for gradient calc
        pkg->AddField("dB", Metadata(cleanup_flags_cell, s_vector));
        // Field divergence as RHS, i.e. including boundary sync
        pkg->AddField("RHS_divB", Metadata(cleanup_flags_node));
    }


    // Optionally take care of B field transport ourselves.  Inadvisable.
    // We've already set a default, so only do this if we're *explicitly* asked
    bool manage_field = pin->GetString("b_field", "solver") == "b_cleanup";
    params.Add("manage_field", manage_field);
    // Set an interval to clean during the run *can be run in addition to a normal solver*!
    // You might want to do this if, e.g., you care about divergence on faces with outflow/constant conditions
    int cleanup_interval = pin->GetOrAddInteger("b_cleanup", "cleanup_interval", manage_field ? 10 : -1);
    params.Add("cleanup_interval", cleanup_interval);

    if (manage_field) {
        // Copy in the field initialization from B_CT and/or B_FluxCT here to declare the right stuff
        throw std::runtime_error("B field cleanup/projection is set as B field transport! This is not implemented!");
    }

    return pkg;
}

bool B_Cleanup::CleanupThisStep(Mesh* pmesh, int nstep)
{
    auto pkg = pmesh->packages.Get("B_Cleanup");
    return (pkg->Param<int>("cleanup_interval") > 0) && (nstep % pkg->Param<int>("cleanup_interval") == 0);
}

// TODO(BSP) Make this add to a TaskCollection rather than operating synchronously
TaskStatus B_Cleanup::CleanupDivergence(std::shared_ptr<MeshData<Real>>& md)
{
    auto pmesh = md->GetMeshPointer();
    auto pkg = pmesh->packages.Get("B_Cleanup");
    auto max_iters = pkg->Param<int>("max_iterations");
    auto check_interval = pkg->Param<int>("check_interval");
    auto rel_tolerance = pkg->Param<Real>("rel_tolerance");
    auto abs_tolerance = pkg->Param<Real>("abs_tolerance");
    auto fail_flag = pkg->Param<bool>("fail_without_convergence");
    auto warn_flag = pkg->Param<bool>("warn_without_convergence");
    auto always_solve = pkg->Param<bool>("always_solve");
    auto solver = pkg->Param<BiCGStabSolver<int>>("solver");
    auto use_normalized = pkg->Param<bool>("use_normalized_divb");

    auto verbose = pmesh->packages.Get("Globals")->Param<int>("verbose");
    const bool use_b_ct = pmesh->packages.AllPackages().count("B_CT");

    if (MPIRank0() && verbose > 0) {
        std::cout << "Cleaning divB to absolute tolerance " << abs_tolerance <<
                     " OR relative tolerance " << rel_tolerance << std::endl;
        if (warn_flag) std::cout << "Convergence failure will produce a warning." << std::endl;
        if (fail_flag) std::cout << "Convergence failure will produce an error." << std::endl;
    }

    // Calculate/print inital max divB exactly as we would during run
    double divb_start; 
    if (use_b_ct) {
        divb_start = B_CT::GlobalMaxDivB(md.get());
    } else {
        divb_start = B_FluxCT::GlobalMaxDivB(md.get(), true);
    }
    if ((divb_start < abs_tolerance  || divb_start < rel_tolerance) && !always_solve) {
        // If divB is "pretty good" and we allow not solving...
        if (MPIRank0())
            std::cout << "Magnetic field divergence of " << divb_start << " is below tolerance. Skipping B field cleanup." << std::endl;
        return TaskStatus::complete;
    } else {
        if(MPIRank0())
            std::cout << "Starting magnetic field divergence: " << divb_start << std::endl;
    }

    // Add a solver container as a shallow copy on the default MeshData
    // msolve is just a sub-set of vars we need from md, making MPI syncs etc faster
    std::vector<std::string> names = KHARMA::GetVariableNames(&pmesh->packages, {Metadata::GetUserFlag("B_Cleanup"), Metadata::GetUserFlag("StartupOnly")});
    auto &msolve = pmesh->mesh_data.AddShallow("solve", md, names);

    // Initialize the divB variable, which we'll be solving against.
    if (use_b_ct) {
        B_CT::CalcDivB(md.get(), "RHS_divB"); // this fn draws from cons.fB, which is not in msolve
    } else {
        // This gets signed divB on all physical corners (total (N+1)^3)
        B_FluxCT::CalcDivB(md.get(), "RHS_divB"); // this fn draws from cons.B, which is not in msolve
    }
    if (use_normalized) {
        // Normalize divB by local metric determinant for fairer weighting of errors
        // Note that laplacian operator will also have to be normalized ofc
        auto divb_rhs = msolve->PackVariables(std::vector<std::string>{"RHS_divB"});
        auto pmb0 = msolve->GetBlockData(0)->GetBlockPointer();
        const IndexRange ib = msolve->GetBoundsI(IndexDomain::entire);
        const IndexRange jb = msolve->GetBoundsJ(IndexDomain::entire);
        const IndexRange kb = msolve->GetBoundsK(IndexDomain::entire);
        pmb0->par_for("normalize_divB", 0, divb_rhs.GetDim(5)-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
                const auto& G = divb_rhs.GetCoords(b);
                if (use_b_ct) {
                    divb_rhs(b, CC, 0, k, j, i) /= G.gdet(Loci::center, j, i);
                } else {
                    divb_rhs(b, NN, 0, k, j, i) /= G.gdet(Loci::corner, j, i);
                }
            }
        );
    }
    // make sure divB_RHS is sync'd
    KHARMADriver::SyncAllBounds(msolve);

    // Create a TaskCollection of just the solve,
    // execute it to perform BiCGStab iteration
    TaskID t_none(0);
    TaskCollection tc;
    auto& tr = tc.AddRegion(1);
    auto t_solve_step = solver.CreateTaskList(t_none, 0, tr, msolve, msolve);
    tc.Execute();
    // Make sure solution's ghost zones are sync'd
    KHARMADriver::SyncAllBounds(msolve);

    // Apply the result
    if (MPIRank0() && verbose > 0) {
        std::cout << "Applying magnetic field correction" << std::endl;
    }

    double divb_end;
    if (use_b_ct) {
        // Update the (conserved) magnetic field on physical zones using our solution
        B_Cleanup::ApplyPFace(md.get(), md.get());
        // Synchronize to update cons.B's ghost zones
        KHARMADriver::SyncAllBounds(md);
        // Make sure prims.B reflects solution
        B_CT::MeshUtoP(md.get(), IndexDomain::entire, false);
        // Recalculate divB max for one last check
        divb_end = B_CT::GlobalMaxDivB(md.get());
    } else {
        // Update the (conserved) magnetic field on physical zones using our solution
        B_Cleanup::ApplyPCenter(md.get(), md.get());
        // Synchronize to update cons.B's ghost zones
        KHARMADriver::SyncAllBounds(md);
        // Make sure prims.B reflects solution
        B_FluxCT::MeshUtoP(md.get(), IndexDomain::entire, false);
        // Recalculate divB max for one last check
        divb_end = B_FluxCT::GlobalMaxDivB(md.get());
    }

    if (MPIRank0()) {
        std::cout << "Magnetic field divergence after cleanup: " << divb_end << std::endl;
    }

    return TaskStatus::complete;
}

TaskStatus B_Cleanup::ApplyPCenter(MeshData<Real> *msolve, MeshData<Real> *md)
{
    // Apply on physical zones only, we'll be syncing/updating ghosts
    const IndexRange3 b = KDomain::GetRange(msolve, IndexDomain::interior, 0, 1);
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    auto P = msolve->PackVariables(std::vector<std::string>{"p"});
    auto B = md->PackVariables(std::vector<std::string>{"cons.B"});

    const int ndim = P.GetNdim();

    // dB = grad(p), defined at cell centers, subtract to make field divergence-free
    pmb0->par_for("gradient_P", 0, P.GetDim(5) - 1, b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
            const auto& G = P.GetCoords(b);
            double b1, b2, b3;
            B_FluxCT::center_grad(G, P(b), k, j, i, ndim > 2, b1, b2, b3);
            B(b, V1, k, j, i) -= b1;
            B(b, V2, k, j, i) -= b2;
            B(b, V3, k, j, i) -= b3;
        }
    );

    return TaskStatus::complete;
}
TaskStatus B_Cleanup::ApplyPFace(MeshData<Real> *msolve, MeshData<Real> *md)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    auto P = msolve->PackVariables(std::vector<std::string>{"p"});
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

TaskStatus B_Cleanup::CornerLaplacian(MeshData<Real>* md, const std::string& p_var, MeshData<Real>* md_again, const std::string& lap_var)
{
    auto pkg = md->GetMeshPointer()->packages.Get("B_Cleanup");
    const auto use_normalized = pkg->Param<bool>("use_normalized_divb");

    // Updating interior is easier to follow -- BiCGStab will sync
    const IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    const IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    auto P = md->PackVariables(std::vector<std::string>{p_var});
    auto lap = md->PackVariables(std::vector<std::string>{lap_var});
    auto dB = md->PackVariables(std::vector<std::string>{"dB"}); // Temp

    const int ndim = P.GetNdim();

    // P is defined on cell corners.  We need enough to take
    // grad -> center, then div -> corner, so one extra in each direction
    const IndexRange ib_l = IndexRange{ib.s-1, ib.e+1};
    const IndexRange jb_l = (ndim > 1) ? IndexRange{jb.s-1, jb.e+1} : jb;
    const IndexRange kb_l = (ndim > 2) ? IndexRange{kb.s-1, kb.e+1} : kb;
    // The div computes corner i,j,k, so needs to be [0,N+1] to cover all physical corners
    const IndexRange ib_r = IndexRange{ib.s, ib.e+1};
    const IndexRange jb_r = (ndim > 1) ? IndexRange{jb.s, jb.e+1} : jb;
    const IndexRange kb_r = (ndim > 2) ? IndexRange{kb.s, kb.e+1} : kb;

    // dB = grad(p), defined at cell centers
    pmb0->par_for("gradient_P", 0, P.GetDim(5) - 1, kb_l.s, kb_l.e, jb_l.s, jb_l.e, ib_l.s, ib_l.e,
        KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
            const auto& G = P.GetCoords(b);
            double b1, b2, b3;
            B_FluxCT::center_grad(G, P(b), k, j, i, ndim > 2, b1, b2, b3);
            dB(b, V1, k, j, i) = b1;
            dB(b, V2, k, j, i) = b2;
            dB(b, V3, k, j, i) = b3;
        }
    );

    // Replace ghost zone calculations with strict boundary conditions
    // Only necessary in j for poles so far, but maybe this should be the condition for outflow too?
    if (pmb0->coords.coords.is_spherical()) {
        for (int i=0; i < md->GetMeshPointer()->GetNumMeshBlocksThisRank(); i++) {
            auto rc = md->GetBlockData(i);
            auto pmb = rc->GetBlockPointer();
            auto dB_block = rc->PackVariables(std::vector<std::string>{"dB"});
            if (pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::user) {
                pmb->par_for("dB_boundary", kb_l.s, kb_l.e, ib_l.s, ib_l.e,
                    KOKKOS_LAMBDA (const int &k, const int &i) {
                        dB_block(V1, k, jb.s-1, i) = dB_block(V1, k, jb.s, i);
                        dB_block(V2, k, jb.s-1, i) = -dB_block(V2, k, jb.s, i);
                        dB_block(V3, k, jb.s-1, i) = dB_block(V3, k, jb.s, i);
                    }
                );
            }
            if (pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::user) {
                pmb->par_for("dB_boundary", kb_l.s, kb_l.e, ib_l.s, ib_l.e,
                    KOKKOS_LAMBDA (const int &k, const int &i) {
                        dB_block(V1, k, jb.e+1, i) = dB_block(V1, k, jb.e, i);
                        dB_block(V2, k, jb.e+1, i) = -dB_block(V2, k, jb.e, i);
                        dB_block(V3, k, jb.e+1, i) = dB_block(V3, k, jb.e, i);
                    }
                );
            }
        }
    }

    // lap = div(dB), defined at cell corners
    pmb0->par_for("laplacian_dB", 0, lap.GetDim(5) - 1, kb_r.s, kb_r.e, jb_r.s, jb_r.e, ib_r.s, ib_r.e,
        KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
            const auto& G = lap.GetCoords(b);
            // This is the inverse diagonal element of a fictional a_ij Laplacian operator
            lap(b, 0, k, j, i) = B_FluxCT::corner_div(G, dB(b), k, j, i, ndim > 2);
            if (use_normalized) {
                lap(b, 0, k, j, i) /= G.gdet(Loci::corner, j, i);
            }
        }
    );

    return TaskStatus::complete;
}

TaskStatus B_Cleanup::CenterLaplacian(MeshData<Real>* md, const std::string& p_var, MeshData<Real>* md_again, const std::string& lap_var)
{
    auto pkg = md->GetMeshPointer()->packages.Get("B_Cleanup");
    const auto use_normalized = pkg->Param<bool>("use_normalized_divb");

    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    auto P = md->PackVariables(std::vector<std::string>{p_var});
    auto lap = md->PackVariables(std::vector<std::string>{lap_var});
    auto dB = md->PackVariables(std::vector<std::string>{"dB"}); // Temp

    const int ndim = P.GetNdim();
    const IndexRange block = IndexRange{0, P.GetDim(5) - 1};

    // dB = grad(p), interpolating to faces
    // Do I know why these have to be ::entire?  No.  Does it work?  Yes.
    // TODO separate gradient functions since we're calculating directions separately anyway
    const IndexRange3 b1 = KDomain::GetRange(md, IndexDomain::entire, F1, 1, -1, false);
    pmb0->par_for("gradient_P", block.s, block.e, b1.ks, b1.ke, b1.js, b1.je, b1.is, b1.ie,
        KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
            const auto& G = P.GetCoords(b);
            dB(b, F1, 0, k, j, i) = B_CT::face_grad<X1DIR>(G, P(b), k, j, i);
        }
    );
    const IndexRange3 b2 = KDomain::GetRange(md, IndexDomain::entire, F2, 1, -1, false);
    pmb0->par_for("gradient_P", block.s, block.e, b2.ks, b2.ke, b2.js, b2.je, b2.is, b2.ie,
        KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
            const auto& G = P.GetCoords(b);
            dB(b, F2, 0, k, j, i) = B_CT::face_grad<X2DIR>(G, P(b), k, j, i);
        }
    );
    if (ndim > 2) {
        const IndexRange3 b3 = KDomain::GetRange(md, IndexDomain::entire, F3, 1, -1, false);
        pmb0->par_for("gradient_P", block.s, block.e, b3.ks, b3.ke, b3.js, b3.je, b3.is, b3.ie,
            KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
                const auto& G = P.GetCoords(b);
                dB(b, F3, 0, k, j, i) = B_CT::face_grad<X3DIR>(G, P(b), k, j, i);
            }
        );
    }

    // Make sure B on poles is zero
    if (pmb0->coords.coords.is_spherical()) {
        for (int i=0; i < md->GetMeshPointer()->GetNumMeshBlocksThisRank(); i++) {
            auto rc = md->GetBlockData(i);
            auto pmb = rc->GetBlockPointer();
            auto dB_block = rc->PackVariables(std::vector<std::string>{"dB"});
            const IndexRange3 bi2 = KDomain::GetRange(md, IndexDomain::interior, F2);
            if (pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::user) {
                pmb->par_for("dB_boundary", b2.ks, b2.ke, bi2.js, bi2.js, b2.is, b2.ie,
                    KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                        dB_block(F2, 0, k, j, i) = 0.;
                    }
                );
            }
            if (pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::user) {
                pmb->par_for("dB_boundary", b2.ks, b2.ke, bi2.je, bi2.je, b2.is, b2.ie,
                    KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                        dB_block(F2, 0, k, j, i) = 0.;
                    }
                );
            }
        }
    }

    // lap = div(dB), interpolating back to cell centers
    const IndexRange3 bc = KDomain::GetRange(md, IndexDomain::entire, CC, false);
    pmb0->par_for("laplacian_dB", block.s, block.e, bc.ks, bc.ke, bc.js, bc.je, bc.is, bc.ie,
        KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
            const auto& G = lap.GetCoords(b);
            // This is the inverse diagonal element of a fictional a_ij Laplacian operator
            lap(b, 0, k, j, i) = B_CT::face_div(G, dB(b), ndim, k, j, i);
            if (use_normalized) {
                lap(b, 0, k, j, i) /= G.gdet(Loci::corner, j, i);
            }
        }
    );

    // Make sure divB on outflows is 0
    // Our outflow conditions guarantee divergence-free last zones, so we shouldn't clean for them
    if (pmb0->packages.Get("Boundaries")->Param<std::string>("inner_x1") == "outflow") {
        for (int i=0; i < md->GetMeshPointer()->GetNumMeshBlocksThisRank(); i++) {
            auto rc = md->GetBlockData(i);
            auto pmb = rc->GetBlockPointer();
            auto lap_block = rc->PackVariables(std::vector<std::string>{lap_var});
            const IndexRange3 bic = KDomain::GetRange(md, IndexDomain::interior);
            if (pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user) {
                pmb->par_for("lap_boundary", bc.ks, bc.ke, bc.js, bc.je, bc.is, bic.is,
                    KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                        lap_block(0, k, j, i) = 0.;
                    }
                );
            }
        }
    }
    if (pmb0->packages.Get("Boundaries")->Param<std::string>("outer_x1") == "outflow") {
        for (int i=0; i < md->GetMeshPointer()->GetNumMeshBlocksThisRank(); i++) {
            auto rc = md->GetBlockData(i);
            auto pmb = rc->GetBlockPointer();
            auto lap_block = rc->PackVariables(std::vector<std::string>{lap_var});
            const IndexRange3 bic = KDomain::GetRange(md, IndexDomain::interior);
            if (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::user) {
                pmb->par_for("lap_boundary", bc.ks, bc.ke, bc.js, bc.je, bic.ie, bc.ie,
                    KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                        lap_block(0, k, j, i) = 0.;
                    }
                );
            }
        }
    }

    return TaskStatus::complete;
}

#endif
