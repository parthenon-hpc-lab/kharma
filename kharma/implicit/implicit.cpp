/* 
 *  File: implicit.cpp
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

#include "implicit.hpp"

#include "grmhd.hpp"
#include "grmhd_functions.hpp"
#include "kharma.hpp"
#include "pack.hpp"
#include "reductions.hpp"
#include "types.hpp"

#if DISABLE_IMPLICIT

// The package should never be loaded if there are not implicitly-evolved variables
// Therefore we yell at load time rather than waiting for the first solve
std::shared_ptr<KHARMAPackage> Implicit::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{ throw std::runtime_error("KHARMA was compiled without implicit stepping support!"); }
// We still need a stub for Step() in order to compile, but it will never be called
TaskStatus Implicit::Step(MeshData<Real> *md_full_step_init, MeshData<Real> *md_sub_step_init, MeshData<Real> *md_flux_src,
                MeshData<Real> *md_linesearch, MeshData<Real> *md_solver, const Real& dt) {}

#else

// Implicit nonlinear solve requires several linear solves per-zone
// Use Kokkos-kernels QR decomposition & triangular solve, they're fast.
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_QR_Decl.hpp>
#include <KokkosBatched_ApplyQ_Decl.hpp>
#include <KokkosBatched_Trsv_Decl.hpp>
#include <KokkosBatched_ApplyPivot_Decl.hpp>

std::vector<std::string> Implicit::GetOrderedNames(MeshBlockData<Real> *rc, const MetadataFlag& flag, bool only_implicit)
{
    auto pmb0 = rc->GetBlockPointer();
    std::vector<std::string> out;
    auto vars = rc->GetVariablesByFlag({Metadata::GetUserFlag("Implicit"), flag}).vars();
    for (int i=0; i < vars.size(); ++i) {
        if (rc->Contains(vars[i]->label())) {
            out.push_back(vars[i]->label());
        }
    }
    if (!only_implicit) {
        vars = rc->GetVariablesByFlag({Metadata::GetUserFlag("Explicit"), flag}).vars();
        for (int i=0; i < vars.size(); ++i) {
            if (rc->Contains(vars[i]->label())) {
                out.push_back(vars[i]->label());
            }
        }
    }
    return out;
}

int Implicit::CountSolverFails(MeshData<Real> *md)
{
    return Reductions::CountFlags(md, "solve_fail", Implicit::status_names, IndexDomain::interior, false)[0];
}

std::shared_ptr<KHARMAPackage> Implicit::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Implicit");
    Params &params = pkg->AllParams();

    // Implicit evolution must use predictor-corrector i.e. "vl2" integrator
    pin->SetString("parthenon/time", "integrator", "vl2");

    // Implicit solver parameters
    Real jacobian_delta = pin->GetOrAddReal("implicit", "jacobian_delta", 4.e-8);
    params.Add("jacobian_delta", jacobian_delta);
    Real rootfind_tol = pin->GetOrAddReal("implicit", "rootfind_tol", 1.e-12);
    params.Add("rootfind_tol", rootfind_tol);
    int min_nonlinear_iter = pin->GetOrAddInteger("implicit", "min_nonlinear_iter", 1);
    params.Add("min_nonlinear_iter", min_nonlinear_iter);
    int max_nonlinear_iter = pin->GetOrAddInteger("implicit", "max_nonlinear_iter", 3);
    params.Add("max_nonlinear_iter", max_nonlinear_iter);
    // The QR decomposition bundled with KHARMA has column pivoting for stability.
    // The alternative LU decomposition does not, and should mostly be used for debugging.
    bool use_qr = pin->GetOrAddBoolean("implicit", "use_qr", true);
    params.Add("use_qr", use_qr);

    bool linesearch = pin->GetOrAddBoolean("implicit", "linesearch", true);
    params.Add("linesearch", linesearch);
    int max_linesearch_iter = pin->GetOrAddInteger("implicit", "max_linesearch_iter", 3);
    params.Add("max_linesearch_iter", max_linesearch_iter);
    Real linesearch_eps = pin->GetOrAddReal("implicit", "linesearch_eps", 1.e-4);
    params.Add("linesearch_eps", linesearch_eps);
    Real linesearch_lambda = pin->GetOrAddReal("implicit", "linesearch_lambda", 1.0);
    params.Add("linesearch_lambda", linesearch_lambda);

    // Allocate the Jacobian and step so we can split the solver kernel
    int nvars_implicit = KHARMA::PackDimension(packages.get(), Metadata::GetUserFlag("Implicit"));
    int nvars_explicit = KHARMA::PackDimension(packages.get(), Metadata::GetUserFlag("Explicit"));
    std::vector<int> s_vars_implicit({nvars_implicit});
    std::vector<int> s_jac_implicit({nvars_implicit, nvars_implicit});
    std::vector<int> s_vars_all({nvars_implicit+nvars_explicit});
    Metadata m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_jac_implicit);
    pkg->AddField("Implicit.jacobian", m);
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_vars_implicit);
    pkg->AddField("Implicit.residual", m);
    // While the *guess* for delta_prim == -residual, we require *solved* delta_prim and original residual in-flight at once
    pkg->AddField("Implicit.delta_prim", m);
    // We also need to carry around the implicit sources
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_vars_all);
    pkg->AddField("Implicit.dU_implicit", m);

    // Allocate additional fields that reflect the success of the solver
    // L2 norm of the residual
    Metadata m_real = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("solve_norm", m_real);
    // Integer field that saves where the solver fails (rho + drho < 0 || u + du < 0)
    m_real = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost});
    pkg->AddField("solve_fail", m_real); // TODO: Replace with m_int once Integer is supported for CellVariable

    // The major call, to Step(), is done manually from the ImEx driver
    // But, we just register the diagnostics function to print out solver failures
    pkg->PostStepDiagnosticsMesh = Implicit::PostStepDiagnostics;

    // List (vector) of HistoryOutputVars that will all be enrolled as output variables
    parthenon::HstVar_list hst_vars = {};
    // Count total floors as a history item
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, CountSolverFails, "ImpSolverFails"));
    // add callbacks for HST output to the Params struct, identified by the `hist_param_key`
    pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

    return pkg;
}

TaskStatus Implicit::Step(MeshData<Real> *md_full_step_init, MeshData<Real> *md_sub_step_init, MeshData<Real> *md_flux_src,
                MeshData<Real> *md_linesearch, MeshData<Real> *md_solver, const Real& dt)
{
    Flag("Implicit::Step");
    // Pull out the block pointers for each sub-step, as we need the *mutable parameters*
    // of the EMHD package.  TODO(BSP) restrict state back to the variables...
    auto pmb_full_step_init = md_full_step_init->GetBlockData(0)->GetBlockPointer();
    auto pmb_sub_step_init  = md_sub_step_init->GetBlockData(0)->GetBlockPointer();
    auto pmb_solver         = md_solver->GetBlockData(0)->GetBlockPointer();
    auto pmb_linesearch     = md_linesearch->GetBlockData(0)->GetBlockPointer();

    // Parameters
    const auto& implicit_par = pmb_full_step_init->packages.Get("Implicit")->AllParams();
    const int iter_min       = implicit_par.Get<int>("min_nonlinear_iter");
    const int iter_max       = implicit_par.Get<int>("max_nonlinear_iter");
    const Real delta         = implicit_par.Get<Real>("jacobian_delta");
    const Real rootfind_tol  = implicit_par.Get<Real>("rootfind_tol");
    const bool use_qr        = implicit_par.Get<bool>("use_qr");
    const auto& globals      = pmb_full_step_init->packages.Get("Globals")->AllParams();
    const int verbose        = globals.Get<int>("verbose");
    const int flag_verbose   = globals.Get<int>("flag_verbose");
    const Real gam           = pmb_full_step_init->packages.Get("GRMHD")->Param<Real>("gamma");

    const bool linesearch         = implicit_par.Get<bool>("linesearch");
    const int max_linesearch_iter = implicit_par.Get<int>("max_linesearch_iter");
    const Real linesearch_eps     = implicit_par.Get<Real>("linesearch_eps");
    const Real linesearch_lambda  = implicit_par.Get<Real>("linesearch_lambda");

    // Misc other constants for inside the kernel
    const bool am_rank0 = MPIRank0();
    const Real tiny(SMALL), alpha(1.0);

    // We need two sets of emhd_params because we need the relaxation scale
    // at the same state in the implicit source terms
    // Need an object of `EMHD_parameters` for the `linesearch` state
    EMHD::EMHD_parameters emhd_params_sub_step_init, emhd_params_solver, emhd_params_linesearch;
    if (pmb_sub_step_init->packages.AllPackages().count("EMHD")) {
        const auto& pars_sub_step_init  = pmb_sub_step_init->packages.Get("EMHD")->AllParams();
        const auto& pars_solver         = pmb_solver->packages.Get("EMHD")->AllParams();
        const auto& pars_linesearch     = pmb_linesearch->packages.Get("EMHD")->AllParams();
        emhd_params_sub_step_init       = pars_sub_step_init.Get<EMHD::EMHD_parameters>("emhd_params");
        emhd_params_solver              = pars_solver.Get<EMHD::EMHD_parameters>("emhd_params");
        emhd_params_linesearch          = pars_linesearch.Get<EMHD::EMHD_parameters>("emhd_params");
    }

    // I don't normally do this, but we *really* care about variable ordering here.
    // The implicit variables need to be first, so we know how to iterate over just them to fill
    // just the residual & Jacobian we care about, which makes the solve faster.
    auto& mbd_full_step_init  = md_full_step_init->GetBlockData(0); // MeshBlockData object, more member functions
    
    auto ordered_prims = GetOrderedNames(mbd_full_step_init.get(), Metadata::GetUserFlag("Primitive"));
    auto ordered_cons  = GetOrderedNames(mbd_full_step_init.get(), Metadata::Conserved);
    //std::cerr << "Ordered prims:"; for(auto prim: ordered_prims) std::cerr << " " << prim; std::cerr << std::endl;
    //std::cerr << "Ordered cons:"; for(auto con: ordered_cons) std::cerr << " " << con; std::cerr << std::endl;

    // Initial state.  Also mapping template
    PackIndexMap prims_map, cons_map;
    auto& P_full_step_init_all = md_full_step_init->PackVariables(ordered_prims, prims_map);
    auto& U_full_step_init_all = md_full_step_init->PackVariables(ordered_cons, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    // Current sub-step starting state.
    auto& P_sub_step_init_all = md_sub_step_init->PackVariables(ordered_prims);
    auto& U_sub_step_init_all = md_sub_step_init->PackVariables(ordered_cons);
    // Flux divergence plus explicit source terms. This is what we'd be adding.
    auto& flux_src_all = md_flux_src->PackVariables(ordered_cons);
    // Guess at initial state. We update only the implicit primitive vars
    auto& P_solver_all     = md_solver->PackVariables(ordered_prims);
    auto& P_linesearch_all = md_linesearch->PackVariables(ordered_prims);

    // Sizes and scratchpads
    const int nblock = U_full_step_init_all.GetDim(5);
    const int nvar   = U_full_step_init_all.GetDim(4);
    // Get number of implicit variables
    auto implicit_vars = GetOrderedNames(mbd_full_step_init.get(), Metadata::GetUserFlag("Primitive"), true);
    //std::cerr << "Ordered implicit:"; for(auto var: implicit_vars) std::cerr << " " << var; std::cerr << std::endl;

    PackIndexMap implicit_prims_map;
    auto& P_full_step_init_implicit = md_full_step_init->PackVariables(implicit_vars, implicit_prims_map);
    const int nfvar = P_full_step_init_implicit.GetDim(4);

    // Pull fields associated with the solver's performance
    auto& solve_norm_all = md_solver->PackVariables(std::vector<std::string>{"solve_norm"});
    auto& solve_fail_all = md_solver->PackVariables(std::vector<std::string>{"solve_fail"});

    auto& jacobian_all = md_solver->PackVariables(std::vector<std::string>{"Implicit.jacobian"});
    auto& delta_prim_all = md_solver->PackVariables(std::vector<std::string>{"Implicit.delta_prim"});
    auto& residual_all = md_solver->PackVariables(std::vector<std::string>{"Implicit.residual"});
    auto& dU_implicit_all = md_solver->PackVariables(std::vector<std::string>{"Implicit.dU_implicit"});

    auto bounds  = pmb_sub_step_init->cellbounds;
    const int n1 = bounds.ncellsi(IndexDomain::entire);
    const int n2 = bounds.ncellsj(IndexDomain::entire);
    const int n3 = bounds.ncellsk(IndexDomain::entire);

    // RETURN if there aren't any implicit variables to evolve
    // TODO(BSP) probably redundant with not loading package, see kharma.cpp
    // std::cerr << "Solve size " << nfvar << " on prim size " << nvar << std::endl;
    if (nfvar == 0) return TaskStatus::complete;

    // The norm of the residual.  We store this to avoid the main kernel
    // also being a 2-stage reduction, which is complex and sucks.
    // TODO keep this around as a field?
    // ParArray4D<Real> norm_all("norm_all", nblock, n3, n2, n1); // EDIT

    // Get meshblock array bounds from Parthenon
    const IndexDomain domain = IndexDomain::interior;
    const IndexRange ib      = bounds.GetBoundsI(domain);
    const IndexRange jb      = bounds.GetBoundsJ(domain);
    const IndexRange kb      = bounds.GetBoundsK(domain);
    const IndexRange block   = IndexRange{0, nblock - 1};

    // Allocate scratch space
    // Only needed for the Kokkos-kernels solve, anymore
    // Otherwise we pull a bad hack and allocate constant temporaries
    const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
    const size_t tensor_size_in_bytes = parthenon::ScratchPad3D<Real>::shmem_size(n1, nfvar, nfvar);
    const size_t fvar_size_in_bytes   = parthenon::ScratchPad2D<Real>::shmem_size(n1, nfvar);
    const size_t fvar_int_size_in_bytes = parthenon::ScratchPad2D<int>::shmem_size(n1, nfvar);
    const size_t total_scratch_bytes = tensor_size_in_bytes + 4 * fvar_size_in_bytes + fvar_int_size_in_bytes;

    // Iterate.  This loop is outside the kokkos kernel in order to print max_norm
    // There are generally a low and similar number of iterations between
    // different zones, so probably acceptable speed loss.
    for (int iter=1; iter <= iter_max; ++iter) {
        // Flags per iter, since debugging here will be rampant
        Flag("ImplicitIteration_"+std::to_string(iter));

#if SPLIT_IMPLICIT_SOLVE
        pmb_solver->par_for("implicit_jacobian",
            block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(const int& b, const int& k, const int& j, const int& i) {
#else
        parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "implicit_solve", pmb_sub_step_init->exec_space,
            total_scratch_bytes, scratch_level, block.s, block.e, kb.s, kb.e, jb.s, jb.e,
            KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& b, const int& k, const int& j) {
                parthenon::par_for_inner(member, ib.s, ib.e,
                    [&](const int& i) {
#endif
                const auto& G = U_full_step_init_all.GetCoords(b);

                // Solver performance diagnostics.
                Real &solve_fail = solve_fail_all(b, 0, k, j, i);
                // Reset (locally and in View) at first iteration!
                if (iter == 1) solve_fail = SolverStatusR::converged;

                // Perform the solve only if it hadn't failed in any of the previous iterations.
                if (solve_fail != SolverStatusR::fail) {
                    // Now that we know that it isn't a bad zone, reset solve_fail for this iteration
                    solve_fail = SolverStatusR::converged;

                    if (m_p.Q >= 0 || m_p.DP >= 0) {
                        Real throwaway;
                        Real &dUq  = (m_u.Q >= 0)  ? dU_implicit_all(b, m_u.Q, k, j, i) : throwaway;
                        Real &dUdP = (m_u.DP >= 0) ? dU_implicit_all(b, m_u.DP, k, j, i): throwaway;
                        Real tau, chi_e, nu_e;
                        EMHD::set_parameters(G, P_sub_step_init_all(b), m_p, emhd_params_sub_step_init,
                                                gam, k, j, i, tau, chi_e, nu_e);
                        EMHD::implicit_sources(G, P_full_step_init_all(b), m_p,
                                                gam, tau, k, j, i, dUq, dUdP);
                    }

                    // Jacobian calculation
                    // Requires calculating the residual anyway, so we grab it here
                    calc_jacobian(G, P_solver_all(b), P_full_step_init_all(b), U_full_step_init_all(b), P_sub_step_init_all(b), 
                                flux_src_all(b), dU_implicit_all(b), m_p, m_u, emhd_params_solver, emhd_params_sub_step_init,
                                nvar, nfvar, k, j, i, delta, gam, dt,
                                jacobian_all(b), residual_all(b));
                }
#if SPLIT_IMPLICIT_SOLVE
            } // End lambda
        ); // End par_for

        parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "implicit_solve", pmb_sub_step_init->exec_space,
            total_scratch_bytes, scratch_level, block.s, block.e, kb.s, kb.e, jb.s, jb.e,
            KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& b, const int& k, const int& j) {
#else
                }); // End par_for_inner
                member.team_barrier();
#endif
                const auto& G = U_full_step_init_all.GetCoords(b);
                // Scratchpads for implicit vars
                ScratchPad3D<Real> jacobian_s(member.team_scratch(scratch_level), n1, nfvar, nfvar);
                ScratchPad2D<Real> delta_prim_s(member.team_scratch(scratch_level), n1, nfvar);
                ScratchPad2D<Real> trans_s(member.team_scratch(scratch_level), n1, nfvar);
                ScratchPad2D<Real> work_s(member.team_scratch(scratch_level), n1, 2*nfvar);
                ScratchPad2D<int> pivot_s(member.team_scratch(scratch_level), n1, nfvar);

                // Copy in to scratchpads
                FLOOP {
                    parthenon::par_for_inner(member, 0, n1-1,
                        [&](const int& i) {
                            delta_prim_s(i, ip) = -residual_all(b)(ip, k, j, i);
                        }
                    );
                }
                FLOOP2 {
                    parthenon::par_for_inner(member, 0, n1-1,
                        [&](const int& i) {
                            jacobian_s(i, ip, jp) = jacobian_all(b)(ip*nfvar+jp, k, j, i);
                        }
                    );
                }
                member.team_barrier();

                // TODO(BSP) even still worth keeping non-QR version?  Much less stable
                if (use_qr) {
                    parthenon::par_for_inner(member, ib.s, ib.e,
                        [&](const int& i) {
                            // Solver variables
                            auto jacobian   = Kokkos::subview(jacobian_s, i, Kokkos::ALL(), Kokkos::ALL());
                            auto delta_prim = Kokkos::subview(delta_prim_s, i, Kokkos::ALL());
                            auto pivot      = Kokkos::subview(pivot_s, i, Kokkos::ALL());
                            auto trans      = Kokkos::subview(trans_s, i, Kokkos::ALL());
                            auto work       = Kokkos::subview(work_s, i, Kokkos::ALL());

                            if (solve_fail_all(b, 0, k, j, i) != SolverStatusR::fail) {
                                // Linear solve by QR decomposition
                                KokkosBatched::SerialQR<KokkosBatched::Algo::QR::Unblocked>::invoke(jacobian, trans, pivot, work);
                                KokkosBatched::SerialApplyQ<KokkosBatched::Side::Left, KokkosBatched::Trans::Transpose,
                                                            KokkosBatched::Algo::ApplyQ::Unblocked>
                                ::invoke(jacobian, trans, delta_prim, work);
                                KokkosBatched::SerialTrsv<KokkosBatched::Uplo::Upper, KokkosBatched::Trans::NoTranspose, 
                                                        KokkosBatched::Diag::NonUnit, KokkosBatched::Algo::Trsv::Unblocked>
                                ::invoke(alpha, jacobian, delta_prim);
                                // Linear solve by QR decomposition
                                KokkosBatched::SerialApplyPivot<KokkosBatched::Side::Left,KokkosBatched::Direct::Backward>
                                    ::invoke(pivot, delta_prim);
                            }
                        }
                    );
                } else {
                    parthenon::par_for_inner(member, ib.s, ib.e,
                        [&](const int& i) {
                            // Solver variables
                            auto jacobian   = Kokkos::subview(jacobian_s, i, Kokkos::ALL(), Kokkos::ALL());
                            auto delta_prim = Kokkos::subview(delta_prim_s, i, Kokkos::ALL());

                            if (solve_fail_all(b, 0, k, j, i) != SolverStatusR::fail) {
                                KokkosBatched::SerialLU<KokkosBatched::Algo::LU::Unblocked>::invoke(jacobian, tiny);
                                KokkosBatched::SerialTrsv<KokkosBatched::Uplo::Upper, KokkosBatched::Trans::NoTranspose, 
                                                        KokkosBatched::Diag::NonUnit, KokkosBatched::Algo::Trsv::Unblocked>
                                ::invoke(alpha, jacobian, delta_prim);
                            }
                        }
                    );
                }
                member.team_barrier();

                // Copy out delta_prim
                FLOOP {
                    parthenon::par_for_inner(member, ib.s, ib.e,
                        [&](const int& i) {
                            delta_prim_all(b)(ip, k, j, i) = delta_prim_s(i, ip);
                        }
                    );
                }
#if SPLIT_IMPLICIT_SOLVE
            } // End lambda
        ); // End par_for

        pmb_solver->par_for("implicit_set_step",
            block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(const int& b, const int& k, const int& j, const int& i) {
#else
                member.team_barrier();
                parthenon::par_for_inner(member, ib.s, ib.e,
                    [&](const int& i) {
#endif
                const auto& G = U_full_step_init_all.GetCoords(b);

                // Solver performance diagnostics
                Real &solve_norm = solve_norm_all(b, 0, k, j, i);
                Real &solve_fail = solve_fail_all(b, 0, k, j, i);

                if (solve_fail == SolverStatusR::fail) return;

                // Copy `solver` prims to `linesearch`. This doesn't matter for the first step of the solver
                // since we do a copy in imex_driver just before, but it is required for the subsequent
                // iterations of the solver.
                if (iter > 1)
                   PLOOP P_linesearch_all(b, ip, k, j, i) = P_solver_all(b, ip, k, j, i);

                // Check for positive definite values of density and internal energy.
                // Ignore zone if manual backtracking is not sufficient.
                // The primitives will be averaged over good neighbors.
                Real lambda = linesearch_lambda;
                if ((P_solver_all(b, m_p.RHO, k, j, i) + lambda*delta_prim_all(b, m_p.RHO, k, j, i) < 0.) ||
                    (P_solver_all(b, m_p.UU, k, j, i) + lambda*delta_prim_all(b, m_p.UU, k, j, i) < 0.)) {
                    solve_fail = SolverStatusR::backtrack;
                    lambda       = 0.1;
                }
                if ((P_solver_all(b, m_p.RHO, k, j, i) + lambda*delta_prim_all(b, m_p.RHO, k, j, i) < 0.) ||
                    (P_solver_all(b, m_p.UU, k, j, i) + lambda*delta_prim_all(b, m_p.UU, k, j, i) < 0.)) {
                    solve_fail = SolverStatusR::fail;
                    // Set all fluid primitives to value at beginning of substep.
                    // We average over neighboring good zones later.
                    FLOOP P_solver_all(b, ip, k, j, i) = P_sub_step_init_all(b, ip, k, j, i);
                    return;
                }

                // Assuming we did not fail...
                // Linesearch
                if (linesearch) {
                    solve_norm        = 0.;
                    FLOOP solve_norm += SQR(residual_all(b, ip, k, j, i));
                    solve_norm        = m::sqrt(solve_norm);

                    Real f0      = 0.5 * solve_norm;
                    Real fprime0 = -2. * f0;

                    for (int linesearch_iter = 0; linesearch_iter < max_linesearch_iter; linesearch_iter++) {
                        // Take step
                        FLOOP P_linesearch_all(b, ip, k, j, i) =
                                P_solver_all(b, ip, k, j, i) + (lambda * delta_prim_all(b, ip, k, j, i));

                        // Compute solve_norm of the residual (loss function)
                        calc_residual(G, P_linesearch_all(b), P_full_step_init_all(b), U_full_step_init_all(b),
                                    P_sub_step_init_all(b), flux_src_all(b), dU_implicit_all(b), 
                                    m_p, m_u, emhd_params_linesearch, emhd_params_solver, nfvar,
                                    k, j, i, gam, dt, residual_all(b));

                        solve_norm        = 0.;
                        FLOOP solve_norm += SQR(residual_all(b, ip, k, j, i));
                        solve_norm        = m::sqrt(solve_norm);
                        Real f1             = 0.5 * solve_norm;

                        // Compute new step length
                        int condition   = f1 > (f0 * (1. - linesearch_eps * lambda) + SMALL);
                        Real denom      = (f1 - f0 - (fprime0 * lambda)) * condition + (1 - condition);
                        Real lambda_new = -fprime0 * lambda * lambda / denom * 0.5;
                        lambda          = lambda * (1 - condition) + (condition * lambda_new);

                        // Check if new solution has converged within required tolerance
                        if (condition == 0) break;                           
                    }
                }

                // Update the guess
                FLOOP P_solver_all(b, ip, k, j, i) += lambda * delta_prim_all(b, ip, k, j, i);

                calc_residual(G, P_solver_all(b), P_full_step_init_all(b), U_full_step_init_all(b),
                            P_sub_step_init_all(b), flux_src_all(b), dU_implicit_all(b),
                            m_p, m_u, emhd_params_solver, emhd_params_sub_step_init, nfvar,
                            k, j, i, gam, dt, residual_all(b));

                // Store for maximum/output
                solve_norm        = 0;
                FLOOP solve_norm += SQR(residual_all(b, ip, k, j, i));
                solve_norm        = m::sqrt(solve_norm);

                // Did we converge to required tolerance? If not, update solve_fail accordingly
                if (m::isnan(solve_norm)) {
                    // TODO(BSP) this can probably be detected/implemented alongside the floors above
                    solve_fail = SolverStatusR::fail;
                    FLOOP P_solver_all(b, ip, k, j, i) = P_sub_step_init_all(b, ip, k, j, i);
                } else if (solve_norm > rootfind_tol) {
                    solve_fail = SolverStatusR::beyond_tol; // TODO was changed from +=. Valid?
                }
#if SPLIT_IMPLICIT_SOLVE
            } // End lambda
        ); // End par_for
#else
                }); // End par_for_inner
            } // End lambda
        ); // End par_for
#endif
        // If we need to print or exit on the max norm...
        if (iter >= iter_min || verbose >= 1) {
            // Take the maximum L2 norm on this rank
            Real lmax_norm = 0.0;
            pmb_sub_step_init->par_reduce("max_norm", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                KOKKOS_LAMBDA (const int& b, const int& k, const int& j, const int& i, Real& local_result) {
                    if (solve_norm_all(b, 0, k, j, i) > local_result) local_result = solve_norm_all(b, 0, k, j, i);
                }
            , Kokkos::Max<Real>(lmax_norm));
            // Then MPI AllReduce to copy the global max to every rank
            Reductions::StartToAll<Real>(md_solver, 4, lmax_norm, MPI_MAX);
            Real max_norm = Reductions::CheckOnAll<Real>(md_solver, 4);

            if (verbose >= 1) {
                // Count total number of solver fails
                int lnfails = 0;
                pmb_sub_step_init->par_reduce("count_solver_fails", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                    KOKKOS_LAMBDA (const int& b, const int& k, const int& j, const int& i, int& local_result) {
                        if (failed(solve_fail_all(b, 0, k, j, i))) ++local_result;
                    }
                , Kokkos::Sum<int>(lnfails));
                // Then reduce to rank 0 to print the iteration by iteration
                Reductions::Start<int>(md_solver, 5, lnfails, MPI_SUM);
                int nfails = Reductions::Check<int>(md_solver, 5);
                if (MPIRank0()) {
                    printf("Iteration %d max L2 norm: %g, failed zones: %d\n", iter, max_norm, nfails);
                }
            }

            // Finally, break if max_norm is less than the total tolerance we set
            // TODO per-zone tolerance with masks?
            if (iter >= iter_min && max_norm < rootfind_tol) {
                EndFlag();
                break;
            }
        }
        EndFlag();
    }

    // if (flag_verbose > 0) {
    //     // Start the reduction as soon as we have the data
    //     // Dangerous, so commented
    //     Reductions::StartFlagReduce(md_solver, "solve_fail", Implicit::status_names, IndexDomain::interior, false, 2);
    // }

    EndFlag();
    return TaskStatus::complete;

}

TaskStatus Implicit::PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md)
{
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Options
    const auto& pars = pmesh->packages.Get("Globals")->AllParams();
    const int flag_verbose = pars.Get<int>("flag_verbose");

    // Debugging/diagnostic info about implicit solver
    if (flag_verbose > 0) {
        Reductions::StartFlagReduce(md, "solve_fail", Implicit::status_names, IndexDomain::interior, false, 2);
        Reductions::CheckFlagReduceAndPrintHits(md, "solve_fail", Implicit::status_names, IndexDomain::interior, false, 2);
    }

    return TaskStatus::complete;
}

#endif
