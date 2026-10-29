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
std::shared_ptr<KHARMAPackage> Implicit::Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages)
{
    throw std::runtime_error("KHARMA was compiled without implicit stepping support!");
}
// We still need a stub for Step() in order to compile, but it will never be called
TaskStatus Implicit::Step(MeshData<Real>* md_full_step_init,
    MeshData<Real>* md_sub_step_init, MeshData<Real>* md_flux_src,
    MeshData<Real>* md_linesearch, MeshData<Real>* md_solver, const Real& dt)
{
    return TaskStatus::complete;
}

#else

// Parthenon's batched QR and solve.  No more kokkos-kernels!
#include "batched_linear_algebra/qr_decomposition.hpp"
#include "slices.hpp"

std::vector<std::string> Implicit::GetOrderedNames(
    MeshBlockData<Real>* rc, const MetadataFlag& flag, bool only_implicit)
{
    auto pmb0 = rc->GetBlockPointer();
    std::vector<std::string> out;
    auto vars = rc->GetVariablesByFlag({Metadata::GetUserFlag("Implicit"), flag}).vars();
    for (int i = 0; i < vars.size(); ++i) {
        if (rc->Contains(vars[i]->label())) {
            out.push_back(vars[i]->label());
        }
    }
    if (!only_implicit) {
        vars = rc->GetVariablesByFlag({Metadata::GetUserFlag("Explicit"), flag}).vars();
        for (int i = 0; i < vars.size(); ++i) {
            if (rc->Contains(vars[i]->label())) {
                out.push_back(vars[i]->label());
            }
        }
    }
    return out;
}

int Implicit::CountSolverFails(MeshData<Real>* md)
{
    return Reductions::CountFlags(
        md, "solve_fail", Implicit::status_names, IndexDomain::interior, false)[0];
}

std::shared_ptr<KHARMAPackage> Implicit::Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Implicit");
    Params& params = pkg->AllParams();

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
    // QR decomposition/solve from Parthenon's batched linear algebra
    // (Householder QR, no column pivoting), executed directly on the solver's
    // global fields. Per-zone temporaries are sized by MAX_VARS.

    bool linesearch = pin->GetOrAddBoolean("implicit", "linesearch", true);
    params.Add("linesearch", linesearch);
    int max_linesearch_iter = pin->GetOrAddInteger("implicit", "max_linesearch_iter", 3);
    params.Add("max_linesearch_iter", max_linesearch_iter);
    Real linesearch_eps = pin->GetOrAddReal("implicit", "linesearch_eps", 1.e-4);
    params.Add("linesearch_eps", linesearch_eps);
    Real linesearch_lambda = pin->GetOrAddReal("implicit", "linesearch_lambda", 1.0);
    params.Add("linesearch_lambda", linesearch_lambda);

    // Allocate the Jacobian and step so we can split the solver kernel
    auto resolved = StateDescriptor::CreateResolvedStateDescriptor(*packages);
    int nvars_implicit = resolved->GetPackDimension(Metadata::GetUserFlag("Implicit"));
    int nvars_explicit = resolved->GetPackDimension(Metadata::GetUserFlag("Explicit"));
    std::vector<int> s_vars_implicit({nvars_implicit});
    std::vector<int> s_jac_implicit({nvars_implicit, nvars_implicit});
    std::vector<int> s_vars_all({nvars_implicit + nvars_explicit});
    // Per-zone solver temporaries (residual/P saves, Householder scratch) are
    // fixed-size arrays of MAX_VARS components
    PARTHENON_REQUIRE(nvars_implicit <= MAX_VARS,
        "Too many implicitly-evolved variables for the implicit solver!");
    Metadata m =
        Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
            s_jac_implicit);
    pkg->AddField("Implicit.jacobian", m);
    // Q factor of the QR decomposition.  A field rather than scratch so the
    // decomposition can slice the global arrays directly
    pkg->AddField("Implicit.qr_factor", m);
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
        s_vars_implicit);
    pkg->AddField("Implicit.residual", m);
    // While the *guess* for delta_prim == -residual, we require *solved* delta_prim and
    // original residual in-flight at once
    pkg->AddField("Implicit.delta_prim", m);
    // We also need to carry around the implicit sources
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
        s_vars_all);
    pkg->AddField("Implicit.dU_implicit", m);

    // Allocate additional fields that reflect the success of the solver
    // L2 norm of the residual
    Metadata m_real =
        Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("solve_norm", m_real);
    // Integer field that saves where the solver fails (rho + drho < 0 || u + du < 0)
    m_real = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived,
        Metadata::OneCopy, Metadata::FillGhost});
    pkg->AddField("solve_fail",
        m_real); // TODO: Replace with m_int once Integer is supported for CellVariable

    // The major call, to Step(), is done manually from the ImEx driver
    // But, we just register the diagnostics function to print out solver failures
    pkg->PostStepDiagnosticsMesh = Implicit::PostStepDiagnostics;

    // List (vector) of HistoryOutputVars that will all be enrolled as output variables
    parthenon::HstVar_list hst_vars = {};
    // Count total floors as a history item
    hst_vars.emplace_back(parthenon::HistoryOutputVar(
        UserHistoryOperation::sum, CountSolverFails, "ImpSolverFails"));
    // add callbacks for HST output to the Params struct, identified by the
    // `hist_param_key`
    pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

    return pkg;
}

TaskStatus Implicit::Step(MeshData<Real>* md_full_step_init,
    MeshData<Real>* md_sub_step_init, MeshData<Real>* md_flux_src,
    MeshData<Real>* md_linesearch, MeshData<Real>* md_solver, const Real& dt)
{
    Flag("Implicit::Step");
    // Pull out the block pointers for each sub-step, as we need the *mutable parameters*
    // of the EMHD package.  TODO(CEP) restrict state back to the variables...
    auto pmb_full_step_init = md_full_step_init->GetBlockData(0)->GetBlockPointer();
    auto pmb_sub_step_init = md_sub_step_init->GetBlockData(0)->GetBlockPointer();
    auto pmb_solver = md_solver->GetBlockData(0)->GetBlockPointer();
    auto pmb_linesearch = md_linesearch->GetBlockData(0)->GetBlockPointer();

    // Parameters
    const auto& implicit_par = pmb_full_step_init->packages.Get("Implicit")->AllParams();
    const int iter_min = implicit_par.Get<int>("min_nonlinear_iter");
    const int iter_max = implicit_par.Get<int>("max_nonlinear_iter");
    const Real delta = implicit_par.Get<Real>("jacobian_delta");
    const Real rootfind_tol = implicit_par.Get<Real>("rootfind_tol");
    const auto& globals = pmb_full_step_init->packages.Get("Globals")->AllParams();
    const int verbose = globals.Get<int>("verbose");
    const int flag_verbose = globals.Get<int>("flag_verbose");

    const auto& eos_params = pmb_full_step_init->packages.Get("eos")->AllParams();
    auto eos = eos_params.Get<Microphysics::EOS::EOS>("d.EOS");

    const bool linesearch = implicit_par.Get<bool>("linesearch");
    const int max_linesearch_iter = implicit_par.Get<int>("max_linesearch_iter");
    const Real linesearch_eps = implicit_par.Get<Real>("linesearch_eps");
    const Real linesearch_lambda = implicit_par.Get<Real>("linesearch_lambda");

    // Misc other constants for inside the kernel
    const bool am_rank0 = MPIRank0();

    // We need two sets of emhd_params because we need the relaxation scale
    // at the same state in the implicit source terms
    // Need an object of `EMHD_parameters` for the `linesearch` state
    EMHD::EMHD_parameters emhd_params_sub_step_init, emhd_params_solver,
        emhd_params_linesearch;
    if (pmb_sub_step_init->packages.AllPackages().count("EMHD")) {
        const auto& pars_sub_step_init =
            pmb_sub_step_init->packages.Get("EMHD")->AllParams();
        const auto& pars_solver = pmb_solver->packages.Get("EMHD")->AllParams();
        const auto& pars_linesearch = pmb_linesearch->packages.Get("EMHD")->AllParams();
        emhd_params_sub_step_init =
            pars_sub_step_init.Get<EMHD::EMHD_parameters>("emhd_params");
        emhd_params_solver = pars_solver.Get<EMHD::EMHD_parameters>("emhd_params");
        emhd_params_linesearch =
            pars_linesearch.Get<EMHD::EMHD_parameters>("emhd_params");
    }

    // I don't normally do this, but we *really* care about variable ordering here.
    // The implicit variables need to be first, so we know how to iterate over just them
    // to fill just the residual & Jacobian we care about, which makes the solve faster.
    auto& mbd_full_step_init =
        md_full_step_init->GetBlockData(0); // MeshBlockData object, more member functions

    auto ordered_prims =
        GetOrderedNames(mbd_full_step_init.get(), Metadata::GetUserFlag("Primitive"));
    auto ordered_cons = GetOrderedNames(mbd_full_step_init.get(), Metadata::Conserved);
    // std::cerr << "Ordered prims:"; for(auto prim: ordered_prims) std::cerr << " " <<
    // prim; std::cerr << std::endl; std::cerr << "Ordered cons:"; for(auto con:
    // ordered_cons) std::cerr << " " << con; std::cerr << std::endl;

    // Initial state.  Also mapping template
    PackIndexMap prims_map, cons_map;
    auto& P_full_step_init_all =
        md_full_step_init->PackVariables(ordered_prims, prims_map);
    auto& U_full_step_init_all = md_full_step_init->PackVariables(ordered_cons, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    // Current sub-step starting state.
    auto& P_sub_step_init_all = md_sub_step_init->PackVariables(ordered_prims);
    auto& U_sub_step_init_all = md_sub_step_init->PackVariables(ordered_cons);
    // Flux divergence plus explicit source terms. This is what we'd be adding.
    auto& flux_src_all = md_flux_src->PackVariables(ordered_cons);
    // Guess at initial state. We update only the implicit primitive vars
    auto& P_solver_all = md_solver->PackVariables(ordered_prims);
    auto& P_linesearch_all = md_linesearch->PackVariables(ordered_prims);

    // Sizes and scratchpads
    const int nblock = U_full_step_init_all.GetDim(5);
    const int nvar = U_full_step_init_all.GetDim(4);
    // Get number of implicit variables
    auto implicit_vars = GetOrderedNames(
        mbd_full_step_init.get(), Metadata::GetUserFlag("Primitive"), true);
    // std::cerr << "Ordered implicit:"; for(auto var: implicit_vars) std::cerr << " " <<
    // var; std::cerr << std::endl;

    PackIndexMap implicit_prims_map;
    auto& P_full_step_init_implicit =
        md_full_step_init->PackVariables(implicit_vars, implicit_prims_map);
    const int nfvar = P_full_step_init_implicit.GetDim(4);

    // Pull fields associated with the solver's performance
    auto& solve_norm_all =
        md_solver->PackVariables(std::vector<std::string>{"solve_norm"});
    auto& solve_fail_all =
        md_solver->PackVariables(std::vector<std::string>{"solve_fail"});

    auto& jacobian_all =
        md_solver->PackVariables(std::vector<std::string>{"Implicit.jacobian"});
    auto& qr_factor_all =
        md_solver->PackVariables(std::vector<std::string>{"Implicit.qr_factor"});
    auto& delta_prim_all =
        md_solver->PackVariables(std::vector<std::string>{"Implicit.delta_prim"});
    auto& residual_all =
        md_solver->PackVariables(std::vector<std::string>{"Implicit.residual"});
    auto& dU_implicit_all =
        md_solver->PackVariables(std::vector<std::string>{"Implicit.dU_implicit"});

    auto bounds = pmb_sub_step_init->cellbounds;

    // RETURN if there aren't any implicit variables to evolve
    // TODO(CEP) probably redundant with not loading package, see kharma.cpp
    // std::cerr << "Solve size " << nfvar << " on prim size " << nvar << std::endl;
    if (nfvar == 0) return TaskStatus::complete;

    // The norm of the residual.  We store this to avoid the main kernel
    // also being a 2-stage reduction, which is complex and sucks.
    // TODO keep this around as a field?
    // ParArray4D<Real> norm_all("norm_all", nblock, n3, n2, n1); // EDIT

    // Get meshblock array bounds from Parthenon
    const IndexDomain domain = IndexDomain::interior;
    const IndexRange ib = bounds.GetBoundsI(domain);
    const IndexRange jb = bounds.GetBoundsJ(domain);
    const IndexRange kb = bounds.GetBoundsK(domain);
    const IndexRange block = IndexRange{0, nblock - 1};

    using QRDecomposition = parthenon::batched_linear_algebra::QRDecomposition;

    // Iterate.  This loop is outside the kokkos kernel in order to print max_norm
    // There are generally a low and similar number of iterations between
    // different zones, so probably acceptable speed loss.
    for (int iter = 1; iter <= iter_max; ++iter) {
        // Flags per iter, since debugging here will be rampant
        Flag("ImplicitIteration_" + std::to_string(iter));

        // Phase 1: Jacobian + residual calculation (and source terms)
        pmb_solver->par_for(
            "implicit_jacobian", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(const int& b, const int& k, const int& j, const int& i)
            {
                const auto& G = U_full_step_init_all.GetCoords(b);

                // Solver performance diagnostics.
                Real& solve_fail = solve_fail_all(b, 0, k, j, i);
                // Reset (locally and in View) at first iteration!
                if (iter == 1) solve_fail = SolverStatusR::converged;

                // Perform the solve only if it hadn't failed in any of the previous
                // iterations.
                if (solve_fail != SolverStatusR::fail) {
                    // Now that we know that it isn't a bad zone, reset solve_fail for
                    // this iteration
                    solve_fail = SolverStatusR::converged;

                    if (m_p.Q >= 0 || m_p.DP >= 0) {
                        Real throwaway;
                        Real& dUq =
                            (m_u.Q >= 0) ? dU_implicit_all(b, m_u.Q, k, j, i) : throwaway;
                        Real& dUdP = (m_u.DP >= 0) ? dU_implicit_all(b, m_u.DP, k, j, i)
                                                   : throwaway;
                        Real tau, chi_e, nu_e;
                        EMHD::set_parameters(G, P_sub_step_init_all(b), m_p,
                            emhd_params_sub_step_init, eos, k, j, i, tau, chi_e, nu_e);
                        EMHD::implicit_sources(G, P_full_step_init_all(b), m_p, eos, tau,
                            k, j, i, dUq, dUdP);
                    }

                    // Jacobian calculation
                    // Requires calculating the residual anyway, so we grab it here
                    calc_jacobian(G, P_solver_all(b), P_full_step_init_all(b),
                        U_full_step_init_all(b), P_sub_step_init_all(b), flux_src_all(b),
                        dU_implicit_all(b), m_p, m_u, emhd_params_solver,
                        emhd_params_sub_step_init, nvar, nfvar, k, j, i, delta, eos, dt,
                        jacobian_all(b), residual_all(b));
                }
#if SPLIT_IMPLICIT_SOLVE
            } // End lambda
        ); // End par_for

        // Phase 2: Linear solve via Parthenon's batched QR
        // Operates directly on global fields (no scratchpad copies needed)
        pmb_solver->par_for(
            "implicit_solve", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(const int& b, const int& k, const int& j, const int& i)
            {
                Real& solve_fail = solve_fail_all(b, 0, k, j, i);
#endif
                if (solve_fail != SolverStatusR::fail) {
                    // Linear solve via Parthenon's batched QR, operating on
                    // per-zone slices. The Jacobian is overwritten in-place with the R factor,
                    // Q lands in qr_factor, the rhs b is a *negated view* of the
                    // residual, and the solution x is written into delta_prim.
                    // Note rhs (b) and delta_prim (x) must not be aliased.
                    auto R = Slices::pack_matrix(jacobian_all, b, k, j, i, nfvar);
                    auto Q = Slices::pack_matrix(qr_factor_all, b, k, j, i, nfvar);
                    auto rhs = Slices::negated_pack_vector(residual_all, b, k, j, i);
                    auto delta_prim = Slices::pack_vector(delta_prim_all, b, k, j, i);
                    // Per-zone Householder scratch: double_scratch_size(nfvar) =
                    // 3*nfvar doubles. nfvar <= MAX_VARS enforced at Initialize
                    double qr_scratch[3 * MAX_VARS];
                    // factor A = Q R, then x = R^{-1} Q^T b
                    int status = QRDecomposition::execute(&R, &Q, qr_scratch);
                    status += QRDecomposition::solve(R, Q, rhs, delta_prim);
                    if (status != 0) {
                        // Singular/ill-conditioned Jacobian. Mark the zone failed so
                        // FixSolve can repair it.
                        solve_fail = SolverStatusR::fail;
                    }
                }

#if SPLIT_IMPLICIT_SOLVE
            } // End lambda
        ); // End par_for

        // Phase 3: Backtracking / linesearch / step update
        pmb_solver->par_for(
            "implicit_set_step", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(const int& b, const int& k, const int& j, const int& i)
            {
                const auto& G = U_full_step_init_all.GetCoords(b);
                Real& solve_fail = solve_fail_all(b, 0, k, j, i);
#endif
                // Solver performance diagnostics
                Real& solve_norm = solve_norm_all(b, 0, k, j, i);

                if (solve_fail == SolverStatusR::fail) return;

                // Copy `solver` prims to `linesearch`. This doesn't matter for the first
                // step of the solver since we do a copy in imex_driver just before, but
                // it is required for the subsequent iterations of the solver.
                if (iter > 1)
                    PLOOP
                        P_linesearch_all(b, ip, k, j, i) = P_solver_all(b, ip, k, j, i);

                // Check for positive definite values of density and internal energy.
                // Ignore zone if manual backtracking is not sufficient.
                // The primitives will be averaged over good neighbors.
                Real lambda = linesearch_lambda;
                if ((P_solver_all(b, m_p.RHO, k, j, i) +
                            lambda * delta_prim_all(b, m_p.RHO, k, j, i) <
                        0.) ||
                    (P_solver_all(b, m_p.UU, k, j, i) +
                            lambda * delta_prim_all(b, m_p.UU, k, j, i) <
                        0.)) {
                    solve_fail = SolverStatusR::backtrack;
                    lambda = 0.1;
                }
                if ((P_solver_all(b, m_p.RHO, k, j, i) +
                            lambda * delta_prim_all(b, m_p.RHO, k, j, i) <
                        0.) ||
                    (P_solver_all(b, m_p.UU, k, j, i) +
                            lambda * delta_prim_all(b, m_p.UU, k, j, i) <
                        0.)) {
                    solve_fail = SolverStatusR::fail;
                    // Set all fluid primitives to value at beginning of substep.
                    // We average over neighboring good zones later.
                    FLOOP P_solver_all(b, ip, k, j, i) =
                        P_sub_step_init_all(b, ip, k, j, i);
                    return;
                }

                // Assuming we did not fail...
                // Linesearch
                if (linesearch) {
                    solve_norm = 0.;
                    FLOOP solve_norm += SQR(residual_all(b, ip, k, j, i));
                    solve_norm = m::sqrt(solve_norm);

                    Real f0 = 0.5 * solve_norm;
                    Real fprime0 = -2. * f0;

                    for (int linesearch_iter = 0; linesearch_iter < max_linesearch_iter;
                        linesearch_iter++) {
                        // Take step
                        FLOOP P_linesearch_all(b, ip, k, j, i) =
                            P_solver_all(b, ip, k, j, i) +
                            (lambda * delta_prim_all(b, ip, k, j, i));

                        // Compute solve_norm of the residual (loss function)
                        calc_residual(G, P_linesearch_all(b), P_full_step_init_all(b),
                            U_full_step_init_all(b), P_sub_step_init_all(b),
                            flux_src_all(b), dU_implicit_all(b), m_p, m_u,
                            emhd_params_linesearch, emhd_params_solver, nfvar, k, j, i,
                            eos, dt, residual_all(b));

                        solve_norm = 0.;
                        FLOOP solve_norm += SQR(residual_all(b, ip, k, j, i));
                        solve_norm = m::sqrt(solve_norm);
                        Real f1 = 0.5 * solve_norm;

                        // Compute new step length
                        int condition =
                            f1 > (f0 * (1. - linesearch_eps * lambda) + SMALL_NUM);
                        Real denom =
                            (f1 - f0 - (fprime0 * lambda)) * condition + (1 - condition);
                        Real lambda_new = -fprime0 * lambda * lambda / denom * 0.5;
                        lambda = lambda * (1 - condition) + (condition * lambda_new);

                        // Check if new solution has converged within required tolerance
                        if (condition == 0) break;
                    }
                }

                // Update the guess
                FLOOP P_solver_all(b, ip, k, j, i) +=
                    lambda * delta_prim_all(b, ip, k, j, i);

                calc_residual(G, P_solver_all(b), P_full_step_init_all(b),
                    U_full_step_init_all(b), P_sub_step_init_all(b), flux_src_all(b),
                    dU_implicit_all(b), m_p, m_u, emhd_params_solver,
                    emhd_params_sub_step_init, nfvar, k, j, i, eos, dt, residual_all(b));

                // Store for maximum/output
                solve_norm = 0;
                FLOOP solve_norm += SQR(residual_all(b, ip, k, j, i));
                solve_norm = m::sqrt(solve_norm);

                // Did we converge to required tolerance? If not, update solve_fail
                // accordingly
                if (m::isnan(solve_norm)) {
                    // TODO(CEP) this can probably be detected/implemented alongside the
                    // floors above
                    solve_fail = SolverStatusR::fail;
                    FLOOP P_solver_all(b, ip, k, j, i) =
                        P_sub_step_init_all(b, ip, k, j, i);
                } else if (solve_norm > rootfind_tol) {
                    solve_fail =
                        SolverStatusR::beyond_tol; // TODO was changed from +=. Valid?
                }
            } // End lambda
        ); // End par_for

        // If we need to print or exit on the max norm...
        if (iter >= iter_min || verbose >= 1) {
            // Take the maximum L2 norm on this rank
            Real lmax_norm = 0.0;
            pmb_sub_step_init->par_reduce(
                "max_norm", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                KOKKOS_LAMBDA(const int& b, const int& k, const int& j, const int& i,
                    Real& local_result)
                {
                    if (solve_norm_all(b, 0, k, j, i) > local_result)
                        local_result = solve_norm_all(b, 0, k, j, i);
                },
                Kokkos::Max<Real>(lmax_norm));
            // Then MPI AllReduce to copy the global max to every rank
            Reductions::StartToAll<Real>(md_solver, 4, lmax_norm, MPI_MAX);
            Real max_norm = Reductions::CheckOnAll<Real>(md_solver, 4);

            if (verbose >= 1) {
                // Count total number of solver fails
                int lnfails = 0;
                pmb_sub_step_init->par_reduce(
                    "count_solver_fails", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s,
                    ib.e,
                    KOKKOS_LAMBDA(const int& b, const int& k, const int& j, const int& i,
                        int& local_result)
                    {
                        if (failed(solve_fail_all(b, 0, k, j, i))) ++local_result;
                    },
                    Kokkos::Sum<int>(lnfails));
                // Then reduce to rank 0 to print the iteration by iteration
                Reductions::Start<int>(md_solver, 5, lnfails, MPI_SUM);
                int nfails = Reductions::Check<int>(md_solver, 5);
                if (MPIRank0()) {
                    printf("Iteration %d max L2 norm: %g, failed zones: %d\n", iter,
                        max_norm, nfails);
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
    //     Reductions::StartFlagReduce(md_solver, "solve_fail", Implicit::status_names,
    //     IndexDomain::interior, false, 2);
    // }

    EndFlag();
    return TaskStatus::complete;
}

TaskStatus Implicit::PostStepDiagnostics(const SimTime& tm, MeshData<Real>* md)
{
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Options
    const auto& pars = pmesh->packages.Get("Globals")->AllParams();
    const int flag_verbose = pars.Get<int>("flag_verbose");

    // Debugging/diagnostic info about implicit solver
    Reductions::StartFlagReduce(
        md, "solve_fail", Implicit::status_names, IndexDomain::interior, false, 2);
    Reductions::CheckFlagReduceAndPrintHits(
        md, "solve_fail", Implicit::status_names, IndexDomain::interior, false, 2);

    // TODO(CEP) Die on too many solver failures (accuracy failures excepted)

    return TaskStatus::complete;
}

#endif
