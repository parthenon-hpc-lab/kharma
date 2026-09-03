/* 
 *  File: force_free.cpp
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
#include "force_free.hpp"

#include <parthenon/parthenon.hpp>

#include "decs.hpp"
#include "domain.hpp"
#include "entropy.hpp"
#include "floors.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"

using namespace parthenon;

int Force_Free::CountFFFlags(MeshData<Real> *md)
{
    return Reductions::CountFlags(md, "flags.force_free", Force_Free::flag_names, IndexDomain::interior, true)[0];
}

std::shared_ptr<KHARMAPackage> Force_Free::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Force_Free");
    Params &params = pkg->AllParams();

    // TODO Will be a while before we evolve non-hybrid.
    // Theoretically, we disable the GRMHD package but leave B, and provide uvec == drift vel
    bool hybrid = pin->GetOrAddBoolean("force_free", "hybrid", packages->AllPackages().count("GRMHD"));
    params.Add("do_hybrid", hybrid);

    if (hybrid) {
        // TODO can we push farther with GRMHD?  Should we?
        Real hybrid_sigcut = pin->GetOrAddReal("force_free", "hybrid_sigcut", 100.);
        params.Add("hybrid_sigcut", hybrid_sigcut);
        // TODO not used yet
        Real hybrid_sigcut_width = pin->GetOrAddReal("force_free", "hybrid_sigcut_width", 0.1);
        params.Add("hybrid_sigcut_width", hybrid_sigcut_width);

        Real ffinv_lower_cutoff, ffinv_upper_cutoff;
        if (hybrid_sigcut_width <= 0.) { // step function cutoff
            ffinv_lower_cutoff = hybrid_sigcut;
            ffinv_upper_cutoff = hybrid_sigcut;
        } else { //cutoff at f(sigma) = 1/64 and f(sigma) = 63/64
            Real fac = m::pow(3., hybrid_sigcut_width) * m::pow(7., 0.5 * hybrid_sigcut_width);
            ffinv_upper_cutoff = hybrid_sigcut * fac;
            ffinv_lower_cutoff = hybrid_sigcut / fac;
        }
        params.Add("ffinv_lower_cutoff", ffinv_lower_cutoff);
        params.Add("ffinv_upper_cutoff", ffinv_upper_cutoff);
    }

    // which equations to start with for parallel solver?
    // Choice is the *attempted* solve, will cascade upward to 5 at worst
#ifdef FORCE_FREE_COLD
    std::vector<std::string> parallel_theory_allowed_vals{"cold"};
#else
    std::vector<std::string> parallel_theory_allowed_vals{"zero", "entropy", "entropy_mhd", "hot_mhd"};
#endif
    std::string parallel_theory_s = pin->GetOrAddString("force_free", "parallel_theory", "zero", parallel_theory_allowed_vals);
    if (parallel_theory_s == "zero") {
        params.Add("parallel_theory", ParallelTheory::zero);
    } else if (parallel_theory_s == "cold") {
        params.Add("parallel_theory", ParallelTheory::cold);
    } else if (parallel_theory_s == "entropy") {
        params.Add("parallel_theory", ParallelTheory::entropy);
    } else if (parallel_theory_s == "entropy_mhd") {
        params.Add("parallel_theory", ParallelTheory::entropy_mhd);
    } else if (parallel_theory_s == "hot_mhd") {
        params.Add("parallel_theory", ParallelTheory::hot_mhd);
    }
    // TODO check this is what KORAL uses in practice
    bool zero_parallel_vel_bh = pin->GetOrAddBoolean("force_free", "zero_parallel_vel_bh", true);
    params.Add("zero_parallel_vel_bh", zero_parallel_vel_bh);

    // FIELDS
    // Vector size: 3x[grid shape]
    std::vector<int> s_vector({NVEC});
    // Flags: always explicit, sync like GRMHD variables (TODO pretty sure we actually want to always sync cons)
    std::vector<MetadataFlag> flags_grmhd = {Metadata::Cell, Metadata::GetUserFlag("Explicit")};
    auto& driver = packages->Get("Driver")->AllParams();
    auto flags_prim = driver.Get<std::vector<MetadataFlag>>("prim_flags");
    flags_prim.insert(flags_prim.end(), flags_grmhd.begin(), flags_grmhd.end());
    auto flags_cons = driver.Get<std::vector<MetadataFlag>>("cons_flags");
    flags_cons.insert(flags_cons.end(), flags_grmhd.begin(), flags_grmhd.end());

    // We must additionally save the primtive variables as the "seed" for the next U->P solve
    // TODO Necessary?  Only if a numerically solved eqn is enabled?
    flags_prim.push_back(Metadata::Restart);

    // Scalars
    // Primitive: parallel velocity
    auto m = Metadata(flags_prim);
    pkg->AddField("prims.ff.upar", m);
    // Conserved: mu b^0
    m = Metadata(flags_cons);
    pkg->AddField("cons.ff.upar", m);

    // Note we also require the specific entropy prims.Ktot & total entropy cons.Ktot,
    // provided by the Entropy package

    // Vectors
    flags_prim.push_back(Metadata::Vector);
    flags_cons.push_back(Metadata::Vector);

    m = Metadata(flags_prim, s_vector);
    pkg->AddField("prims.ff.uvec", m);
    m = Metadata(flags_cons, s_vector);
    pkg->AddField("cons.ff.uvec", m);

    // Additional cache of local magnetization, decides FF zones for next step.
    m = Metadata({Metadata::Real, Metadata::Derived, Metadata::Cell, Metadata::OneCopy});
    pkg->AddField("Force_Free.sigma", m);

    // Flag for status of force-free operation.  Records which zones are FF, and inversion results
    pkg->AddField("flags.force_free", m);

    // CALLBACKS

    // Also ensure that prims get filled
    pkg->MeshUtoP = Force_Free::MeshUtoP;
    pkg->BlockUtoP = Force_Free::BlockUtoP;

    // This doesn't just apply our floors, it also resets the FF variables based on the current
    // GRMHD state (and calculates sigma for determining where to use FF in next UtoP)
    // TODO should probably split that...
    pkg->BlockApplyFloors = Force_Free::ApplyFloors;

    // This is the function that resets FF variables from GRMHD prims, which is what we want 
    //pkg->BoundaryUtoP = Force_Free::ApplyFloors;

    // Register the other callbacks
    pkg->PostStepDiagnosticsMesh = Force_Free::PostStepDiagnostics;

    // TODO history/stats
    // // List (vector) of HistoryOutputVars that will all be enrolled as output variables
    // parthenon::HstVar_list hst_vars = {};
    // hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, B_FluxCT::MaxDivB, "MaxDivB"));
    // // Event horizon magnetization.  Might be the same or different for different representations?
    // if (pin->GetBoolean("coordinates", "spherical")) {
    //     hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, ReducePhi0, "Phi_0"));
    //     hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, ReducePhi5, "Phi_EH"));
    // }
    // // add callbacks for HST output to the Params struct, identified by the `hist_param_key`
    // pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

    return pkg;
}

void Force_Free::ApplyFloors(MeshBlockData<Real> *mbd, IndexDomain domain)
{
    auto pmb = mbd->GetBlockPointer();

    // We pack all primitive vars here so we can get rho to calculate sigma
    PackIndexMap prims_map;
    auto P = mbd->PackVariables(std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prims_map);
    const VarMap m_p(prims_map, false);
    auto sigma = mbd->PackVariables(std::vector<std::string>{"Force_Free.sigma"});

    const auto& G = pmb->coords;

    // TODO RECORD FLOOR HITS

    IndexRange3 b = KDomain::GetRange(mbd, domain);

    // Update all Force-free equation state based on current (floored) GRMHD
    // primitive variable state
    // Also record sigma to determine which cells are FF next step
    pmb->par_for("Reset_Force_Free", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            // Update the primitive FF uvec
            FourVectors D;
            GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, D);
            const Real gamma = GRMHD::lorentz_calc(G, P, m_p, k, j, i);
            const Real inv_gamma2 = 1. / SQR(gamma);
  
            // take dot product and find perpindicular gamma
            const Real Bsq = dot(D.bcon, D.bcov);
            sigma(0, k, j, i) = Bsq / P(m_p.RHO, k, j, i);
            const Real Bmag  = m::sqrt(Bsq);
            const Real udotB = dot(D.ucon,D.bcov); // = gamma*|B|*vpar
            const Real vpar = udotB / (gamma*Bmag);
            const Real gammapar = 1. / m::sqrt(1 - SQR(vpar));
            P(m_p.UUFF, k, j, i) = gammapar * vpar;
  
            const Real inv_gammaperp2 = inv_gamma2  + vpar*vpar; 
            const Real gammaperp = 1. / m::sqrt(inv_gammaperp2);

            // find perpindicular 4-velocity
            P(m_p.U1FF, k, j, i) = gammaperp * ((D.ucon[1] / gamma) - vpar * D.bcon[1] / Bmag);
            P(m_p.U2FF, k, j, i) = gammaperp * ((D.ucon[2] / gamma) - vpar * D.bcon[2] / Bmag);
            P(m_p.U3FF, k, j, i) = gammaperp * ((D.ucon[3] / gamma) - vpar * D.bcon[3] / Bmag);
        }
    );

    // Then apply any of our own floors
    pmb->par_for("Floors_Force_Free", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            // Limit vpar
            Real abs_vpar = m::abs(P(m_p.UUFF, k, j, i));
            if(abs_vpar > 1) {
                P(m_p.UUFF, k, j, i) /= abs_vpar;
            }

            // TODO take GAMMA_MAX from floors, calc gamma_perp etc as below
            // Limit vperp
            // Real vsqmax = 1. - 1. / SQR(GAMMA_MAX);
            // if(!isfinite(gamma_perp) || vsq_perp > vsqmax) {
            //     ldouble vfac = m::sqrt(vsqmax/vsq_perp);
            //     vcon_perp[1] *= vfac;
            //     vcon_perp[2] *= vfac;
            //     vcon_perp[3] *= vfac;
            //     vsq_perp = vsqmax;
            //     gamma2_perp = 1. / (1 - vsq_perp);
            //     gamma_perp = m::sqrt(gamma2_perp);
            //     hitgammaceil=1;
            // }
        }
    );

    // Also update the entropy -- pretty sure the new package takes care of this tho
    //Entropy::MeshUpdateEntropy(mbd);
}

TaskStatus Force_Free::BlockUtoP(MeshBlockData<Real> *mbd, IndexDomain domain, bool coarse)
{
    auto pmb = mbd->GetBlockPointer();

    PackIndexMap prims_map, cons_map;
    auto U = mbd->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map);
    auto P = mbd->PackVariables(std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prims_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    auto sigma = mbd->PackVariables(std::vector<std::string>{"Force_Free.sigma"});

    auto ffflag = mbd->PackVariables(std::vector<std::string>{"flags.force_free"});

    const auto& G = pmb->coords;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    auto& params = pmb->packages.Get("Force_Free")->AllParams();
    const Real sigcut = params.Get<Real>("hybrid_sigcut");
    const ParallelTheory parallel_theory_chosen = params.Get<ParallelTheory>("parallel_theory");
    const bool zero_parallel_vel_bh = params.Get<bool>("zero_parallel_vel_bh");;
    const Real r_horizon = G.coords.get_horizon();

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const IndexRange ib = bounds.GetBoundsI(domain);
    const IndexRange jb = bounds.GetBoundsJ(domain);
    const IndexRange kb = bounds.GetBoundsK(domain);
    pmb->par_for("UtoP_MHD_From_FF", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            // Skip if we're not initialized or if sigma is low (most zones!)
            // TODO generalize as floors_force_free?
            if (sigma(0, k, j, i) == 0. || sigma(0, k, j, i) < sigcut) {
                ffflag(0, k, j, i) = (Real) FFFlag::none;
                return;
            }

            // FORCE-FREE SOLVE

            // Convert from conserved variables to four-vectors
            const Real alpha  = 1. / m::sqrt(-G.gcon(Loci::center, j, i, 0, 0));
            const Real alpha2 = 1. / (-G.gcon(Loci::center, j, i, 0, 0));
            const Real gdet = G.gdet(Loci::center, j, i);
            const Real a_over_g = alpha / gdet;
            // beta vector components
            double beta[GR_DIM] = {0};
            beta[1] = alpha2 * G.gcon(Loci::center, j, i, 0, 1);
            beta[2] = alpha2 * G.gcon(Loci::center, j, i, 0, 2);
            beta[3] = alpha2 * G.gcon(Loci::center, j, i, 0, 3);

            // McKinney defn of B^mu = B^mu HARM * alpha
            // We can assume primitive B is filled
            double Bcon[GR_DIM], Bcov[GR_DIM];
            Bcon[0] = 0.;
            Bcon[1] = U(m_u.B1, k, j, i) * a_over_g;
            Bcon[2] = U(m_u.B2, k, j, i) * a_over_g;
            Bcon[3] = U(m_u.B3, k, j, i) * a_over_g;
            G.lower(Bcon, Bcov, k, j, i, Loci::center);
            const Real Bsq = dot(Bcon, Bcov);
            const Real Bmag = m::sqrt(Bsq);

            // Q_mu = alpha*T^t_mu 
            Real Qcov[GR_DIM];
            Qcov[0] = 0.; 
            Qcov[1] = U(m_u.U1FF, k, j, i) * a_over_g;
            Qcov[2] = U(m_u.U2FF, k, j, i) * a_over_g;
            Qcov[3] = U(m_u.U3FF, k, j, i) * a_over_g;
            
            // Qtilde_mu = Q_mu + n_mu (n.Q)
            Real Qtildecov[GR_DIM], Qtildecon[GR_DIM];
            Qtildecov[0] = Qcov[1]*beta[1] + Qcov[2]*beta[2] + Qcov[3]*beta[3];
            Qtildecov[1] = Qcov[1];
            Qtildecov[2] = Qcov[2];
            Qtildecov[3] = Qcov[3];
            G.raise(Qtildecov, Qtildecon, k, j, i, Loci::center);

            // get three velocity
            Real vcon_perp[GR_DIM], vcov_perp[GR_DIM];
            vcon_perp[0] = 0.;
            vcon_perp[1] = Qtildecon[1] / Bsq;
            vcon_perp[2] = Qtildecon[2] / Bsq;
            vcon_perp[3] = Qtildecon[3] / Bsq;
            G.lower(vcon_perp, vcov_perp, k, j, i, Loci::center);
            const Real vsq_perp = dot(vcon_perp, vcov_perp);
            const Real gamma2_perp = 1. / (1 - vsq_perp);
            const Real gamma_perp = sqrt(gamma2_perp);

            // HYDRO SOLVE
            Real inv_gamma2_perp = 1. / gamma2_perp;
            Real afac = (gam - 1.) / gam;

            // MHD conserveds
            Real D  = U(m_u.RHO, k, j, i) * a_over_g; // uu[RHO] = gdet rho ut, so D = gamma * rho
            Real Sc = U(m_u.KTOT, k, j, i) * a_over_g; // uu[ENTR] = gdet S ut, so Sc = gamma * S

            // We try to stick to the chosen theory, but cascade fallbacks for stability
            ParallelTheory parallel_theory = parallel_theory_chosen;

            // no parallel velocity under BH
            if (zero_parallel_vel_bh && G.r(k, j, i) < r_horizon) {
                parallel_theory = ParallelTheory::zero;
            }

            // Parallel velocity solve
            Etype etype = Etype::entropy;
            Real rho, uu, W = 0;
            Real gamma, gamma_par, vpar, vcon_par[GR_DIM];
            do {
                if (parallel_theory == ParallelTheory::zero) {
                    // no parallel velocity
                    vpar = 0.;
                } else if (parallel_theory == ParallelTheory::cold) {
                    // cold parallel equation, conservation of b^0

                    // specific enthalpy is included in the conserved quantity
                    // if we don't define  FORCEFREE_PARALLEL_COLD
                    // so these equations are inconsistent
                    // as a guess, divide out the CURRENT specific enthalpy
                    // #ifndef FORCEFREE_PARALLEL_COLD
                    // printf("whicheqs_parallel=4 but FORCEFREE_PARALLEL_COLD not defined!\n");
                    // exit(-1);
                    // //w_s = 1 + gam*pp[UU]/pp[RHO]; // specific enthalpy
                    // #endif

                    const Real upar = U(m_u.UUFF, k, j, i) * a_over_g / Bmag;
                    const Real upar_sq = SQR(upar);
                    //const Real gamma2 = gamma2_perp * (1 + upar_sq);
                    vpar = m::copysign(m::sqrt(inv_gamma2_perp * upar_sq / (1 + upar_sq)), upar);
                } else {
                    // Numerical inversion using hot or entropy eqns

                    // initial guess for W is based on current primitives
                    Real Y = 0, Z = 0;
                    const Real gamma_prev = GRMHD::lorentz_calc(G, P, m_p, k, j, i, Loci::center);
                    W = (P(m_p.RHO, k, j, i) + gam * P(m_p.UU, k, j, i)) * SQR(gamma_prev);

                    if (parallel_theory == ParallelTheory::entropy) {
                        // inversion based on adiabatically evolved specific enthalpy
                        Real M = U(m_u.UUFF, k, j, i) * a_over_g / Bmag;
                        Y = M * D;
                        Z = 0.;
                    } else if (parallel_theory == ParallelTheory::entropy_mhd ||
                                parallel_theory == ParallelTheory::hot_mhd) {
                        // inversion constants based on full MHD conserveds
                        Qcov[0] = U(m_u.UU, k, j, i) * a_over_g - U(m_u.RHO, k, j, i) * a_over_g;
                        Qcov[1] = U(m_u.U1, k, j, i) * a_over_g;
                        Qcov[2] = U(m_u.U2, k, j, i) * a_over_g;
                        Qcov[3] = U(m_u.U3, k, j, i) * a_over_g;
                        Real Qcon[GR_DIM];
                        G.raise(Qcov, Qcon, k, j, i, Loci::center);
                        Real QdotB = dot(Bcon,Qcov);
                        Real QdotEta = -alpha * Qcon[0];

                        Y = QdotB / Bmag;
                        Z = QdotEta + 0.5*Bsq*(1. + vsq_perp);

                        if (parallel_theory == ParallelTheory::hot_mhd)
                            etype = Etype::hot;
                    }

                    // solver constants
                    Real cons[FF_CONSTANTS_LEN];
                    cons[0] = D;
                    cons[1] = Y;
                    cons[2] = Z;
                    cons[3] = inv_gamma2_perp;
                    cons[4] = afac;
                    cons[5] = Sc;
                
                    // solve for W
                    FFFlag solver_status;
                    if (etype == Etype::hot) {
                        solver_status = u2p_solver_ff_parallel<Etype::hot>(W, cons);
                    } else {
                        solver_status = u2p_solver_ff_parallel<Etype::entropy>(W, cons);
                    }
                    if (solver_status != FFFlag::success) ffflag(0, k, j, i) = (Real) solver_status;
                    vpar = Y / W;
                    break;
                } // switch

        
                // get parallel 3-velocity from vpar solution
                vcon_par[0] = 0.;
                vcon_par[1] = vpar * (Bcon[1] / Bmag);
                vcon_par[2] = vpar * (Bcon[2] / Bmag);
                vcon_par[3] = vpar * (Bcon[3] / Bmag);
                const Real vsq_par = vpar*vpar;

                // parallel lorentz factor
                const Real gamma2_par = 1. / (1. - vsq_par);
                gamma_par = m::sqrt(gamma2_par);
                // total lorentz factor
                const Real gamma2 = 1. / (1. - vsq_perp - vsq_par);
                gamma = m::sqrt(gamma2);

                // determine the density
                // Will be floored later if needed
                rho = D / gamma;
                // determine the entropy and energy density
                Real entr = Sc/gamma;
                if(etype == Etype::hot) {
                    uu = (W / gamma2 - rho) / gam;
                } else { // etype == Etype::entropy
                    uu = calc_ufromS(entr, rho, gam);
                }

                // is the solution for uu acceptable?
                // if not, continue with next set of equations
                if(!isfinite(uu) || uu < 0) {
                    if((int) parallel_theory < 5) {
                        // skip cold parallel evolution if adiabatic evolution fails
                        // we are not conserving the right quantity for cold evolution
                        if (parallel_theory == ParallelTheory::entropy) {
                            parallel_theory = ParallelTheory::zero;
                        } else {
                            parallel_theory = (ParallelTheory) ((int) parallel_theory + 1);
                        }
                        // if(verbose > 0) {
                        //     printf("neg uu in parallel solver: whicheqs %d -> %d \n",
                        //             whicheqs_parallel_old, whicheqs_parallel);
                        // }
                        continue;
                    } else {
                        //if(verbose>0) printf("neg uu in parallel solver: whicheqs 5 -> fail \n");
                        break;
                    }
                } else {
                    break; // acceptable solution for uu
                }
            } while ((int) parallel_theory < 6);

            // Final checks, flags
            if(rho < 0. || !isfinite(rho)) {
                ffflag(0, k, j, i) = (Real) FFFlag::neg_rho;
            } else if(uu < 0. || !isfinite(uu)) {
                ffflag(0, k, j, i) = (Real) FFFlag::neg_u;
            } else {
                ffflag(0, k, j, i) = (Real) FFFlag::success;
            }
  
            // total velocity (VELR)
            Real uprim[NVEC];
            uprim[0] = (vcon_perp[1] + vcon_par[1]) * gamma;
            uprim[1] = (vcon_perp[2] + vcon_par[2]) * gamma;
            uprim[2] = (vcon_perp[3] + vcon_par[3]) * gamma;
            
            // New GRMHD primitives
            P(m_p.RHO, k, j, i) = rho;
            P(m_p.UU, k, j, i) = uu;
            P(m_p.U1, k, j, i) = (vcon_perp[1] + vcon_par[1]) * gamma;
            P(m_p.U2, k, j, i) = (vcon_perp[2] + vcon_par[2]) * gamma;
            P(m_p.U3, k, j, i) = (vcon_perp[3] + vcon_par[3]) * gamma;
            // New force-free primitives
            P(m_p.UUFF, k, j, i) = vpar * gamma_par; 
            P(m_p.U1FF, k, j, i) = vcon_perp[1] * gamma_perp;
            P(m_p.U2FF, k, j, i) = vcon_perp[2] * gamma_perp;
            P(m_p.U3FF, k, j, i) = vcon_perp[3] * gamma_perp;

        }
    );

    return TaskStatus::complete;
}

TaskStatus Force_Free::PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md)
{
    auto pmesh = md->GetMeshPointer();
    // Options
    const auto& globals = pmesh->packages.Get("Globals")->AllParams();
    const int flag_verbose = globals.Get<int>("flag_verbose");

    // Debugging/diagnostic info about force-free zones
    if (flag_verbose > 0) {
        Reductions::StartFlagReduce(md, "flags.force_free", Force_Free::flag_names, IndexDomain::interior, false, 9);
        // Debugging/diagnostic info about floor and inversion flags
        Reductions::CheckFlagReduceAndPrintHits(md, "flags.force_free", Force_Free::flag_names, IndexDomain::interior, false, 9);
    }

    return TaskStatus::complete;
}