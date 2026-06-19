/* 
 *  File: radM1.cpp
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
#include "radM1.hpp"
#include "radM1_solvers.hpp"

#include "kharma_driver.hpp"
#include "units.hpp"
#include "kharma.hpp"
#include "inverter.hpp"
#include <limits> 
#include <stdexcept>


std::shared_ptr<KHARMAPackage> RadM1::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("RadM1");
    Params &params = pkg->AllParams();

    auto& driver = packages->Get("Driver")->AllParams();
    auto driver_type = driver.Get<DriverType>("type");
    // TODO follow GRMHD::implicit?
    bool implicit_radm1 = (driver_type == DriverType::imex && pin->GetOrAddBoolean("RadM1", "implicit", true));
    //CHANGE BACK PEDRO
    // bool implicit_radm1 = false;
    
    if (!implicit_radm1) PARTHENON_WARN("M1 implementation will be uncoupled unless ImEx driver is used!");

    //Registers a new label called "RADM1" for the variables to be grouped together later?
    Metadata::AddUserFlag("RADM1");
    // Properties of these new flags
    // I believe Metadata::Cell means these variables are defined at cell centers, but I should ask Cora.
    // We also add the "areWeImplicit" flag, which is either "Implicit" or "Explicit" based on the user's choice in the input file.
    std::vector<MetadataFlag> flags_radm1 = {Metadata::Cell, Metadata::GetUserFlag("Explicit"), Metadata::GetUserFlag("RADM1")};

    //Retrieves the existing flags for the primitive and conserved variables, and adds the new radM1 flags to them.
    //Then adds the new variables for the radiation primitives and conserved variables with these flags.
    // auto flags_prim = driver.Get<std::vector<MetadataFlag>>("prim_flags");
    // flags_prim.insert(flags_prim.end(), flags_radm1.begin(), flags_radm1.end());

    auto flags_prim = driver.Get<std::vector<MetadataFlag>>("prim_flags");
    flags_prim.insert(flags_prim.end(), flags_radm1.begin(), flags_radm1.end());

    // Explicitly tag these as Primitive so your PackVariables call finds them!
    flags_prim.push_back(Metadata::GetUserFlag("Primitive")); 

    // I think this will push this metadata to restart files
    flags_prim.push_back(Metadata::Restart);

    // sync variables across boundaries (ASK CORA)
    if (pin->GetOrAddBoolean("RadM1", "sync_utop_seed", true)) { 
        flags_prim.push_back(Metadata::FillGhost);
    }
    //
    auto flags_cons = driver.Get<std::vector<MetadataFlag>>("cons_flags");
    flags_cons.insert(flags_cons.end(), flags_radm1.begin(), flags_radm1.end());

    auto m_prim_scalar = Metadata(flags_prim);
    pkg->AddField("prims.u_rad", m_prim_scalar);

    auto m_cons_scalar = Metadata(flags_cons);
    pkg->AddField("cons.u_rad", m_cons_scalar);

    auto flags_prim_vec(flags_prim);
    flags_prim_vec.push_back(Metadata::Vector);

    auto flags_cons_vec(flags_cons);
    flags_cons_vec.push_back(Metadata::Vector);

    std::vector<int> s_vector({NVEC});
    auto m_prim_vector = Metadata(flags_prim_vec, s_vector);
    pkg->AddField("prims.uvec_rad", m_prim_vector);

    auto m_cons_vector = Metadata(flags_cons_vec, s_vector);
    pkg->AddField("cons.uvec_rad", m_cons_vector);

    // Flag denoting RadM1 implicit solver failures
    Metadata m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("rflag", m);

    Real u_rad_floor = pin->GetOrAddReal("radM1", "u_rad_floor", 1.e-8);
    pkg->AllParams().Add("u_rad_floor", u_rad_floor);

    Real src_rootfind_eps = pin->GetOrAddReal("RadM1", "src_rootfind_eps", 1e-8);
    Real src_rootfind_tol = pin->GetOrAddReal("RadM1", "src_rootfind_tol", 1e-8);
    int src_rootfind_maxiter = pin->GetOrAddInteger("RadM1", "src_rootfind_maxiter", 50);
    pkg->AllParams().Add("src_rootfind_eps", src_rootfind_eps);
    pkg->AllParams().Add("src_rootfind_tol", src_rootfind_tol);
    pkg->AllParams().Add("src_rootfind_maxiter", src_rootfind_maxiter);


    //Right now, to execute the torus problem with radM1, we need to initialize the radiation primitives in fm_torus.cpp (this is stupid) (ASK Cora)
    //I think this should be moved to RadM1 method, maybe call an initialization method like a task straight after initializing the torus?
    //Especially because we'll want to initialize the radiation field for other problems. 

    // //This method should allow you to add source terms to both plasma and radiation variables separately.
    // pkg->AddSource = RadM1::AddImplicitRadiationSourceTerms;

    //Add inversion to the tasks
    pkg->BlockUtoP = RadM1::BlockUtoP;

    if(pin->GetBoolean("floors", "on"))
        pkg->BlockApplyFloors = RadM1::ApplyRadM1Floors;

    pkg->PostStepDiagnosticsMesh = RadM1::PostStepDiagnostics;

    return pkg;
}

void RadM1::ApplyRadM1Floors(MeshBlockData<Real> *rc, IndexDomain domain)
{
    auto pmb = rc->GetBlockPointer();
    auto& params = pmb->packages.Get("RadM1")->AllParams();

    const Real erad_floor   = params.Get<Real>("u_rad_floor");
    PackIndexMap prims_map;
    auto P = rc->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    const VarMap m_p(prims_map, false);

    // We need to check if we actually have B fields enabled to avoid segfaults
    const bool has_b_field = pmb->packages.AllPackages().count("B_FluxCT") || 
                             pmb->packages.AllPackages().count("B_CD");

    auto bounds = pmb->cellbounds;
    const IndexRange ib = bounds.GetBoundsI(domain);
    const IndexRange jb = bounds.GetBoundsJ(domain);
    const IndexRange kb = bounds.GetBoundsK(domain);

    pmb->par_for("ApplyRadM1Floors", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            if (P(m_p.UU_RAD, k, j, i) < erad_floor) {
                P(m_p.UU_RAD, k, j, i) = erad_floor;
            }
        }
    );
}


TaskStatus RadM1::BlockPtoU(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();
    const auto& G = pmb->coords;

    // Pack Conserved Variables (Destination)
    PackIndexMap cons_map;
    auto& U = rc->PackVariables(std::vector<std::string>{"cons.u_rad", "cons.uvec_rad"}, cons_map);
    VarMap m_u(cons_map, true);

    // Pack Primitive Variables (Source)
    PackIndexMap prim_map;
    auto P = rc->PackVariables(std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prim_map);
    const VarMap m_p(prim_map, false);

    // Get Loop Bounds
    IndexRange3 b = KDomain::GetRange(rc, domain, coarse);

    // Parallel Loop
    pmb->par_for("RadM1_PtoU", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {

        Real Erf = P(m_p.UU_RAD, k, j, i);
        Real uvec_radframe[4] = {0, P(m_p.U1_RAD, k, j, i), P(m_p.U2_RAD, k, j, i), P(m_p.U3_RAD, k, j, i)};
        const Real gamma = GRMHD::lorentz_calc(G, uvec_radframe, k, j, i, Loci::center);
        Real ucon_rad[GR_DIM];
        //GRMHD::calc_ucon(G, uvec_radframe, k, j, i, Loci::center, ucon_rad);
        calc_ucon_rad(G, P, m_p, k, j, i, Loci::center, ucon_rad);
        // R^t^mu
        Real R_t_con[GR_DIM];

        for (int mu=0; mu<GR_DIM; ++mu) {
            R_t_con[mu] = 4./3. * Erf * ucon_rad[0] * ucon_rad[mu] + 1./3. * Erf * G.gcon(Loci::center, j, i, 0, mu);
        }

        // Calculat R^t_mu
        Real R_t_cov[GR_DIM];
        G.lower(R_t_con, R_t_cov, k, j, i, Loci::center);

        U(m_u.UU_RAD, k, j, i) = R_t_cov[0] * G.gdet(Loci::center, j, i); // cons.u_rad
        U(m_u.U1_RAD, k, j, i) = R_t_cov[1] * G.gdet(Loci::center, j, i); // cons.uvec_rad 1
        U(m_u.U2_RAD, k, j, i) = R_t_cov[2] * G.gdet(Loci::center, j, i); // cons.uvec_rad 2
        U(m_u.U3_RAD, k, j, i) = R_t_cov[3] * G.gdet(Loci::center, j, i); // cons.uvec_rad 3
    });
    return TaskStatus::complete;
}


TaskStatus RadM1::BlockUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();
    const auto& G = pmb->coords;

    //Pack Variables
    auto U = rc->PackVariables(std::vector<std::string>{"cons.u_rad", "cons.uvec_rad"});

    // Pack Primitive Variables
    PackIndexMap prim_map;
    auto P = rc->PackVariables(std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prim_map);
    const VarMap m_p(prim_map, false);

    //Get Loop Bounds
    IndexRange3 b = KDomain::GetRange(rc, domain, coarse);

    auto& params = pmb->packages.Get("RadM1")->AllParams();
    //TODO: FLOOR! CHANGE THIS
    const Real min_erad   = 10.*1.e-80;

    //Parallel Loop
    pmb->par_for("RadM1_UtoP", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {


            
            Real gdet = G.gdet(Loci::center, j, i);

            // Reconstruct Radiation Stress-Energy Tensor R^{mu, nu} 
            // For the M1 scheme, we assume the radiation is isotropic and satisfies the Eddington approximation (P^{ij} = (1/3) E delta^{ij} in the fluid frame)
            // but in the radiation frame. Therefore, \bar{R}^{tt} = E, \bar{R}^{ii} = \bar{E}/3, and every other component is zero. 
            // So we have Equation 27 in Sadowski et al. 2013 in the radiation rest frame, but since it's covariant, it's valid for every other frame (including lab frame).
            // The equation goes as follows:
            // R^{mu nu} = (4/3) \bar{E} u_R^mu u_R^nu + (1/3) \bar{E} g^{mu nu}, \bar{E} is always in the radiation rest frame.


            //Recover R^t_mu
            // U(0) is R^t_t, U(1..3) are R^t_i
            Real R_t_cov[GR_DIM] = {
                U(0, k, j, i) / gdet,
                U(1, k, j, i) / gdet,
                U(2, k, j, i) / gdet,
                U(3, k, j, i) / gdet
            };

            // Get R^{t\mu} from R^t_\mu by doing R^{t\mu} = g^{t\nu} R_{\nu\mu}
            // This is R^t_\mu
            Real R_t_con[GR_DIM];
            G.raise(R_t_cov, R_t_con, k, j, i, Loci::center);

            //Calculate Invariant Scalar S = R^t_mu * R^{t\mu}
            // Real invariant_scalar = 0.0;
            // for(int mu=0; mu<4; ++mu) {
            //      // Note: R_t_cov is R^t_mu (mixed), R_t_con is R^{t\mu} (upper). 
            //      // S = g_{mu nu} R^{t mu} R^{t nu} 
            //      for(int nu=0; nu<4; ++nu) {
            //         invariant_scalar += G.gcov(Loci::center, j, i, mu, nu) * R_t_con[mu] * R_t_con[nu];
            //      }
            // }
            Real invariant_scalar = 0.0;
            for(int mu=0; mu<4; ++mu) {
                invariant_scalar += R_t_cov[mu] * R_t_con[mu];
            }

            // From Sadowski et al. Eq 32 and 33
            // Solving by isolating Isolaring u^t_R^2 results in a much better equation, but I think (and I could be wrong)
            // that solving by isolating \bar{E} results in a much more numerical stable solution.

            // Calculating u^t_R^2
            Real gammarel2 = CalculateGammaRel2(G, R_t_cov, invariant_scalar, j, i);

            // Pre-calculate alpha_sq so E_rf can use it
            Real alpha_sq = -1.0 / G.gcon(Loci::center, j, i, 0, 0); 
            Real alpha = m::sqrt(alpha_sq);
            Real E_rf = (3.0 * R_t_con[0] * alpha_sq) / (4.0 * gammarel2 - 1.0);
            
            // TODO: FLOOR! CHANGE THIS
            Real GAMMAMAX = 20.;
            int nonfailure = gammarel2 >= 1.0 && E_rf > min_erad && gammarel2 <= (GAMMAMAX * GAMMAMAX)/(min_erad * min_erad);
            Real uvec_radframe_con[4] = {0};

            if(nonfailure){
                for(int mu=0; mu<4; ++mu) {
                    uvec_radframe_con[mu] = alpha * (R_t_con[mu] + 1./3. * E_rf * G.gcon(Loci::center, j, i, 0, mu) * (4.0 * gammarel2 - 1.0)) / (4./3. * E_rf * m::sqrt(gammarel2));
                }
            } else {
                // Attempt Cold Closure
                Real gammarel2_slow = m::pow(1.0 +10.0 * std::numeric_limits<double>::epsilon(), 2.0);
                Real gammarel2_fast = GAMMAMAX * GAMMAMAX;

                Real R_t_t_slow, Erf_slow;
                ApplyColdClosureFix(G, R_t_cov, gammarel2_slow, j, i, R_t_t_slow, Erf_slow);

                Real R_t_t_fast, Erf_fast;
                ApplyColdClosureFix(G, R_t_cov, gammarel2_fast, j, i, R_t_t_fast, Erf_fast);

                Real R_t_t_new, gammarel2_new;
                
                if (m::abs(R_t_t_slow - R_t_cov[0]) > m::abs(R_t_t_fast - R_t_cov[0])) {
                    R_t_t_new = R_t_t_fast;
                    E_rf = Erf_fast;
                    gammarel2_new = gammarel2_fast;
                } else {
                    R_t_t_new = R_t_t_slow;
                    E_rf = Erf_slow;
                    gammarel2_new = gammarel2_slow;
                }

                Real R_t_cov_new[GR_DIM] = {R_t_t_new, R_t_cov[1], R_t_cov[2], R_t_cov[3]};

                Real R_t_con_new[GR_DIM];
                G.raise(R_t_cov_new, R_t_con_new, k, j, i, Loci::center);

                // Calculate primitive velocities with the new state
                if (E_rf > 0.0) {
                    for(int mu=0; mu<4; ++mu) {
                        uvec_radframe_con[mu] = alpha * (R_t_con_new[mu] + 1./3. * E_rf * G.gcon(Loci::center, j, i, 0, mu) * (4.0 * gammarel2_new - 1.0)) / (4./3. * E_rf * m::sqrt(gammarel2_new));
                    }
                } else {
                    for(int mu=0; mu<4; ++mu) uvec_radframe_con[mu] = 0.0;
                }
            }

            P(m_p.UU_RAD, k, j, i) = E_rf;
            P(m_p.U1_RAD, k, j, i) = uvec_radframe_con[1];
            P(m_p.U2_RAD, k, j, i) = uvec_radframe_con[2];
            P(m_p.U3_RAD, k, j, i) = uvec_radframe_con[3];

    });
    return TaskStatus::complete;
}

TaskStatus RadM1::Step(MeshData<Real> *md_full_init, MeshData<Real> *md_sub_init, MeshData<Real> *md_sub_final, const Real dt)
{
    for (int b = 0; b < md_sub_final->NumBlocks(); ++b) {
        auto pmb_data = md_sub_final->GetBlockData(b);
        auto pmb      = pmb_data->GetBlockPointer();
        auto& params  = pmb->packages.Get("RadM1")->AllParams();

        // Fetch parameters
        const Real src_rootfind_eps   = params.Get<Real>("src_rootfind_eps");
        const Real src_rootfind_tol   = params.Get<Real>("src_rootfind_tol");
        const int src_rootfind_maxiter = params.Get<int>("src_rootfind_maxiter");
        const Real gam                 = pmb->packages.Get("GRMHD")->AllParams().Get<Real>("gamma");
        const auto& G                  = pmb->coords;

        PackIndexMap prims_map, cons_map;
        auto P_new = pmb_data->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
        auto U_new = pmb_data->PackVariables({Metadata::Conserved, Metadata::Cell}, cons_map);
        const VarMap m_p(prims_map, false);
        const VarMap m_u(cons_map, true);

        auto rflag = pmb_data->PackVariables(std::vector<std::string>{"rflag"});

        auto pmb_init_data = md_sub_init->GetBlockData(b);
        
        PackIndexMap dummy_map;
        auto P_init = pmb_init_data->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
        auto U_init = pmb_init_data->PackVariables({Metadata::Conserved, Metadata::Cell}, cons_map);

        auto bounds = pmb->cellbounds;
        const IndexRange ib = bounds.GetBoundsI(IndexDomain::interior);
        const IndexRange jb = bounds.GetBoundsJ(IndexDomain::interior);
        const IndexRange kb = bounds.GetBoundsK(IndexDomain::interior);

        pmb->par_for("RadM1_Implicit_Solver4D", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                // if (i == 110) {
                //     printf("The initial set of U_init is [%.15e, %.15e, %.15e, %.15e]\n", U_init(m_u.UU, k, j, i), U_init(m_u.U1, k, j, i), U_init(m_u.U2, k, j, i), U_init(m_u.U3, k, j, i));
                // }
                int rflagl = solve_radiation_4d(
                    G, U_init, P_init, P_new, U_new, m_p, m_u, k, j, i,
                    dt, gam, src_rootfind_eps, src_rootfind_tol, src_rootfind_maxiter
                );
                // if(i == 110) {
                //     printf("The new set of U_new is [%.15e, %.15e, %.15e, %.15e]\n\n", U_new(m_u.UU, k, j, i), U_new(m_u.U1, k, j, i), U_new(m_u.U2, k, j, i), U_new(m_u.U3, k, j, i));
                // }
                rflag(0, k, j, i) = rflagl;
            }
        );
    }

    std::cerr << "Took RadM1 Step!" << std::endl;

    return TaskStatus::complete;
}

TaskStatus RadM1::PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md)
{
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Options
    const auto& pars = pmesh->packages.Get("Globals")->AllParams();
    const int flag_verbose = pars.Get<int>("flag_verbose");

    // Debugging/diagnostic info about implicit step flags
    // Mirrors Inverter function, logic should be the same
    if (flag_verbose >= 1) {
        Reductions::StartFlagReduce(md, "rflag", RadM1::status_names, IndexDomain::interior, false, 1);
        auto total_flag_counts = Reductions::CheckFlagReduceAndPrintHits(md, "rflag", RadM1::status_names, IndexDomain::interior, false, 1);
        Reductions::PrintFlagPercentages(md, "rflag", RadM1::status_names, IndexDomain::interior, total_flag_counts);
        
    }

    return TaskStatus::complete;
}