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

#include "domain.hpp"
#include "inverter.hpp"
#include "kharma.hpp"
#include "kharma_driver.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include <limits>
#include <stdexcept>

std::shared_ptr<KHARMAPackage> RadM1::Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("RadM1");
    Params& params = pkg->AllParams();

    auto& driver = packages->Get("Driver")->AllParams();
    auto driver_type = driver.Get<DriverType>("type");
    // TODO(PNM): Make it as an option to also add kharma driver eventually
    bool implicit_radm1 = (driver_type == DriverType::imex &&
                           pin->GetOrAddBoolean("RadM1", "implicit", true));

    if (!implicit_radm1)
        PARTHENON_WARN("M1 implementation will be uncoupled unless ImEx driver is used!");

    Metadata::AddUserFlag("RADM1");
    std::vector<MetadataFlag> flags_radm1 = {Metadata::Cell,
        Metadata::GetUserFlag("Explicit"), Metadata::GetUserFlag("RADM1")};

    auto flags_prim = driver.Get<std::vector<MetadataFlag>>("prim_flags");
    flags_prim.insert(flags_prim.end(), flags_radm1.begin(), flags_radm1.end());

    //Save primitive variables to restart files
    //TODO (PNM): Is this really necessary? Kharma only restarts using conserved variables.
    flags_prim.push_back(Metadata::Restart);

    auto flags_cons = driver.Get<std::vector<MetadataFlag>>("cons_flags");
    flags_cons.insert(flags_cons.end(), flags_radm1.begin(), flags_radm1.end());

    //TODO (PNM): Eventually, just collapse all the conserved and prim variables on a single vector, instead of dividing t component
    //from spatial components.
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
    Metadata m =
        Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("rflag", m);

    Real u_rad_floor = pin->GetOrAddReal("radM1", "u_rad_floor", 1.e-8);
    pkg->AllParams().Add("u_rad_floor", u_rad_floor);

    Real src_rootfind_eps = pin->GetOrAddReal("RadM1", "src_rootfind_eps", 1e-8);
    Real src_rootfind_tol = pin->GetOrAddReal("RadM1", "src_rootfind_tol", 1e-8);
    int src_rootfind_maxiter = pin->GetOrAddInteger("RadM1", "src_rootfind_maxiter", 50);
    pkg->AllParams().Add("src_rootfind_eps", src_rootfind_eps);
    pkg->AllParams().Add("src_rootfind_tol", src_rootfind_tol);
    pkg->AllParams().Add("src_rootfind_maxiter", src_rootfind_maxiter);

    // Opacity model selector (see RadM1::OpacityModel in radM1.hpp).
    // Determine the problem ID
    std::string problem_id = pin->GetString("parthenon/job", "problem_id");

    // Set the default opacity model based on the problem ID using an if/else chain
    std::string default_opacity_model = "default";
    if (problem_id == "shock") {
        default_opacity_model = "shocktube_constant";
    } else if (problem_id == "bondi_rad") {
        default_opacity_model = "bondi_opacs";
    } else if (problem_id == "beam_of_light") {
        default_opacity_model = "beam_light_zero";
    } else if (problem_id == "thermal_equilibrium") {
        default_opacity_model = "thermal_equilibrium";
    }

    // fallback to the default if no input
    // TODO (PNM): add a vector of strings as the last parameter here to yell at the user what the options are
    std::string opacity_model_str = pin->GetOrAddString("radM1", "opacity_model", default_opacity_model);

    int opacity_model = (int) OpacityModel::Default;
    if (opacity_model_str == "shocktube_constant") {
        opacity_model = (int) OpacityModel::ShocktubeConstant;
    } else if (opacity_model_str == "bondi_opacs") {
        opacity_model = (int) OpacityModel::Bondi;
    } else if (opacity_model_str == "beam_light_zero") {
        opacity_model = (int) OpacityModel::BeamLight;
    } else if (opacity_model_str == "thermal_equilibrium") {
        opacity_model = (int) OpacityModel::ThermalEquilibrium;
    }

    // Read Shocktube constants
    // TODO(PNM): Make these parameters part of the shocktube problem. Important!
    Real shocktube_sigma_rad = pin->GetOrAddReal("radM1", "sigma_rad", 3.470e7);
    Real shocktube_kappa_rho = pin->GetOrAddReal("radM1", "kappa_rho", 0.08);
    Real shocktube_kappa_scat = pin->GetOrAddReal("radM1", "kappa_scat", 0.0);

    // Add everything to the package parameters
    pkg->AllParams().Add("opacity_model", opacity_model);

    pkg->AllParams().Add("shocktube_sigma_rad", shocktube_sigma_rad);
    pkg->AllParams().Add("shocktube_kappa_rho", shocktube_kappa_rho);
    pkg->AllParams().Add("shocktube_kappa_scat", shocktube_kappa_scat);

    // Initialize units needed for radm1
    // TODO (PNM): Use a proper units package to bundle these together.
    auto unit_conv = phoebus::UnitConversions(pin);
    UnitScales units_cgs;
    units_cgs.length_cgs      = unit_conv.GetLengthCodeToCGS();
    printf("RadM1: length_cgs = %e\n", units_cgs.length_cgs);
    units_cgs.time_cgs        = unit_conv.GetTimeCodeToCGS();
    printf("RadM1: time_cgs = %e\n", units_cgs.time_cgs);
    units_cgs.mass_cgs        = unit_conv.GetMassCodeToCGS();
    printf("RadM1: mass_cgs = %e\n", units_cgs.mass_cgs);
    units_cgs.energy_cgs      = unit_conv.GetEnergyCodeToCGS();
    printf("RadM1: energy_cgs = %e\n", units_cgs.energy_cgs);

    // Tg = mp * c^2 * 1/kb * 1/m_scale (gamma -1) * ug */(rho)
    units_cgs.temperature_cgs = unit_conv.GetTemperatureCodeToCGS() * pc::mp * pc::c * pc::c;
    printf("RadM1: temperature_cgs = %e\n", units_cgs.temperature_cgs);

    // Mean molecular weight, used only by OpacityModel::Default (see the comment on
    // UnitScales::mu) to get a real temperature in Kelvin: T_K = Tg * mu *
    // units_cgs.temperature_cgs. Existing branches (Bondi, ThermalEquilibrium, ...)
    // are untouched and keep their original (implicit mu=1) behavior.
    units_cgs.mu = pin->GetOrAddReal("radM1", "mu", 0.6);
    printf("RadM1: mu = %e\n", units_cgs.mu);
    pkg->AllParams().Add("units_cgs", units_cgs);


    // TODO (PNM): Currently attached to the floors package. Make this a separate option only for radiation package.
    bool floors_on_default = true;
    if (pin->DoesParameterExist("floors", "disable_floors")) {
        floors_on_default = !pin->GetBoolean("floors", "disable_floors");
    }
    if (pin->GetOrAddBoolean("floors", "on", floors_on_default))
        pkg->BlockApplyFloors = RadM1::ApplyRadM1Floors;

    pkg->PostStepDiagnosticsMesh = RadM1::PostStepDiagnostics;

    return pkg;
}

void RadM1::ApplyRadM1Floors(MeshBlockData<Real>* rc, IndexDomain domain)
{
    auto pmb = rc->GetBlockPointer();
    auto& params = pmb->packages.Get("RadM1")->AllParams();

    const Real erad_floor = params.Get<Real>("u_rad_floor");
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
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
        {
            if (P(m_p.UU_RAD, k, j, i) < erad_floor) {
                P(m_p.UU_RAD, k, j, i) = erad_floor;
            }
        });
}

TaskStatus RadM1::BlockPtoU(MeshBlockData<Real>* rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();
    const auto& G = pmb->coords;

    // Pack Conserved Variables (Destination)
    PackIndexMap cons_map;
    auto& U = rc->PackVariables(
        std::vector<std::string>{"cons.u_rad", "cons.uvec_rad"}, cons_map);
    VarMap m_u(cons_map, true);

    // Pack Primitive Variables (Source)
    PackIndexMap prim_map;
    auto P = rc->PackVariables(
        std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prim_map);
    const VarMap m_p(prim_map, false);

    // Get Loop Bounds
    IndexRange3 b = KDomain::GetRange(rc, domain, coarse);

    // Parallel Loop
    pmb->par_for("RadM1_PtoU", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
        {
            Real Erf = P(m_p.UU_RAD, k, j, i);
            Real uvec_radframe[4] = {0, P(m_p.U1_RAD, k, j, i), P(m_p.U2_RAD, k, j, i),
                P(m_p.U3_RAD, k, j, i)};
            const Real gamma =
                GRMHD::lorentz_calc(G, uvec_radframe, k, j, i, Loci::center);
            Real ucon_rad[GR_DIM];
            // GRMHD::calc_ucon(G, uvec_radframe, k, j, i, Loci::center, ucon_rad);
            calc_ucon_rad(G, P, m_p, k, j, i, Loci::center, ucon_rad);
            // R^t^mu
            Real R_t_con[GR_DIM];

            for (int mu = 0; mu < GR_DIM; ++mu) {
                R_t_con[mu] = 4. / 3. * Erf * ucon_rad[0] * ucon_rad[mu] +
                              1. / 3. * Erf * G.gcon(Loci::center, j, i, 0, mu);
            }

            // Calculat R^t_mu
            Real R_t_cov[GR_DIM];
            G.lower(R_t_con, R_t_cov, k, j, i, Loci::center);

            U(m_u.UU_RAD, k, j, i) =
                R_t_cov[0] * G.gdet(Loci::center, j, i); // cons.u_rad
            U(m_u.U1_RAD, k, j, i) =
                R_t_cov[1] * G.gdet(Loci::center, j, i); // cons.uvec_rad 1
            U(m_u.U2_RAD, k, j, i) =
                R_t_cov[2] * G.gdet(Loci::center, j, i); // cons.uvec_rad 2
            U(m_u.U3_RAD, k, j, i) =
                R_t_cov[3] * G.gdet(Loci::center, j, i); // cons.uvec_rad 3
        });
    return TaskStatus::complete;
}

TaskStatus RadM1::Step(MeshData<Real>* md_sub_init,
    MeshData<Real>* md_sub_final, const Real dt)
{
    for (int b = 0; b < md_sub_final->NumBlocks(); ++b) {
        auto pmb_data = md_sub_final->GetBlockData(b);
        auto pmb = pmb_data->GetBlockPointer();
        auto& params = pmb->packages.Get("RadM1")->AllParams();

        // Fetch parameters
        const Real src_rootfind_eps = params.Get<Real>("src_rootfind_eps");
        const Real src_rootfind_tol = params.Get<Real>("src_rootfind_tol");
        const int src_rootfind_maxiter = params.Get<int>("src_rootfind_maxiter");
        const Real gam = pmb->packages.Get("GRMHD")->AllParams().Get<Real>("gamma");
        const int opacity_model = params.Get<int>("opacity_model");
        const Real shocktube_sigma_rad = params.Get<Real>("shocktube_sigma_rad");
        const Real shocktube_kappa_rho = params.Get<Real>("shocktube_kappa_rho");
        const Real shocktube_kappa_scat = params.Get<Real>("shocktube_kappa_scat");
        const UnitScales units_cgs = params.Get<UnitScales>("units_cgs");

        Microphysics::Opacities opacities;
        if (pmb->packages.AllPackages().count("opacity")) {
            opacities = pmb->packages.Get("opacity")->AllParams().Get<Microphysics::Opacities>("opacities");
        }

        const auto& G = pmb->coords;

        PackIndexMap prims_map, cons_map;
        auto P_new =
            pmb_data->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
        auto U_new =
            pmb_data->PackVariables({Metadata::WithFluxes, Metadata::Cell}, cons_map);
        const VarMap m_p(prims_map, false);
        const VarMap m_u(cons_map, true);

        auto rflag = pmb_data->PackVariables(std::vector<std::string>{"rflag"});
        auto pflag = pmb_data->PackVariables(std::vector<std::string>{"pflag"});

        auto pmb_init_data = md_sub_init->GetBlockData(b);

        auto P_init =
            pmb_init_data->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
        auto U_init =
            pmb_init_data->PackVariables({Metadata::WithFluxes, Metadata::Cell}, cons_map);

        auto bounds = pmb->cellbounds;
        const IndexRange ib = bounds.GetBoundsI(IndexDomain::interior);
        const IndexRange jb = bounds.GetBoundsJ(IndexDomain::interior);
        const IndexRange kb = bounds.GetBoundsK(IndexDomain::interior);

        //TODO (PNM): Split it, Cora thinks this is too large to be good. Probably too slow.
        pmb->par_for("RadM1_Implicit_Solver4D", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
            {
                int rflagl =
                    solve_radiation_4d(G, U_init, P_init, P_new, U_new, m_p, m_u, k, j, i,
                        dt, gam, src_rootfind_eps, src_rootfind_tol, src_rootfind_maxiter,
                        opacity_model, shocktube_sigma_rad, shocktube_kappa_rho,
                        shocktube_kappa_scat, units_cgs, opacities, pflag);
                rflag(0, k, j, i) = rflagl;
            });
    }

    std::cerr << "Took RadM1 Step!" << std::endl;

    return TaskStatus::complete;
}

TaskStatus RadM1::PostStepDiagnostics(const SimTime& tm, MeshData<Real>* md)
{
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Options
    const auto& pars = pmesh->packages.Get("Globals")->AllParams();
    const int flag_verbose = pars.Get<int>("flag_verbose");


    if (flag_verbose >= 1) {
        Reductions::StartFlagReduce(
            md, "rflag", RadM1::status_names, IndexDomain::interior, false, 3);
        auto total_flag_counts = Reductions::CheckFlagReduceAndPrintHits(
            md, "rflag", RadM1::status_names, IndexDomain::interior, false, 3);
        Reductions::PrintFlagPercentages(
            md, "rflag", RadM1::status_names, IndexDomain::interior, total_flag_counts);
    }

    return TaskStatus::complete;
}
