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
#include "units.hpp"
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
                           pin->GetOrAddBoolean("radM1", "implicit", true));

    if (!implicit_radm1)
        PARTHENON_WARN("M1 implementation will be uncoupled unless ImEx driver is used!");

    Metadata::AddUserFlag("RADM1");
    std::vector<MetadataFlag> flags_radm1 = {Metadata::Cell,
        Metadata::GetUserFlag("Explicit"), Metadata::GetUserFlag("RADM1")};

    auto flags_prim = driver.Get<std::vector<MetadataFlag>>("prim_flags");
    flags_prim.insert(flags_prim.end(), flags_radm1.begin(), flags_radm1.end());

    // Save primitive variables to restart files
    // TODO (PNM): Is this really necessary? Kharma only restarts using conserved
    // variables.
    flags_prim.push_back(Metadata::Restart);

    auto flags_cons = driver.Get<std::vector<MetadataFlag>>("cons_flags");
    flags_cons.insert(flags_cons.end(), flags_radm1.begin(), flags_radm1.end());

    // TODO (PNM): Eventually, just collapse all the conserved and prim variables on a
    // single vector, instead of dividing t component from spatial components.
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
    pkg->AddField("rimplflag", m);
    pkg->AddField("rinvflag", m);
    Real u_rad_floor = pin->GetOrAddReal("radM1", "u_rad_floor", 1.e-40);
    pkg->AllParams().Add("u_rad_floor", u_rad_floor);

    Real src_rootfind_eps = pin->GetOrAddReal("radM1", "src_rootfind_eps", 1e-8);
    Real src_rootfind_tol = pin->GetOrAddReal("radM1", "src_rootfind_tol", 1e-8);
    int src_rootfind_maxiter = pin->GetOrAddInteger("radM1", "src_rootfind_maxiter", 50);
    pkg->AllParams().Add("src_rootfind_eps", src_rootfind_eps);
    pkg->AllParams().Add("src_rootfind_tol", src_rootfind_tol);
    pkg->AllParams().Add("src_rootfind_maxiter", src_rootfind_maxiter);

    pkg->AllParams().Add("current_stage_dt", 0.0, true);

    // Opacity model selector (see rad_opacities.hpp).
    // Determine the problem ID
    std::string problem_id = pin->GetString("parthenon/job", "problem_id");

    // Set the default opacity model based on the problem ID using an if/else chain
    // Default here will be handled by singularity opac
    std::string default_opacity_type = "default";
    if (problem_id == "shock") {
        default_opacity_type = "shocktube_constant";
    } else if (problem_id == "bondi_rad") {
        default_opacity_type = "bondi_opacs";
    } else if (problem_id == "beam_of_light") {
        default_opacity_type = "transparent";
    } else if (problem_id == "thermal_equilibrium") {
        default_opacity_type = "thermal_equilibrium";
    } else if (problem_id == "radmhdmodes") {
        default_opacity_type = "shocktube_constant";
    }

    // user can override the default opacity model in the input file, but if not, we use
    // the default based on the problem ID.
    std::string opacity_type_str =
        pin->GetOrAddString("opac", "type", default_opacity_type);

    std::set<std::string> known_opacity_types = {"default", "bondi_opacs", "transparent",
        "thermal_equilibrium", "shocktube_constant", "constant"};

    if (!known_opacity_types.count(opacity_type_str)) {
        std::stringstream msg;
        msg << "Opacity type \"" << opacity_type_str << "\" not recognized!";
        PARTHENON_FAIL(msg);
    }

    int opacity_type = (int)OpacityType::Default;
    if (opacity_type_str == "shocktube_constant") {
        opacity_type = (int)OpacityType::ShocktubeConstant;
    } else if (opacity_type_str == "bondi_opacs") {
        opacity_type = (int)OpacityType::Bondi;
    } else if (opacity_type_str == "transparent") {
        opacity_type = (int)OpacityType::Transparent;
    } else if (opacity_type_str == "thermal_equilibrium") {
        opacity_type = (int)OpacityType::ThermalEquilibrium;
    } else if (opacity_type_str == "constant") {
        opacity_type = (int)OpacityType::Constant;
    }

    // These parameters are only valid when singularity-opac is not in use! When it's in
    // use, we just pass the responsability of handling opacities to it. Check
    // rad_opacities.hpp
    Real const_sigma = pin->GetOrAddReal("opac", "sigma_rad", 0.0);
    Real const_kappa_a = pin->GetOrAddReal("opac", "kappa_a", 0.0);
    Real const_kappa_sc = pin->GetOrAddReal("opac", "kappa_sc", 0.0);

    // Add everything to the package parameters
    pkg->AllParams().Add("opacity_type", opacity_type);

    pkg->AllParams().Add("const_sigma", const_sigma, true);
    pkg->AllParams().Add("const_kappa_a", const_kappa_a, true);
    pkg->AllParams().Add("const_kappa_sc", const_kappa_sc, true);

    // TODO (PNM): Currently attached to the floors package. Make this a separate option
    // only for radiation package.
    bool floors_on_default = true;
    if (pin->DoesParameterExist("floors", "disable_floors")) {
        floors_on_default = !pin->GetBoolean("floors", "disable_floors");
    }
    if (pin->GetOrAddBoolean("floors", "on", floors_on_default))
        pkg->BlockApplyFloors = RadM1::ApplyRadM1Floors;

    pkg->BlockUtoP = RadM1::BlockUtoP;
    pkg->PostStepDiagnosticsMesh = RadM1::PostStepDiagnostics;

    pkg->AddSource = RadM1::AddSourceImplicitly;

    return pkg;
}

void RadM1::ApplyRadM1Floors(MeshBlockData<Real>* rc, IndexDomain domain)
{
    auto pmb = rc->GetBlockPointer();
    const auto& G = pmb->coords;
    auto& params = pmb->packages.Get("RadM1")->AllParams();

    const Real erad_floor = params.Get<Real>("u_rad_floor");
    PackIndexMap prims_map;
    auto P = rc->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    const VarMap m_p(prims_map, false);

    PackIndexMap cons_map;
    auto U = rc->PackVariables(
        std::vector<std::string>{"cons.u_rad", "cons.uvec_rad"}, cons_map);
    const VarMap m_u(cons_map, true);

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
            GReal Xembed_fix[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, Xembed_fix);
            const GReal r_hor_fix = G.coords.get_horizon();
            const bool inside_horizon = (r_hor_fix > 0.0) && (Xembed_fix[1] < r_hor_fix);


            if (P(m_p.UU_RAD, k, j, i) < erad_floor || inside_horizon) {
                P(m_p.UU_RAD, k, j, i) = erad_floor;
                P(m_p.U1_RAD, k, j, i) = 0.0;
                P(m_p.U2_RAD, k, j, i) = 0.0;
                P(m_p.U3_RAD, k, j, i) = 0.0;

                Real Prad[4] = {P(m_p.UU_RAD, k, j, i), P(m_p.U1_RAD, k, j, i),
                    P(m_p.U2_RAD, k, j, i), P(m_p.U3_RAD, k, j, i)};
                Real Urad[4];
                RadM1::calc_tensor(G, Prad, 0, j, i, Urad);

                const Real gdet = G.gdet(Loci::center, j, i);
                U(m_u.UU_RAD, k, j, i) = Urad[0] * gdet;
                U(m_u.U1_RAD, k, j, i) = Urad[1] * gdet;
                U(m_u.U2_RAD, k, j, i) = Urad[2] * gdet;
                U(m_u.U3_RAD, k, j, i) = Urad[3] * gdet;
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

TaskStatus RadM1::BlockUtoP(MeshBlockData<Real>* rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();
    const auto& G = pmb->coords;

    // Pack Conserved Variables (Source)
    PackIndexMap cons_map;
    auto& U = rc->PackVariables(
        std::vector<std::string>{"cons.u_rad", "cons.uvec_rad"}, cons_map);
    VarMap m_u(cons_map, true);

    // Pack Primitive Variables (Destination)
    PackIndexMap prim_map;
    auto P = rc->PackVariables(
        std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prim_map);
    const VarMap m_p(prim_map, false);

    // Get Loop Bounds
    IndexRange3 b = KDomain::GetRange(rc, domain, coarse);

    // Parallel Loop
    pmb->par_for("RadM1_UtoP", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
        {
            Real Prad[4];
            Real Urad[4] = {U(m_u.UU_RAD, k, j, i), U(m_u.U1_RAD, k, j, i),
                U(m_u.U2_RAD, k, j, i), U(m_u.U3_RAD, k, j, i)};
            RadM1::u_to_p_rad(G, Urad, Prad, k, j, i);

            P(m_p.UU_RAD, k, j, i) = Prad[0];
            P(m_p.U1_RAD, k, j, i) = Prad[1];
            P(m_p.U2_RAD, k, j, i) = Prad[2];
            P(m_p.U3_RAD, k, j, i) = Prad[3];
        });
    return TaskStatus::complete;
}

// TaskStatus RadM1::Step(
//     MeshData<Real>* md_sub_init, MeshData<Real>* md_sub_final, const Real dt)
// {
//     for (int b = 0; b < md_sub_final->NumBlocks(); ++b) {
//         auto pmb_data = md_sub_final->GetBlockData(b);
//         auto pmb = pmb_data->GetBlockPointer();
//         auto& params = pmb->packages.Get("RadM1")->AllParams();

//         // Fetch parameters
//         const Real src_rootfind_eps = params.Get<Real>("src_rootfind_eps");
//         const Real src_rootfind_tol = params.Get<Real>("src_rootfind_tol");
//         const int src_rootfind_maxiter = params.Get<int>("src_rootfind_maxiter");
//         const auto& eos_params = pmb->packages.Get("eos")->AllParams();
//         auto eos = eos_params.Get<Microphysics::EOS::EOS>("d.EOS");
//         RadOpac rad_opac;
//         rad_opac.opacity_type = params.Get<int>("opacity_type");
//         rad_opac.const_sigma = params.Get<Real>("const_sigma");
//         rad_opac.const_kappa_a = params.Get<Real>("const_kappa_a");
//         rad_opac.const_kappa_sc = params.Get<Real>("const_kappa_sc");
//         rad_opac.units_cgs =
//             pmb->packages.Get("Units")->AllParams().Get<Units::UnitConversions>(
//                 "unit_conv");
//         if (pmb->packages.AllPackages().count("opacity")) {
//             rad_opac.sing_opac =
//                 pmb->packages.Get("opacity")->AllParams().Get<Microphysics::Opacities>(
//                     "opacities");
//         }

//         const auto& G = pmb->coords;

//         PackIndexMap prims_map, cons_map;
//         auto P_new =
//             pmb_data->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
//         auto U_new =
//             pmb_data->PackVariables({Metadata::WithFluxes, Metadata::Cell}, cons_map);
//         const VarMap m_p(prims_map, false);
//         const VarMap m_u(cons_map, true);

//         auto rimplflag = pmb_data->PackVariables(std::vector<std::string>{"rimplflag"});
//         auto pflag = pmb_data->PackVariables(std::vector<std::string>{"pflag"});

//         auto rinvflag = pmb_data->PackVariables(std::vector<std::string>{"rinvflag"});

//         auto pmb_init_data = md_sub_init->GetBlockData(b);

//         auto P_init =
//             pmb_init_data->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
//         auto U_init = pmb_init_data->PackVariables(
//             {Metadata::WithFluxes, Metadata::Cell}, cons_map);

//         auto bounds = pmb->cellbounds;
//         const IndexRange ib = bounds.GetBoundsI(IndexDomain::interior);
//         const IndexRange jb = bounds.GetBoundsJ(IndexDomain::interior);
//         const IndexRange kb = bounds.GetBoundsK(IndexDomain::interior);

//         // TODO (PNM): Split it, Cora thinks this is too large to be good. Probably too
//         // slow.
//         pmb->par_for("RadM1_Implicit_Solver4D", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
//             KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
//             {
//                 const Real U_entry[8] = {U_new(m_u.UU, k, j, i), U_new(m_u.U1, k, j, i),
//                     U_new(m_u.U2, k, j, i), U_new(m_u.U3, k, j, i),
//                     U_new(m_u.UU_RAD, k, j, i), U_new(m_u.U1_RAD, k, j, i),
//                     U_new(m_u.U2_RAD, k, j, i), U_new(m_u.U3_RAD, k, j, i)};

//                 const Real P_entry[9] = {P_new(m_p.RHO, k, j, i), P_new(m_p.UU, k, j, i),
//                     P_new(m_p.U1, k, j, i), P_new(m_p.U2, k, j, i),
//                     P_new(m_p.U3, k, j, i), P_new(m_p.UU_RAD, k, j, i),
//                     P_new(m_p.U1_RAD, k, j, i), P_new(m_p.U2_RAD, k, j, i),
//                     P_new(m_p.U3_RAD, k, j, i)};
//                 int rflagl;

//                 rflagl = solve_4d_pmhd(G, U_init, P_init, P_new, U_new, m_p, m_u, k, j, i,
//                     dt, eos, src_rootfind_eps, src_rootfind_tol, src_rootfind_maxiter,
//                     rad_opac, pflag, rinvflag, U_entry);

//                 // If the solver converged, but the final U_to_p for the fluid failed, we
//                 // should not be dealing with this, just accept this as it worked and send
//                 // straight to fixup.
//                 if (rflagl == static_cast<int>(StatusImplicitStep::success)) {
//                     rimplflag(0, k, j, i) = rflagl;
//                     return;
//                 }

//                 rflagl = solve_4d_prad(G, U_init, P_init, P_new, U_new, m_p, m_u, k, j, i,
//                     dt, eos, src_rootfind_eps, src_rootfind_tol, src_rootfind_maxiter,
//                     rad_opac, pflag, rinvflag, U_entry);

//                 if (rflagl == static_cast<int>(StatusImplicitStep::success)) {
//                     rimplflag(0, k, j, i) =
//                         static_cast<int>(StatusImplicitStep::pradfallback_success);
//                     return;
//                 }

//                 // Because of how Prad needs to do multiple kaustaun and kaustaun will
//                 // write to P_new, we need to reset P_new to the original values before
//                 // calling the 1D fallback. We don't need to do the same for pmhd because
//                 // we roll it back inside the function in case it fails (since it's only 1
//                 // u_to_p call for the plasma).
//                 P_new(m_p.RHO, k, j, i) = P_entry[0];
//                 P_new(m_p.UU, k, j, i) = P_entry[1];
//                 P_new(m_p.U1, k, j, i) = P_entry[2];
//                 P_new(m_p.U2, k, j, i) = P_entry[3];
//                 P_new(m_p.U3, k, j, i) = P_entry[4];
//                 P_new(m_p.UU_RAD, k, j, i) = P_entry[5];
//                 P_new(m_p.U1_RAD, k, j, i) = P_entry[6];
//                 P_new(m_p.U2_RAD, k, j, i) = P_entry[7];
//                 P_new(m_p.U3_RAD, k, j, i) = P_entry[8];

//                 auto status_1d = solve_radiation_1d(G, U_init, P_init, m_p, m_u, U_new,
//                     P_new, eos, rad_opac, k, j, i, dt, src_rootfind_tol,
//                     src_rootfind_maxiter, pflag, rinvflag, U_entry);

//                 if (status_1d == StatusImplicitStep::success) {
//                     rimplflag(0, k, j, i) =
//                         static_cast<int>(StatusImplicitStep::onedfallback_success);
//                     return;
//                 }

//                 rimplflag(0, k, j, i) =
//                     static_cast<int>(StatusImplicitStep::onedfallback_failure);

//                 assume_no_interaction(G, U_init, P_init, m_p, m_u, U_new, P_new, eos,
//                     rad_opac, k, j, i, dt, src_rootfind_tol, src_rootfind_maxiter, pflag,
//                     rinvflag, U_entry);
//             });
//     }

//     return TaskStatus::complete;
// }



void RadM1::AddSourceImplicitly(
    MeshData<Real>* md_sub_init, MeshData<Real>* md_flux_src, IndexDomain domain)
{
    for (int b = 0; b < md_sub_init->NumBlocks(); ++b) {
        auto pmb_data = md_sub_init->GetBlockData(b);
        auto pmb = pmb_data->GetBlockPointer();
        auto& params = pmb->packages.Get("RadM1")->AllParams();

        // Fetch parameters
        const Real dt = pmb->packages.Get("RadM1")->Param<Real>("current_stage_dt");
        const Real src_rootfind_eps = params.Get<Real>("src_rootfind_eps");
        const Real src_rootfind_tol = params.Get<Real>("src_rootfind_tol");
        const int src_rootfind_maxiter = params.Get<int>("src_rootfind_maxiter");
        const auto& eos_params = pmb->packages.Get("eos")->AllParams();
        auto eos = eos_params.Get<Microphysics::EOS::EOS>("d.EOS");
        RadOpac rad_opac;
        rad_opac.opacity_type = params.Get<int>("opacity_type");
        rad_opac.const_sigma = params.Get<Real>("const_sigma");
        rad_opac.const_kappa_a = params.Get<Real>("const_kappa_a");
        rad_opac.const_kappa_sc = params.Get<Real>("const_kappa_sc");
        rad_opac.units_cgs =
            pmb->packages.Get("Units")->AllParams().Get<Units::UnitConversions>(
                "unit_conv");
        if (pmb->packages.AllPackages().count("opacity")) {
            rad_opac.sing_opac =
                pmb->packages.Get("opacity")->AllParams().Get<Microphysics::Opacities>(
                    "opacities");
        }

        const auto& G = pmb->coords;

        PackIndexMap prims_map, cons_map;
        auto P_init_substep =
            pmb_data->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
        auto U_init_substep =
            pmb_data->PackVariables({Metadata::WithFluxes, Metadata::Cell}, cons_map);

        // Is this declaration correct? ASK CORA
        auto dU_substep = md_flux_src->GetBlockData(b)->PackVariables(
            {Metadata::WithFluxes, Metadata::Cell}, cons_map);
        const VarMap m_p(prims_map, false);
        const VarMap m_u(cons_map, true);

        auto rimplflag = pmb_data->PackVariables(std::vector<std::string>{"rimplflag"});
        auto pflag = pmb_data->PackVariables(std::vector<std::string>{"pflag"});

        auto rinvflag = pmb_data->PackVariables(std::vector<std::string>{"rinvflag"});

        auto bounds = pmb->cellbounds;
        const IndexRange ib = bounds.GetBoundsI(IndexDomain::interior);
        const IndexRange jb = bounds.GetBoundsJ(IndexDomain::interior);
        const IndexRange kb = bounds.GetBoundsK(IndexDomain::interior);

        // TODO (PNM): Split it, Cora thinks this is too large to be good. Probably too
        // slow.
        pmb->par_for("RadM1_Implicit_Solver4D", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
            {
                // Check if it's within the horizon, if it is, just assume no interaction and dU_subinit = 0;
                // PNM: I've been having some trouble getting it to stay controled within the horizon.
                
                GReal Xembed[GR_DIM];
                G.coord_embed(k, j, i, Loci::center, Xembed);
                const GReal r = Xembed[1];
                const GReal r_hor = G.coords.get_horizon();
                // If there is no horizon, r_hor = 0.0. For some of the tests, we don't have a horizon, and infact, we have negative values
                // so we don't want this check.
                if (r_hor > 0.0 && r < r_hor) {
                    rimplflag(0, k, j, i) = static_cast<int>(StatusImplicitStep::success);
                    return;
                }


                const Real U_entry[8] = {U_init_substep(m_u.UU, k, j, i), U_init_substep(m_u.U1, k, j, i),
                    U_init_substep(m_u.U2, k, j, i), U_init_substep(m_u.U3, k, j, i),
                    U_init_substep(m_u.UU_RAD, k, j, i), U_init_substep(m_u.U1_RAD, k, j, i),
                    U_init_substep(m_u.U2_RAD, k, j, i), U_init_substep(m_u.U3_RAD, k, j, i)};

                const Real P_entry[9] = {P_init_substep(m_p.RHO, k, j, i), P_init_substep(m_p.UU, k, j, i),
                    P_init_substep(m_p.U1, k, j, i), P_init_substep(m_p.U2, k, j, i),
                    P_init_substep(m_p.U3, k, j, i), P_init_substep(m_p.UU_RAD, k, j, i),
                    P_init_substep(m_p.U1_RAD, k, j, i), P_init_substep(m_p.U2_RAD, k, j, i),
                    P_init_substep(m_p.U3_RAD, k, j, i)};

                Real dS_subinit[5] = {0., 0., 0., 0., 0.};
                int rflagl;


                rflagl = solve_4d_pmhd(G, P_init_substep, m_p, m_u, k, j, i,
                    dt, eos, src_rootfind_eps, src_rootfind_tol, src_rootfind_maxiter,
                    rad_opac, pflag, rinvflag, U_entry, dS_subinit);

                // If the solver converged, but the final U_to_p for the fluid failed, we
                // should not be dealing with this, just accept this as it worked and send
                // straight to fixup.
                if (rflagl == static_cast<int>(StatusImplicitStep::success)) {
                    dU_substep(m_u.UU, k, j, i) -= dS_subinit[0];
                    dU_substep(m_u.U1, k, j, i) -= dS_subinit[1];
                    dU_substep(m_u.U2, k, j, i) -= dS_subinit[2];
                    dU_substep(m_u.U3, k, j, i) -= dS_subinit[3];
                    dU_substep(m_u.UU_RAD, k, j, i) += dS_subinit[0];
                    dU_substep(m_u.U1_RAD, k, j, i) += dS_subinit[1];
                    dU_substep(m_u.U2_RAD, k, j, i) += dS_subinit[2];
                    dU_substep(m_u.U3_RAD, k, j, i) += dS_subinit[3];
                    if (m_u.KTOT >= 0) dU_substep(m_u.KTOT, k, j, i) += dS_subinit[4];
                    rimplflag(0, k, j, i) = rflagl;
                    return;
                }

                rflagl = solve_4d_prad(G, U_init_substep, P_init_substep, m_p, m_u, k, j, i,
                    dt, eos, src_rootfind_eps, src_rootfind_tol, src_rootfind_maxiter,
                    rad_opac, pflag, rinvflag, U_entry, dS_subinit);

                // Prad alters the P_init. So we gotta revert it back.
                U_init_substep(m_u.UU, k, j, i) = U_entry[0];
                U_init_substep(m_u.U1, k, j, i) = U_entry[1];
                U_init_substep(m_u.U2, k, j, i) = U_entry[2];
                U_init_substep(m_u.U3, k, j, i) = U_entry[3];
                P_init_substep(m_p.RHO, k, j, i) = P_entry[0];
                P_init_substep(m_p.UU, k, j, i) = P_entry[1];
                P_init_substep(m_p.U1, k, j, i) = P_entry[2];
                P_init_substep(m_p.U2, k, j, i) = P_entry[3];
                P_init_substep(m_p.U3, k, j, i) = P_entry[4];

                if (rflagl == static_cast<int>(StatusImplicitStep::success)) {
                    dU_substep(m_u.UU, k, j, i) -= dS_subinit[0];
                    dU_substep(m_u.U1, k, j, i) -= dS_subinit[1];
                    dU_substep(m_u.U2, k, j, i) -= dS_subinit[2];
                    dU_substep(m_u.U3, k, j, i) -= dS_subinit[3];
                    dU_substep(m_u.UU_RAD, k, j, i) += dS_subinit[0];
                    dU_substep(m_u.U1_RAD, k, j, i) += dS_subinit[1];
                    dU_substep(m_u.U2_RAD, k, j, i) += dS_subinit[2];
                    dU_substep(m_u.U3_RAD, k, j, i) += dS_subinit[3];
                    if (m_u.KTOT >= 0) dU_substep(m_u.KTOT, k, j, i) += dS_subinit[4];

                    rimplflag(0, k, j, i) =
                        static_cast<int>(StatusImplicitStep::pradfallback_success);
                    return;
                }


                auto status_1d = solve_radiation_1d(G, P_init_substep, m_p, m_u, eos, rad_opac, k, j, i, dt, src_rootfind_tol,
                    src_rootfind_maxiter, pflag, rinvflag, U_entry, dS_subinit);

                if (status_1d == StatusImplicitStep::success) {
                    dU_substep(m_u.UU, k, j, i) -= dS_subinit[0];
                    dU_substep(m_u.U1, k, j, i) -= dS_subinit[1];
                    dU_substep(m_u.U2, k, j, i) -= dS_subinit[2];
                    dU_substep(m_u.U3, k, j, i) -= dS_subinit[3];
                    dU_substep(m_u.UU_RAD, k, j, i) += dS_subinit[0];
                    dU_substep(m_u.U1_RAD, k, j, i) += dS_subinit[1];
                    dU_substep(m_u.U2_RAD, k, j, i) += dS_subinit[2];
                    dU_substep(m_u.U3_RAD, k, j, i) += dS_subinit[3];
                    
                    rimplflag(0, k, j, i) =
                        static_cast<int>(StatusImplicitStep::onedfallback_success);
                    return;
                }

                rimplflag(0, k, j, i) =
                    static_cast<int>(StatusImplicitStep::onedfallback_failure);
                
            });
    }

}

TaskStatus RadM1::PostStepDiagnostics(const SimTime& tm, MeshData<Real>* md)
{
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    const auto& pars = pmesh->packages.Get("Globals")->AllParams();
    const int flag_verbose = pars.Get<int>("flag_verbose");

    if (flag_verbose >= 1) {
        Reductions::StartFlagReduce(md, "rimplflag", RadM1::status_names_implicit,
            IndexDomain::interior, false, 3);
        auto total_flag_counts = Reductions::CheckFlagReduceAndPrintHits(md, "rimplflag",
            RadM1::status_names_implicit, IndexDomain::interior, false, 3);
        Reductions::PrintFlagPercentages(md, "rimplflag", RadM1::status_names_implicit,
            IndexDomain::interior, total_flag_counts);
        // Radiation inversion flags
        Reductions::StartFlagReduce(md, "rinvflag", RadM1::status_names_inversion,
            IndexDomain::interior, false, 4);
        auto rad_inv_counts = Reductions::CheckFlagReduceAndPrintHits(md, "rinvflag",
            RadM1::status_names_inversion, IndexDomain::interior, false, 4);
        Reductions::PrintFlagPercentages(md, "rinvflag", RadM1::status_names_inversion,
            IndexDomain::interior, rad_inv_counts);
    }

    return TaskStatus::complete;
}
