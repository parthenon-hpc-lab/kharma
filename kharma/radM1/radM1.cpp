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
#include "floors_functions.hpp"
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
    pkg->AddField("rad_imex_src",
        Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
            std::vector<int>{5}));
    // Cell-centered total opacity, for the radiation signal speeds in the fluxes
    pkg->AddField("rad_kappa",
        Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy}));
    // Newton iterations per cell in the last implicit call: pmhd, prad, ff (diagnostic)
    pkg->AddField("solver_count",
        Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
            std::vector<int>{3}));
    Real u_rad_floor = pin->GetOrAddReal("radM1", "u_rad_floor", 1.e-40);
    pkg->AllParams().Add("u_rad_floor", u_rad_floor);

    Real src_rootfind_eps = pin->GetOrAddReal("radM1", "src_rootfind_eps", 1e-8);
    Real src_rootfind_tol = pin->GetOrAddReal("radM1", "src_rootfind_tol", 1e-8);
    int src_rootfind_maxiter = pin->GetOrAddInteger("radM1", "src_rootfind_maxiter", 50);
    pkg->AllParams().Add("src_rootfind_eps", src_rootfind_eps);
    pkg->AllParams().Add("src_rootfind_tol", src_rootfind_tol);
    pkg->AllParams().Add("src_rootfind_maxiter", src_rootfind_maxiter);

    // Separate iteration limits for the PMHD and fluid-frame solvers. It defaults to
    // src_rootfind_maxiter. P_rad still keeps src_rootfind_maxiter. The names are pretty
    // much self explanatory.
    int pmhd_maxiter =
        pin->GetOrAddInteger("radM1", "pmhd_maxiter", src_rootfind_maxiter);
    pkg->AllParams().Add("pmhd_maxiter", pmhd_maxiter);

    int ff_maxiter = pin->GetOrAddInteger("radM1", "ff_maxiter", src_rootfind_maxiter);
    pkg->AllParams().Add("ff_maxiter", ff_maxiter);

    // Start the fluid-frame solver from the gas-radiation equilibrium temperature when
    // that guess has a smaller residual than the star state.
    // PNM: I found that this is useful for optically thick flows, but I'm not sure how
    // much this small test impacts performance in optically thin flows. I think it's not
    // that important though, so maybe just keep it on.
    bool ff_eq_guess = pin->GetOrAddBoolean("radM1", "ff_eq_guess", false);
    pkg->AllParams().Add("ff_eq_guess", ff_eq_guess);

    // When every solver fails, accept the last solver's best state if its error reached
    // this (relaxed_success). 0 turns it off
    Real accept_tol = pin->GetOrAddReal("radM1", "accept_tol", 1.e-4);
    pkg->AllParams().Add("accept_tol", accept_tol);

    // PMHD iterations for cells the fluid frame solver took in the last implicit call.
    // They still try PMHD every call, just with this budget.
    // If it failed last stage/step, it will maybe fail this one too, so instead of giving
    // it the full budget, just give it a smaller budget to save time.
    int pmhd_maxiter_after_ff =
        pin->GetOrAddInteger("radM1", "pmhd_maxiter_after_ff", src_rootfind_maxiter);
    pkg->AllParams().Add("pmhd_maxiter_after_ff", pmhd_maxiter_after_ff);

    // Try P_rad on cells the fluid-frame solver fails. If false, they are failures
    // P_rad is very slow due to multiple kaustaun calls, so this is off by default, but I
    // leave the choice up to the user.
    bool prad_fallback = pin->GetOrAddBoolean("radM1", "prad_fallback", false);

    pkg->AllParams().Add("prad_fallback", prad_fallback);

    // Second order stuff, we update these every step.
    pkg->AllParams().Add("current_stage_dt", 0.0, true);
    pkg->AllParams().Add("current_stage_gam0", 0.0, true);
    pkg->AllParams().Add("current_stage_gam1", 1.0, true);
    pkg->AllParams().Add("current_stage", 1, true);
    pkg->AllParams().Add("current_dt_full", 0.0, true);

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
    pkg->AllParams().Add("mean_molecular_weight", pin->GetOrAddReal("radM1", "mu", 1.0));
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
            // Inside the horizon too, only the floor (no reset to it)
            if (P(m_p.UU_RAD, k, j, i) < erad_floor) {
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

    PackIndexMap cons_map;
    auto& U = rc->PackVariables(
        std::vector<std::string>{"cons.u_rad", "cons.uvec_rad"}, cons_map);
    VarMap m_u(cons_map, true);

    PackIndexMap prim_map;
    auto P = rc->PackVariables(
        std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prim_map);
    const VarMap m_p(prim_map, false);

    IndexRange3 b = KDomain::GetRange(rc, domain, coarse);

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
    auto rinvflag = rc->PackVariables(std::vector<std::string>{"rinvflag"});
    // Parallel Loop
    pmb->par_for("RadM1_UtoP", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
        {
            Real Prad[4];
            Real Urad[4] = {U(m_u.UU_RAD, k, j, i), U(m_u.U1_RAD, k, j, i),
                U(m_u.U2_RAD, k, j, i), U(m_u.U3_RAD, k, j, i)};
            const auto status =
                RadM1::u_to_p_rad(G, Urad, Prad, k, j, i, nullptr, RadLimiterType::BASIC);
            rinvflag(0, k, j, i) = static_cast<int>(status);
            P(m_p.UU_RAD, k, j, i) = Prad[0];
            P(m_p.U1_RAD, k, j, i) = Prad[1];
            P(m_p.U2_RAD, k, j, i) = Prad[2];
            P(m_p.U3_RAD, k, j, i) = Prad[3];
        });
    return TaskStatus::complete;
}

TaskStatus RadM1::FixRadiation(MeshBlockData<Real>* rc)
{
    Flag("RadM1::FixRadiation");
    auto pmb = rc->GetBlockPointer();
    const auto& G = pmb->coords;

    PackIndexMap rad_prims_map;
    auto Prad = rc->PackVariables(
        std::vector<std::string>{"prims.u_rad", "prims.uvec_rad"}, rad_prims_map);
    const VarMap m_prad(rad_prims_map, false);

    GridScalar rinvflag = rc->Get("rinvflag").data;
    GridScalar rimplflag = rc->Get("rimplflag").data;
    GridScalar pflag = rc->Get("pflag").data;
    PackIndexMap mhd_prims_map, mhd_cons_map;
    auto Pg = GRMHD::PackMHDPrims(rc, mhd_prims_map);
    auto Ug = GRMHD::PackMHDCons(rc, mhd_cons_map);
    const VarMap m_pg(mhd_prims_map, false), m_ug(mhd_cons_map, true);
    auto eos = pmb->packages.Get("eos")->AllParams().Get<Microphysics::EOS::EOS>("d.EOS");
    const int impl_fail = static_cast<int>(StatusImplicitStep::failure);

    const Real erad_floor = pmb->packages.Get("RadM1")->Param<Real>("u_rad_floor");
    const GReal r_hor = G.coords.get_horizon();

    const IndexRange3 b = KDomain::GetPhysicalRange(rc);

    pmb->par_for("fix_radiation", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
        {
            GReal Xembed[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, Xembed);
            const bool inside_horizon = r_hor > 0.0 && Xembed[1] < r_hor;

            const bool bad =
                !inside_horizon && (rad_inversion_failed(rinvflag(0, k, j, i)) ||
                                       rimplflag(0, k, j, i) ==
                                           static_cast<int>(StatusImplicitStep::failure));
            if (bad) {
                // Implicit failures also average u, u^i (not rho, B). I took this from
                // Koral Lite, previously I was just avg rad variables.
                const bool gas_bad =
                    !inside_horizon && rimplflag(0, k, j, i) == impl_fail;
                double gsum = 0.;
                double gas[4] = {0.};
                double wsum = 0.;
                double sum[4] = {0.};
                for (int n = -1; n <= 1; n++) {
                    for (int m = -1; m <= 1; m++) {
                        for (int l = -1; l <= 1; l++) {
                            int ii = i + l, jj = j + m, kk = k + n;
                            if (KDomain::inside(kk, jj, ii, b)) {
                                GReal Xembed_n[GR_DIM];
                                G.coord_embed(kk, jj, ii, Loci::center, Xembed_n);
                                const bool n_inside_horizon =
                                    r_hor > 0.0 && Xembed_n[1] < r_hor;
                                const bool nbad =
                                    n_inside_horizon ||
                                    rad_inversion_failed(rinvflag(0, kk, jj, ii)) ||
                                    rimplflag(0, kk, jj, ii) ==
                                        static_cast<int>(StatusImplicitStep::failure);
                                if (gas_bad && rimplflag(0, kk, jj, ii) != impl_fail &&
                                    !Inverter::failed(pflag(kk, jj, ii))) {
                                    const double w =
                                        1. / (m::abs(l) + m::abs(m) + m::abs(n) + 1);
                                    gsum += w;
                                    // u/rho, so a low-density cell gets its neighbors'
                                    // temperature rather than their energy density
                                    const Real rho_n = Pg(m_pg.RHO, kk, jj, ii);
                                    gas[0] += w * Pg(m_pg.UU, kk, jj, ii) / rho_n;
                                    for (int v = 0; v < NVEC; v++)
                                        gas[1 + v] += w * Pg(m_pg.U1 + v, kk, jj, ii);
                                }
                                if (!nbad) {
                                    double w =
                                        1. / (m::abs(l) + m::abs(m) + m::abs(n) + 1);
                                    wsum += w;
                                    // Average T^t_mu, not E_rf and u^i
                                    const Real Pn[4] = {Prad(m_prad.UU_RAD, kk, jj, ii),
                                        Prad(m_prad.U1_RAD, kk, jj, ii),
                                        Prad(m_prad.U2_RAD, kk, jj, ii),
                                        Prad(m_prad.U3_RAD, kk, jj, ii)};
                                    Real Tn[GR_DIM];
                                    RadM1::calc_tensor(G, Pn, 0, jj, ii, Tn);
                                    for (int mu = 0; mu < GR_DIM; mu++)
                                        sum[mu] += w * Tn[mu];
                                }
                            }
                        }
                    }
                }

                if (wsum < 1.e-10) {
                    Prad(m_prad.UU_RAD, k, j, i) = erad_floor;
                    Prad(m_prad.U1_RAD, k, j, i) = 0.;
                    Prad(m_prad.U2_RAD, k, j, i) = 0.;
                    Prad(m_prad.U3_RAD, k, j, i) = 0.;
                } else {
                    const Real gdet = G.gdet(Loci::center, j, i);
                    Real U_avg[4], P_avg[4];
                    for (int mu = 0; mu < GR_DIM; mu++) U_avg[mu] = sum[mu] / wsum * gdet;
                    const auto st = RadM1::u_to_p_rad(
                        G, U_avg, P_avg, k, j, i, nullptr, RadLimiterType::BASIC);
                    if (rad_inversion_failed(static_cast<int>(st))) {
                        P_avg[0] = erad_floor;
                        P_avg[1] = P_avg[2] = P_avg[3] = 0.;
                    }
                    Prad(m_prad.UU_RAD, k, j, i) = P_avg[0];
                    Prad(m_prad.U1_RAD, k, j, i) = P_avg[1];
                    Prad(m_prad.U2_RAD, k, j, i) = P_avg[2];
                    Prad(m_prad.U3_RAD, k, j, i) = P_avg[3];
                }
                if (gas_bad && gsum > 1.e-10) {
                    Pg(m_pg.UU, k, j, i) = Pg(m_pg.RHO, k, j, i) * gas[0] / gsum;
                    for (int v = 0; v < NVEC; v++)
                        Pg(m_pg.U1 + v, k, j, i) = gas[1 + v] / gsum;
                    GRMHD::p_to_u(G, Pg, m_pg, eos, k, j, i, Ug, m_ug);
                }
            }
        });

    RadM1::BlockPtoU(rc, IndexDomain::entire);

    EndFlag();
    return TaskStatus::complete;
}

RadM1::RadOpac RadM1::GetRadOpac(Packages_t& packages)
{
    const auto& params = packages.Get("RadM1")->AllParams();
    const auto& opac_pars = packages.Get("opacity")->AllParams();
    RadOpac rad_opac;
    rad_opac.opacity_type = params.Get<int>("opacity_type");
    rad_opac.const_sigma = params.Get<Real>("const_sigma");
    rad_opac.const_kappa_a = params.Get<Real>("const_kappa_a");
    rad_opac.const_kappa_sc = params.Get<Real>("const_kappa_sc");
    rad_opac.mean_molecular_weight = params.Get<Real>("mean_molecular_weight");
    rad_opac.regime = opac_pars.Get<RadOpac::FitType>("regime");
    rad_opac.use_bremsstrahlung = opac_pars.Get<bool>("use_bremsstrahlung");
    rad_opac.use_synchrotron = opac_pars.Get<bool>("use_synchrotron");
    rad_opac.use_tops_table = opac_pars.Get<bool>("use_tops_table");
    rad_opac.units_cgs =
        packages.Get("Units")->AllParams().Get<Units::UnitConversions>("unit_conv");
    rad_opac.sing_opac = opac_pars.Get<Microphysics::Opacities>("opacities");
    return rad_opac;
}

TaskStatus RadM1::CalcKappa(MeshData<Real>* md)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    const RadOpac rad_opac = GetRadOpac(pmb0->packages);
    const auto& eos_params = pmb0->packages.Get("eos")->AllParams();
    auto eos = eos_params.Get<Microphysics::EOS::EOS>("d.EOS");
    PackIndexMap prims_map;
    const auto& P = md->PackVariables(
        std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prims_map);
    const VarMap m_p(prims_map, false);
    const auto& kappa = md->PackVariables(std::vector<std::string>{"rad_kappa"});
    const IndexRange3 b = KDomain::GetRange(md, IndexDomain::entire);
    pmb0->par_for("rad_kappa", 0, P.GetDim(5) - 1, b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA(const int& bl, const int& k, const int& j, const int& i)
        {
            const auto& G = P.GetCoords(bl);
            FourVectors D;
            GRMHD::calc_4vecs(G, P(bl), m_p, k, j, i, Loci::center, D);
            kappa(bl, 0, k, j, i) = inside_horizon(G, k, j, i, Loci::center)
                                        ? 0.
                                        : calc_kappa_tot(G, P(bl), m_p, D, eos, rad_opac,
                                              k, j, i, Loci::center);
        });
    return TaskStatus::complete;
}

void RadM1::AddSourceImplicitly(
    MeshData<Real>* md_sub_init, MeshData<Real>* md_flux_src, IndexDomain domain)
{
    using SIS = StatusImplicitStep;
    for (int b = 0; b < md_sub_init->NumBlocks(); ++b) {
        auto pmb_data = md_sub_init->GetBlockData(b);
        auto pmb = pmb_data->GetBlockPointer();
        auto& params = pmb->packages.Get("RadM1")->AllParams();

        // Fetch parameters
        // 2nd order parameters
        const Real dt = pmb->packages.Get("RadM1")->Param<Real>("current_stage_dt");
        const Real gam0 = pmb->packages.Get("RadM1")->Param<Real>("current_stage_gam0");
        const Real gam1 = pmb->packages.Get("RadM1")->Param<Real>("current_stage_gam1");
        const int stage = pmb->packages.Get("RadM1")->Param<int>("current_stage");
        const Real dt_full = pmb->packages.Get("RadM1")->Param<Real>("current_dt_full");
        const Real dt_impl = (1.0 - 1.0 / std::sqrt(2.0)) * dt_full;
        const Real w_solve = dt_impl / dt;
        const Real w_hist = (stage == 2) ? std::sqrt(2.0) * dt_full / dt : 0.0;

        // Solver parameters, such as iteration limits, tolerances, FD stepsize, etc...
        const Real src_rootfind_eps = params.Get<Real>("src_rootfind_eps");
        const Real src_rootfind_tol = params.Get<Real>("src_rootfind_tol");
        const int src_rootfind_maxiter = params.Get<int>("src_rootfind_maxiter");
        const int pmhd_maxiter = params.Get<int>("pmhd_maxiter");
        const int ff_maxiter = params.Get<int>("ff_maxiter");
        // Try to start the fluid-frame solver from the gas-radiation equilibrium
        const bool ff_eq_guess = params.Get<bool>("ff_eq_guess");
        const Real accept_tol = params.Get<Real>("accept_tol");
        const int pmhd_maxiter_after_ff = params.Get<int>("pmhd_maxiter_after_ff");
        const bool prad_fallback = params.Get<bool>("prad_fallback");
        const auto& eos_params = pmb->packages.Get("eos")->AllParams();
        auto eos = eos_params.Get<Microphysics::EOS::EOS>("d.EOS");
        const RadOpac rad_opac = GetRadOpac(pmb->packages);

        const auto& G = pmb->coords;

        PackIndexMap prims_map, cons_map;
        auto P_init_substep =
            pmb_data->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
        auto U_init_substep =
            pmb_data->PackVariables({Metadata::WithFluxes, Metadata::Cell}, cons_map);

        auto dU_substep = md_flux_src->GetBlockData(b)->PackVariables(
            {Metadata::WithFluxes, Metadata::Cell}, cons_map);
        const VarMap m_p(prims_map, false);
        const VarMap m_u(cons_map, true);
        PackIndexMap cons_map_base;
        auto U_base = pmb->meshblock_data.Get()->PackVariables(
            {Metadata::WithFluxes, Metadata::Cell}, cons_map_base);
        const VarMap m_ub(cons_map_base, true);

        // Assinging flags, this is defined in radm1.hpp, but basically rimplflag are the
        // implicit solver flags, rinvflag are the inversion flags, and pflag are the
        // primitive variable flags.
        auto rimplflag = pmb_data->PackVariables(std::vector<std::string>{"rimplflag"});
        auto pflag = pmb_data->PackVariables(std::vector<std::string>{"pflag"});
        auto rinvflag = pmb_data->PackVariables(std::vector<std::string>{"rinvflag"});

        // rad_imex_src stores the stage-1 implicit source of the two-stage IMEX scheme,
        // cause we want 2nd order accuracy.
        auto imex_src = pmb_data->PackVariables(std::vector<std::string>{"rad_imex_src"});

        // How many iterations it took for each cell to exit the solver.
        auto solver_count =
            pmb_data->PackVariables(std::vector<std::string>{"solver_count"});
        const bool use_floors = pmb->packages.AllPackages().count("Floors");
        const Floors::Prescription floors =
            use_floors
                ? pmb->packages.Get("Floors")->Param<Floors::Prescription>("prescription")
                : Floors::Prescription();

        // In this solver, cells can be ill conditioned, since we are considering
        // post-flux update states. Sometimes, we need to fix it, before calculating the
        // implicit solver and later actually going to floors + fixup.

        // the face-centred magnetic field cons.fB (gdet·B^i on faces), taken from the
        // current stage's start. It's weighted by gam0.
        auto Bf_sub = pmb_data->PackVariables(std::vector<std::string>{"cons.fB"});
        // Same field as Bf_sub, but taken from the start of the full timestep. It's
        // weighted by gam1.
        auto Bf_base =
            pmb->meshblock_data.Get()->PackVariables(std::vector<std::string>{"cons.fB"});

        // This is this stage's dB/dt of the face field, multiplied by dt.
        auto dBf = md_flux_src->GetBlockData(b)->PackVariables(
            std::vector<std::string>{"cons.fB"});
        // Just a boolean to check if we have a face field or not.
        const bool face_b = Bf_sub.GetDim(4) > 0;

        // the coefficient that adds the stage-1 explicit tendency again at stage 2. It's
        // the delta of the ARS(2,2,2) IMEX scheme
        const Real ars_d1 =
            (stage == 2) ? (1.0 - 1.0 / (2.0 * (1.0 - 1.0 / std::sqrt(2.0)))) * dt_full
                         : 0.0;
        PackIndexMap cons_map_d1;
        // the stage-1 explicit flux divergence plus explicit sources for the cell
        // variables, and its index map.
        auto dU1 = pmb->meshblock_data.Get("dUdt_s1")->PackVariables(
            {Metadata::WithFluxes, Metadata::Cell}, cons_map_d1);
        const VarMap m_d1(cons_map_d1, true);
        // Same stage 1 explicit for the face field.
        auto dBf1 = pmb->meshblock_data.Get("dUdt_s1")->PackVariables(
            std::vector<std::string>{"cons.fB"});
        const int ndim = pmb->pmy_mesh->ndim;

        // eos adiabatic index
        const Real gm1 = eos_params.Get<Real>("gm1");
        const Real gam = gm1 + 1.0;
        auto bounds = pmb->cellbounds;
        const IndexRange ib = bounds.GetBoundsI(IndexDomain::interior);
        const IndexRange jb = bounds.GetBoundsJ(IndexDomain::interior);
        const IndexRange kb = bounds.GetBoundsK(IndexDomain::interior);

        // KOKKOS function to add the stage 1 term first, in order to make it second
        // order. This is applied for failed cells, they only get the first stage's source
        // term.
        auto apply_history = KOKKOS_LAMBDA(const int k, const int j, const int i)
        {
            const int iug[4] = {m_u.UU, m_u.U1, m_u.U2, m_u.U3};
            const int iur[4] = {m_u.UU_RAD, m_u.U1_RAD, m_u.U2_RAD, m_u.U3_RAD};
            Real G2[5];
            for (int n = 0; n < 5; n++) G2[n] = (stage == 2) ? imex_src(n, k, j, i) : 0.0;
            for (int n = 0; n < 4; n++) {
                dU_substep(iug[n], k, j, i) -= w_hist * G2[n];
                dU_substep(iur[n], k, j, i) += w_hist * G2[n];
            }
            if (m_u.KTOT_ADV >= 0)
                dU_substep(m_u.KTOT_ADV, k, j, i) +=
                    (gm1) / m::pow(P_init_substep(m_p.RHO, k, j, i), gm1) * w_hist *
                    G2[4];
        };

        // For cells that actually succeed, we combine both stages update.
        auto apply_source = KOKKOS_LAMBDA(
            const int k, const int j, const int i, Real dS[5], const int flag)
        {
            for (int n = 0; n < 5; n++) {
                const Real g2 = (stage == 2) ? imex_src(n, k, j, i) : 0.0;
                if (stage == 1) imex_src(n, k, j, i) = dS[n];
                dS[n] = w_solve * dS[n] + w_hist * g2;
            }
            dU_substep(m_u.UU, k, j, i) -= dS[0];
            dU_substep(m_u.U1, k, j, i) -= dS[1];
            dU_substep(m_u.U2, k, j, i) -= dS[2];
            dU_substep(m_u.U3, k, j, i) -= dS[3];
            dU_substep(m_u.UU_RAD, k, j, i) += dS[0];
            dU_substep(m_u.U1_RAD, k, j, i) += dS[1];
            dU_substep(m_u.U2_RAD, k, j, i) += dS[2];
            dU_substep(m_u.U3_RAD, k, j, i) += dS[3];
            if (m_u.KTOT_ADV >= 0)
                dU_substep(m_u.KTOT_ADV, k, j, i) +=
                    (gm1) / m::pow(P_init_substep(m_p.RHO, k, j, i), gm1) * dS[4];
            rimplflag(0, k, j, i) = flag;
        };

        // If failed, restore the original state.
        auto restore =
            KOKKOS_LAMBDA(const int k, const int j, const int i, const Real U_save[9],
                const Real P_save[9], const Real B_save[3], const Real PB_save[3])
        {
            const int iu[9] = {m_u.RHO, m_u.UU, m_u.U1, m_u.U2, m_u.U3, m_u.UU_RAD,
                m_u.U1_RAD, m_u.U2_RAD, m_u.U3_RAD};
            const int ip[9] = {m_p.RHO, m_p.UU, m_p.U1, m_p.U2, m_p.U3, m_p.UU_RAD,
                m_p.U1_RAD, m_p.U2_RAD, m_p.U3_RAD};
            for (int n = 0; n < 9; n++) {
                U_init_substep(iu[n], k, j, i) = U_save[n];
                P_init_substep(ip[n], k, j, i) = P_save[n];
            }
            if (m_u.B1 >= 0 && m_p.B1 >= 0) {
                for (int v = 0; v < 3; v++) {
                    U_init_substep(m_u.B1 + v, k, j, i) = B_save[v];
                    P_init_substep(m_p.B1 + v, k, j, i) = PB_save[v];
                }
            }
        };

        // When doing U->P conversion, maybe the state does not exist, so we fix it,
        // generate a new primitive state, and then continue with the implicit solver, but
        // with the OG U. If it ends up fail at the end, send it to fallbacks.
        auto build_star = KOKKOS_LAMBDA(const int k, const int j, const int i,
            Real U_save[9], Real P_save[9], Real B_save[3], Real PB_save[3]) -> bool
        {
            const int iug[4] = {m_u.UU, m_u.U1, m_u.U2, m_u.U3};
            const int iur[4] = {m_u.UU_RAD, m_u.U1_RAD, m_u.U2_RAD, m_u.U3_RAD};
            const int ifx[5] = {m_u.RHO, m_u.UU, m_u.U1, m_u.U2, m_u.U3};
            const int iu[9] = {m_u.RHO, m_u.UU, m_u.U1, m_u.U2, m_u.U3, m_u.UU_RAD,
                m_u.U1_RAD, m_u.U2_RAD, m_u.U3_RAD};
            const int iub[9] = {m_ub.RHO, m_ub.UU, m_ub.U1, m_ub.U2, m_ub.U3, m_ub.UU_RAD,
                m_ub.U1_RAD, m_ub.U2_RAD, m_ub.U3_RAD};
            const int ip[9] = {m_p.RHO, m_p.UU, m_p.U1, m_p.U2, m_p.U3, m_p.UU_RAD,
                m_p.U1_RAD, m_p.U2_RAD, m_p.U3_RAD};
            const int iud1[9] = {m_d1.RHO, m_d1.UU, m_d1.U1, m_d1.U2, m_d1.U3,
                m_d1.UU_RAD, m_d1.U1_RAD, m_d1.U2_RAD, m_d1.U3_RAD};
            Real G2[5];
            for (int n = 0; n < 5; n++) G2[n] = (stage == 2) ? imex_src(n, k, j, i) : 0.0;

            // Save the original state, in case we need to restore it.
            for (int n = 0; n < 9; n++) {
                U_save[n] = U_init_substep(iu[n], k, j, i);
                P_save[n] = P_init_substep(ip[n], k, j, i);
            }

            // Conserved state update, with the stage 1 explicit source term added back in
            // for stage 2.
            for (int n = 0; n < 9; n++)
                U_init_substep(iu[n], k, j, i) =
                    gam0 * U_save[n] + gam1 * U_base(iub[n], k, j, i) +
                    dt * dU_substep(iu[n], k, j, i) +
                    ((ars_d1 != 0.0) ? ars_d1 * dU1(iud1[n], k, j, i) : 0.0);

            // Add the stage 1 implicit source term
            for (int n = 0; n < 4; n++) {
                U_init_substep(iug[n], k, j, i) -= w_hist * dt * G2[n];
                U_init_substep(iur[n], k, j, i) += w_hist * dt * G2[n];
            }

            // Now we have to do the same thing with B, reconstruct the explicit update of
            // this stage + previous stage stuff.
            for (int v = 0; v < 3; v++) B_save[v] = PB_save[v] = 0.;
            if (m_u.B1 >= 0 && m_p.B1 >= 0) {
                const Real gdet_c = G.gdet(Loci::center, j, i);
                for (int v = 0; v < 3; v++) {
                    // Save the original state, in case we need to restore it.
                    B_save[v] = U_init_substep(m_u.B1 + v, k, j, i);
                    PB_save[v] = P_init_substep(m_p.B1 + v, k, j, i);

                    // Reconstruct the updated state, with the stage 1 explicit source
                    // term added back in for stage 2.
                    U_init_substep(m_u.B1 + v, k, j, i) =
                        gam0 * B_save[v] + gam1 * U_base(m_ub.B1 + v, k, j, i) +
                        dt * dU_substep(m_u.B1 + v, k, j, i) +
                        ((ars_d1 != 0.0) ? ars_d1 * dU1(m_d1.B1 + v, k, j, i) : 0.0);
                    P_init_substep(m_p.B1 + v, k, j, i) =
                        U_init_substep(m_u.B1 + v, k, j, i) / gdet_c;
                }

                // Now if we are doing face-centered B, we need to reconstruct it as well.
                if (face_b) {
                    const Real b1l =
                        gam0 * Bf_sub(F1, 0, k, j, i) + gam1 * Bf_base(F1, 0, k, j, i) +
                        dt * dBf(F1, 0, k, j, i) +
                        ((ars_d1 != 0.0) ? ars_d1 * dBf1(F1, 0, k, j, i) : 0.0);
                    const Real b1r =
                        gam0 * Bf_sub(F1, 0, k, j, i + 1) +
                        gam1 * Bf_base(F1, 0, k, j, i + 1) +
                        dt * dBf(F1, 0, k, j, i + 1) +
                        ((ars_d1 != 0.0) ? ars_d1 * dBf1(F1, 0, k, j, i + 1) : 0.0);
                    const Real b2l =
                        gam0 * Bf_sub(F2, 0, k, j, i) + gam1 * Bf_base(F2, 0, k, j, i) +
                        dt * dBf(F2, 0, k, j, i) +
                        ((ars_d1 != 0.0) ? ars_d1 * dBf1(F2, 0, k, j, i) : 0.0);
                    const Real b3l =
                        gam0 * Bf_sub(F3, 0, k, j, i) + gam1 * Bf_base(F3, 0, k, j, i) +
                        dt * dBf(F3, 0, k, j, i) +
                        ((ars_d1 != 0.0) ? ars_d1 * dBf1(F3, 0, k, j, i) : 0.0);
                    Real Bc[3];
                    Bc[0] = (b1l / G.gdet(Loci::face1, j, i) +
                                b1r / G.gdet(Loci::face1, j, i + 1)) /
                            2;
                    if (ndim > 1) {
                        const Real b2r =
                            gam0 * Bf_sub(F2, 0, k, j + 1, i) +
                            gam1 * Bf_base(F2, 0, k, j + 1, i) +
                            dt * dBf(F2, 0, k, j + 1, i) +
                            ((ars_d1 != 0.0) ? ars_d1 * dBf1(F2, 0, k, j + 1, i) : 0.0);
                        Bc[1] = (b2l / G.gdet(Loci::face2, j, i) +
                                    b2r / G.gdet(Loci::face2, j + 1, i)) /
                                2;
                    } else {
                        Bc[1] = b2l / G.gdet(Loci::face2, j, i);
                    }
                    if (ndim > 2) {
                        const Real b3r =
                            gam0 * Bf_sub(F3, 0, k + 1, j, i) +
                            gam1 * Bf_base(F3, 0, k + 1, j, i) +
                            dt * dBf(F3, 0, k + 1, j, i) +
                            ((ars_d1 != 0.0) ? ars_d1 * dBf1(F3, 0, k + 1, j, i) : 0.0);
                        Bc[2] = (b3l / G.gdet(Loci::face3, j, i) +
                                    b3r / G.gdet(Loci::face3, j, i)) /
                                2;
                    } else {
                        Bc[2] = b3l / G.gdet(Loci::face3, j, i);
                    }
                    for (int v = 0; v < 3; v++) {
                        P_init_substep(m_p.B1 + v, k, j, i) = Bc[v];
                        U_init_substep(m_u.B1 + v, k, j, i) = Bc[v] * gdet_c;
                    }
                }
            }

            // Then now that we have the updated Umhd and Urad, we can try to invert them.
            const int status_star =
                Inverter::u_to_p<Inverter::Type::kastaun>(G, U_init_substep, m_u, eos, k,
                    j, i, P_init_substep, m_p, Loci::center, 25, 1e-12);
            const Real Urad_star[4] = {U_init_substep(m_u.UU_RAD, k, j, i),
                U_init_substep(m_u.U1_RAD, k, j, i), U_init_substep(m_u.U2_RAD, k, j, i),
                U_init_substep(m_u.U3_RAD, k, j, i)};
            Real Prad_star[4];
            const auto rad_status_star = u_to_p_rad(G, Urad_star, Prad_star, k, j, i);
            // If gas is not invertible, we do the entropy inversion + geo floors, as the
            // Newton guess only. The star U is kept, so the exchange stays conservative
            const bool star_fixed =
                !(status_star == static_cast<int>(Inverter::Status::success) &&
                    P_init_substep(m_p.UU, k, j, i) > 0.0);

            // it failed >:(
            if (star_fixed) {
                Real Ustar[5];
                for (int n = 0; n < 5; n++) Ustar[n] = U_init_substep(ifx[n], k, j, i);
                Real Kent = (P_save[0] > 0. && P_save[1] > 0.)
                                ? gm1 * P_save[1] * m::pow(P_save[0], -gam)
                                : -1.;
                if (m_u.KTOT >= 0) {
                    const Real Kc =
                        gam0 * U_init_substep(m_u.KTOT, k, j, i) +
                        gam1 * U_base(m_ub.KTOT, k, j, i) +
                        dt * dU_substep(m_u.KTOT, k, j, i) +
                        ((ars_d1 != 0.0) ? ars_d1 * dU1(m_d1.KTOT, k, j, i) : 0.0);
                    if (Kc > 0. && Ustar[0] > 0.) Kent = Kc / Ustar[0];
                }
                Real Bcons[NVEC] = {0., 0., 0.};
                if (m_u.B1 >= 0)
                    for (int v = 0; v < NVEC; v++)
                        Bcons[v] = U_init_substep(m_u.B1 + v, k, j, i);
                const Real Ucov[NVEC] = {Ustar[2], Ustar[3], Ustar[4]};
                Real rho_f, u_f, uvec_f[NVEC];
                if (!u_to_p_entropy(G, Ustar[0], Ucov, Bcons, Kent, gam, k, j, i, rho_f,
                        u_f, uvec_f)) {
                    for (int v = 0; v < NVEC; v++) uvec_f[v] = P_save[2 + v];
                    rho_f = rho_from_D(
                        G, Ustar[0] / G.gdet(Loci::center, j, i), uvec_f, k, j, i);
                    if (!(rho_f > 0.)) rho_f = P_save[0];
                    u_f = (Kent > 0.) ? Kent * m::pow(rho_f, gam) / gm1 : P_save[1];
                }
                if (use_floors) {
                    Real rhoflr, uflr;
                    Floors::determine_geo_floors(
                        G, P_init_substep, m_p, k, j, i, floors, rhoflr, uflr);
                    rho_f = m::max(rho_f, rhoflr);
                    u_f = m::max(u_f, uflr);
                }
                P_init_substep(m_p.RHO, k, j, i) = rho_f;
                P_init_substep(m_p.UU, k, j, i) = u_f;
                for (int v = 0; v < NVEC; v++)
                    P_init_substep(m_p.U1 + v, k, j, i) = uvec_f[v];
            }

            // If the radiation inversion failed, we restore the original radiation state.
            if (rad_status_star == StatusRadiationInversion::success) {
                P_init_substep(m_p.UU_RAD, k, j, i) = Prad_star[0];
                P_init_substep(m_p.U1_RAD, k, j, i) = Prad_star[1];
                P_init_substep(m_p.U2_RAD, k, j, i) = Prad_star[2];
                P_init_substep(m_p.U3_RAD, k, j, i) = Prad_star[3];
            } else {
                for (int n = 5; n < 9; n++) P_init_substep(ip[n], k, j, i) = P_save[n];
            }

            {
                Real uvec_s[NVEC] = {P_init_substep(m_p.U1, k, j, i),
                    P_init_substep(m_p.U2, k, j, i), P_init_substep(m_p.U3, k, j, i)};
                Real ucon_s[GR_DIM];
                GRMHD::calc_ucon(G, uvec_s, k, j, i, Loci::center, ucon_s);
                P_init_substep(m_p.RHO, k, j, i) =
                    U_init_substep(m_u.RHO, k, j, i) /
                    (G.gdet(Loci::center, j, i) * ucon_s[0]);
            }
            return star_fixed;
        };

        // Solver state the star build left in U_init/P_init
        auto get_entry = KOKKOS_LAMBDA(
            const int k, const int j, const int i, Real U_entry[8], Real P_entry[9])
        {
            const int iu8[8] = {m_u.UU, m_u.U1, m_u.U2, m_u.U3, m_u.UU_RAD, m_u.U1_RAD,
                m_u.U2_RAD, m_u.U3_RAD};
            const int ip[9] = {m_p.RHO, m_p.UU, m_p.U1, m_p.U2, m_p.U3, m_p.UU_RAD,
                m_p.U1_RAD, m_p.U2_RAD, m_p.U3_RAD};
            for (int n = 0; n < 8; n++) U_entry[n] = U_init_substep(iu8[n], k, j, i);
            for (int n = 0; n < 9; n++) P_entry[n] = P_init_substep(ip[n], k, j, i);
        };

        // Now we start the kernels for the different implicit solvers.
        // Dynamic schedule (CPU): the solve cost per cell varies a lot
        const int nj = jb.e - jb.s + 1, ni = ib.e - ib.s + 1;
        const int ncells = (kb.e - kb.s + 1) * nj * ni;
        using DynPolicy = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>;

        // one-time per-cell work, skipped cells, mark the rest
        pmb->par_for("RadM1_Implicit_Setup", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
            {
                for (int n = 0; n < 3; n++) solver_count(n, k, j, i) = 0.;
                for (int n = 0; n < 5; n++)
                    if (stage == 1) imex_src(n, k, j, i) = 0.0;

                // Inside the horizon: no interaction. If there is no horizon, r_hor = 0.
                GReal Xembed[GR_DIM];
                G.coord_embed(k, j, i, Loci::center, Xembed);
                const GReal r_hor = G.coords.get_horizon();
                if (r_hor > 0.0 && Xembed[1] < r_hor) {
                    rimplflag(0, k, j, i) = static_cast<int>(SIS::success);
                    return;
                }

                // Cells that fell on backstop, stage 1 term only
                if ((P_init_substep(m_p.U1_RAD, k, j, i) == 0.0 &&
                        P_init_substep(m_p.U3_RAD, k, j, i) == 0.0) ||
                    (P_init_substep(m_p.U1, k, j, i) == 0.0 &&
                        P_init_substep(m_p.U3, k, j, i) == 0.0)) {
                    apply_history(k, j, i);
                    rimplflag(0, k, j, i) = static_cast<int>(SIS::success);
                    return;
                }
                // rimplflag still holds the last call's outcome here
                const bool was_ff = static_cast<int>(rimplflag(0, k, j, i)) ==
                                    static_cast<int>(SIS::ufluid_success);
                rimplflag(0, k, j, i) = static_cast<int>(
                    was_ff ? SIS::pending_pmhd_short : SIS::pending_pmhd);
            });

        // 4D Newton on the MHD primitives
        Kokkos::parallel_for("RadM1_Implicit_PMHD", DynPolicy(pmb->exec_space, 0, ncells),
            KOKKOS_LAMBDA (const int idx)
            {
                const int k = kb.s + idx / (nj * ni), j = jb.s + (idx / ni) % nj,
                          i = ib.s + idx % ni;
                const int flag = static_cast<int>(rimplflag(0, k, j, i));
                const bool short_try =
                    (flag == static_cast<int>(SIS::pending_pmhd_short));
                if (flag != static_cast<int>(SIS::pending_pmhd) && !short_try) return;
                Real U_save[9], P_save[9], B_save[3], PB_save[3];
                const bool star_fixed =
                    build_star(k, j, i, U_save, P_save, B_save, PB_save);
                Real U_entry[8], P_entry[9];
                get_entry(k, j, i, U_entry, P_entry);

                Real dS[5] = {0., 0., 0., 0., 0.};
                int nit = 0;
                const int rflagl = solve_4d_pmhd(G, P_init_substep, m_p, m_u, k, j, i,
                    dt_impl, eos, src_rootfind_eps, src_rootfind_tol,
                    short_try ? pmhd_maxiter_after_ff : pmhd_maxiter, rad_opac, pflag,
                    rinvflag, U_entry, dS, &nit, star_fixed);
                solver_count(0, k, j, i) = nit;

                if (rflagl == static_cast<int>(SIS::success)) {
                    apply_source(k, j, i, dS,
                        star_fixed ? static_cast<int>(SIS::starfix_success) : rflagl);
                } else {
                    rimplflag(0, k, j, i) = static_cast<int>(
                        star_fixed ? SIS::pending_prad_starfix : SIS::pending_prad);
                }
                restore(k, j, i, U_save, P_save, B_save, PB_save);
            });

        // fallbacks -> fluid-frame solver, then P_rad Newton (I disabled it cause I think
        // it's not necessary in current version), then failure handling
        Kokkos::View<int*> todo(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "RadM1_fallback_cells"),
            ncells);
        int nfall = 0;
        Kokkos::parallel_scan(
            "RadM1_Implicit_Compact", Kokkos::RangePolicy<>(pmb->exec_space, 0, ncells),
            KOKKOS_LAMBDA(const int idx, int& n, const bool final)
            {
                const int k = kb.s + idx / (nj * ni), j = jb.s + (idx / ni) % nj,
                          i = ib.s + idx % ni;
                const int flag = static_cast<int>(rimplflag(0, k, j, i));
                if (flag == static_cast<int>(SIS::pending_prad) ||
                    flag == static_cast<int>(SIS::pending_prad_starfix)) {
                    if (final) todo(n) = idx;
                    n++;
                }
            },
            nfall);
        if (nfall == 0) continue;

        // Fluid-frame solver first!  It end up rescuing most cells the PMHD solve leaves,
        // and costs far less than P_rad, which inverts the gas at every residual
        // evaluation
        Kokkos::parallel_for("RadM1_Implicit_FluidFrame",
            DynPolicy(pmb->exec_space, 0, nfall),
            KOKKOS_LAMBDA (const int t)
            {
                const int idx = todo(t);
                const int k = kb.s + idx / (nj * ni), j = jb.s + (idx / ni) % nj,
                          i = ib.s + idx % ni;
                Real U_save[9], P_save[9], B_save[3], PB_save[3];
                build_star(k, j, i, U_save, P_save, B_save, PB_save);
                Real U_entry[8], P_entry[9];
                get_entry(k, j, i, U_entry, P_entry);

                Real dS[5] = {0., 0., 0., 0., 0.};
                int nit = 0;
                int why = 0;
                const int rflagl = solve_4d_pmhd_ff(G, P_init_substep, m_p, k, j, i,
                    dt_impl, eos, src_rootfind_tol, ff_maxiter, rad_opac, U_entry, dS,
                    &nit, &why, ff_eq_guess, prad_fallback ? 0. : accept_tol);
                solver_count(2, k, j, i) = (why > 0) ? -why : nit;

                // On failure the flag stays pending_prad(_starfix) for the P_rad kernel,
                // unless that fallback is off. Then FF is the last solver, and a failed
                // solve whose best state reached accept_tol is accepted
                if (rflagl == static_cast<int>(SIS::success)) {
                    apply_source(k, j, i, dS, static_cast<int>(SIS::ufluid_success));
                } else if (rflagl == static_cast<int>(SIS::relaxed_success)) {
                    apply_source(k, j, i, dS, rflagl);
                } else if (!prad_fallback) {
                    apply_history(k, j, i);
                    rimplflag(0, k, j, i) = static_cast<int>(SIS::failure);
                }
                restore(k, j, i, U_save, P_save, B_save, PB_save);
            });
        if (!prad_fallback) continue;

        Kokkos::View<int*> todo_prad(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "RadM1_prad_cells"), nfall);
        int nprad = 0;
        Kokkos::parallel_scan(
            "RadM1_Implicit_Compact_Prad",
            Kokkos::RangePolicy<>(pmb->exec_space, 0, nfall),
            KOKKOS_LAMBDA(const int t, int& n, const bool final)
            {
                const int idx = todo(t);
                const int k = kb.s + idx / (nj * ni), j = jb.s + (idx / ni) % nj,
                          i = ib.s + idx % ni;
                const int flag = static_cast<int>(rimplflag(0, k, j, i));
                if (flag == static_cast<int>(SIS::pending_prad) ||
                    flag == static_cast<int>(SIS::pending_prad_starfix)) {
                    if (final) todo_prad(n) = idx;
                    n++;
                }
            },
            nprad);
        if (nprad == 0) continue;

        Kokkos::parallel_for("RadM1_Implicit_Fallback",
            DynPolicy(pmb->exec_space, 0, nprad),
            KOKKOS_LAMBDA (const int t)
            {
                const int idx = todo_prad(t);
                const int k = kb.s + idx / (nj * ni), j = jb.s + (idx / ni) % nj,
                          i = ib.s + idx % ni;
                const int flag = static_cast<int>(rimplflag(0, k, j, i));
                const bool star_fixed =
                    (flag == static_cast<int>(SIS::pending_prad_starfix));
                Real U_save[9], P_save[9], B_save[3], PB_save[3];
                build_star(k, j, i, U_save, P_save, B_save, PB_save);
                Real U_entry[8], P_entry[9];
                get_entry(k, j, i, U_entry, P_entry);

                Real dS[5] = {0., 0., 0., 0., 0.};
                int nit = 0;
                int rflagl = solve_4d_prad(G, U_init_substep, P_init_substep, m_p, m_u, k,
                    j, i, dt_impl, eos, src_rootfind_eps, src_rootfind_tol,
                    src_rootfind_maxiter, rad_opac, pflag, rinvflag, U_entry, dS, &nit);
                solver_count(1, k, j, i) = nit;

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

                if (rflagl == static_cast<int>(SIS::success)) {
                    apply_source(k, j, i, dS,
                        static_cast<int>(star_fixed ? SIS::starfix_success
                                                    : SIS::pradfallback_success));
                } else {
                    apply_history(k, j, i);
                    rimplflag(0, k, j, i) = static_cast<int>(SIS::failure);
                }
                restore(k, j, i, U_save, P_save, B_save, PB_save);
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
