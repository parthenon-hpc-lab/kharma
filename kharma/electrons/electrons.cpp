/*
 *  File: electrons.cpp
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
#include "electrons.hpp"

#include "decs.hpp"
#include "domain.hpp"
#include "kharma_driver.hpp"
#include "flux.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"
#include "gaussian.hpp"

#include <parthenon/parthenon.hpp>
#include <utils/string_utils.hpp>

#include <string>

using namespace parthenon;

// Used only in Howes model
#define ME (9.1093826e-28  ) // Electron mass
#define MP (1.67262171e-24 ) // Proton mass

namespace Electrons
{

std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Electrons");
    Params &params = pkg->AllParams();

    // Evolution parameters
    Real gamma_e = pin->GetOrAddReal("electrons", "gamma_e", 4./3);
    params.Add("gamma_e", gamma_e);
    Real gamma_p = pin->GetOrAddReal("electrons", "gamma_p", 5./3);
    params.Add("gamma_p", gamma_p);

    // Whether to enforce that dissipation be positive, i.e. increasing entropy
    // Probably more accurate to keep off.
    bool enforce_positive_dissipation = pin->GetOrAddBoolean("electrons", "enforce_positive_dissipation", false);
    params.Add("enforce_positive_dissipation", enforce_positive_dissipation);

    // This is used only in constant model
    Real fel_const = pin->GetOrAddReal("electrons", "fel_constant", 0.1);
    params.Add("fel_constant", fel_const);

    // This prevented spurious heating when heat_electrons used pre-floored dissipation
    bool suppress_highb_heat = pin->GetOrAddBoolean("electrons", "suppress_highb_heat", false);
    params.Add("suppress_highb_heat", suppress_highb_heat);

    // Initialization
    bool init_to_fel_0 = pin->GetOrAddBoolean("electrons", "init_to_fel_0", true);
    params.Add("init_to_fel_0", init_to_fel_0);
    Real fel_0 = pin->GetOrAddReal("electrons", "fel_0", 0.01);
    params.Add("fel_0", fel_0);

    // Floors
    // Whether to limit electron entropy K with following two floors
    bool limit_kel = pin->GetOrAddBoolean("electrons", "limit_kel", true);
    params.Add("limit_kel", limit_kel);
    Real tp_over_te_min = pin->GetOrAddReal("electrons", "tp_over_te_min", 0.001);
    params.Add("tp_over_te_min", tp_over_te_min);
    Real tp_over_te_max = pin->GetOrAddReal("electrons", "tp_over_te_max", 1000.0);
    params.Add("tp_over_te_max", tp_over_te_max);

    // Model options
    bool do_constant = pin->GetOrAddBoolean("electrons", "constant", false);
    params.Add("do_constant", do_constant);
    bool do_howes = pin->GetOrAddBoolean("electrons", "howes", false);
    params.Add("do_howes", do_howes);
    bool do_kawazura = pin->GetOrAddBoolean("electrons", "kawazura", false);
    params.Add("do_kawazura", do_kawazura);
    bool do_werner = pin->GetOrAddBoolean("electrons", "werner", false);
    params.Add("do_werner", do_werner);
    bool do_rowan = pin->GetOrAddBoolean("electrons", "rowan", false);
    params.Add("do_rowan", do_rowan);
    bool do_sharma = pin->GetOrAddBoolean("electrons", "sharma", false);
    params.Add("do_sharma", do_sharma);

    // Parse various mass and density units to set the different cooling rates
    // TODO actually respect them of course
    std::vector<Real> masses = pin->GetOrAddVector<Real>("electrons", "masses", std::vector<Real>{});
    if (masses.size() > 0) {
        std::vector<std::string> mass_names = pin->GetVector<std::string>("electrons", "masses");
        std::vector<std::vector<Real>> munits;
        for (auto mass_name : mass_names) {
            munits.push_back(pin->GetVector<Real>("electrons", "munits_"+mass_name));
        }
    }

    // Evolving e- implicitly is not tested.  Shouldn't be necessary even in EMHD
    auto& driver = packages->Get("Driver")->AllParams();
    auto driver_type = driver.Get<DriverType>("type");
    bool implicit_e = (driver_type == DriverType::imex && pin->GetOrAddBoolean("electrons", "implicit", false));
    params.Add("implicit", implicit_e);

    Metadata::AddUserFlag("Elec");
    MetadataFlag areWeImplicit = (implicit_e) ? Metadata::GetUserFlag("Implicit")
                                              : Metadata::GetUserFlag("Explicit");
    std::vector<MetadataFlag> flags_elec = {Metadata::Cell, areWeImplicit, Metadata::GetUserFlag("Elec")};

    auto flags_prim = driver.Get<std::vector<MetadataFlag>>("prim_flags");
    flags_prim.insert(flags_prim.end(), flags_elec.begin(), flags_elec.end());
    auto flags_cons = driver.Get<std::vector<MetadataFlag>>("cons_flags");
    flags_cons.insert(flags_cons.end(), flags_elec.begin(), flags_elec.end());

    // Total entropy, used to track changes
    int nKs = 1;
    pkg->AddField("cons.Ktot", flags_cons);
    pkg->AddField("prims.Ktot", flags_prim);

    if ("driven_turbulence" == packages->Get("Globals")->Param<std::string>("problem")) {
        std::vector<int> s_vector({2});
        Metadata m_vector = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_vector);
        Metadata m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
        pkg->AddField("grf_normalized", m_vector);
        pkg->AddField("alfven_speed", m);
    }

    // Individual models
    // TO ADD A MODEL:
    // 1. Define fields here
    // 2. Define names in types.hpp
    // 3. Add clauses in p_to_u in electrons.hpp and prim_to_flux in flux_functions.hpp
    // 4. Add heating model in ApplyElectronHeating, below
    if (do_constant) {
        nKs += 1;
        pkg->AddField("cons.Kel_Constant", flags_cons);
        pkg->AddField("prims.Kel_Constant", flags_prim);
    }
    if (do_howes) {
        nKs += 1;
        pkg->AddField("cons.Kel_Howes", flags_cons);
        pkg->AddField("prims.Kel_Howes", flags_prim);
    }
    if (do_kawazura) {
        nKs += 1;
        pkg->AddField("cons.Kel_Kawazura", flags_cons);
        pkg->AddField("prims.Kel_Kawazura", flags_prim);
    }
    if (do_werner) {
        nKs += 1;
        pkg->AddField("cons.Kel_Werner", flags_cons);
        pkg->AddField("prims.Kel_Werner", flags_prim);
    }
    if (do_rowan) {
        nKs += 1;
        pkg->AddField("cons.Kel_Rowan", flags_cons);
        pkg->AddField("prims.Kel_Rowan", flags_prim);
    }
    if (do_sharma) {
        nKs += 1;
        pkg->AddField("cons.Kel_Sharma", flags_cons);
        pkg->AddField("prims.Kel_Sharma", flags_prim);
    }
    // TODO if nKs == 1 then rename Kel_Whatever -> Kel?
    // TODO record nKs and find a nice way to loop/vector the device-side layout?

    // Problem-specific fields
    if (packages->Get("Globals")->Param<std::string>("problem") == "driven_turbulence") {
        std::vector<int> s_vector({2});
        Metadata m_vector = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_vector);
        Metadata m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
        pkg->AddField("grf_normalized", m_vector);
        pkg->AddField("alfven_speed", m);
    }

    pkg->BlockUtoP = Electrons::BlockUtoP;
    pkg->BoundaryUtoP = Electrons::BlockUtoP;
    // If we wanted to apply the domian boundaries to primitive Electrons variables
    //pkg->DomainBoundaryPtoU = Electrons::BlockPtoU;

    return pkg;
}

TaskStatus InitElectrons(MeshBlockData<Real> *rc, ParameterInput *pin)
{
    Flag("InitElectrons");
    auto pmb = rc->GetBlockPointer();

    // Don't initialize entropies if we've already done so e.g. in Hubble problem
    if (!pmb->packages.Get("Electrons")->Param<bool>("init_to_fel_0")) {
        return TaskStatus::complete;
    }

    // Need to distinguish KTOT from the other variables, so we record which it is
    PackIndexMap prims_map;
    auto& e_P = rc->PackVariables({Metadata::GetUserFlag("Elec"), Metadata::GetUserFlag("Primitive")}, prims_map);
    const int ktot_index = prims_map["prims.Ktot"].first;
    // Just need these two from the rest of Prims
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real game = pmb->packages.Get("Electrons")->Param<Real>("gamma_e");
    const Real fel0 = pmb->packages.Get("Electrons")->Param<Real>("fel_0");

    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    pmb->par_for("UtoP_electrons", 0, e_P.GetDim(4)-1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA (const int &p, const int &k, const int &j, const int &i) {
            if (p == ktot_index) {
                // Initialize total entropy by definition,
                e_P(p, k, j, i) = (gam - 1.) * u(k, j, i) * m::pow(rho(k, j, i), -gam);
            } else {
                // and e- entropy by given constant initial fraction
                e_P(p, k, j, i) = (game - 1.) * fel0 * u(k, j, i) * m::pow(rho(k, j, i), -game);
            }
        }
    );

    EndFlag();
    return TaskStatus::complete;
}

void BlockUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();

    // No need for a "map" here, we just want everything that fits these
    auto& e_P = rc->PackVariables({Metadata::GetUserFlag("Elec"), Metadata::GetUserFlag("Primitive")});
    auto& e_U = rc->PackVariables({Metadata::GetUserFlag("Elec"), Metadata::Conserved});
    // And then the local density
    GridScalar rho_U = rc->Get("cons.rho").data;

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    int is = bounds.is(domain), ie = bounds.ie(domain);
    int js = bounds.js(domain), je = bounds.je(domain);
    int ks = bounds.ks(domain), ke = bounds.ke(domain);
    pmb->par_for("UtoP_electrons", 0, e_P.GetDim(4)-1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA (const int &p, const int &k, const int &j, const int &i) {
            e_P(p, k, j, i) = e_U(p, k, j, i) / rho_U(k, j, i);
        }
    );
}

void BlockPtoU(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();

    PackIndexMap prims_map, cons_map;
    auto& P = rc->PackVariables({Metadata::GetUserFlag("Primitive"), Metadata::Cell}, prims_map);
    auto& U = rc->PackVariables({Metadata::Conserved, Metadata::Cell}, cons_map);
    const VarMap m_p(prims_map, false), m_u(cons_map, true);

    const auto& G = pmb->coords;

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    int is = bounds.is(domain), ie = bounds.ie(domain);
    int js = bounds.js(domain), je = bounds.je(domain);
    int ks = bounds.ks(domain), ke = bounds.ke(domain);
    pmb->par_for("PtoU_electrons", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            Electrons::p_to_u(G, P, m_p, k, j, i, U, m_u);
        }
    );
}

TaskStatus ApplyElectronHeating(MeshBlockData<Real> *rc_old, MeshBlockData<Real> *rc, bool generate_grf)
{
    // Need to distinguish different electron models
    // So far, Parthenon's maps of the same sets of variables are consistent,
    // so we only bother with one map of the primitives
    // TODO Parthenon can definitely build a pack from a map, though
    PackIndexMap prims_map, cons_map;
    auto& P = rc_old->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    auto& P_new = rc->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    auto& U_new = rc->PackVariables({Metadata::Conserved}, cons_map);
    const VarMap m_p(prims_map, false), m_u(cons_map, true);

    auto pmb = rc->GetBlockPointer();
    const auto& G = pmb->coords;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real gamp = pmb->packages.Get("Electrons")->Param<Real>("gamma_p");
    const Real game = pmb->packages.Get("Electrons")->Param<Real>("gamma_e");
    const Real fel_const = pmb->packages.Get("Electrons")->Param<Real>("fel_constant");
    const bool suppress_highb_heat = pmb->packages.Get("Electrons")->Param<bool>("suppress_highb_heat");
    // Floors
    const Real tptemin = pmb->packages.Get("Electrons")->Param<Real>("tp_over_te_min");
    const Real tptemax = pmb->packages.Get("Electrons")->Param<Real>("tp_over_te_max");
    const bool enforce_positive_diss = pmb->packages.Get("Electrons")->Param<bool>("enforce_positive_dissipation");
    const bool limit_kel = pmb->packages.Get("Electrons")->Param<bool>("limit_kel");

    // This function (and any primitive-variable sources) needs to be run over the entire domain,
    // because the boundary zones have already been updated and so the same calculations must be applied
    // in order to keep them consistent.
    // See kharma_step.cpp for the full picture of what gets updated when.
    const IndexRange ib = rc->GetBoundsI(IndexDomain::entire);
    const IndexRange jb = rc->GetBoundsJ(IndexDomain::entire);
    const IndexRange kb = rc->GetBoundsK(IndexDomain::entire);
    pmb->par_for("heat_electrons", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            FourVectors Dtmp;
            GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
            Real bsq = dot(Dtmp.bcon, Dtmp.bcov);

            // Calculate the new total entropy in this cell considering heating
            const Real k_energy_conserving = (gam-1.) * P_new(m_p.UU, k, j, i) / m::pow(P_new(m_p.RHO, k, j, i), gam);

            // Dissipation is the real entropy k_energy_conserving minus any advected entropy from the previous (sub-)step P_new(KTOT)
            Real diss_tmp = (game-1.) / (gam-1.) * m::pow(P(m_p.RHO, k, j, i), gam - game) * (k_energy_conserving - P_new(m_p.KTOT, k, j, i));
            //this is eq27                  ratio of heating: Qi/Qe                           advected entropy from prev step
            // ^ denotes the solution corresponding to entropy conservation

            // Under the flag "suppress_highb_heat", we set all dissipation to zero at sigma > 1.
            diss_tmp = (suppress_highb_heat && (bsq / P(m_p.RHO, k, j, i) > 1.)) ? 0.0 : diss_tmp;

            // Default is True diss_sign == Enforce nonnegative
            // Due to floors we can end up with diss==0 or even *slightly* <0, so we require it to be positive here
            const Real diss = enforce_positive_diss ? m::max(diss_tmp, 0.0) : diss_tmp;

            // Reset the entropy to measure next (sub-)step's dissipation
            P_new(m_p.KTOT, k, j, i) = k_energy_conserving;

            // We'll be applying floors inline as we heat electrons, so
            // we cache the floors as entropy limits so they'll be cheaper to apply.
            // Note tp_te_min -> kel_max & vice versa
            const Real kel_max = P(m_p.KTOT, k, j, i) * m::pow(P(m_p.RHO, k, j, i), gam - game) /
                                    (tptemin * (gam - 1.) / (gamp-1.) + (gam-1.) / (game-1.)); //0.001
            const Real kel_min = P(m_p.KTOT, k, j, i) * m::pow(P(m_p.RHO, k, j, i), gam - game) /
                                    (tptemax * (gam - 1.) / (gamp-1.) + (gam-1.) / (game-1.)); //1000
            // Note this differs a little from Ressler '15, who ensure u_e/u_g > 0.01 rather than use temperatures

            // The ion temperature is useful for a few models, cache it too.
            // The minimum values on Tpr & Tel here ensure that for un-initialized zones,
            // Tpr/Tel == Tel/Tpr == 1 != NaN.  This condition should not be hit after step 1
            const Real Tpr = m::max((gamp - 1.) * P(m_p.UU, k, j, i) / P(m_p.RHO, k, j, i), SMALL);

            // Heat different electron passives based on different dissipation fraction models
            // Expressions here closely adapted (read: stolen) from implementation in iharm3d
            // courtesy of Cesar Diaz, see https://github.com/AFD-Illinois/iharm3d

            // In all of these the electron entropy stored value is the entropy conserving solution
                                 // and then when updated it becomes the energy conserving solution
            if (m_p.K_CONSTANT >= 0) {
                const Real fel = fel_const;
                // Default is true then enforce kel limits with clamp/clip, else no restrictions on kel
                if (limit_kel) {
                    P_new(m_p.K_CONSTANT, k, j, i) = clip(P_new(m_p.K_CONSTANT, k, j, i) + fel * diss, kel_min, kel_max);
                } else {
                    P_new(m_p.K_CONSTANT, k, j, i) += fel * diss;
                }
            }
            if (m_p.K_HOWES >= 0) {
                const Real Tel = m::max(P(m_p.K_HOWES, k, j, i) * m::pow(P(m_p.RHO, k, j, i), game-1), SMALL);

                const Real Trat = Tpr / Tel;
                const Real pres = P(m_p.RHO, k, j, i) * Tpr; // Proton pressure
                const Real beta = m::min(pres / bsq * 2, 1.e20);// If somebody enables electrons in a GRHD sim

                const Real logTrat = log10(Trat);
                const Real mbeta = 2. - 0.2*logTrat;

                const Real c2 = (Trat <= 1.) ? 1.6/Trat : 1.2/Trat;
                const Real c3 = (Trat <= 1.) ? 18. + 5.*logTrat : 18.;

                const Real beta_pow = m::pow(beta, mbeta);
                const Real qrat = 0.92 * (c2*c2 + beta_pow)/(c3*c3 + beta_pow) * m::exp(-1./beta) * m::sqrt(MP/ME * Trat);
                const Real fel = 1./(1. + qrat);
                P_new(m_p.K_HOWES, k, j, i) = clip(P_new(m_p.K_HOWES, k, j, i) + fel * diss, kel_min, kel_max);
            }
            if (m_p.K_KAWAZURA >= 0) {
                // Equation (2) in http://www.pnas.org/lookup/doi/10.1073/pnas.1812491116
                const Real Tel = m::max(P(m_p.K_KAWAZURA, k, j, i) * m::pow(P(m_p.RHO, k, j, i), game-1), SMALL);

                const Real Trat = Tpr / Tel;
                const Real pres = P(m_p.RHO, k, j, i) * Tpr; // Proton pressure
                const Real beta = m::min(pres / bsq * 2, 1.e20);// If somebody enables electrons in a GRHD sim

                const Real QiQe = 35. / (1. + m::pow(beta/15., -1.4) * m::exp(-0.1 / Trat));
                const Real fel = 1./(1. + QiQe);
                P_new(m_p.K_KAWAZURA, k, j, i) = clip(P_new(m_p.K_KAWAZURA, k, j, i) + fel * diss, kel_min, kel_max);
            }
            // TODO KAWAZURA 19/20/21 separately?
            if (m_p.K_WERNER >= 0) {
                // Equation (3) in http://academic.oup.com/mnras/article/473/4/4840/4265350
                const Real sigma = bsq / P(m_p.RHO, k, j, i);
                const Real fel = 0.25 * (1 + m::sqrt((sigma/5.) / (2 + (sigma/5.))));
                P_new(m_p.K_WERNER, k, j, i) = clip(P_new(m_p.K_WERNER, k, j, i) + fel * diss, kel_min, kel_max);
            }
            if (m_p.K_ROWAN >= 0) {
                // Equation (34) in https://iopscience.iop.org/article/10.3847/1538-4357/aa9380
                const Real pres = (gamp - 1.) * P(m_p.UU, k, j, i); // Proton pressure
                const Real pg = (gam - 1) * P(m_p.UU, k, j, i);
                const Real beta = pres / bsq * 2;
                const Real sigma = bsq / (P(m_p.RHO, k, j, i) + P(m_p.UU, k, j, i) + pg);
                const Real betamax = 0.25 / sigma;
                const Real fel = 0.5 * m::exp(-m::pow(1 - beta/betamax, 3.3) / (1 + 1.2*m::pow(sigma, 0.7)));
                P_new(m_p.K_ROWAN, k, j, i) = clip(P_new(m_p.K_ROWAN, k, j, i) + fel * diss, kel_min, kel_max);
            }
            if (m_p.K_SHARMA >= 0) {
                // Equation for \delta on  pg. 719 (Section 4) in https://iopscience.iop.org/article/10.1086/520800
                const Real Tel = m::max(P(m_p.K_SHARMA, k, j, i) * m::pow(P(m_p.RHO, k, j, i), game-1), SMALL);

                const Real Trat_inv = Tel / Tpr; // Inverse of the temperature ratio in KAWAZURA
                const Real QeQi = 0.33 * m::sqrt(Trat_inv);
                const Real fel = 1./(1.+1./QeQi);
                P_new(m_p.K_SHARMA, k, j, i) = clip(P_new(m_p.K_SHARMA, k, j, i) + fel * diss, kel_min, kel_max);
            }
            // Conserved variables are updated at the end of the step
        }
    );

    const IndexRange myib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    const IndexRange myjb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange mykb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    // A couple of the electron test problems add source terms to the *fluid*.
    // we bundle them here because they're generally relevant alongside e- heating,
    // and should be applied at the same time
    const std::string prob = pmb->packages.Get("Globals")->Param<std::string>("problem");
    if (prob == "driven_turbulence") { // Gaussian random field:
        const auto& G = pmb->coords;
        GridScalar rho = rc->Get("prims.rho").data;
        GridVector uvec = rc->Get("prims.uvec").data;
        GridVector grf_normalized = rc->Get("grf_normalized").data;
        const Real t = pmb->packages.Get("Globals")->Param<Real>("time");
        Real counter = pmb->packages.Get("GRMHD")->Param<Real>("counter");
        const Real dt_kick=  pmb->packages.Get("GRMHD")->Param<Real>("dt_kick");
        if (generate_grf && counter < t) {
            counter += dt_kick;
            pmb->packages.Get("GRMHD")->UpdateParam<Real>("counter", counter);
            printf("Kick applied at time %.32f\n", t);

            const Real lx1=  pmb->packages.Get("GRMHD")->Param<Real>("lx1");
            const Real lx2=  pmb->packages.Get("GRMHD")->Param<Real>("lx2");
            const Real edot= pmb->packages.Get("GRMHD")->Param<Real>("drive_edot");
            GridScalar alfven_speed = rc->Get("alfven_speed").data;

            int Nx1 = pmb->cellbounds.ncellsi(IndexDomain::interior);
            int Nx2 = pmb->cellbounds.ncellsj(IndexDomain::interior);
            Real *dv0 =  (Real*) malloc(sizeof(Real)*Nx1*Nx2);
            Real *dv1 =  (Real*) malloc(sizeof(Real)*Nx1*Nx2);
            create_grf(Nx1, Nx2, lx1, lx2, dv0, dv1);

            Real mean_velocity_num0 = 0;    Kokkos::Sum<Real> mean_velocity_num0_reducer(mean_velocity_num0);
            Real mean_velocity_num1 = 0;    Kokkos::Sum<Real> mean_velocity_num1_reducer(mean_velocity_num1);
            Real tot_mass = 0;              Kokkos::Sum<Real> tot_mass_reducer(tot_mass);
            pmb->par_reduce("forced_mhd_normal_kick_centering_mean_vel0", mykb.s, mykb.e, myjb.s, myjb.e, myib.s, myib.e,
                KOKKOS_LAMBDA(const int k, const int j, const int i, Real &local_result) {
                    Real cell_mass = (rho(k, j, i) * G.Dxc<3>(k) * G.Dxc<2>(j) * G.Dxc<1>(i));
                    local_result += cell_mass * dv0[(i-4)*Nx1+(j-4)];
                }
            , mean_velocity_num0_reducer);
            pmb->par_reduce("forced_mhd_normal_kick_centering_mean_vel1", mykb.s, mykb.e, myjb.s, myjb.e, myib.s, myib.e,
                KOKKOS_LAMBDA(const int k, const int j, const int i, Real &local_result) {
                    Real cell_mass = (rho(k, j, i) * G.Dxc<3>(k) * G.Dxc<2>(j) * G.Dxc<1>(i));
                    local_result += cell_mass * dv1[(i-4)*Nx1+(j-4)];
                }
            , mean_velocity_num1_reducer);
            pmb->par_reduce("forced_mhd_normal_kick_centering_tot_mass", mykb.s, mykb.e, myjb.s, myjb.e, myib.s, myib.e,
                KOKKOS_LAMBDA(const int k, const int j, const int i, Real &local_result) {
                    local_result += (rho(k, j, i) * G.Dxc<3>(k) * G.Dxc<2>(j) * G.Dxc<1>(i));
                }
            , tot_mass_reducer);
            Real mean_velocity0 = mean_velocity_num0/tot_mass;
            Real mean_velocity1 = mean_velocity_num1/tot_mass;
            #pragma omp parallel for simd collapse(2)
            for (size_t i = 0; i < Nx1 ; i ++) {
                for (size_t j = 0; j < Nx2 ; j ++) {
                    dv0[i*Nx1+j] -= mean_velocity0;
                    dv1[i*Nx1+j] -= mean_velocity1;
                }
            }

            Real Bhalf = 0; Real A = 0; Real init_e = 0;
            Kokkos::Sum<Real> Bhalf_reducer(Bhalf); Kokkos::Sum<Real> A_reducer(A); Kokkos::Sum<Real> init_e_reducer(init_e);
            pmb->par_reduce("forced_mhd_normal_kick_normalization_Bhalf", mykb.s, mykb.e, myjb.s, myjb.e, myib.s, myib.e,
                KOKKOS_LAMBDA(const int k, const int j, const int i, Real &local_result) {
                    Real cell_mass = (rho(k, j, i) * G.Dxc<3>(k) * G.Dxc<2>(j) * G.Dxc<1>(i));
                    local_result += cell_mass * (dv0[(i-4)*Nx1+(j-4)]*uvec(0, k, j, i) + dv1[(i-4)*Nx1+(j-4)]*uvec(1, k, j, i));
                }
            , Bhalf_reducer);
            pmb->par_reduce("forced_mhd_normal_kick_normalization_A", mykb.s, mykb.e, myjb.s, myjb.e, myib.s, myib.e,
                KOKKOS_LAMBDA(const int k, const int j, const int i, Real &local_result) {
                    Real cell_mass = (rho(k, j, i) * G.Dxc<3>(k) * G.Dxc<2>(j) * G.Dxc<1>(i));
                    local_result += cell_mass * (pow(dv0[(i-4)*Nx1+(j-4)], 2) + pow(dv1[(i-4)*Nx1+(j-4)], 2));
                }
            , A_reducer);
            pmb->par_reduce("forced_mhd_normal_kick_init_e", mykb.s, mykb.e, myjb.s, myjb.e, myib.s, myib.e,
                KOKKOS_LAMBDA(const int k, const int j, const int i, Real &local_result) {
                    Real cell_mass = (rho(k, j, i) * G.Dxc<3>(k) * G.Dxc<2>(j) * G.Dxc<1>(i));
                    local_result += 0.5 * cell_mass * (pow(uvec(0, k, j, i), 2) + pow(uvec(1, k, j, i), 2));
                }
            , init_e_reducer);

            Real norm_const = (-Bhalf + pow(pow(Bhalf,2) + A*2*dt_kick*edot, 0.5))/A;  // going from k:(0, 0), j:(4, 515), i:(4, 515) inclusive
            pmb->par_for("forced_mhd_normal_kick_setting", mykb.s, mykb.e, myjb.s, myjb.e, myib.s, myib.e,
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    grf_normalized(0, k, j, i) = (dv0[(i-4)*Nx1+(j-4)]*norm_const);
                    grf_normalized(1, k, j, i) = (dv1[(i-4)*Nx1+(j-4)]*norm_const);
                    uvec(0, k, j, i) += grf_normalized(0, k, j, i);
                    uvec(1, k, j, i) += grf_normalized(1, k, j, i);
                    FourVectors Dtmp;
                    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
                    Real bsq = dot(Dtmp.bcon, Dtmp.bcov);
                    alfven_speed(k,j,i) = bsq/rho(k, j, i); //saving alfven speed for analysis purposes
                }
            );

            Real finl_e = 0;    Kokkos::Sum<Real> finl_e_reducer(finl_e);
            pmb->par_reduce("forced_mhd_normal_kick_finl_e", mykb.s, mykb.e, myjb.s, myjb.e, myib.s, myib.e,
                KOKKOS_LAMBDA(const int k, const int j, const int i, Real &local_result) {
                    Real cell_mass = (rho(k, j, i) * G.Dxc<3>(k) * G.Dxc<2>(j) * G.Dxc<1>(i));
                    local_result += 0.5 * cell_mass * (pow(uvec(0, k, j, i), 2) + pow(uvec(1, k, j, i), 2));
                }
            , finl_e_reducer);
            printf("%.32f\n", A); printf("%.32f\n", Bhalf); printf("%.32f\n", norm_const);
            printf("%.32f\n", (finl_e-init_e)/dt_kick);
            free(dv0); free(dv1);
        }
        // This could be only the GRMHD vars, for this problem, but speed isn't really an issue
        Flux::BlockPtoU(rc, IndexDomain::interior);
    }
    EndFlag();
    return TaskStatus::complete;
}

void ApplyFloors(MeshBlockData<Real> *mbd, IndexDomain domain)
{
    auto pmb                 = mbd->GetBlockPointer();
    auto packages            = pmb->packages;

    PackIndexMap prims_map, cons_map;
    auto P = mbd->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    const VarMap m_p(prims_map, false);

    auto fflag = mbd->PackVariables(std::vector<std::string>{"fflag"}, prims_map);

    const auto& G = pmb->coords;

    const Real gam = packages.Get("GRMHD")->Param<Real>("gamma");
    const Floors::Prescription floors       = packages.Get("Floors")->Param<Floors::Prescription>("prescription");
    const Floors::Prescription floors_inner = packages.Get("Floors")->Param<Floors::Prescription>("prescription_inner");

    const IndexRange3 b = KDomain::GetRange(mbd, domain);
    pmb->par_for("apply_electrons_floors", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {

            // Also apply the ceiling to the advected entropy KTOT, if we're keeping track of that
            // (either for electrons, or robust primitive inversions in future)
            Real ktot_max;
            if (m_p.KTOT >= 0) {
                if (floors.radius_dependent_floors && G.coords.is_spherical()
                    && G.r(k, j, i) < floors.floors_switch_r) {
                    ktot_max = floors_inner.ktot_max;
                } else {
                    ktot_max = floors.ktot_max;
                }

                if (P(m_p.KTOT, k, j, i) > ktot_max) {
                    fflag(0, k, j, i) = Floors::FFlag::KTOT | (int) fflag(0, k, j, i);
                    P(m_p.KTOT, k, j, i) = ktot_max;
                }
            }

            // TODO(BSP) restore Ressler adjustment option
            // Ressler adjusts KTOT & KEL to conserve u whenever adjusting rho
            // but does *not* recommend adjusting them when u hits floors/ceilings
            // This is in contrast to ebhlight, which heats electrons before applying *any* floors,
            // and resets KTOT during floor application without touching KEL
            // if (floors.adjust_k && (fflag() & FFlag::GEOM_RHO || fflag() & FFlag::B_RHO)) {
            //     const Real reduce   = m::pow(rho / P(m_p.RHO, k, j, i), gam);
            //     const Real reduce_e = m::pow(rho / P(m_p.RHO, k, j, i), 4./3); // TODO pipe in real gam_e
            //     if (m_p.KTOT >= 0) P(m_p.KTOT, k, j, i) *= reduce;
            //     if (m_p.K_CONSTANT >= 0) P(m_p.K_CONSTANT, k, j, i) *= reduce_e;
            //     if (m_p.K_HOWES >= 0)    P(m_p.K_HOWES, k, j, i)    *= reduce_e;
            //     if (m_p.K_KAWAZURA >= 0) P(m_p.K_KAWAZURA, k, j, i) *= reduce_e;
            //     if (m_p.K_WERNER >= 0)   P(m_p.K_WERNER, k, j, i)   *= reduce_e;
            //     if (m_p.K_ROWAN >= 0)    P(m_p.K_ROWAN, k, j, i)    *= reduce_e;
            //     if (m_p.K_SHARMA >= 0)   P(m_p.K_SHARMA, k, j, i)   *= reduce_e;
            // }
        }
    );
    Flux::BlockPtoU(mbd, domain);
}

TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *rc)
{
    Flag("PostStepDiagnostics");

    // Output any diagnostics after a step completes

    EndFlag();
    return TaskStatus::complete;
}

void FillOutput(MeshBlock *pmb, ParameterInput *pin)
{
    // Any variables or diagnostics that should be computed especially for output to a file,
    // but which are not otherwise updated.
}

} // namespace Electrons
