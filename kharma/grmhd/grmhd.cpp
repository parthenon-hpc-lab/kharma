/*
 *  File: grmhd.cpp
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

#include "grmhd.hpp"

#include "decs.hpp"

// TODO eliminate when Parthenon gets reduction types
#include "Kokkos_Core.hpp"

#include "boundaries.hpp"
#include "current.hpp"
#include "floors.hpp"
#include "flux.hpp"
#include "gr_coordinates.hpp"
#include "grmhd_functions.hpp"
#include "inverter.hpp"
#include "kharma.hpp"
#include "kharma_driver.hpp"

#include <memory>

/**
 * GRMHD package.  Global operations on General Relativistic Magnetohydrodynamic systems.
 */
namespace GRMHD
{

std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    // This function builds and returns a "KHARMAPackage" object, which is a light
    // superset of Parthenon's "StateDescriptor" class for packages.
    // The most important part of this object is a member of type "Params",
    // which acts more or less like a Python dictionary:
    // it puts values into a map of names->objects, where "objects" are usually
    // floats, strings, and ints, but can be arbitrary classes.
    // This "dictionary" is mostly immutable, and should always be treated as immutable,
    // except in the "Globals" package.
    auto pkg = std::make_shared<KHARMAPackage>("GRMHD");
    Params &params = pkg->AllParams();

    // GRMHD PARAMETERS
    // Fluid gamma for ideal EOS.  Don't guess this.
    // Only ideal EOS are supported, though modifying gamma based on
    // local temperatures would be straightforward.
    double gamma = pin->GetReal("GRMHD", "gamma");
    params.Add("gamma", gamma);

    // Proportion of courant condition for timesteps
    double cfl = pin->GetOrAddReal("GRMHD", "cfl", 0.9);
    params.Add("cfl", cfl);

    // TIME PARAMETERS
    // These parameters are put in "parthenon/time" to match others, but ultimately we should
    // override the parthenon timestep chooser
    // Minimum timestep, if something about the sound speed goes wonky. Probably won't save you :)
    double dt_min = pin->GetOrAddReal("parthenon/time", "dt_min", 1.e-5);
    params.Add("dt_min", dt_min);
    // Starting timestep: guaranteed step 1 timestep returned by EstimateTimestep,
    // usually matters most for restarts
    double dt_start = pin->GetOrAddReal("parthenon/time", "dt", dt_min);
    params.Add("dt_start", dt_start);
    double max_dt_increase = pin->GetOrAddReal("parthenon/time", "max_dt_increase", 2.0);
    params.Add("max_dt_increase", max_dt_increase);

    // Alternatively, you can start with (or just always use) the light (phase) speed crossing time
    // of the smallest zone. Useful when you're not sure of/modeling the characteristic velocities
    bool start_dt_light = pin->GetOrAddBoolean("parthenon/time", "start_dt_light", false);
    params.Add("start_dt_light", start_dt_light);
    bool use_dt_light = pin->GetOrAddBoolean("parthenon/time", "use_dt_light", false);
    params.Add("use_dt_light", use_dt_light);
    bool use_dt_light_phase_speed = pin->GetOrAddBoolean("parthenon/time", "use_dt_light_phase_speed", false);
    params.Add("use_dt_light_phase_speed", use_dt_light_phase_speed);

    // IMPLICIT PARAMETERS
    // The ImEx driver is necessary to evolve implicitly, but doesn't require it.  Using explicit
    // updates for GRMHD vars is useful for testing, or if adding just a couple of implicit variables
    // Doing EGRMHD requires implicit evolution of GRMHD variables, of course
    auto& driver = packages->Get("Driver")->AllParams();
    auto implicit_grmhd = (driver.Get<DriverType>("type") == DriverType::imex) &&
                          (pin->GetBoolean("emhd", "on") || pin->GetOrAddBoolean("GRMHD", "implicit", false));
    params.Add("implicit", implicit_grmhd);
    // Explicitly-evolved ideal MHD variables as guess for Extended MHD runs
    // TODO move to EMHD package, guard reads on package presence
    const bool ideal_guess = pin->GetOrAddBoolean("emhd", "ideal_guess", false);
    params.Add("ideal_guess", ideal_guess);

    // AMR PARAMETERS
    // Adaptive mesh refinement options
    // Only active if "refinement" and "numlevel" parameters allow
    Real refine_tol = pin->GetOrAddReal("GRMHD", "refine_tol", 0.5);
    params.Add("refine_tol", refine_tol);
    Real derefine_tol = pin->GetOrAddReal("GRMHD", "derefine_tol", 0.05);
    params.Add("derefine_tol", derefine_tol);

    // =================================== FIELDS ===================================

    // In addition to "params", the StateDescriptor/Package object carries "Fields"
    // These represent any variables we want to keep track of across the grid, and
    // generally inherit the size of the MeshBlock (for "Cell" fields) or some
    // closely-related size (for "Face" and "Edge" fields)

    // Add flags to distinguish groups of fields.
    // Hydrodynamics (everything we directly handle in this package)
    Metadata::AddUserFlag("HD");
    // Magnetohydrodynamics (all HD fields plus B field, which we'll need to make use of)
    Metadata::AddUserFlag("MHD");
    // Mark whether to evolve our variables via the explicit or implicit step inside the driver
    MetadataFlag areWeImplicit = (implicit_grmhd) ? Metadata::GetUserFlag("Implicit")
                                                  : Metadata::GetUserFlag("Explicit");
    std::vector<MetadataFlag> flags_grmhd = {Metadata::Cell, areWeImplicit, Metadata::GetUserFlag("HD"), Metadata::GetUserFlag("MHD")};

    auto flags_prim = driver.Get<std::vector<MetadataFlag>>("prim_flags");
    flags_prim.insert(flags_prim.end(), flags_grmhd.begin(), flags_grmhd.end());
    auto flags_cons = driver.Get<std::vector<MetadataFlag>>("cons_flags");
    flags_cons.insert(flags_cons.end(), flags_grmhd.begin(), flags_grmhd.end());

    // Mark whether the ideal MHD variables are to be updated explicitly for the guess to the solver
    if (ideal_guess) {
        flags_prim.push_back(Metadata::GetUserFlag("IdealGuess"));
        flags_cons.push_back(Metadata::GetUserFlag("IdealGuess"));
    }

    // We must additionally save the primtive variables as the "seed" for the next U->P solve
    flags_prim.push_back(Metadata::Restart);

    // We must additionally fill ghost zones of primitive variables in GRMHD, to seed the solver
    // Only necessary to add here if syncing conserved vars
    // Note some startup behavior relies on having the GRHD prims marked for syncing,
    // so disable sync_utop_seed at your peril
    // TODO work out disabling this automatically if Kastaun solver is enabled (requires no seed to converge)
    if (!driver.Get<bool>("sync_prims") && pin->GetOrAddBoolean("GRMHD", "sync_utop_seed", true)) {
        flags_prim.push_back(Metadata::FillGhost);
    }

    // With the flags sorted & explained, actually declaring fields is easy.
    // These will be initialized & cleaned automatically for each meshblock
    auto m = Metadata(flags_prim);
    pkg->AddField("prims.rho", m);
    pkg->AddField("prims.u", m);
    // We add the "Vector" flag and a size to vectors
    auto flags_prim_vec(flags_prim);
    flags_prim_vec.push_back(Metadata::Vector);
    std::vector<int> s_vector({NVEC});
    m = Metadata(flags_prim_vec, s_vector);
    pkg->AddField("prims.uvec", m);

    m = Metadata(flags_cons);
    pkg->AddField("cons.rho", m);
    pkg->AddField("cons.u", m);
    auto flags_cons_vec(flags_cons);
    flags_cons_vec.push_back(Metadata::Vector);
    m = Metadata(flags_cons_vec, s_vector);
    pkg->AddField("cons.uvec", m);

    // No magnetic fields here. KHARMA should operate fine in GRHD without them,
    // so they are allocated only by B field packages.

    // A KHARMAPackage also contains quite a few "callbacks," or functions called at
    // specific points in a step if the package is loaded.
    // Generally, see the headers for function descriptions.
    pkg->CheckRefinementBlock    = GRMHD::CheckRefinement;
    pkg->EstimateTimestepMesh    = GRMHD::EstimateTimestep;
    pkg->PostStepDiagnosticsMesh = GRMHD::PostStepDiagnostics;
    // NOTE: PtoU and UtoP for the five fluid variables are taken care of by the "Inverter"
    // package, which registers the relevant callbacks
    // This is so that the whole package can be disabled when running implicitly or w/EMHD

    // List (vector) of HistoryOutputVars that will all be enrolled as output variables
    parthenon::HstVar_list hst_vars = {};
    bool do_all = KHARMA::FieldIsOutput(pin, "all_reductions");
    // Common tracking variables
    if (do_all || KHARMA::FieldIsOutput(pin, "conserved_vars")) {
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::Total<Reductions::Var::rhou0>, "Mass"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::Total<Reductions::Var::mix_T00>, "Egas"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::Total<Reductions::Var::mix_T01>, "X1_Mom"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::Total<Reductions::Var::mix_T02>, "X2_Mom"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::Total<Reductions::Var::mix_T03>, "Ang_Mom"));
    }
    // TODO these are probably more useful at/within/without certain radii
    if (do_all || KHARMA::FieldIsOutput(pin, "luminosities")) {
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::Total<Reductions::Var::eht_lum>, "EHT_Lum_Proxy"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::Total<Reductions::Var::jet_lum>, "Jet_Lum"));
    }
    // Event horizon fluxes
    if (pin->GetBoolean("coordinates", "domain_intersects_eh")) {
        if (do_all || KHARMA::FieldIsOutput(pin, "eh_fluxes_cell")) {
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAt0<Reductions::Var::mdot>, "Mdot"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAtEH<Reductions::Var::mdot>, "Mdot_EH"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAt5M<Reductions::Var::mdot>, "Mdot_5M"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAt0<Reductions::Var::edot>, "Edot"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAtEH<Reductions::Var::edot>, "Edot_EH"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAt5M<Reductions::Var::edot>, "Edot_5M"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAt0<Reductions::Var::ldot>, "Ldot"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAtEH<Reductions::Var::ldot>, "Ldot_EH"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAt5M<Reductions::Var::ldot>, "Ldot_5M"));
        }

        if (do_all || KHARMA::FieldIsOutput(pin, "eh_fluxes_flux")) {
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAt0<Reductions::Var::mdot_flux>, "Mdot_Flux"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAtEH<Reductions::Var::mdot_flux>, "Mdot_EH_Flux"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAt5M<Reductions::Var::mdot_flux>, "Mdot_5M_Flux"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAt0<Reductions::Var::edot_flux>, "Edot_Flux"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAtEH<Reductions::Var::edot_flux>, "Edot_EH_Flux"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAt5M<Reductions::Var::edot_flux>, "Edot_5M_Flux"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAt0<Reductions::Var::ldot_flux>, "Ldot_0_Flux"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAtEH<Reductions::Var::ldot_flux>, "Ldot_EH_Flux"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, Reductions::SumAt5M<Reductions::Var::ldot_flux>, "Ldot_5M_Flux"));
        }
    }
    // add callbacks for HST output to the Params struct, identified by the `hist_param_key`
    pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

    return pkg;
}

Real EstimateTimestep(MeshData<Real> *md)
{
    // Normally the caller would place this flag before calling us, but this is from Parthenon
    // This function is a nice demo of why client-side flagging
    // like this is inadvisable: you have to EndFlag() at every different return
    Flag("EstimateTimestep");
    auto pmesh = md->GetMeshPointer();
    auto& globals = pmesh->packages.Get("Globals")->AllParams();
    const auto& grmhd_pars = pmesh->packages.Get("GRMHD")->AllParams();

    // If we have to recompute ctop anywhere, we do it now
    UpdateAveragedCtop(md);

    // Other things we might have to return (light-crossing, pre-set timestep, etc.)
    // TODO move these options to SetGlobalTimestep
    if (!globals.Get<bool>("in_loop")) {
        if (grmhd_pars.Get<bool>("start_dt_light") ||
            grmhd_pars.Get<bool>("use_dt_light")) {
            // Estimate based on light crossing time
            double dt = EstimateRadiativeTimestep(md);
            // This records a per-rank minimum,
            // but Parthenon finds the global minimum anyway
            if (globals.hasKey("dt_light")) {
                if (dt < globals.Get<double>("dt_light"))
                    globals.Update<double>("dt_light", dt);
            } else {
                globals.Add<double>("dt_light", dt);
            }
            EndFlag();
            return dt;
        } else {
            // Or Just take from parameters
            double dt = grmhd_pars.Get<double>("dt_start");
            // Record this, since we'll use it to determine the max step increase
            globals.Update<double>("dt_last", dt);
            EndFlag();
            return dt;
        }
    }
    // If we're still using the light crossing time, skip the rest
    if (grmhd_pars.Get<bool>("use_dt_light")) {
        EndFlag();
        return globals.Get<double>("dt_light");
    }

    // Actually compute the timestep if we have to
    const IndexRange3 b = KDomain::GetRange(md, IndexDomain::interior);

    // TODO version preserving location, with switch to keep this fast one
    // TODO maybe split normal, ISMR timesteps? Excised pole/recalculated ctop too?
    double min_ndt = std::numeric_limits<double>::max();
    for (auto &pmb : pmesh->block_list) {
        auto rc = pmb->meshblock_data.Get(md->StageName()).get();
        // We only need this block-wise to check boundary flags for ISMR, could special-case that
        const bool polar_inner_x2 = pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::user;
        const bool polar_outer_x2 = pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::user;

        const auto& cmax  = rc->PackVariables(std::vector<std::string>{"Flux.cmax"});
        const auto& cmin  = rc->PackVariables(std::vector<std::string>{"Flux.cmin"});

        double block_min_ndt = 0.;
        pmb->par_reduce("ndt_min", b.ks, b.ke, b.js, b.je, b.is, b.ie,
            KOKKOS_LAMBDA (const int k, const int j, const int i,
                        double &local_result) {
                const auto& G = cmax.GetCoords();
                int ismr_factor = 1;
                double courant_limit = 1.0;

                double ndt_zone = courant_limit / (1 / (G.Dxc<1>(i) /  m::max(cmax(V1, k, j, i), cmin(V1, k, j, i))) +
                                    1 / (G.Dxc<2>(j) /  m::max(cmax(V2, k, j, i), cmin(V2, k, j, i))) +
                                    1 / (G.Dxc<3>(k) * ismr_factor /  m::max(cmax(V3, k, j, i), cmin(V3, k, j, i))));

                if (!m::isnan(ndt_zone) && (ndt_zone < local_result)) {
                    local_result = ndt_zone;
                }
            }
        , Kokkos::Min<double>(block_min_ndt));
        if (block_min_ndt < min_ndt) min_ndt = block_min_ndt;
        //std::cerr << "Got block timestep: " << block_min_ndt << std::endl;
    }
    //std::cerr << "Got min timestep: " << min_ndt << std::endl;

    // Apply limits (TODO move into KHARMADriver::SetGlobalTimestep)
    const double cfl = grmhd_pars.Get<double>("cfl");
    const double dt_min = grmhd_pars.Get<double>("dt_min");
    const double dt_last = globals.Get<double>("dt_last");
    const double dt_max = grmhd_pars.Get<double>("max_dt_increase") * dt_last;
    const double ndt = clip(min_ndt * cfl, dt_min, dt_max);

    EndFlag();
    return ndt;
}

Real EstimateRadiativeTimestep(MeshData<Real> *md)
{
    Flag("EstimateRadiativeTimestep");
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    const auto& grmhd_pars = pmb0->packages.Get("GRMHD")->AllParams();
    const bool phase_speed = grmhd_pars.Get<bool>("use_dt_light_phase_speed");

    // Doesn't actually matter what we pack here, we're just pulling G
    const auto& dummy  = md->PackVariables(std::vector<std::string>{});

    const IndexRange3 b = KDomain::GetRange(md, IndexDomain::interior);
    const IndexRange block = IndexRange{0, dummy.GetDim(5)-1};

    // Leaving minmax in case the max phase speed is useful
    typename Kokkos::MinMax<Real>::value_type minmax;
    pmb0->par_reduce("ndt_min", block.s, block.e, b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA(const int& b, const int& k, const int& j, const int& i,
                      typename Kokkos::MinMax<Real>::value_type& lminmax) {
            const auto& G = dummy.GetCoords(b);

            double light_phase_speed = SMALL;
            double dt_light_local = 0.;

            if (phase_speed) {
                double local_phase_speed[GR_DIM];
                for (int mu = 1; mu < GR_DIM; mu++) {
                    if(SQR(G.gcon(Loci::center, j, i, 0, mu)) -
                        G.gcon(Loci::center, j, i, mu, mu)*G.gcon(Loci::center, j, i, 0, 0) >= 0.) {

                        double cplus = m::abs((-G.gcon(Loci::center, j, i, 0, mu) +
                                            m::sqrt(SQR(G.gcon(Loci::center, j, i, 0, mu)) -
                                                G.gcon(Loci::center, j, i, mu, mu)*G.gcon(Loci::center, j, i, 0, 0)))/
                                            G.gcon(Loci::center, j, i, 0, 0));

                        double cminus = m::abs((-G.gcon(Loci::center, j, i, 0, mu) -
                                            m::sqrt(SQR(G.gcon(Loci::center, j, i, 0, mu)) -
                                                G.gcon(Loci::center, j, i, mu, mu)*G.gcon(Loci::center, j, i, 0, 0)))/
                                            G.gcon(Loci::center, j, i, 0, 0));

                        local_phase_speed[mu] = m::max(cplus,cminus);
                    } else {
                        local_phase_speed[mu] = SMALL;
                    }
                }
                dt_light_local = 1./(G.Dxc<1>(0)/local_phase_speed[1]) +
                                 1./(G.Dxc<2>(0)/local_phase_speed[2]) +
                                 1./(G.Dxc<3>(0)/local_phase_speed[3]);
                light_phase_speed = m::max(local_phase_speed[1], m::max(local_phase_speed[2], local_phase_speed[3]));
            } else {
                dt_light_local = 1./G.Dxc<1>(0) +
                                 1./G.Dxc<2>(0) +
                                 1./G.Dxc<3>(0);
            }
            dt_light_local = 1/dt_light_local;

            if (!m::isnan(dt_light_local) && (dt_light_local < lminmax.min_val))
                lminmax.min_val = dt_light_local;
            if (!m::isnan(light_phase_speed) && (light_phase_speed > lminmax.max_val))
                lminmax.max_val = light_phase_speed;
        }
    , Kokkos::MinMax<Real>(minmax));

    // Just spit out dt
    const double cfl = grmhd_pars.Get<double>("cfl");
    const double ndt = minmax.min_val * cfl;

    EndFlag();
    return ndt;
}

AmrTag CheckRefinement(MeshBlockData<Real> *rc)
{
    auto pmb = rc->GetBlockPointer();
    auto v = rc->Get("prims.rho").data;

    IndexDomain domain = IndexDomain::interior;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);

    typename Kokkos::MinMax<Real>::value_type minmax;
    pmb->par_reduce("check_refinement", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i,
                      typename Kokkos::MinMax<Real>::value_type &lminmax) {
        lminmax.min_val =
            v(k, j, i) < lminmax.min_val ? v(k, j, i) : lminmax.min_val;
        lminmax.max_val =
            v(k, j, i) > lminmax.max_val ? v(k, j, i) : lminmax.max_val;
        }
    , Kokkos::MinMax<Real>(minmax));

    auto pkg = pmb->packages.Get("GRMHD");
    const auto &refine_tol   = pkg->Param<Real>("refine_tol");
    const auto &derefine_tol = pkg->Param<Real>("derefine_tol");

    if (minmax.max_val - minmax.min_val > refine_tol) return AmrTag::refine;
    if (minmax.max_val - minmax.min_val < derefine_tol) return AmrTag::derefine;
    return AmrTag::same;
}

TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md)
{
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Options
    const auto& pars = pmesh->packages.Get("Globals")->AllParams();
    const int extra_checks = pars.Get<int>("extra_checks");

    // Checking for any negative values.  Floors should
    // prevent this, so we save it for dire debugging
    if (extra_checks >= 2) {
        // Not sure when I'd do the check to hide latency, it's a step-end sort of deal
        // Just as well it's behind extra_checks 2
        // This may happen while ch0-1 are in flight from floors, but ch2-4 are now reusable
        Reductions::DomainReduction<Reductions::Var::neg_rho, int>(md, UserHistoryOperation::sum, 2);
        Reductions::DomainReduction<Reductions::Var::neg_u, int>(md, UserHistoryOperation::sum, 3);
        Reductions::DomainReduction<Reductions::Var::neg_rhout, int>(md, UserHistoryOperation::sum, 4);
        int nless_rho = Reductions::Check<int>(md, 2);
        int nless_u = Reductions::Check<int>(md, 3);
        int nless_rhout = Reductions::Check<int>(md, 4);

        if (MPIRank0()) {
            if (nless_rhout > 0) {
                std::cout << "Number of negative conserved rho: " << nless_rhout << std::endl;
            }
            if (nless_rho > 0 || nless_u > 0) {
                std::cout << "Number of negative primitive rho, u: " << nless_rho << "," << nless_u << std::endl;
            }
        }
    }

    return TaskStatus::complete;
}

void CancelBoundaryU3(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    // We're sometimes called on coarse buffers with or without AMR.
    // Use of transmitting polar conditions when coarse buffers matter (e.g., refinement
    // boundary touching the pole) is UNSUPPORTED
    if (coarse) return;

    // Pull boundary properties
    auto pmb = rc->GetBlockPointer();
    const BoundaryFace bface = KBoundaries::BoundaryFaceOf(domain);
    const bool binner = KBoundaries::BoundaryIsInner(bface);
    const auto bname = KBoundaries::BoundaryName(bface);
    const auto bdir = KBoundaries::BoundaryDirection(bface);
    if (bdir != 2) throw std::runtime_error("T3 Cancellation is only implemented for polar X2 boundaries!");

    // Pull variables (TODO take packs & maps, see boundaries.cpp)
    PackIndexMap prims_map, cons_map;
    auto P = rc->PackVariables({Metadata::GetUserFlag("Primitive"), Metadata::Cell}, prims_map);
    auto U = rc->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved, Metadata::Cell}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    const auto &G = pmb->coords;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    const bool sync_prims = pmb->packages.Get("Driver")->Param<bool>("sync_prims");

    const Floors::Prescription floors = pmb->packages.Get("Floors")->Param<Floors::Prescription>("prescription");
    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb->packages);

    // Subtract the average B3 as "reconnection" through the pole
    IndexRange3 b = KDomain::GetRange(rc, domain, coarse);
    IndexRange3 bi = KDomain::GetRange(rc, IndexDomain::interior, coarse);
    const int jf = (binner) ? bi.js : bi.je; // j index of last zone next to pole
    parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "reduce_U3_" + bname, pmb->exec_space,
        0, 1, b.is, b.ie,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& i) {
            if (!sync_prims) {
                // Recover primitive GRMHD variables to sync them
                parthenon::par_for_inner(member, bi.ks, bi.ke,
                    [&](const int& k) {
                    Inverter::u_to_p<Inverter::Type::kastaun>(G, U, m_u, gam, k, jf, i, P, m_p, Loci::center,
                                                                floors, 8, 1e-8);
                    }
                );
            }
            member.team_barrier();

            // Sum the first rank of U3
            Real U3_sum = 0.;
            Kokkos::Sum<Real> sum_reducer(U3_sum);
            parthenon::par_reduce_inner(member, bi.ks, bi.ke,
                [&](const int& k, Real& local_result) {
                    local_result += isnan(P(m_p.U3, k, jf, i)) ? 0. : P(m_p.U3, k, jf, i);
                }
            , sum_reducer);
            member.team_barrier();

            // Subtract the average, floor, restore conserved vars, update ctop
            const Real U3_avg = U3_sum / (bi.ke - bi.ks + 1);
            parthenon::par_for_inner(member, b.ks, b.ke,
                [&](const int& k) {
                    P(m_p.U3, k, jf, i) -= U3_avg;

                    // Apply floors
                    Floors::apply_geo_floors(G, P, m_p, gam, k, jf, i, floors, floors, Loci::center);

                    // Always PtoU, we modified P.  Accommodate EMHD
                    Flux::p_to_u_mhd(G, P, m_p, emhd_params, gam, k, jf, i, U, m_u);
                }
            );
        }
    );
}

void CancelBoundaryT3(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    // We're sometimes called on coarse buffers with or without AMR.
    // Use of transmitting polar conditions when coarse buffers matter (e.g., refinement
    // boundary touching the pole) is UNSUPPORTED
    if (coarse) return;

    // Pull boundary properties
    auto pmb = rc->GetBlockPointer();
    const BoundaryFace bface = KBoundaries::BoundaryFaceOf(domain);
    const bool binner = KBoundaries::BoundaryIsInner(bface);
    const auto bname = KBoundaries::BoundaryName(bface);
    const auto bdir = KBoundaries::BoundaryDirection(bface);
    if (bdir != 2) throw std::runtime_error("T3 Cancellation is only implemented for polar X2 boundaries!");

    // Pull variables (TODO take packs & maps, see boundaries.cpp)
    PackIndexMap prims_map, cons_map;
    auto P = rc->PackVariables({Metadata::GetUserFlag("Primitive"), Metadata::Cell}, prims_map);
    auto U = rc->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved, Metadata::Cell}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    const auto &G = pmb->coords;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    const bool sync_prims = pmb->packages.Get("Driver")->Param<bool>("sync_prims");

    const Floors::Prescription floors = pmb->packages.Get("Floors")->Param<Floors::Prescription>("prescription");
    // Don't be fooled, this function does *not* support/preserve EMHD values
    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb->packages);

    // Subtract the average B3 as "reconnection"
    IndexRange3 b = KDomain::GetRange(rc, domain, coarse);
    IndexRange3 bi = KDomain::GetRange(rc, IndexDomain::interior, coarse);
    const int jf = (binner) ? bi.js : bi.je; // j index of last zone next to pole
    parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "reduce_T3_" + bname, pmb->exec_space,
        0, 1, b.is, b.ie,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& i) {
            if (sync_prims) {
                parthenon::par_for_inner(member, bi.ks, bi.ke,
                    [&](const int& k) {
                        p_to_u(G, P, m_p, gam, k, jf, i, U, m_u, Loci::center);
                    }
                );
            }
            member.team_barrier();

            // Sum the first rank of the angular momentum T3
            Real T3_sum = 0.;
            Kokkos::Sum<Real> sum_reducer(T3_sum);
            parthenon::par_reduce_inner(member, bi.ks, bi.ke,
                [&](const int& k, Real& local_result) {
                    local_result += isnan(U(m_u.U3, k, jf, i)) ? 0. : U(m_u.U3, k, jf, i);
                }
            , sum_reducer);
            member.team_barrier();

            // Calculate the average and subtract it
            const Real T3_avg = T3_sum / (bi.ke - bi.ks + 1);
            parthenon::par_for_inner(member, b.ks, b.ke,
                [&](const int& k) {
                    U(m_u.U3, k, jf, i) -= T3_avg;
                    // Recover primitive GRMHD variables from our modified U
                    Inverter::u_to_p<Inverter::Type::kastaun>(G, U, m_u, gam, k, jf, i, P, m_p, Loci::center,
                                                              floors, 8, 1e-8);
                    // Floor them
                    int fflag = Floors::apply_geo_floors(G, P, m_p, gam, k, jf, i, floors, floors, Loci::center);
                    // Recalculate U on anything we floored
                    if (fflag)
                        p_to_u(G, P, m_p, gam, k, jf, i, U, m_u, Loci::center);
                }
            );
        }
    );
}

void UpdateAveragedCtop(MeshData<Real> *md)
{
    auto pmesh = md->GetMeshPointer();
    auto& params = pmesh->packages.Get<KHARMAPackage>("Boundaries")->AllParams();
    for (auto &pmb : pmesh->block_list) {
        auto &rc = pmb->meshblock_data.Get(md->StageName());
        for (int i = 0; i < BOUNDARY_NFACES; i++) {
            BoundaryFace bface = (BoundaryFace)i;
            auto bname = KBoundaries::BoundaryName(bface);
            const auto bdir = KBoundaries::BoundaryDirection(bface);
            const auto binner = KBoundaries::BoundaryIsInner(bface);
            const auto bdomain = KBoundaries::BoundaryDomain(bface);

            if (bdir > pmesh->ndim) continue;

            bool b3_is_reconnected = (pmesh->packages.AllPackages().count("B_CT")) ?
                                      params.Get<bool>("reconnect_B3_" + bname) :
                                      false;

            // If we've modified values on the pole...
            if (params.Get<bool>("cancel_T3_" + bname) ||
                params.Get<bool>("cancel_U3_" + bname) ||
                b3_is_reconnected) {
                // ...and if this face of the block corresponds to a global boundary...
                if (pmb->boundary_flag[bface] == BoundaryFlag::user) {
                    PackIndexMap prims_map, cons_map;
                    auto P = rc->PackVariables({Metadata::GetUserFlag("Primitive"), Metadata::Cell}, prims_map);
                    const VarMap m_p(prims_map, false);
                    const auto& cmax  = rc->PackVariables(std::vector<std::string>{"Flux.cmax"});
                    const auto& cmin  = rc->PackVariables(std::vector<std::string>{"Flux.cmin"});

                    const auto& G = pmb->coords;
                    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
                    const Floors::Prescription floors = pmb->packages.Get("Floors")->Param<Floors::Prescription>("prescription");
                    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb->packages);

                    // If we calculated the flux assuming half-size cells,
                    // we modify ctop rather than special-case in EstimateTimestep
                    const bool half_cells = params.Get<bool>("excise_flux_" + bname);

                    // Recompute ctop in zones affected by averaging
                    IndexRange3 b = KDomain::GetRange(rc, bdomain);
                    IndexRange3 bi = KDomain::GetRange(rc, IndexDomain::interior);
                    // TODO this part wouldn't be hard to generalize if polar boundary moves
                    const int jf = (binner) ? bi.js : bi.je;
                    pmb->par_for("update_ctop_in_averaged", b.ks, b.ke, b.is, b.ie,
                        KOKKOS_LAMBDA(const int& k, const int& i) {
                            FourVectors Dtmp;
                            GRMHD::calc_4vecs(G, P, m_p, k, jf, i, Loci::center, Dtmp);
                            // Remember our 'cmin' array stores *positive* values!
                            Real cmin_minus;
                            Flux::vchar_global(G, P, m_p, Dtmp, gam, emhd_params, k, jf, i, Loci::center, X1DIR,
                                        cmax(V1, k, jf, i), cmin_minus);
                            cmin(V1, k, jf, i) = -cmin_minus;
                            Flux::vchar_global(G, P, m_p, Dtmp, gam, emhd_params, k, jf, i, Loci::center, X2DIR,
                                        cmax(V2, k, jf, i), cmin_minus);
                            cmin(V2, k, jf, i) = -cmin_minus;
                            Flux::vchar_global(G, P, m_p, Dtmp, gam, emhd_params, k, jf, i, Loci::center, X3DIR,
                                        cmax(V3, k, jf, i), cmin_minus);
                            cmin(V3, k, jf, i) = -cmin_minus;
                            if (half_cells) {
                                cmin(bdir-1, k, jf, i) *= 0.5;
                                cmax(bdir-1, k, jf, i) *= 0.5;
                            }
                        }
                    );

                    if (params.Get<bool>("excise_flux_" + bname)) {

                    }
                }
            }
        }
    }
}

} // namespace GRMHD
