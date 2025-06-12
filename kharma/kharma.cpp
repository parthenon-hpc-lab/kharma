/* 
 *  File: kharma.cpp
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
#include "kharma.hpp"

#include <iostream>

#include <parthenon/parthenon.hpp>

#include "decs.hpp"
#include "version.hpp"

// Packages
#include "b_flux_ct.hpp"
#include "b_cd.hpp"
#include "b_cleanup.hpp"
#include "b_cleanup_gmg.hpp"
#include "b_ct.hpp"
#include "coord_output.hpp"
#include "current.hpp"
#include "kharma_driver.hpp"
#include "electrons.hpp"
#include "implicit.hpp"
#include "inverter.hpp"
#include "ismr.hpp"
#include "floors.hpp"
#include "flux.hpp"
#include "grmhd.hpp"
#include "reductions.hpp"
#include "emhd.hpp"
#include "wind.hpp"

#include "bondi.hpp"
#include "boundaries.hpp"
#include "resize_restart.hpp"
#include "resize_restart_kharma.hpp"

std::shared_ptr<KHARMAPackage> KHARMA::InitializeGlobals(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    // All truly global state.  Mostly mutable state in order to avoid scope creep
    auto pkg = std::make_shared<KHARMAPackage>("Globals");
    Params &params = pkg->AllParams();
    // Current time in the simulation.  For ramping things up, ramping things down,
    // or preventing bad outcomes at known times
    params.Add("time", 0.0, true);
    // Last step's dt (Parthenon SimTime tm.dt), which must be preserved to output jcon
    params.Add("dt_last", 0.0, true);
    // Whether we are computing initial outputs/timestep, or versions in the execution loop
    params.Add("in_loop", false, true);

    // Log levels, the other acceptable global
    // Made mutable in case we want to bump global log level on certain events
    // TODO allow a "go_verbose" file watch
    int verbose = pin->GetOrAddInteger("debug", "verbose", 0);
    params.Add("verbose", verbose, true);
    int flag_verbose = pin->GetOrAddInteger("debug", "flag_verbose", 0);
    params.Add("flag_verbose", flag_verbose, true);
    int extra_checks = pin->GetOrAddInteger("debug", "extra_checks", 0);
    params.Add("extra_checks", extra_checks, true);

    // Record the problem name, just in case we need to special-case for different problems.
    // Please favor packages & options before using this, and modify problem-specific code
    // to be more general as it matures.
    std::string problem_name = pin->GetString("parthenon/job", "problem_id");
    params.Add("problem", problem_name);

    // Finally, the code version.  Recorded so it gets passed to output files & for printing
    params.Add("version", KHARMA::Version::GIT_VERSION);
    params.Add("SHA1", KHARMA::Version::GIT_SHA1);
    params.Add("branch", KHARMA::Version::GIT_REFSPEC);

    // Update the times with callbacks
    pkg->PreStepWork = KHARMA::PreStepWork;
    pkg->PostStepWork = KHARMA::PostStepWork;

    return pkg;
}

void KHARMA::ResetGlobals(ParameterInput *pin, Mesh *pmesh)
{
    // The globals package was loaded & exists, retrieve it
    auto pkg = pmesh->packages.Get("Globals");
    Params &params = pkg->AllParams();
    // This needs to be reset to guarantee that EstimateTimestep doesn't try to
    // calculate a new dt from a blank 'ctop' variable,
    // just uses whatever the next step was going to be at reset
    params.Update("in_loop", false);

    // Everything else is a per-step variable, not per-run, so they're fine
    // to be restored by Parthenon
}

void KHARMA::PreStepWork(Mesh *pmesh, ParameterInput *pin, const SimTime &tm)
{
    auto& globals = pmesh->packages.Get("Globals")->AllParams();
    if (!globals.Get<bool>("in_loop")) {
        globals.Update<bool>("in_loop", true);
    }
    globals.Update<double>("dt_last", tm.dt);
    globals.Update<double>("time", tm.time);
}

void KHARMA::PostStepWork(Mesh *pmesh, ParameterInput *pin, const SimTime &tm)
{
    // Knowing that this works took a little digging into Parthenon's EvolutionDriver.
    // The order of operations after calling Step() is:
    // 1. Call PostStepWork and PostStepDiagnostics (this function and following)
    // 2. Set the timestep tm.dt to the minimum from the EstimateTimestep calls
    // 3. Generate any outputs, e.g. jcon
    // Thus we preserve tm.dt (which has not yet been reset) as dt_last for Current::FillOutput
    auto& globals = pmesh->packages.Get("Globals")->AllParams();
    globals.Update<double>("dt_last", tm.dt);
    globals.Update<double>("time", tm.time);
}

void KHARMA::FixParameters(ParameterInput *pin, bool is_parthenon_restart)
{
    Flag("Fixing parameters");
    // Parthenon sets 2 ghost zones as a default.
    // We set a better default with our own parameter, and inform Parthenon.
    // This means that ONLY driver/nghost will be respected
    // Driver::Initialize will check we set enough for our reconstruction
    Globals::nghost = pin->GetOrAddInteger("driver", "nghost", 4);
    pin->SetInteger("parthenon/mesh", "nghost", Globals::nghost);

    // If we're restarting (not via Parthenon), read the restart file to get most parameters
    std::string prob = pin->GetString("parthenon/job", "problem_id");
    if (!is_parthenon_restart) {
        if (prob == "resize_restart") {
            ReadIharmRestartHeader(pin->GetString("resize_restart", "fname"), pin);
        }
        if (prob == "resize_restart_kharma") {
            ReadKharmaRestartHeader(pin->GetString("resize_restart", "fname"), pin);
        }
    } else if (prob == "resize_restart") {
        // If this is a Parthenon restart of a problem named `resize_restart`,
        // we don't want to trigger all the resizing stuff again.
        // So we rename the problem, and undo the custom stuff we needed for
        // resizing.
        pin->SetString("parthenon/job", "problem_id", "resized_restart");
        // Don't automatically clean B on subsequent restarts, either!
        pin->SetBoolean("b_cleanup", "on", false);
        // Finally, we probably set nlim=0 or 1 for the restarting phase, clear that
        if (pin->GetInteger("parthenon/time", "nlim") <= 1)
            pin->SetInteger("parthenon/time", "nlim", -1);
    }

    // Construct a CoordinateEmbedding object.  See coordinate_embedding.hpp for supported systems/tags
    CoordinateEmbedding tmp_coords(pin);
    // Record whether we're in spherical as we'll need that
    pin->SetBoolean("coordinates", "spherical", tmp_coords.is_spherical());

    // Do a bunch of autodetection/setting in spherical coordinates
    // Note frequent use of "GetOrAddX": this sets a default if not present but allows overriding
    if (tmp_coords.is_spherical()) {
        // Spherical systems can specify r_out and optionally r_in,
        // instead of xNmin/max.
        if (!pin->DoesParameterExist("parthenon/mesh", "x1min") ||
            !pin->DoesParameterExist("parthenon/mesh", "x1max")) {
            // Outer radius is always specified
            GReal Rout = pin->GetReal("coordinates", "r_out");
            GReal x1max = tmp_coords.r_to_native(Rout);
            pin->GetOrAddReal("parthenon/mesh", "x1max", x1max);

            if (mpark::holds_alternative<SphMinkowskiCoords>(tmp_coords.base)) {
                // In Minkowski coordinates, require Rin so the singularity is at user option
                GReal Rin = pin->GetReal("coordinates", "r_in");
                GReal x1min = tmp_coords.r_to_native(Rin);
                pin->GetOrAddReal("parthenon/mesh", "x1min", x1min);
            } else { // Any spherical BH metric: KS, BL, and derivatives
                // Set inner radius if not specified
                if (pin->DoesParameterExist("coordinates", "r_in")) {
                    GReal Rin = pin->GetReal("coordinates", "r_in");
                    GReal x1min = tmp_coords.r_to_native(Rin);
                    pin->GetOrAddReal("parthenon/mesh", "x1min", x1min);
                    if (Rin < 2.0){ // warn if there are fewer than 5 zones inside the event horizon
                        GReal dx = (x1max - x1min) / pin->GetInteger("parthenon/mesh", "nx1");
                        if (tmp_coords.X1_to_embed(x1min + 5*dx) > tmp_coords.get_horizon()) {
                            std::cerr << "WARNING: inner radius is near/in the EH, but does not allow 5 zones inside!" << std::endl;
                        }
                    }
                } else {
                    int nx1 = pin->GetInteger("parthenon/mesh", "nx1");
                    // Allow overriding Rhor for bondi_viscous problem
                    const GReal Rhor = pin->GetOrAddReal("coordinates", "Rhor", tmp_coords.get_horizon());
                    const GReal x1hor = tmp_coords.r_to_native(Rhor);

                    // Set Rin such that we have 5 zones completely inside the event horizon
                    // If xeh = log(Rhor), xin = log(Rin), and xout = log(Rout),
                    // then we want xeh = xin + 5.5 * (xout - xin) / N1TOT:
                    const GReal x1min = (nx1 * x1hor / 5.5 - x1max) / (-1. + nx1 / 5.5);
                    if (x1min < 0.0) {
                        throw std::invalid_argument("Not enough radial zones were specified to put 5 zones inside EH!");
                    }
                    pin->GetOrAddReal("parthenon/mesh", "x1min", x1min);
                    pin->GetOrAddReal("coordinates", "r_in", tmp_coords.X1_to_embed(x1min));
                }
            }
        } else {
            // Add the coordinate versions if they don't exist (usually restarts)
            pin->GetOrAddReal("coordinates", "r_in", tmp_coords.X1_to_embed(pin->GetReal("parthenon/mesh", "x1min")));
            pin->GetOrAddReal("coordinates", "r_out", tmp_coords.X1_to_embed(pin->GetReal("parthenon/mesh", "x1max")));
        }

        // If the simulation domain extends inside the EH, we change some boundary options
        pin->SetBoolean("coordinates", "domain_intersects_eh", pin->GetReal("coordinates", "r_in") < tmp_coords.get_horizon());

        // Spherical systems will also want KHARMA's spherical boundary conditions.
        // Note boundaries are now exclusively set by KBoundaries package
        pin->GetOrAddString("boundaries", "inner_x1", "outflow");
        pin->GetOrAddString("boundaries", "outer_x1", "outflow");
        pin->GetOrAddString("boundaries", "inner_x2", "reflecting");
        pin->GetOrAddString("boundaries", "outer_x2", "reflecting");
        pin->GetOrAddString("boundaries", "inner_x3", "periodic");
        pin->GetOrAddString("boundaries", "outer_x3", "periodic");
    } else {
        // This will never happen in Minkowski, but sometimes is checked later
        pin->SetReal("coordinates", "r_in", 0.);
        pin->SetBoolean("coordinates", "domain_intersects_eh", false);
        // We can set reasonable default boundary conditions for Cartesian sims,
        // but not default domain bounds
        pin->GetOrAddString("boundaries", "inner_x1", "periodic");
        pin->GetOrAddString("boundaries", "outer_x1", "periodic");
        pin->GetOrAddString("boundaries", "inner_x2", "periodic");
        pin->GetOrAddString("boundaries", "outer_x2", "periodic");
        pin->GetOrAddString("boundaries", "inner_x3", "periodic");
        pin->GetOrAddString("boundaries", "outer_x3", "periodic");
    }

    // Default boundaries are to cover the domain of our native coordinate system
    // std::cout << "Coordinate transform has boundaries: "
    //             << tmp_coords.startx(1) << " "
    //             << tmp_coords.startx(2) << " "
    //             << tmp_coords.startx(3) << " to "
    //             << tmp_coords.stopx(1) << " "
    //             << tmp_coords.stopx(2) << " "
    //             << tmp_coords.stopx(3) << std::endl;
    // In any coordinate system which sets boundaries (i.e. not Cartesian),
    // stopx > startx > 0. In Cartesian xNmin/xNmax are required
    if (tmp_coords.startx(1) >= 0)
        pin->GetOrAddReal("parthenon/mesh", "x1min", tmp_coords.startx(1));
    if (tmp_coords.stopx(1) >= 0)
        pin->GetOrAddReal("parthenon/mesh", "x1max", tmp_coords.stopx(1));
    if (tmp_coords.startx(2) >= 0)
        pin->GetOrAddReal("parthenon/mesh", "x2min", tmp_coords.startx(2));
    if (tmp_coords.stopx(2) >= 0)
        pin->GetOrAddReal("parthenon/mesh", "x2max", tmp_coords.stopx(2));
    if (tmp_coords.startx(3) >= 0)
        pin->GetOrAddReal("parthenon/mesh", "x3min", tmp_coords.startx(3));
    if (tmp_coords.stopx(3) >= 0)
        pin->GetOrAddReal("parthenon/mesh", "x3max", tmp_coords.stopx(3));

    // Also set x1 refinements as a proportion of size
    // TODO all regions!
    if (pin->DoesBlockExist("parthenon/static_refinement0")) {
        Real startx1 = pin->GetReal("parthenon/mesh", "x1min");
        Real stopx1 = pin->GetReal("parthenon/mesh", "x1max");
        Real lx1 = stopx1 - startx1;
        Real startx1_prop = pin->GetReal("parthenon/static_refinement0", "x1min");
        Real stopx1_prop = pin->GetReal("parthenon/static_refinement0", "x1max");
        //std::cerr << "StartX1 " << startx1 << " lx1 " << lx1 << "Prop " << startx1_prop << " " << stopx1_prop << std::endl;
        //std::cerr << "Adjust X1 " << startx1_prop*lx1 + startx1 << " to " << stopx1_prop*lx1 + startx1 << std::endl;
        pin->SetReal("parthenon/static_refinement0", "x1min", std::max(startx1_prop*lx1 + startx1, startx1));
        pin->SetReal("parthenon/static_refinement0", "x1max", std::min(stopx1_prop*lx1 + startx1, stopx1));

        if (pin->DoesParameterExist("parthenon/static_refinement0", "x2min")) {
            Real startx2 = pin->GetReal("parthenon/mesh", "x2min");
            Real stopx2 = pin->GetReal("parthenon/mesh", "x2max");
            Real lx2 = stopx2 - startx2;
            Real startx2_prop = pin->GetReal("parthenon/static_refinement0", "x2min");
            Real stopx2_prop = pin->GetReal("parthenon/static_refinement0", "x2max");
            pin->SetReal("parthenon/static_refinement0", "x2min", std::max(startx2_prop*lx2 + startx2, startx2));
            pin->SetReal("parthenon/static_refinement0", "x2max", std::min(stopx2_prop*lx2 + startx2, stopx2));
        }

        if (pin->DoesParameterExist("parthenon/static_refinement0", "x3min")) {
            Real startx3 = pin->GetReal("parthenon/mesh", "x3min");
            Real stopx3 = pin->GetReal("parthenon/mesh", "x3max");
            Real lx3 = stopx3 - startx3;
            Real startx3_prop = pin->GetReal("parthenon/static_refinement0", "x3min");
            Real stopx3_prop = pin->GetReal("parthenon/static_refinement0", "x3max");
            pin->SetReal("parthenon/static_refinement0", "x3min", std::max(startx3_prop*lx3 + startx3, startx3));
            pin->SetReal("parthenon/static_refinement0", "x3max", std::min(stopx3_prop*lx3 + startx3, stopx3));
        }
    }

    EndFlag();
}

TaskStatus KHARMA::AddPackage(std::shared_ptr<Packages_t>& packages,
                              std::function<std::shared_ptr<KHARMAPackage>(ParameterInput*, std::shared_ptr<Packages_t>&)> package_init,
                              ParameterInput *pin)
{
    // TODO package names before initialization
    const auto& pkg = package_init(pin, packages);
    packages->Add(pkg);
    Flag("AddPackage_"+pkg->label());
    EndFlag();
    return TaskStatus::complete;
}

Packages_t KHARMA::ProcessPackages(std::unique_ptr<ParameterInput> &pin)
{
    Flag("ProcessPackages");

    // Allocate the packages list as a shared pointer, to be updated in various tasks
    // TODO print what we're doing here & do some sanity checks, if verbose
    auto packages = std::make_shared<Packages_t>();

    TaskCollection tc;
    auto& tr = tc.AddRegion(1);
    auto& tl = tr[0];
    TaskID t_none(0);
    // The globals package will never have dependencies
    auto t_globals = tl.AddTask(t_none, KHARMA::AddPackage, packages, KHARMA::InitializeGlobals, pin.get());
    // Neither will grid output, as any mesh will get GRCoordinates objects
    // FieldIsOutput actually just checks for substring match, so this matches any coords. variable
    if (FieldIsOutput(pin.get(), "coords")) {
        auto t_coord_out = tl.AddTask(t_none, KHARMA::AddPackage, packages, CoordinateOutput::Initialize, pin.get());
    }
    // Driver package is the foundation
    auto t_driver = tl.AddTask(t_none, KHARMA::AddPackage, packages, KHARMADriver::Initialize, pin.get());
    // GRMHD needs globals to mark packages
    auto t_grmhd = tl.AddTask(t_globals | t_driver, KHARMA::AddPackage, packages, GRMHD::Initialize, pin.get());
    // Only load the inverter if GRMHD/EMHD isn't being evolved implicitly
    // Unless we want to use the explicitly-evolved ideal MHD variables as a guess for the solver
    // Or we want first-order flux corrections, which rely on a UtoP guess
    // Note we only accept fofc/on here, not legacy versions/defaults --
    // FOFC should be explicitly enabled in EMHD!
    auto t_inverter = t_grmhd;
    if (!pin->GetOrAddBoolean("GRMHD", "implicit", pin->GetOrAddBoolean("emhd", "on", false)) ||
        pin->GetOrAddBoolean("emhd", "ideal_guess", false) || pin->GetOrAddBoolean("fofc", "on", false)) {
        t_inverter = tl.AddTask(t_grmhd, KHARMA::AddPackage, packages, Inverter::Initialize, pin.get());
    }
    // Floors package is only loaded if floors aren't disabled
    // Respect legacy version for a while
    bool floors_on_default = true;
    if (pin->DoesParameterExist("floors", "disable_floors")) {
        floors_on_default = !pin->GetBoolean("floors", "disable_floors");
    }
    if (pin->GetOrAddBoolean("floors", "on", floors_on_default)) {
        auto t_floors = tl.AddTask(t_inverter, KHARMA::AddPackage, packages, Floors::Initialize, pin.get());
    }
    // Reductions, needed by most other packages
    auto t_reductions = tl.AddTask(t_none, KHARMA::AddPackage, packages, Reductions::Initialize, pin.get());

    // B field solvers, to ensure divB ~= 0.
    // Bunch of logic here: basically we want to load <=1 solver with an encoded order of preference:
    // 0. Anything user-specified
    // 1. Prefer B_CT if AMR since it's compatible
    // 2. Prefer B_Flux_CT otherwise since it's well-tested
    auto t_b_field = t_none;
    bool have_b_transport = false;
    bool face_centered_b  = false;
    bool multilevel = pin->GetOrAddString("parthenon/mesh", "refinement", "none") != "none";
    std::string b_field_solver = pin->GetOrAddString("b_field", "solver", "face_ct");
    if (b_field_solver == "none" || b_field_solver == "cleanup" || b_field_solver == "b_cleanup") {
        // Don't add a B field here
    } else if (b_field_solver == "constrained_transport" || b_field_solver == "face_ct") {
        t_b_field = tl.AddTask(t_grmhd, KHARMA::AddPackage, packages, B_CT::Initialize, pin.get());
        have_b_transport = true;
        face_centered_b = true;
    } else if (b_field_solver == "constraint_damping" || b_field_solver == "cd") {
        // Constraint damping. NON-WORKING
        t_b_field = tl.AddTask(t_grmhd, KHARMA::AddPackage, packages, B_CD::Initialize, pin.get());
        have_b_transport = true;
    } else if (b_field_solver == "flux_ct") {
        t_b_field = tl.AddTask(t_grmhd, KHARMA::AddPackage, packages, B_FluxCT::Initialize, pin.get());
        have_b_transport = true;
    } else {
        throw std::invalid_argument("Invalid solver! Must be e.g., flux_ct, face_ct, cd, cleanup...");
    }
    // Cleanup for the B field, using an elliptic solve for eliminating divB
    // Almost always loaded explicitly in addition to another transport, just for cleaning at simulation start
    // Enable b_cleanup package if we want it explicitly
    bool b_cleanup_package = pin->GetOrAddBoolean("b_cleanup", "on", (b_field_solver == "b_cleanup"));
    // OR if we need it for resizing a dump
    bool is_resize = pin->GetString("parthenon/job", "problem_id") == "resize_restart" &&
                     !pin->GetOrAddBoolean("resize_restart", "skip_b_cleanup", false);
    // OR if we ordered an initial cleanup pass for some other reason
    bool initial_cleanup = pin->GetOrAddBoolean("b_field", "initial_cleanup", false);
    bool use_b_cleanup = b_cleanup_package || is_resize || initial_cleanup;
    pin->SetBoolean("b_cleanup", "on", use_b_cleanup);
    auto t_b_cleanup = t_none;
    if (use_b_cleanup) {
        if (face_centered_b) {
            t_b_cleanup = tl.AddTask(t_grmhd, KHARMA::AddPackage, packages, B_CleanupGMG::Initialize, pin.get());
        } else {
            t_b_cleanup = tl.AddTask(t_grmhd, KHARMA::AddPackage, packages, B_Cleanup::Initialize, pin.get());
            // If we're the transport, assign us to the transport setup task too
            if (!have_b_transport) t_b_field = t_b_cleanup;
        }
    }

    // Optional standalone packages
    // Electrons are boring but not impossible without a B field (TODO add a test?)
    if (pin->GetOrAddBoolean("electrons", "on", false)) {
        auto t_electrons = tl.AddTask(t_grmhd, KHARMA::AddPackage, packages, Electrons::Initialize, pin.get());
    }
    if (pin->GetBoolean("emhd", "on")) { // Set above when deciding to load inverter
        auto t_emhd = tl.AddTask(t_grmhd, KHARMA::AddPackage, packages, EMHD::Initialize, pin.get());
    }
    if (pin->GetOrAddBoolean("wind", "on", false)) {
        auto t_wind = tl.AddTask(t_grmhd, KHARMA::AddPackage, packages, Wind::Initialize, pin.get());
    }
    // Enable calculating jcon iff it is in any list of outputs (and there's even B to calculate it).
    // Since it is never required to restart, this is the only time we'd write (hence, need) it
    if (FieldIsOutput(pin.get(), "jcon") && have_b_transport) {
        auto t_current = tl.AddTask(t_b_field, KHARMA::AddPackage, packages, Current::Initialize, pin.get());
    }

    // Execute the whole collection (just in case we do something fancy?)
    tc.Execute(); // TODO check return if Exe ever returns errors

    // There are some packages which must be loaded after all physics
    // Easier to load them separately than list dependencies

    // Flux temporaries must be full size
    KHARMA::AddPackage(packages, Flux::Initialize, pin.get());

    // ISMR temporaries must be full size
    if (pin->GetOrAddBoolean("ismr", "on", false)) {
        KHARMA::AddPackage(packages, ISMR::Initialize, pin.get());
    }

    // And any dirichlet/constant boundaries
    // TODO avoid init if Parthenon will be handling all boundaries?
    KHARMA::AddPackage(packages, KBoundaries::Initialize, pin.get());

    // Load the implicit package last, if there are *any* variables that need implicit evolution
    // This lets us just count by flag, rather than checking all the possible parameters that would
    // trigger this
    int n_implicit = StateDescriptor::CreateResolvedStateDescriptor(*packages)->GetPackDimension(Metadata::GetUserFlag("Implicit"));
    if (n_implicit > 0) {
        KHARMA::AddPackage(packages, Implicit::Initialize, pin.get());
    }

#if DEBUG
    // Carry the ParameterInput with us, for generating outputs whenever we want
    packages->Get("Globals")->AllParams().Add("pin", pin.get());
#endif

    EndFlag();
    return std::move(*packages);
}
