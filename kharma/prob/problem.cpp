/*
 *  File: problem.cpp
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

#include "problem.hpp"

#include "boundaries.hpp"
#include "electrons.hpp"
#include "floors.hpp"
#include "flux.hpp"
#include "gr_coordinates.hpp"
#include "grmhd.hpp"
#include "grmhd_functions.hpp"
#include "perturbation.hpp"
#include "types.hpp"

// Problem initialization headers
#include "bondi.hpp"
#include "explosion.hpp"
#include "fm_torus.hpp"
#include "resize_restart.hpp"
#include "resize_restart_kharma.hpp"
#include "kelvin_helmholtz.hpp"
#include "mhdmodes.hpp"
#include "orszag_tang.hpp"
#include "shock_tube.hpp"
#include "gizmo.hpp"
// EMHD problem headers
#include "emhd/anisotropic_conduction.hpp"
#include "emhd/emhdmodes.hpp"
#include "emhd/emhdshock.hpp"
#include "emhd/conducting_atmosphere.hpp"
// Electron problem headers
#include "elec/driven_turbulence.hpp"
#include "elec/hubble.hpp"
#include "elec/noh.hpp"

using namespace parthenon;

void KHARMA::ProblemGenerator(MeshBlock *pmb, ParameterInput *pin)
{
    auto rc = pmb->meshblock_data.Get("base");
    auto prob = pin->GetString("parthenon/job", "problem_id"); // Required parameter
    Flag("ProblemGenerator_"+prob);
    // Also just print this, it's important
    if (MPIRank0()) {
        // We have no way of tracking whether this is the first block we're initializing
        static bool printed_msg = false;
        if (!printed_msg) std::cout << "Initializing problem: " << prob << std::endl;
        printed_msg = true;
    }

    // Breakout to call the appropriate initialization function,
    // defined in accompanying headers.
    TaskStatus status = TaskStatus::fail;
    // MHD
    if (prob == "mhdmodes") {
        status = InitializeMHDModes(rc, pin);
    } else if (prob == "orszag_tang") {
        status = InitializeOrszagTang(rc, pin);
    } else if (prob == "explosion") {
        status = InitializeExplosion(rc, pin);
    } else if (prob == "kelvin_helmholtz") {
        status = InitializeKelvinHelmholtz(rc, pin);
    } else if (prob == "shock") {
        status = InitializeShockTube(rc, pin);
    // GRMHD
    } else if (prob == "bondi") {
        status = InitializeBondi(rc, pin);
    // Electrons
    } else if (prob == "noh") {
        status = InitializeNoh(rc, pin);
    } else if (prob == "hubble") {
        status = InitializeHubble(rc, pin);
    } else if (prob == "driven_turbulence") {
        status = InitializeDrivenTurbulence(rc, pin);
    // Extended GRMHD
    } else if (prob == "emhdmodes") {
        status = InitializeEMHDModes(rc, pin);
    } else if (prob == "anisotropic_conduction") {
        status = InitializeAnisotropicConduction(rc, pin);
    } else if (prob == "emhdshock") {
        status = InitializeEMHDShock(rc, pin);
    } else if (prob == "conducting_atmosphere") {
        status = InitializeAtmosphere(rc, pin);
    // Everything
    } else if (prob == "torus") {
        status = InitializeFMTorus(rc, pin);
    } else if (prob == "resize_restart") {
        status = ReadIharmRestart(rc, pin);
    } else if (prob == "resize_restart_kharma") {
        status = ReadKharmaRestart(rc, pin);
    } else if (prob == "gizmo") {
        status = InitializeGIZMO(rc, pin);
    } else if (prob == "vacuum" || prob == "bz_monopole") {
        // No need for a separate initializer, just seed w/floors
        status = Floors::ApplyInitialFloors(pin, rc.get(), IndexDomain::interior);
    }

    // If we didn't initialize a problem, yell
    if (status != TaskStatus::complete) {
        throw std::invalid_argument("Invalid or incomplete problem: "+prob);
    }

    // If we're not restarting, do any grooming of the initial conditions
    if ((prob != "resize_restart")) {
        // Perturb the internal energy a bit to encourage accretion
        // Note this defaults to zero & is basically turned on only for torii
        if (pin->GetOrAddReal("perturbation", "u_jitter", 0.0) > 0.0) {
            PerturbU(rc, pin);
        }

        // Initialize electron entropies to defaults if enabled
        if (pmb->packages.AllPackages().count("Electrons")) {
            Electrons::InitElectrons(rc.get(), pin);
        }

        if (pmb->packages.AllPackages().count("EMHD")) {
            EMHD::InitEMHDVariables(rc, pin);
        }
    }

    // Floors are NOT automatically applied at this point anymore.
    // If needed, they are applied within the problem-specific call.
    // See InitializeFMTorus in fm_torus.cpp for the details for torus problems.

    // Note we no longer call PtoU here either, as GRMHD variables' PtoU requires
    // the magnetic field, which is added in PostInitialize, after all blocks
    // are filled with other variables (it can be related to density averages which
    // require correct ghost zones)
    // If the B field will depend on the conserved variables (for some reason?)
    // they must be computed by the particular problem.

    EndFlag();
}
