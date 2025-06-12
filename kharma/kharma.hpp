/* 
 *  File: kharma.hpp
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
#pragma once

#include "decs.hpp"
#include "types.hpp"

/**
 * General preferences for KHARMA.  Anything semi-driver-independent, like loading packages, etc.
 */
namespace KHARMA {

/**
 * Initialize a "package" of global variables: quantities needed randomly in several places.
 * Some are physical e.g. time, step times. Others track program state like initialization vs. stepping.
 */
std::shared_ptr<KHARMAPackage> InitializeGlobals(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

/**
 * Version for restarts, called in PostInitialize if we're restarting from a Parthenon restart file
 * Note this doesn't do very much -- Parthenon is good about restoring things the way we'd like
 */
void ResetGlobals(ParameterInput *pin, Mesh *pmesh);

/**
 * Update variables in Globals package based on Parthenon state incl. SimTime struct
 */
void PreStepWork(Mesh *pmesh, ParameterInput *pin, const SimTime &tm);
/**
 * Update variables in Globals package based on Parthenon state incl. SimTime struct
 */
void PostStepWork(Mesh *pmesh, ParameterInput *pin, const SimTime &tm);

/**
 * Task to add a package.  Lets us queue up all the packages we want in a task list, *then* load them
 * with correct dependencies and everything!
 */
TaskStatus AddPackage(std::shared_ptr<Packages_t>& packages,
                      std::function<std::shared_ptr<KHARMAPackage>(ParameterInput*, std::shared_ptr<Packages_t>&)> package_init,
                      ParameterInput *pin);

/**
 * This function messes with all Parthenon's parameters in-place before we hand them to the Mesh,
 * so that KHARMA decks can omit/infer some things parthenon needs.
 * This includes boundaries in spherical coordinates, coordinate system translations, etc.
 * This function also handles setting parameters from restart files
 */
void FixParameters(ParameterInput *pin, bool is_parthenon_restart);

/**
 * Load any packages specified in the input parameters
 */
Packages_t ProcessPackages(std::unique_ptr<ParameterInput>& pin);

// TODO(BSP) not sure where to put these

/**
 * Check whether a given field is anywhere in outputs.
 * Used to avoid calculating expensive fields (jcon, divB) if they
 * will not even be written.
 * Note this compares the field name as a substring rather than
 * an exact match to a vector element, so sub-names like `prims.`
 * or `coords.` will match any field which contains them.
 */
inline bool FieldIsOutput(ParameterInput *pin, std::string name)
{
    InputBlock *pib = pin->pfirst_block;
    while (pib != nullptr) {
        // For every output block with a 'variables' entry...
        if (pib->block_name.find("parthenon/output") != std::string::npos &&
            pin->DoesParameterExist(pib->block_name, "variables")) {
            std::string allvars = pin->GetString(pib->block_name, "variables");
            if (allvars.find(name) != std::string::npos) {
                return true;
            }
        }
        pib = pib->pnext;
    }
    return false;
}

/**
 * This fn calculates the size a VariablePack *would* be, without making one --
 * it uses only the package list, and counts through each variable in each package.
 * Mostly useful for initialization.
 */
inline std::vector<std::string> GetVariableNames(Packages_t* packages, Metadata::FlagCollection fc)
{
    // Count dimensions (1 for scalars + vector lengths) of each package's variables
    std::vector<std::string> names;
    for (auto pkg : packages->AllPackages()) {
        std::vector<std::string> pnames = pkg.second->GetVariableNames(fc);
        names.insert(names.end(), pnames.begin(), pnames.end());
    }
    return names;
}

}
