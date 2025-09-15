/* 
 *  File: floors.hpp
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

#include "b_flux_ct.hpp"
#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "emhd.hpp"
#include "reductions.hpp"

namespace Floors {

namespace FFlag {
// Floor codes are non-exclusive, so it makes little sense to use an enum
// Instead, we use bitflags, starting high enough that we can stick the pflag in the bottom 5 bits
// See floors.hpp for explanations of the flags
// This is the namespaced, typed equivalent of #define
static constexpr int GEOM_RHO = 32;
static constexpr int GEOM_U = 64;
static constexpr int B_RHO = 128;
static constexpr int B_U = 256;
static constexpr int TEMP = 512;
static constexpr int GAMMA = 1024;
static constexpr int KTOT = 2048;
// Separate flags for when the floors are applied after reconstruction.
// Not yet used, as this will likely have some speed penalty paid even if
// the flags aren't written
static constexpr int GEOM_RHO_FLUX = 4096;
static constexpr int GEOM_U_FLUX = 8192;
// Yet more flags for floors hit during inversion
static constexpr int INVERTER_RHO = 16384;
static constexpr int INVERTER_U = 32768;
static constexpr int INVERTER_GAMMA = 65536;
static constexpr int INVERTER_U_MAX = 131072;
// Lowest flag value. Needed for combining floor and other return flags
static constexpr int MINIMUM = GEOM_RHO;

// Other advantage of a namespace is including full lists for iterating over
// TODO
// 1. prettier names?
// 2. What deep majicks would allow this to be constexpr?
static const std::map<int, std::string> flag_names = {
    {GEOM_RHO, "GEOM_RHO"},
    {GEOM_U, "GEOM_U"},
    {B_RHO, "B_RHO"},
    {B_U, "B_U"},
    {GAMMA, "GAMMA"},
    {TEMP, "TEMPERATURE"},
    {KTOT, "ENTROPY"},
    {GEOM_RHO_FLUX, "GEOM_RHO_ON_RECON"},
    {GEOM_U_FLUX, "GEOM_U_ON_RECON"},
    {INVERTER_RHO, "GEOM_RHO_ON_INVERT"},
    {INVERTER_U, "GEOM_U_ON_INVERT"},
    {INVERTER_GAMMA, "GAMMA_ON_INVERT"},
    {INVERTER_U_MAX, "U_MAX_ON_INVERT"}
};
}

enum class InjectionFrame{fluid=0, normal_onedw, normal_kastaun, mixed_fluid_normal, mixed_normal_drift, drift};

/**
 * Struct to hold floor values without cumbersome dictionary/string logistics.
 * Hopefully faster than dragging the full Params object device side,
 * similar reasoning to VarMap.
 */
class Prescription {
    public:
        // Constant sanity limits
        Real rho_min_const, u_min_const;
        // Purely geometric limits
        Real rho_min_geom, u_min_geom, r_char;
        // Dynamic limits on magnetization/temperature
        Real bsq_over_rho_max, bsq_over_u_max, u_over_rho_max;
        // Limit entropy
        Real ktot_max;
        // Limit fluid Lorentz factor
        Real gamma_max;
        // Floor options (frame was MOVED to templating)
        bool use_r_char, temp_adjust_u, adjust_k;
        // Radius dependent floors?
        bool radius_dependent_floors;
        Real floors_switch_r;
};

inline Prescription MakePrescription(parthenon::ParameterInput *pin, std::string block="floors")
{
    Prescription p;
    // Floor parameters
    if (pin->GetBoolean("coordinates", "spherical")) {
        // In spherical systems, floors drop as r^2, so set them higher by default
        p.rho_min_geom = pin->GetOrAddReal(block, "rho_min_geom", 1.e-6);
        p.u_min_geom = pin->GetOrAddReal(block, "u_min_geom", 1.e-8);
        // Some constant for large distances. New, out of the way by default
        p.rho_min_const = pin->GetOrAddReal(block, "rho_min_const", 1.e-20);
        p.u_min_const = pin->GetOrAddReal(block, "u_min_const", 1.e-20);
    } else { // TODO spherical cart will also have both
        // Accept old names
        Real rho_min_const_default = pin->DoesParameterExist(block, "rho_min_geom") ?
                                        pin->GetReal(block, "rho_min_geom") : 1.e-8;
        Real u_min_const_default = pin->DoesParameterExist(block, "u_min_geom") ?
                                    pin->GetReal(block, "u_min_geom") : 1.e-10;
        p.rho_min_const = pin->GetOrAddReal(block, "rho_min_const", rho_min_const_default);
        p.u_min_const = pin->GetOrAddReal(block, "u_min_const", u_min_const_default);
    }

    // In iharm3d, overdensities would run away; one proposed solution was
    // to decrease the density floor more with radius.  However, in practice
    // 1. This proved to be a result of the floor vs bsq, not the geometric one
    // 2. interior density floors are dominated by the floor vs bsq
    p.use_r_char = pin->GetOrAddBoolean(block, "use_r_char", false);
    p.r_char = pin->GetOrAddReal(block, "r_char", 10);

    // Floors vs magnetic field.  Most commonly hit & most temperamental
    p.bsq_over_rho_max = pin->GetOrAddReal(block, "bsq_over_rho_max", 1e20);
    p.bsq_over_u_max = pin->GetOrAddReal(block, "bsq_over_u_max", 1e20);

    // Limit temperature or entropy, optionally by siphoning off extra rather
    // than by adding material.
    p.u_over_rho_max = pin->GetOrAddReal(block, "u_over_rho_max", 1e20);
    p.ktot_max = pin->GetOrAddReal(block, "ktot_max", 1e20);
    p.temp_adjust_u = pin->GetOrAddBoolean(block, "temp_adjust_u", false);
    // Adjust electron entropy values when applying density floors to conserve
    // internal energy, as in Ressler+ but not more recent implementations
    p.adjust_k = pin->GetOrAddBoolean(block, "adjust_k", true);

    // Limit the fluid Lorentz factor gamma
    p.gamma_max = pin->GetOrAddReal(block, "gamma_max", 50.);

    p.radius_dependent_floors = pin->GetOrAddBoolean("floors", "radius_dependent_floors", false); 
    p.floors_switch_r = pin->GetOrAddReal("floors", "floors_switch_r", 50.);

    return p;
}

/**
 * Set prescription struct for inner domain (r < floor_switch_r) if 'radius_dependent_floors' is enabled.
 * Sets values provided in the 'floors_inner' block in input par file if provided,
 * else sets it equal to the values in the whole/outer domain.
 */
inline Prescription MakePrescriptionInner(parthenon::ParameterInput *pin, Prescription p_outer, std::string block="floors_inner")
{
    // TODO(BSP) I wonder if there's an easier way to "set if parameter exists" from pin, that would be broadly useful
    Prescription p_inner;

    // Floor parameters
    if (pin->GetBoolean("coordinates", "spherical")) {
        // In spherical systems, floors drop as r^2, so set them higher by default
        p_inner.rho_min_geom = pin->GetOrAddReal(block, "rho_min_geom", p_outer.rho_min_geom);
        p_inner.u_min_geom   = pin->GetOrAddReal(block, "u_min_geom", p_outer.u_min_geom);
        // Some constant for large distances. New, out of the way by default
        p_inner.rho_min_const = pin->GetOrAddReal(block, "rho_min_const", p_outer.rho_min_const);
        p_inner.u_min_const   = pin->GetOrAddReal(block, "u_min_const", p_outer.u_min_const);
    } else { // TODO spherical cart will also have both
        // Accept old names
        Real rho_min_const_default = pin->DoesParameterExist(block, "rho_min_geom") ?
                                        pin->GetReal(block, "rho_min_geom") : p_outer.rho_min_geom;
        Real u_min_const_default   = pin->DoesParameterExist(block, "u_min_geom") ?
                                        pin->GetReal(block, "u_min_geom") : p_outer.u_min_geom;
        p_inner.rho_min_const = pin->GetOrAddReal(block, "rho_min_const", rho_min_const_default);
        p_inner.u_min_const   = pin->GetOrAddReal(block, "u_min_const", u_min_const_default);
    }

    p_inner.use_r_char = pin->GetOrAddBoolean(block, "use_r_char", p_outer.use_r_char);
    p_inner.r_char     = pin->GetOrAddReal(block, "r_char", p_outer.r_char);

    p_inner.bsq_over_rho_max = pin->GetOrAddReal(block, "bsq_over_rho_max", p_outer.bsq_over_rho_max);
    p_inner.bsq_over_u_max = pin->GetOrAddReal(block, "bsq_over_u_max", p_outer.bsq_over_u_max);

    p_inner.u_over_rho_max = pin->GetOrAddReal(block, "u_over_rho_max", p_outer.u_over_rho_max);
    p_inner.ktot_max = pin->GetOrAddReal(block, "ktot_max", p_outer.ktot_max);
    p_inner.temp_adjust_u = pin->GetOrAddBoolean(block, "temp_adjust_u", p_outer.temp_adjust_u);
    p_inner.adjust_k = pin->GetOrAddBoolean(block, "adjust_k", p_outer.adjust_k);

    p_inner.gamma_max = pin->GetOrAddReal(block, "gamma_max", p_outer.gamma_max);

    // Always grab these from p_outer, they should never differ between outer/inner floors
    p_inner.radius_dependent_floors = p_outer.radius_dependent_floors; 
    p_inner.floors_switch_r = p_outer.floors_switch_r;

    return p_inner;
}

/**
 * Initialization.  Set parameters.
 */
std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

/**
 * Apply density and internal energy floors and ceilings
 * 
 * This function definitely applies floors over the stated domain.
 * If "Floors" package is not loaded, it is not registered.
 * 
 * LOCKSTEP: this function respects P and returns consistent P<->U
 */
TaskStatus ApplyGRMHDFloors(MeshData<Real> *md, IndexDomain domain);

/**
 * Determine just the floor values and flags for the current state, i.e.
 * 1. floor_vals fields: floor value corresponding to current conditions
 * 2. fflag, which floors were hit by the current state
 * This is what ApplyFloors uses to determine the floor values/locations
 */
TaskStatus DetermineGRMHDFloors(MeshData<Real> *md, IndexDomain domain,
    const Floors::Prescription& floors, const Floors::Prescription& floors_inner);

/**
 * Apply the same floors as above, in the same way, except:
 * 1. No ceilings
 * 2. Don't record results to 'fflag' or 'pflag'
 * Used for problems where some part of the domain is initialized to
 * "whatever the floor value is."
 * *This function can be called even if the Floors package is not initialized, and ignores "floors/on=false"*
 */
TaskStatus ApplyInitialFloors(ParameterInput *pin, MeshBlockData<Real> *mbd, IndexDomain domain);

/**
 * Count up all nonzero FFlags on md.  Used for history file reductions.
 */
int CountFFlags(MeshData<Real> *md);

/**
 * Print a summary of floors which were hit
 */
TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md);


} // namespace Floors
