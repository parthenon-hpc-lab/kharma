/*
 *  File: units.cpp
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

#include "units.hpp"

namespace Units
{

parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;

UnitConversions::UnitConversions(ParameterInput* pin)
{
    // if scale_free parameter is set, every conversion factor is set to 1.
    // This basically means that we don't care about the physical units and we can just
    // use code units. This is also parsed to singularity_opac, which helps with how it
    // interprets the units.
    scale_free_ = pin->GetOrAddBoolean("units", "scale_free", true);

    if (scale_free_) {
        mass_ = 1.;
        length_ = 1.;
        time_ = 1.;
        energy_ = 1.;
        number_density_ = 1.;
        mass_density_ = 1.;
        temperature_ = 1.;
        return;
    } else {
        int geom_mass_g_exists =
            pin->DoesParameterExist("units", "geom_mass_g"); // reads mass of the BH in g
        int geom_mass_msun_exists = pin->DoesParameterExist(
            "units", "geom_mass_msun"); // reads mass of the BH in solar masses
        int geom_length_cm_exists =
            pin->DoesParameterExist("units", "geom_length_cm"); // Sets the length unit.
        int fluid_density_cgs_exists =
            pin->DoesParameterExist("units", "fluid_density_cgs");
        int fluid_mass_g_exists = pin->DoesParameterExist("units", "fluid_mass_g");

        PARTHENON_REQUIRE(
            geom_mass_g_exists + geom_mass_msun_exists + geom_length_cm_exists == 1,
            "Must provide exactly one of geom_mass_g, geom_mass_msun, "
            "geom_length_cm!");

        PARTHENON_REQUIRE(fluid_mass_g_exists + fluid_density_cgs_exists == 1,
            "Cannot provide both fluid_mass_g and fluid_density_cgs");

        if (geom_mass_g_exists) {
            Real geom_mass_ = pin->GetReal("units", "geom_mass_g");
            length_ = pc.g_newt * geom_mass_ / pow(pc.c, 2);
        }

        if (geom_mass_msun_exists) {
            Real geom_mass_ = pin->GetReal("units", "geom_mass_msun") * solar_mass;
            length_ = pc.g_newt * geom_mass_ / pow(pc.c, 2);
        }

        if (geom_length_cm_exists) {
            length_ = pin->GetReal("units", "geom_length_cm");
        }

        if (fluid_mass_g_exists) {
            mass_ = pin->GetReal("units", "fluid_mass_g");
        }

        if (fluid_density_cgs_exists) {
            mass_ =
                length_ * length_ * length_ * pin->GetReal("units", "fluid_density_cgs");
        }
    }

    time_ = length_ / pc.c;

    energy_ = mass_ * pow(pc.c, 2);

    number_density_ = pow(length_, -3);

    mass_density_ = mass_ * number_density_;
    Real mu = pin->GetOrAddReal("radM1", "mu", 1.0);
    temperature_ = mu * pc.mp * pc.c * pc.c / pc.kb;
}

std::shared_ptr<KHARMAPackage> Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Units");
    Params& params = pkg->AllParams();

    // Only purpose of this function is to create the unit_conv parameter of the class
    // Units.
    UnitConversions unit_conv(pin);
    params.Add("unit_conv", unit_conv);

    printf("Units initialized. Scale free: %s\n",
        unit_conv.IsScaleFree() ? "true" : "false");
    if (!unit_conv.IsScaleFree()) {
        printf("Derived units: length_unit = %e cm, mass_unit = %e g, time_unit = %e s, "
               "temperature_unit = %e\n",
            unit_conv.GetLengthCodeToCGS(), unit_conv.GetMassCodeToCGS(),
            unit_conv.GetTimeCodeToCGS(), unit_conv.GetTemperatureCodeToCGS());
    }

    return pkg;
}

} // namespace Units
