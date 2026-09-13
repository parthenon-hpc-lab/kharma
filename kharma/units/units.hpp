/*
 *  File: units.hpp
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

#include <basic_types.hpp>
#include <memory>
#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

#include "kharma_package.hpp"

namespace Units
{

class UnitConversions
{
  public:
    UnitConversions() = default;
    UnitConversions(ParameterInput* pin);

    KOKKOS_INLINE_FUNCTION bool IsScaleFree() const { return scale_free_; }

    KOKKOS_INLINE_FUNCTION Real GetMassCodeToCGS() const { return mass_; }
    KOKKOS_INLINE_FUNCTION Real GetMassCGSToCode() const { return 1. / mass_; }

    KOKKOS_INLINE_FUNCTION Real GetLengthCodeToCGS() const { return length_; }
    KOKKOS_INLINE_FUNCTION Real GetLengthCGSToCode() const { return 1. / length_; }

    KOKKOS_INLINE_FUNCTION Real GetTimeCodeToCGS() const { return time_; }
    KOKKOS_INLINE_FUNCTION Real GetTimeCGSToCode() const { return 1. / time_; }

    KOKKOS_INLINE_FUNCTION Real GetEnergyCodeToCGS() const { return energy_; }
    KOKKOS_INLINE_FUNCTION Real GetEnergyCGSToCode() const { return 1. / energy_; }

    KOKKOS_INLINE_FUNCTION Real GetNumberDensityCodeToCGS() const
    {
        return number_density_;
    }
    KOKKOS_INLINE_FUNCTION Real GetNumberDensityCGSToCode() const
    {
        return 1. / number_density_;
    }

    KOKKOS_INLINE_FUNCTION Real GetMassDensityCodeToCGS() const { return mass_density_; }
    KOKKOS_INLINE_FUNCTION Real GetMassDensityCGSToCode() const
    {
        return 1. / mass_density_;
    }

    KOKKOS_INLINE_FUNCTION Real GetTemperatureCodeToCGS() const { return temperature_; }
    KOKKOS_INLINE_FUNCTION Real GetTemperatureCGSToCode() const
    {
        return 1. / temperature_;
    }

    KOKKOS_INLINE_FUNCTION Real GetEntropyCodeToCGS() const
    {
        return (energy_ / mass_) / temperature_;
    }
    KOKKOS_INLINE_FUNCTION Real GetEntropyCGSToCode() const
    {
        return mass_ * temperature_ / energy_;
    }

  private:
    bool scale_free_;
    Real mass_;
    Real length_;
    Real time_;
    Real energy_;
    Real number_density_;
    Real mass_density_;
    Real temperature_;
};

class CodeConstants
{
    using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;

  public:
    CodeConstants(UnitConversions unit_conv)
        : CodeConstants(MakeCodeConstants(unit_conv))
    {}

    CodeConstants(CodeConstants&& mE) = default;

    CodeConstants(CodeConstants& mE) = default;

    const Real h;
    const Real c;
    const Real kb;
    const Real mp;

  private:
    CodeConstants(const Real h_, const Real c_, const Real kb_, const Real mp_)
        : h(h_)
        , c(c_)
        , kb(kb_)
        , mp(mp_)
    {}

    static CodeConstants MakeCodeConstants(UnitConversions unit_conv)
    {
        const Real TIME = unit_conv.GetTimeCGSToCode();
        const Real MASS = unit_conv.GetMassCGSToCode();
        const Real LENGTH = unit_conv.GetLengthCGSToCode();
        const Real ENERGY = MASS * LENGTH * LENGTH / (TIME * TIME);
        const Real TEMPERATURE = unit_conv.GetTemperatureCGSToCode();
        return CodeConstants(pc::h * MASS * LENGTH * LENGTH / TIME, pc::c * LENGTH / TIME,
            pc::kb * ENERGY / TEMPERATURE, pc::mp * MASS);
    }
};

constexpr Real solar_mass = 1.989e33; // g

std::shared_ptr<KHARMAPackage> Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages);

} // namespace Units
