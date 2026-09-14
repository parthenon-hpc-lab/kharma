// © 2021. Triad National Security, LLC. All rights reserved.  This
// program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.

#include "unit_conversions.hpp"

namespace phoebus {

parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;

// Construct unit conversion factors based on a mass/length scale for the
// geometry and a mass scale for the fluid. Assume kb = 1 in code units.
UnitConversions::UnitConversions(ParameterInput *pin) {
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
    int geom_mass_g_exists = pin->DoesParameterExist("units", "geom_mass_g");
    int geom_mass_msun_exists = pin->DoesParameterExist("units", "geom_mass_msun");
    int geom_length_cm_exists = pin->DoesParameterExist("units", "geom_length_cm");
    int fluid_density_cgs_exists = pin->DoesParameterExist("units", "fluid_density_cgs");
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
      mass_ = length_ * length_ * length_ * pin->GetReal("units", "fluid_density_cgs");
    }
  }

  time_ = length_ / pc.c;

  energy_ = mass_ * pow(pc.c, 2);

  number_density_ = pow(length_, -3);

  mass_density_ = mass_ * number_density_;

  temperature_ = 1. / pc.kb;
}

} // namespace phoebus
