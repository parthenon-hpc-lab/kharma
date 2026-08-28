// © 2021-2023. Triad National Security, LLC. All rights reserved.  This
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

// system includes
#include <cmath>
#include <memory>
#include <string>
#include <vector>

// parthenon includes
#include <parthenon/package.hpp>

// phoebus includes
#include "microphysics/eos_kharma/eos_kharma.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "phoebus_utils/variables.hpp"

using namespace singularity;

namespace Microphysics {
namespace EOS {

parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;

using names_t = std::vector<std::string>;
const names_t valid_eos_names = {IdealGas::EosType()
#ifdef SPINER_USE_HDF
                                     ,
                                 StellarCollapse::EosType()
#endif
};

std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages) {
  auto pkg = std::make_shared<KHARMAPackage>("eos");
  Params &params = pkg->AllParams();

  const std::string block_name = "eos";

  phoebus::UnitConversions unit_conv(pin);
  const Real time_unit = unit_conv.GetTimeCodeToCGS();
  const Real mass_unit = unit_conv.GetMassCodeToCGS();
  const Real length_unit = unit_conv.GetLengthCodeToCGS();
  const Real temp_unit = unit_conv.GetTemperatureCodeToCGS();

  // If using StellarCollapse, we need additional variables.
  // We also need table max and min values, regardless of the EOS.
  // These can be used for floors/ceilings or for root find bounds

  Real lambda[2] = {0.};
  Real rho_min;
  Real sie_min;
  Real T_min;
  Real rho_max;
  Real sie_max;
  Real T_max;
  Real ye_min;
  Real ye_max;
  Real h_min;                             // lower enthalpy bound
  Real T, dT, rho, drho, ye, dye, eps, P; // for our bound search, temps
  int n;

  //std::string eos_type = pin->GetString(block_name, std::string("type"));
  std::string eos_type = pin->GetOrAddString(block_name, std::string("type"), "IdealGas");
  params.Add("type", eos_type);
  bool needs_ye = false;
  bool provides_entropy = false;
  if (eos_type.compare(IdealGas::EosType()) == 0) {
    Real gamma1;
    // Check if there is gamma in <GRMHD> block for backwards compatibility
    if(!pin -> DoesParameterExist("GRMHD", "gamma") && !pin -> DoesParameterExist(block_name, "gamma")) {
      throw std::runtime_error("IdealGas EOS requires that gamma be specified in <GRMHD> or <eos> block!");
    }

    if(pin -> DoesParameterExist("GRMHD", "gamma")) {
      gamma1 = pin->GetReal("GRMHD", std::string("gamma"));
    } else {
      gamma1 = pin->GetReal(block_name, std::string("gamma"));
    }

    const Real gm1 = gamma1 - 1.0;
    const Real Cv = pin->GetOfAddReal(block_name, "Cv", 1.0);
    params.Add("gm1", gm1);
    params.Add("Cv", Cv);

    EOS eos_host = UnitSystem<IdealGas>(IdealGas(gm1, Cv),
                                        eos_units_init::length_time_units_init_tag,
                                        time_unit, mass_unit, length_unit, temp_unit);
    EOS eos_device = eos_host.GetOnDevice();

    params.Add("d.EOS", eos_device);
    params.Add("h.EOS", eos_host);
    
    // Can specify rho_min, etc, in <eos>
    rho_min = pin->GetOrAddReal("eos", "rho_min", 0.0);
    sie_min = pin->GetOrAddReal("eos", "sie_min", 0.0);
    T_min = eos_host.TemperatureFromDensityInternalEnergy(rho_min, sie_min, lambda);
    rho_max = pin->GetOrAddReal("eos", "rho_max", 1e18);
    sie_max = pin->GetOrAddReal("eos", "sie_max", 1e35);
    T_max = eos_host.TemperatureFromDensityInternalEnergy(rho_max, sie_max, lambda);
    ye_min = 0.01;
    ye_max = 1.0;
    h_min = 1.0; // is this the right guess for an ideal gas case?

#ifdef SPINER_USE_HDF
  } else if (eos_type == StellarCollapse::EosType()) {
    // We request that Ye and temperature exist, but do not provide them.
    Metadata m = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Derived,
                           Metadata::OneCopy, Metadata::Requires});

    // TODO: prob fix this later when we start using nuclear eos
    // pkg->AddField(fluid_prim::ye::name(), m);
    // pkg->AddField(fluid_prim::temperature::name(), m);

    const std::string filename = pin->GetString(block_name, "filename");
    const bool use_sp5 = pin->GetOrAddBoolean(block_name, "use_sp5", true);
    const bool filter_bmod = pin->GetOrAddBoolean(block_name, "filter_bmod", true);
    const bool use_ye = pin->GetOrAddBoolean("fluid", "Ye", false);
    bool save_sp5 = false;
    if (!use_sp5) {
      save_sp5 = pin->GetOrAddBoolean(block_name, "save_sp5", false);
    }

    provides_entropy = true;
    PARTHENON_REQUIRE_THROWS(use_ye,
                             "\"StellarCollapse\" EOS requires that Ye be enabled!");
    needs_ye = use_ye;
    params.Add("filename", filename);
    params.Add("use_sp5", use_sp5);
    params.Add("filter_bmod", filter_bmod);
    params.Add("save_sp5", save_sp5);

    EOS eos_host =
        UnitSystem<StellarCollapse>(StellarCollapse(filename, use_sp5, filter_bmod),
                                    eos_units_init::length_time_units_init_tag, time_unit,
                                    mass_unit, length_unit, temp_unit);
    StellarCollapse eos_sc =
        eos_host.GetUnmodifiedObject().get<singularity::StellarCollapse>();
    EOS eos_device = eos_host.GetOnDevice();

    if (!use_sp5 && save_sp5) {
      const std::string sp5_name =
          pin->GetOrAddString(block_name, "sp5_save_name", "eos.sp5");
      eos_sc.Save(sp5_name);
      params.Add("sp5_save_name", sp5_name);
    }

    params.Add("d.EOS", eos_device);
    params.Add("h.EOS", eos_host);

    Real M_unit = unit_conv.GetMassCodeToCGS();
    Real L_unit = unit_conv.GetLengthCodeToCGS();
    Real rho_unit = M_unit / std::pow(L_unit, 3);
    Real T_unit = unit_conv.GetTemperatureCodeToCGS();
    // Always C^2
    Real sie_unit = std::pow(pc.c, 2);
    Real press_unit = rho_unit * sie_unit;

    // TODO(JMM): To get around current limitations of
    sie_min = eos_sc.sieMin() / sie_unit;
    sie_max = eos_sc.sieMax() / sie_unit;
    T_min = eos_sc.TMin() / T_unit;
    T_max = eos_sc.TMax() / T_unit;
    rho_min = eos_sc.rhoMin() / rho_unit;
    rho_max = eos_sc.rhoMax() / rho_unit;
    ye_min = eos_sc.YeMin();
    ye_max = eos_sc.YeMax();

    // new calculation to find lower bound for enthalpy!
    n = 200;     // maybe this shouldn't be hardcoded in the future?
    h_min = 1.0; // initial guess for minimum enthalpy, sufficient in ideal cases.
    eps = 0.0;

    // spacing for each dimension (logspace for rho, T; linspace for ye)
    // we need to use physical (i.e. not scaled) units here for eos_sc calcs
    dT = (log10(eos_sc.TMax()) - log10(eos_sc.TMin())) / (n - 1);
    drho = (log10(eos_sc.rhoMax()) - log10(eos_sc.rhoMin())) / (n - 1);
    dye = (ye_max - ye_min) / ((n / 2) - 1); // we don't need to resolve ye as much

    // initial conditions
    T = eos_sc.TMin();
    rho = eos_sc.rhoMin();
    ye = ye_min;

    for (int y = 0; y < n / 2; y++) {
      lambda[0] = ye;
      for (int r = 0; r < n; r++) {
        for (int t = 0; t < n; t++) {
          eps = eos_sc.InternalEnergyFromDensityTemperature(rho, T, lambda) / sie_unit;
          P = eos_sc.PressureFromDensityTemperature(rho, T, lambda) / press_unit;
          h_min = std::min(h_min, 1 + eps + robust::ratio(P, rho));
          T *= pow(10.0, dT); // log spacing
        }
        T = eos_sc.TMin();      // "reset"
        rho *= pow(10.0, drho); // log spacing
      }
      rho = eos_sc.rhoMin(); // "reset"
      ye += dye;             // linear spacing
    }

#endif
  } else {
    std::stringstream error_mesg;
    error_mesg << __func__ << ": " << eos_type << " is an invalid EOS selection."
               << " Valid EOS names are:\n";
    for (const auto &name : valid_eos_names) {
      error_mesg << "\t" << name << "\n";
    }
    error_mesg << std::endl;
    PARTHENON_THROW(error_mesg);
  }

  params.Add("needs_ye", needs_ye);
  params.Add("provides_entropy", provides_entropy);

  params.Add("sie_min", sie_min);
  params.Add("sie_max", sie_max);
  params.Add("T_min", T_min);
  params.Add("T_max", T_max);
  params.Add("rho_min", rho_min);
  params.Add("rho_max", rho_max);
  params.Add("ye_min", ye_min);
  params.Add("ye_max", ye_max);
  params.Add("h_min", h_min);

  return pkg;
}
} // namespace EOS
} // namespace Microphysics