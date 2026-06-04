// © 2021-2022. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
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
#include <memory>
#include <string>
#include <vector>

// parthenon includes
#include "utils/constants.hpp"
#include <parthenon/package.hpp>

// singularity includes
#include <singularity-opac/neutrinos/mean_opacity_neutrinos.hpp>
#include <singularity-opac/neutrinos/mean_s_opacity_neutrinos.hpp>
#include <singularity-opac/neutrinos/opac_neutrinos.hpp>
#include <singularity-opac/neutrinos/s_opac_neutrinos.hpp>

// phoebus includes
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "phoebus_utils/unit_conversions.hpp"

#include "opac_phoebus.hpp"

using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;

namespace Microphysics {
namespace Opacity {
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  using namespace singularity::neutrinos;

  auto pkg = std::make_shared<StateDescriptor>("opacity");
  Params &params = pkg->AllParams();

  bool do_rad = pin->GetBoolean("physics", "rad");
  if (!do_rad) {
    params.Add("opacities", Opacities());
    return pkg;
  }

  const bool scale_free = pin->GetOrAddBoolean("units", "scale_free", true);

  const std::string block_name = "opacity";

  auto unit_conv = phoebus::UnitConversions(pin);
  double time_unit = unit_conv.GetTimeCodeToCGS();
  double mass_unit = unit_conv.GetMassCodeToCGS();
  double length_unit = unit_conv.GetLengthCodeToCGS();
  double temp_unit = unit_conv.GetTemperatureCodeToCGS();

  std::string opacity_type = pin->GetOrAddString(block_name, "type", "none");
  std::set<std::string> known_opacity_types = {"none", "tophat", "gray", "tabular"};
  if (!known_opacity_types.count(opacity_type)) {
    std::stringstream msg;
    msg << "Opacity model \"" << opacity_type << "\" not recognized!";
    PARTHENON_FAIL(msg);
  }

  params.Add("type", opacity_type);

  if (opacity_type == "none") {
    // Just return 0 for everything. Still have units vs scale-free because we use the
    // distribution function provided by this class for example if we have a definite
    // scattering opacity.
    const Real kappa = 0.;
    if (scale_free) {
      singularity::neutrinos::Opacity opacity_host = ScaleFree(kappa);
      auto opacity_device = opacity_host.GetOnDevice();
      params.Add("h.opacity_baseunits", opacity_host);
      params.Add("h.opacity", opacity_host);
      params.Add("d.opacity", opacity_device);
    } else {
      singularity::neutrinos::Opacity opacity_host =
          NonCGSUnits<Gray>(Gray(kappa), time_unit, mass_unit, length_unit, temp_unit);
      auto opacity_device = opacity_host.GetOnDevice();
      singularity::neutrinos::Opacity opacity_host_baseunits = Gray(kappa);
      params.Add("h.opacity_baseunits", opacity_host_baseunits);
      params.Add("h.opacity", opacity_host);
      params.Add("d.opacity", opacity_device);
    }
  } else if (opacity_type == "tophat") {
    const Real C = pin->GetReal("opacity", "tophat_C");
    const Real numin = pin->GetReal("opacity", "tophat_numin");
    const Real numax = pin->GetReal("opacity", "tophat_numax");
    params.Add("C", C);
    params.Add("numin", numin);
    params.Add("numax", numax);

    PARTHENON_REQUIRE(!scale_free, "Must have CGS scaling for tophat opacities!");

    singularity::neutrinos::Opacity opacity_host = NonCGSUnits<Tophat>(
        Tophat(C, numin, numax), time_unit, mass_unit, length_unit, temp_unit);
    auto opacity_device = opacity_host.GetOnDevice();
    singularity::neutrinos::Opacity opacity_host_baseunits = Tophat(C, numin, numax);
    params.Add("h.opacity_baseunits", opacity_host_baseunits);
    params.Add("h.opacity", opacity_host);
    params.Add("d.opacity", opacity_device);
  } else if (opacity_type == "gray") {
    const Real kappa = pin->GetReal("opacity", "gray_kappa");
    params.Add("gray_kappa", kappa);

    if (scale_free) {
      singularity::neutrinos::Opacity opacity_host = ScaleFree(kappa);
      auto opacity_device = opacity_host.GetOnDevice();
      params.Add("h.opacity_baseunits", opacity_host);
      params.Add("h.opacity", opacity_host);
      params.Add("d.opacity", opacity_device);
    } else {
      singularity::neutrinos::Opacity opacity_host =
          NonCGSUnits<Gray>(Gray(kappa), time_unit, mass_unit, length_unit, temp_unit);
      auto opacity_device = opacity_host.GetOnDevice();
      singularity::neutrinos::Opacity opacity_host_baseunits = Gray(kappa);
      params.Add("h.opacity_baseunits", opacity_host_baseunits);
      params.Add("h.opacity", opacity_host);
      params.Add("d.opacity", opacity_device);
    }
  } else if (opacity_type == "tabular") {
#ifdef SPINER_USE_HDF
    const std::string filename = pin->GetString("opacity", "filename");
    params.Add("filename", filename);

    PARTHENON_REQUIRE(!scale_free, "Must have CGS scaling for tabular opacities!");

    singularity::neutrinos::Opacity opacity_host = NonCGSUnits<SpinerOpac>(
        SpinerOpac(filename), time_unit, mass_unit, length_unit, temp_unit);
    auto opacity_device = opacity_host.GetOnDevice();
    singularity::neutrinos::Opacity opacity_host_baseunits = SpinerOpac(filename);
    params.Add("h.opacity_baseunits", opacity_host_baseunits);
    params.Add("h.opacity", opacity_host);
    params.Add("d.opacity", opacity_device);
#else
    PARTHENON_FAIL("Tabular opacities requested but HDF5 is disabled!");
#endif
  }

  {
    auto opacity_host =
        params.Get<singularity::neutrinos::Opacity>("h.opacity_baseunits");
    const Real YeMin = pin->GetOrAddReal("mean_opacity", "yemin", 0.1);
    const Real YeMax = pin->GetOrAddReal("mean_opacity", "yemax", 0.5);
    const int NYe = pin->GetOrAddInteger("mean_opacity", "nye", 10);
    if (scale_free) {
      const Real lRhoMin = pin->GetOrAddReal("mean_opacity", "lrhomin", std::log10(0.1));
      const Real lRhoMax = pin->GetOrAddReal("mean_opacity", "lrhomax", std::log10(10.));
      const int NRho = pin->GetOrAddInteger("mean_opacity", "nrho", 10);
      const Real lTMin = pin->GetOrAddReal("mean_opacity", "ltmin", std::log10(0.1));
      const Real lTMax = pin->GetOrAddReal("mean_opacity", "ltmax", std::log10(10.));
      const int NT = pin->GetOrAddInteger("mean_opacity", "nt", 10);
      const Real lNuMin = std::log10(pin->GetOrAddReal("mean_opacity", "numin", 0.1));
      const Real lNuMax = std::log10(pin->GetOrAddReal("mean_opacity", "numax", 10.));
      const int NNu = pin->GetOrAddInteger("mean_opacity", "nnu", 100);
      MeanOpacity mean_opac_host =
          MeanOpacityScaleFree(opacity_host, lRhoMin, lRhoMax, NRho, lTMin, lTMax, NT,
                               YeMin, YeMax, NYe, lNuMin, lNuMax, NNu);
      auto mean_opac_device = mean_opac_host.GetOnDevice();
      params.Add("h.mean_opacity", mean_opac_host);
      params.Add("d.mean_opacity", mean_opac_device);
    } else {
      const Real lRhoMin = pin->GetOrAddReal("mean_opacity", "lrhomin", std::log10(1.e5));
      const Real lRhoMax =
          pin->GetOrAddReal("mean_opacity", "lrhomax", std::log10(1.e14));
      const int NRho = pin->GetOrAddInteger("mean_opacity", "nrho", 10);
      const Real lTMin = pin->GetOrAddReal("mean_opacity", "ltmin", std::log10(1.e5));
      const Real lTMax = pin->GetOrAddReal("mean_opacity", "ltmax", std::log10(1.e12));
      const int NT = pin->GetOrAddInteger("mean_opacity", "nt", 10);
      const Real lNuMin = std::log10(pin->GetOrAddReal("mean_opacity", "numin", 1.e10));
      const Real lNuMax = std::log10(pin->GetOrAddReal("mean_opacity", "numax", 1.e24));
      const int NNu = pin->GetOrAddInteger("mean_opacity", "nnu", 100);
      MeanOpacityCGS cgs_mean_opacity =
          MeanOpacityCGS(opacity_host, lRhoMin, lRhoMax, NRho, lTMin, lTMax, NT, YeMin,
                         YeMax, NYe, lNuMin, lNuMax, NNu);
      MeanOpacity mean_opac_host =
          MeanNonCGSUnits<MeanOpacityCGS>(std::forward<MeanOpacityCGS>(cgs_mean_opacity),
                                          time_unit, mass_unit, length_unit, temp_unit);
      auto mean_opac_device = mean_opac_host.GetOnDevice();
      params.Add("h.mean_opacity", mean_opac_host);
      params.Add("d.mean_opacity", mean_opac_device);
    }
  }

  const std::string s_block_name = "s_opacity";
  std::string s_opacity_type = pin->GetOrAddString(s_block_name, "type", "none");
  std::set<std::string> known_s_opacity_types = {"none", "gray"};
  if (!known_s_opacity_types.count(s_opacity_type)) {
    std::stringstream msg;
    msg << "Scattering opacity model \"" << s_opacity_type << "\" not recognized!";
    PARTHENON_FAIL(msg);
  }

  PARTHENON_REQUIRE(
      !(s_opacity_type == "scalefree" && !unit_conv.IsScaleFree()),
      "Scale free opacity only supported for scale-free phoebus simulations!");

  params.Add("s_type", s_opacity_type);

  const Real avg_particle_mass = pc::mp;

  if (s_opacity_type == "none") {
    const Real kappa = 0.;
    if (scale_free) {
      singularity::neutrinos::SOpacity opacity_host = ScaleFreeS(kappa, 1.);
      auto opacity_device = opacity_host.GetOnDevice();
      params.Add("h.s_opacity_baseunits", opacity_host);
      params.Add("h.s_opacity", opacity_host);
      params.Add("d.s_opacity", opacity_device);
    } else {
      singularity::neutrinos::SOpacity opacity_host =
          NonCGSUnitsS<GrayS>(GrayS(kappa * avg_particle_mass, avg_particle_mass),
                              time_unit, mass_unit, length_unit, temp_unit);
      singularity::neutrinos::SOpacity opacity_host_baseunits =
          GrayS(kappa * avg_particle_mass, avg_particle_mass);
      auto opacity_device = opacity_host.GetOnDevice();
      params.Add("h.s_opacity_baseunits", opacity_host_baseunits);
      params.Add("h.s_opacity", opacity_host);
      params.Add("d.s_opacity", opacity_device);
    }
  } else if (s_opacity_type == "gray") {
    const Real kappa = pin->GetReal(s_block_name, "gray_kappa");
    params.Add("s_gray_kappa", kappa);

    if (scale_free) {
      singularity::neutrinos::SOpacity opacity_host = ScaleFreeS(kappa, 1.);
      auto opacity_device = opacity_host.GetOnDevice();
      params.Add("h.s_opacity_baseunits", opacity_host);
      params.Add("h.s_opacity", opacity_host);
      params.Add("d.s_opacity", opacity_device);
    } else {
      singularity::neutrinos::SOpacity opacity_host =
          NonCGSUnitsS<GrayS>(GrayS(kappa * avg_particle_mass, avg_particle_mass),
                              time_unit, mass_unit, length_unit, temp_unit);
      singularity::neutrinos::SOpacity opacity_host_baseunits =
          GrayS(kappa * avg_particle_mass, avg_particle_mass);
      auto opacity_device = opacity_host.GetOnDevice();
      params.Add("h.s_opacity_baseunits", opacity_host_baseunits);
      params.Add("h.s_opacity", opacity_host);
      params.Add("d.s_opacity", opacity_device);
    }
  }

  {
    auto opacity_host =
        params.Get<singularity::neutrinos::SOpacity>("h.s_opacity_baseunits");
    const Real YeMin = pin->GetOrAddReal("mean_opacity", "yemin", 0.1);
    const Real YeMax = pin->GetOrAddReal("mean_opacity", "yemax", 0.5);
    const int NYe = pin->GetOrAddInteger("mean_opacity", "nye", 10);
    if (scale_free) {
      const Real lRhoMin = pin->GetOrAddReal("mean_opacity", "lrhomin", std::log10(0.1));
      const Real lRhoMax = pin->GetOrAddReal("mean_opacity", "lrhomax", std::log10(10.));
      const int NRho = pin->GetOrAddInteger("mean_opacity", "nrho", 10);
      const Real lTMin = pin->GetOrAddReal("mean_opacity", "ltmin", std::log10(0.1));
      const Real lTMax = pin->GetOrAddReal("mean_opacity", "ltmax", std::log10(10.));
      const int NT = pin->GetOrAddInteger("mean_opacity", "nt", 10);
      MeanSOpacity mean_opac_host = MeanSOpacityScaleFree(
          opacity_host, lRhoMin, lRhoMax, NRho, lTMin, lTMax, NT, YeMin, YeMax, NYe);
      auto mean_opac_device = mean_opac_host.GetOnDevice();
      params.Add("h.mean_s_opacity", mean_opac_host);
      params.Add("d.mean_s_opacity", mean_opac_device);
    } else {
      const Real lRhoMin = pin->GetOrAddReal("mean_opacity", "lrhomin", std::log10(1.e5));
      const Real lRhoMax =
          pin->GetOrAddReal("mean_opacity", "lrhomax", std::log10(1.e14));
      const int NRho = pin->GetOrAddInteger("mean_opacity", "nrho", 10);
      const Real lTMin = pin->GetOrAddReal("mean_opacity", "ltmin", std::log10(1.e5));
      const Real lTMax = pin->GetOrAddReal("mean_opacity", "ltmax", std::log10(1.e12));
      const int NT = pin->GetOrAddInteger("mean_opacity", "nt", 10);
      auto cgs_mean_opacity = MeanSOpacityCGS(opacity_host, lRhoMin, lRhoMax, NRho, lTMin,
                                              lTMax, NT, YeMin, YeMax, NYe);
      MeanSOpacity mean_opac_host = MeanNonCGSUnitsS<MeanSOpacityCGS>(
          std::forward<MeanSOpacityCGS>(cgs_mean_opacity), time_unit, mass_unit,
          length_unit, temp_unit);
      auto mean_opac_device = mean_opac_host.GetOnDevice();
      params.Add("h.mean_s_opacity", mean_opac_host);
      params.Add("d.mean_s_opacity", mean_opac_device);
    }
  }

  auto opacity_device = params.Get<singularity::neutrinos::Opacity>("d.opacity");
  auto &mean_opac_device = params.Get<MeanOpacity>("d.mean_opacity");
  auto &s_opacity_device = params.Get<SOpacity>("d.s_opacity");
  auto &mean_s_opac_device = params.Get<MeanSOpacity>("d.mean_s_opacity");
  Opacities opacities(opacity_device, mean_opac_device, s_opacity_device,
                      mean_s_opac_device);
  params.Add("opacities", opacities);

  return pkg;
}

} // namespace Opacity
} // namespace Microphysics
