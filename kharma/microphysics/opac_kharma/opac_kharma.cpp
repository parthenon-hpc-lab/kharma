// system includes
#include <memory>
#include <string>
#include <vector>

// parthenon includes
#include "utils/constants.hpp"
#include <parthenon/package.hpp>

// singularity includes -- photon opacities
#include <singularity-opac/photons/mean_opacity_photons.hpp>
#include <singularity-opac/photons/mean_s_opacity_photons.hpp>
#include <singularity-opac/photons/opac_photons.hpp>
#include <singularity-opac/photons/s_opac_photons.hpp>

// phoebus includes
#include "microphysics/eos_kharma/eos_kharma.hpp"

#include "opac_kharma.hpp"
#include "radM1.hpp"

using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;

namespace Microphysics
{
namespace Opacity
{
std::shared_ptr<KHARMAPackage> Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages)
{
    using namespace singularity::photons;
    using MeanSOpacity = singularity::photons::impl::MeanSVariant<MeanSOpacityBase,
        MeanNonCGSUnitsS<MeanSOpacityBase>>;

    auto pkg = std::make_shared<KHARMAPackage>("opacity");
    Params& params = pkg->AllParams();

    bool do_rad = pin->GetOrAddBoolean("radM1", "on", false);
    if (!do_rad) {
        params.Add("opacities", Opacities());
        return pkg;
    }

    const std::string block_name = "opac";

    auto& rad_params = packages->Get("RadM1")->AllParams();
    auto opacity_type = rad_params.Get<int>("opacity_type");
    // if we are not using singularity opac, just default the constructor and proceed.
    if (opacity_type != (int)RadM1::OpacityType::Default) {
        std::string opacity_kind = pin->GetOrAddString(block_name, "opac_kind", "none");
        if (opacity_kind != "none") {
            std::stringstream msg;
            msg << "Opacity kind \"" << opacity_kind
                << "\" is set but opacity type is not default! If you want "
                   "singularity-opac "
                   "opacities, set opac/type to default in the input file.";
            PARTHENON_FAIL(msg);
        }
        params.Add("opacities", Opacities());
        return pkg;
    }

    const bool scale_free = pin->GetOrAddBoolean("units", "scale_free", true);

    auto unit_conv = Units::UnitConversions(pin);
    double time_unit = unit_conv.GetTimeCodeToCGS();
    double mass_unit = unit_conv.GetMassCodeToCGS();
    double length_unit = unit_conv.GetLengthCodeToCGS();
    double temp_unit = unit_conv.GetTemperatureCodeToCGS();

    std::string opacity_kind = pin->GetOrAddString(block_name, "opac_kind", "none");

    // Currently bremsstrahlung only, as we add more stuff, we complete here.
    std::set<std::string> known_opacity_kinds = {"none", "bremsstrahlung"};

    if (!known_opacity_kinds.count(opacity_kind)) {
        std::stringstream msg;
        msg << "Opacity model \"" << opacity_kind << "\" not recognized!";
        PARTHENON_FAIL(msg);
    }

    // Currently if type is set to default but kind = "none", the user will get 0
    // opacities without any warning. Let's try to mediate that:
    if (opacity_type == (int)RadM1::OpacityType::Default && opacity_kind == "none") {
        std::stringstream msg;
        msg << "Opacity kind is set to \"none\" but opacity type is set to default! If "
               "you want no opacities, set type to transparent.";
        PARTHENON_FAIL(msg);
    }

    params.Add("opac_kind", opacity_kind);

    if (opacity_kind == "bremsstrahlung") {
        PARTHENON_REQUIRE(
            !scale_free, "Must have CGS scaling for bremsstrahlung opacities!");

        singularity::photons::Opacity opacity_host =
            NonCGSUnits<EPBremss>(EPBremss(PlanckDistribution<>()), time_unit, mass_unit,
                length_unit, temp_unit);
        auto opacity_device = opacity_host.GetOnDevice();
        singularity::photons::Opacity opacity_host_baseunits =
            EPBremss(PlanckDistribution<>());
        params.Add("h.opacity_baseunits", opacity_host_baseunits);
        params.Add("h.opacity", opacity_host);
        params.Add("d.opacity", opacity_device);
    }

    // I'll comment this out, this comes from phoebus, we have to slowly unlock each new
    // type of radiation. I'll just leave it here for now, I'll comeback to it eventually.
    //   if (opacity_kind == "none") {
    //     // Just return 0 for everything. Still have units vs scale-free because we use
    //     the
    //     // distribution function provided by this class for example if we have a
    //     definite
    //     // scattering opacity.
    //     const Real kappa = 0.;
    //     if (scale_free) {
    //       singularity::photons::Opacity opacity_host = ScaleFree(kappa);
    //       auto opacity_device = opacity_host.GetOnDevice();
    //       params.Add("h.opacity_baseunits", opacity_host);
    //       params.Add("h.opacity", opacity_host);
    //       params.Add("d.opacity", opacity_device);
    //     } else {
    //       singularity::photons::Opacity opacity_host =
    //           NonCGSUnits<Gray>(Gray(kappa), time_unit, mass_unit, length_unit,
    //           temp_unit);
    //       auto opacity_device = opacity_host.GetOnDevice();
    //       singularity::photons::Opacity opacity_host_baseunits = Gray(kappa);
    //       params.Add("h.opacity_baseunits", opacity_host_baseunits);
    //       params.Add("h.opacity", opacity_host);
    //       params.Add("d.opacity", opacity_device);
    //     }
    //   } else if (opacity_kind == "gray") {
    //     const Real kappa = pin->GetReal(block_name, "gray_kappa");
    //     params.Add("gray_kappa", kappa);

    //     if (scale_free) {
    //       singularity::photons::Opacity opacity_host = ScaleFree(kappa);
    //       auto opacity_device = opacity_host.GetOnDevice();
    //       params.Add("h.opacity_baseunits", opacity_host);
    //       params.Add("h.opacity", opacity_host);
    //       params.Add("d.opacity", opacity_device);
    //     } else {
    //       singularity::photons::Opacity opacity_host =
    //           NonCGSUnits<Gray>(Gray(kappa), time_unit, mass_unit, length_unit,
    //           temp_unit);
    //       auto opacity_device = opacity_host.GetOnDevice();
    //       singularity::photons::Opacity opacity_host_baseunits = Gray(kappa);
    //       params.Add("h.opacity_baseunits", opacity_host_baseunits);
    //       params.Add("h.opacity", opacity_host);
    //       params.Add("d.opacity", opacity_device);
    //     }
    //   } else if (opacity_kind == "bremsstrahlung") {
    //     PARTHENON_REQUIRE(!scale_free, "Must have CGS scaling for bremsstrahlung
    //     opacities!");

    //     singularity::photons::Opacity opacity_host =
    //     NonCGSUnits<EPBremss>(EPBremss(PlanckDistribution<>()),
    //                                                  time_unit, mass_unit, length_unit,
    //                                                  temp_unit);
    //     auto opacity_device = opacity_host.GetOnDevice();
    //     singularity::photons::Opacity opacity_host_baseunits =
    //     EPBremss(PlanckDistribution<>()); params.Add("h.opacity_baseunits",
    //     opacity_host_baseunits); params.Add("h.opacity", opacity_host);
    //     params.Add("d.opacity", opacity_device);
    //   } else if (opacity_kind == "tabular") {
    // #ifdef SPINER_USE_HDF
    //     const std::string filename = pin->GetString(block_name, "filename");
    //     params.Add("filename", filename);

    //     PARTHENON_REQUIRE(!scale_free, "Must have CGS scaling for tabular opacities!");

    //     singularity::photons::Opacity opacity_host = NonCGSUnits<SpinerOpac>(
    //         SpinerOpac(filename), time_unit, mass_unit, length_unit, temp_unit);
    //     auto opacity_device = opacity_host.GetOnDevice();
    //     singularity::photons::Opacity opacity_host_baseunits = SpinerOpac(filename);
    //     params.Add("h.opacity_baseunits", opacity_host_baseunits);
    //     params.Add("h.opacity", opacity_host);
    //     params.Add("d.opacity", opacity_device);
    // #else
    //     PARTHENON_FAIL("Tabular opacities requested but HDF5 is disabled!");
    // #endif
    //   }

    {
        auto opacity_host =
            params.Get<singularity::photons::Opacity>("h.opacity_baseunits");
        // Gray (ngroups=1) mean opacity spanning the full spectrum: [0, +inf).
        const std::vector<Real> group_bounds = {
            0., std::numeric_limits<Real>::infinity()};
        const int ngroups = 1;
        const int NNuPerGroup = pin->GetOrAddInteger("mean_opacity", "nnu", 100);
        if (scale_free) {
            const Real lRhoMin =
                pin->GetOrAddReal("mean_opacity", "lrhomin", std::log10(0.1));
            const Real lRhoMax =
                pin->GetOrAddReal("mean_opacity", "lrhomax", std::log10(10.));
            const int NRho = pin->GetOrAddInteger("mean_opacity", "nrho", 10);
            const Real lTMin =
                pin->GetOrAddReal("mean_opacity", "ltmin", std::log10(0.1));
            const Real lTMax =
                pin->GetOrAddReal("mean_opacity", "ltmax", std::log10(10.));
            const int NT = pin->GetOrAddInteger("mean_opacity", "nt", 10);
            auto mean_opac_host = MeanOpacityBase(opacity_host, lRhoMin, lRhoMax, NRho,
                lTMin, lTMax, NT, group_bounds, ngroups, NNuPerGroup);
            MeanOpacity mean_opac_device = mean_opac_host.GetOnDevice();
            params.Add("h.mean_opacity", mean_opac_host);
            params.Add("d.mean_opacity", mean_opac_device);
        } else {
            const Real lRhoMin =
                pin->GetOrAddReal("mean_opacity", "lrhomin", std::log10(1.e-10));
            const Real lRhoMax =
                pin->GetOrAddReal("mean_opacity", "lrhomax", std::log10(1.e5));
            const int NRho = pin->GetOrAddInteger("mean_opacity", "nrho", 10);
            const Real lTMin =
                pin->GetOrAddReal("mean_opacity", "ltmin", std::log10(1.e2));
            const Real lTMax =
                pin->GetOrAddReal("mean_opacity", "ltmax", std::log10(1.e12));
            const int NT = pin->GetOrAddInteger("mean_opacity", "nt", 10);
            auto cgs_mean_opacity = MeanOpacityBase(opacity_host, lRhoMin, lRhoMax, NRho,
                lTMin, lTMax, NT, group_bounds, ngroups, NNuPerGroup);
            auto mean_opac_host = MeanNonCGSUnits<MeanOpacityBase>(
                std::forward<MeanOpacityBase>(cgs_mean_opacity), time_unit, mass_unit,
                length_unit, temp_unit);
            MeanOpacity mean_opac_device = mean_opac_host.GetOnDevice();
            params.Add("h.mean_opacity", mean_opac_host);
            params.Add("d.mean_opacity", mean_opac_device);
        }
    }

    const Real avg_particle_mass = pc::mp;
    // Pick which scattering opacity the user wants
    const std::string s_block_name = "opac";
    std::string s_opacity_kind = pin->GetOrAddString(s_block_name, "s_opac_kind", "none");
    std::set<std::string> known_s_opacity_kinds = {"thomson"};

    // Currently just thompson.
    if (!known_s_opacity_kinds.count(s_opacity_kind)) {
        std::stringstream msg;
        msg << "Scattering opacity model \"" << s_opacity_kind << "\" not recognized!";
        PARTHENON_FAIL(msg);
    }

    if (s_opacity_kind == "thomson") {
        // Thermal electron (Thomson) scattering.
        PARTHENON_REQUIRE(!scale_free, "Must have CGS scaling for Thomson scattering!");

        SOpacity opacity_host = NonCGSUnitsS<ThomsonS>(
            ThomsonS(avg_particle_mass), time_unit, mass_unit, length_unit, temp_unit);
        SOpacity opacity_host_baseunits = ThomsonS(avg_particle_mass);
        auto opacity_device = opacity_host.GetOnDevice();
        params.Add("h.s_opacity_baseunits", opacity_host_baseunits);
        params.Add("h.s_opacity", opacity_host);
        params.Add("d.s_opacity", opacity_device);
    }

    //   PARTHENON_REQUIRE(
    //       !(s_opacity_kind == "scalefree" && !unit_conv.IsScaleFree()),
    //       "Scale free opacity only supported for scale-free phoebus simulations!");

    //   params.Add("s_type", s_opacity_kind);

    //   const Real avg_particle_mass = pc::mp;

    //   if (s_opacity_kind == "none") {
    //     const Real kappa = 0.;
    //     if (scale_free) {
    //       SOpacity opacity_host = ScaleFreeS(kappa, 1.);
    //       auto opacity_device = opacity_host.GetOnDevice();
    //       params.Add("h.s_opacity_baseunits", opacity_host);
    //       params.Add("h.s_opacity", opacity_host);
    //       params.Add("d.s_opacity", opacity_device);
    //     } else {
    //       SOpacity opacity_host =
    //           NonCGSUnitsS<GrayS>(GrayS(kappa * avg_particle_mass, avg_particle_mass),
    //                               time_unit, mass_unit, length_unit, temp_unit);
    //       SOpacity opacity_host_baseunits =
    //           GrayS(kappa * avg_particle_mass, avg_particle_mass);
    //       auto opacity_device = opacity_host.GetOnDevice();
    //       params.Add("h.s_opacity_baseunits", opacity_host_baseunits);
    //       params.Add("h.s_opacity", opacity_host);
    //       params.Add("d.s_opacity", opacity_device);
    //     }
    //   } else if (s_opacity_kind == "gray") {
    //     const Real kappa = pin->GetReal(s_block_name, "s_gray_kappa");
    //     params.Add("s_gray_kappa", kappa);

    //     if (scale_free) {
    //       SOpacity opacity_host = ScaleFreeS(kappa, 1.);
    //       auto opacity_device = opacity_host.GetOnDevice();
    //       params.Add("h.s_opacity_baseunits", opacity_host);
    //       params.Add("h.s_opacity", opacity_host);
    //       params.Add("d.s_opacity", opacity_device);
    //     } else {
    //       SOpacity opacity_host =
    //           NonCGSUnitsS<GrayS>(GrayS(kappa * avg_particle_mass, avg_particle_mass),
    //                               time_unit, mass_unit, length_unit, temp_unit);
    //       SOpacity opacity_host_baseunits =
    //           GrayS(kappa * avg_particle_mass, avg_particle_mass);
    //       auto opacity_device = opacity_host.GetOnDevice();
    //       params.Add("h.s_opacity_baseunits", opacity_host_baseunits);
    //       params.Add("h.s_opacity", opacity_host);
    //       params.Add("d.s_opacity", opacity_device);
    //     }
    //   } else if (s_opacity_kind == "thomson") {
    //     // Thermal electron (Thomson) scattering.
    //     PARTHENON_REQUIRE(!scale_free, "Must have CGS scaling for Thomson
    //     scattering!");

    //     SOpacity opacity_host = NonCGSUnitsS<ThomsonS>(
    //         ThomsonS(avg_particle_mass), time_unit, mass_unit, length_unit, temp_unit);
    //     SOpacity opacity_host_baseunits = ThomsonS(avg_particle_mass);
    //     auto opacity_device = opacity_host.GetOnDevice();
    //     params.Add("h.s_opacity_baseunits", opacity_host_baseunits);
    //     params.Add("h.s_opacity", opacity_host);
    //     params.Add("d.s_opacity", opacity_device);
    //   }

    // Declaring mean opacities..
    {
        auto opacity_host = params.Get<SOpacity>("h.s_opacity_baseunits");
        const std::vector<Real> group_bounds = {
            0., std::numeric_limits<Real>::infinity()};
        const int ngroups = 1;
        const int NNuPerGroup = pin->GetOrAddInteger("mean_opacity", "nnu", 100);
        if (scale_free) {
            const Real lRhoMin =
                pin->GetOrAddReal("mean_opacity", "lrhomin", std::log10(0.1));
            const Real lRhoMax =
                pin->GetOrAddReal("mean_opacity", "lrhomax", std::log10(10.));
            const int NRho = pin->GetOrAddInteger("mean_opacity", "nrho", 10);
            const Real lTMin =
                pin->GetOrAddReal("mean_opacity", "ltmin", std::log10(0.1));
            const Real lTMax =
                pin->GetOrAddReal("mean_opacity", "ltmax", std::log10(10.));
            const int NT = pin->GetOrAddInteger("mean_opacity", "nt", 10);
            MeanSOpacityBase mean_opac_host(opacity_host, lRhoMin, lRhoMax, NRho, lTMin,
                lTMax, NT, group_bounds, ngroups, NNuPerGroup);
            auto mean_opac_device = mean_opac_host.GetOnDevice();
            params.Add("h.mean_s_opacity", mean_opac_host);
            params.Add("d.mean_s_opacity", mean_opac_device);
        } else {
            const Real lRhoMin =
                pin->GetOrAddReal("mean_opacity", "lrhomin", std::log10(1.e-10));
            const Real lRhoMax =
                pin->GetOrAddReal("mean_opacity", "lrhomax", std::log10(1.e5));
            const int NRho = pin->GetOrAddInteger("mean_opacity", "nrho", 10);
            const Real lTMin =
                pin->GetOrAddReal("mean_opacity", "ltmin", std::log10(1.e2));
            const Real lTMax =
                pin->GetOrAddReal("mean_opacity", "ltmax", std::log10(1.e12));
            const int NT = pin->GetOrAddInteger("mean_opacity", "nt", 10);
            MeanSOpacityBase cgs_mean_opacity(opacity_host, lRhoMin, lRhoMax, NRho, lTMin,
                lTMax, NT, group_bounds, ngroups, NNuPerGroup);
            MeanSOpacity mean_opac_host =
                MeanNonCGSUnitsS<MeanSOpacityBase>(std::move(cgs_mean_opacity), time_unit,
                    mass_unit, length_unit, temp_unit);
            auto mean_opac_device = mean_opac_host.GetOnDevice();
            params.Add("h.mean_s_opacity", mean_opac_host);
            params.Add("d.mean_s_opacity", mean_opac_device);
        }
    }

    auto opacity_device = params.Get<singularity::photons::Opacity>("d.opacity");
    auto& mean_opac_device = params.Get<MeanOpacity>("d.mean_opacity");
    auto& s_opacity_device = params.Get<SOpacity>("d.s_opacity");
    auto& mean_s_opac_device = params.Get<MeanSOpacity>("d.mean_s_opacity");
    Opacities opacities(
        opacity_device, mean_opac_device, s_opacity_device, mean_s_opac_device);
    params.Add("opacities", opacities);

    return pkg;
}

} // namespace Opacity
} // namespace Microphysics
