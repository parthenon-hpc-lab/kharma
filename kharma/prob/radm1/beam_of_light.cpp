#include "beam_of_light.hpp"
#include "floors.hpp"

#include "boundaries.hpp"
#include "radM1.hpp"

#include <cmath>

void AddBeamOfLightParameters(ParameterInput* pin, Packages_t& packages)
{
    // McKinney et al. 2014, Table 3 pars, where the defaults correspond to Model 3 (the flashlight is farthest away from the BH).
    // Get the params of our boundary conditions
    const Real r_beam = pin->GetOrAddReal("beam_of_light", "r_beam", 16.0);
    const Real dr_beam = pin->GetOrAddReal("beam_of_light", "dr_beam", 0.5);
    const Real T_ambient_K = pin->GetOrAddReal("beam_of_light", "T_ambient_K", 1.e7);
    const Real T_beam_over_Ta = pin->GetOrAddReal("beam_of_light", "T_beam_over_Ta", 1000.0);

    //The flux of the beam is set by the target f = F/E. This sets how directional the radiation is. f = 0 -> isotropic
    // f = 1 -> perfectly directional.
    const Real f_target = pin->GetOrAddReal("beam_of_light", "f_target", 0.9999);

    //Background density that fills the whole domain
    const Real rho_ambient = pin->GetOrAddReal("beam_of_light", "rho_ambient", 1.0);

    //The boost here is f_target
    const Real gamma_beam = 1.0 / std::sqrt(1.0 - f_target * f_target);

    // u^theta of the radiation in an orthonormal frame (we are firing the radiation in the r-theta frame)
    const Real u_theta_beam_ortho = gamma_beam * f_target;



    //This block is actually not necessary. The units are arbitrary.
    const RadM1::UnitScales units_cgs =
        packages.Get("RadM1")->AllParams().Get<RadM1::UnitScales>("units_cgs");
    const Real energy_density_scale =
        units_cgs.energy_cgs / (units_cgs.length_cgs * units_cgs.length_cgs * units_cgs.length_cgs);
    constexpr Real sigma_sb_cgs = 5.670374419e-5;
    constexpr Real c_cgs = 2.99792458e10;
    constexpr Real arad_cgs = 4.0 * sigma_sb_cgs / c_cgs;

    // Find out the beam temperature to set E, by using E = a * T^4.
    const Real T_beam_K = T_beam_over_Ta * T_ambient_K;
    const Real E_ambient = arad_cgs * m::pow(T_ambient_K, 4.0) / energy_density_scale;
    const Real E_beam = arad_cgs * m::pow(T_beam_K, 4.0) / energy_density_scale;

    //Temperature of the gas needed to find ug (which doesn't really matter, but we do it anyway).
    // The gas in this problem does not interact with the radiation at all, so.
    const Real T_ambient = T_ambient_K / units_cgs.temperature_cgs;

    if (!packages.Get("GRMHD")->AllParams().hasKey("beam_r"))
        packages.Get("GRMHD")->AddParam<Real>("beam_r", r_beam);
    if (!packages.Get("GRMHD")->AllParams().hasKey("beam_dr"))
        packages.Get("GRMHD")->AddParam<Real>("beam_dr", dr_beam);
    if (!packages.Get("GRMHD")->AllParams().hasKey("beam_T_ambient"))
        packages.Get("GRMHD")->AddParam<Real>("beam_T_ambient", T_ambient);
    if (!packages.Get("GRMHD")->AllParams().hasKey("beam_rho_ambient"))
        packages.Get("GRMHD")->AddParam<Real>("beam_rho_ambient", rho_ambient);
    if (!packages.Get("GRMHD")->AllParams().hasKey("beam_E_ambient"))
        packages.Get("GRMHD")->AddParam<Real>("beam_E_ambient", E_ambient);
    if (!packages.Get("GRMHD")->AllParams().hasKey("beam_E_beam"))
        packages.Get("GRMHD")->AddParam<Real>("beam_E_beam", E_beam);
    if (!packages.Get("GRMHD")->AllParams().hasKey("beam_gamma"))
        packages.Get("GRMHD")->AddParam<Real>("beam_gamma", gamma_beam);
    if (!packages.Get("GRMHD")->AllParams().hasKey("beam_u_theta_ortho"))
        packages.Get("GRMHD")->AddParam<Real>("beam_u_theta_ortho", u_theta_beam_ortho);
}

TaskStatus InitializeBeamOfLight(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();

    AddBeamOfLightParameters(pin, pmb->packages);

    SetBeamOfLight<IndexDomain::entire>(rc);

    auto bound_pkg = pmb->packages.Get<KHARMAPackage>("Boundaries");
    if (pin->GetOrAddBoolean("beam_of_light", "set_inner_x2_bound", true)) {
        bound_pkg->KBoundaries[BoundaryFace::inner_x2] =
            SetBeamOfLight<IndexDomain::inner_x2>;
    }

    return TaskStatus::complete;
}

TaskStatus SetBeamOfLightImpl(
    std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();

    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    const bool use_rad = pmb->packages.AllPackages().count("RadM1");

    GridScalar u_rad;
    GridVector uvec_rad;
    if (use_rad) {
        u_rad = rc->Get("prims.u_rad").data;
        uvec_rad = rc->Get("prims.uvec_rad").data;
    }

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real rho_ambient = pmb->packages.Get("GRMHD")->Param<Real>("beam_rho_ambient");
    const Real T_ambient = pmb->packages.Get("GRMHD")->Param<Real>("beam_T_ambient");
    const Real E_ambient = pmb->packages.Get("GRMHD")->Param<Real>("beam_E_ambient");
    const Real E_beam = pmb->packages.Get("GRMHD")->Param<Real>("beam_E_beam");
    const Real gamma_beam = pmb->packages.Get("GRMHD")->Param<Real>("beam_gamma");
    const Real u_theta_ortho = pmb->packages.Get("GRMHD")->Param<Real>("beam_u_theta_ortho");
    const Real r_beam = pmb->packages.Get("GRMHD")->Param<Real>("beam_r");
    const Real dr_beam = pmb->packages.Get("GRMHD")->Param<Real>("beam_dr");

    const Real Erf_beam = E_beam / ((4.0 / 3.0) * gamma_beam * gamma_beam - (1.0 / 3.0));
    const bool inject_beam = (domain == IndexDomain::inner_x2);

    const GRCoordinates& G = pmb->coords;

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const IndexRange ib = bounds.GetBoundsI(domain);
    const IndexRange jb = bounds.GetBoundsJ(domain);
    const IndexRange kb = bounds.GetBoundsK(domain);
    const int j_active = bounds.GetBoundsJ(IndexDomain::interior).s;

    pmb->par_for("beam_of_light_boundary", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            rho(k, j, i) = rho_ambient;
            u(k, j, i) = rho_ambient * T_ambient / (gam - 1.0);
            uvec(0, k, j, i) = 0.;
            uvec(1, k, j, i) = 0.;
            uvec(2, k, j, i) = 0.;

            if (!use_rad) return;

            //Not on the injection face at all
            //runs whenever this boundary function is being used to fill somewhere that isn't the flashlight boundary 
            if (!inject_beam) {
                u_rad(k, j, i) = E_ambient;
                uvec_rad(0, k, j, i) = 0.;
                uvec_rad(1, k, j, i) = 0.;
                uvec_rad(2, k, j, i) = 0.;
                return;
            }

            GReal Xembed[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, Xembed);
            const GReal r = Xembed[1];


            //If cell is within the radial interval where the beam is emitted.
            if (m::abs(r - r_beam) <= dr_beam) {
                const Real gcov_11 = G.gcov(Loci::center, j, i, 1, 1);
                const Real gcov_22 = G.gcov(Loci::center, j, i, 2, 2);
                const Real gcon_tr = G.gcon(Loci::center, j, i, 0, 1);
                const Real alpha = 1. / m::sqrt(-G.gcon(Loci::center, j, i, 0, 0));

                //Since the boost was defined in the orthornormal frame, this will convert the boost into u^theta as a coordinate component.
                const Real u1_th = u_theta_ortho / m::sqrt(gcov_22);

                // Now, we want the 4-velocity component u^r to be zero (movement only in the theta direction)
                // Since g_tr is not 0, we have radial shift, when calculating u^mu from the primitive eulerian velocities
                // This is given by u_rad^r = uvec_rad[0] − γ·α·g^{tr}
                // so just normalize it to be 0, by setting uvec_rad[0] = γ·α·g^{tr}
                const Real qsq_th = gcov_22 * u1_th * u1_th;
                const Real C = alpha * gcon_tr;
                const Real u0_r = C * m::sqrt((1. + qsq_th) / (1. - C * C * gcov_11));

                u_rad(k, j, i) = Erf_beam;
                uvec_rad(0, k, j, i) = u0_r;
                uvec_rad(1, k, j, i) = u1_th;
                uvec_rad(2, k, j, i) = 0.;
            } else {
                u_rad(k, j, i) = u_rad(k, j_active, i);
                uvec_rad(0, k, j, i) = uvec_rad(0, k, j_active, i);
                uvec_rad(1, k, j, i) = uvec_rad(1, k, j_active, i);
                uvec_rad(2, k, j, i) = uvec_rad(2, k, j_active, i);
            }
        });

    return TaskStatus::complete;
}
