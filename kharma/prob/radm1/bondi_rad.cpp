#include "bondi_rad.hpp"
#include "floors.hpp"

#include "boundaries.hpp"
#include "utils/constants.hpp"

using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;

void AddBondiRadParameters(ParameterInput* pin, Packages_t& packages)
{
    // McKinney et al. 2014, Table 4 pars.
    const Real mdot = pin->GetOrAddReal("bondi_rad", "mdot", 1.0);
    const Real T_out_K = pin->GetOrAddReal("bondi_rad", "T_out", 1.e6);
    const Real fp = pin->GetOrAddReal("bondi_rad", "fp", 1.2e-4);

    const Real T_out = T_out_K * pc::kb / (pc::mp * pc::c * pc::c);

    // Outer radius the analytic profile is anchored to (rho_0,out in eq. 94).
    const Real r_out = pin->GetReal("coordinates", "r_out");

    const Real r_in = pin->GetReal("coordinates", "r_in");
    const Real r_in_bondi = pin->GetOrAddReal("bondi_rad", "r_in_bondi", 1.2 * r_in);
    const bool fill_interior = pin->GetOrAddBoolean("bondi_rad", "fill_interior", true);

    // Add these to package properties, since they continue to be needed on boundaries
    if (!packages.Get("GRMHD")->AllParams().hasKey("mdot_rad"))
        packages.Get("GRMHD")->AddParam<Real>("mdot_rad", mdot);
    if (!packages.Get("GRMHD")->AllParams().hasKey("T_out_rad"))
        packages.Get("GRMHD")->AddParam<Real>("T_out_rad", T_out);
    if (!packages.Get("GRMHD")->AllParams().hasKey("fp_rad"))
        packages.Get("GRMHD")->AddParam<Real>("fp_rad", fp);
    if (!packages.Get("GRMHD")->AllParams().hasKey("r_out_rad"))
        packages.Get("GRMHD")->AddParam<Real>("r_out_rad", r_out);
    if (!packages.Get("GRMHD")->AllParams().hasKey("r_in_bondi_rad"))
        packages.Get("GRMHD")->AddParam<Real>("r_in_bondi_rad", r_in_bondi);
    if (!packages.Get("GRMHD")->AllParams().hasKey("fill_interior_rad"))
        packages.Get("GRMHD")->AddParam<bool>("fill_interior_rad", fill_interior);
}

/**
 * Initialization of the McKinney radiative Bondi problem
 */
TaskStatus InitializeRadiativeBondi(
    std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();

    // Add parameters we'll need throughout the run to 'GRMHD' package
    AddBondiRadParameters(pin, pmb->packages);

    //PostInitialize will fill all ghosts
    SetBondiRad<IndexDomain::interior>(rc);
    const Real r_in = pin->GetReal("coordinates", "r_in");
    const Real rin_bondi = pin->GetOrAddReal("bondi_rad", "r_in_bondi", 1.2 * r_in);
    const bool fill_interior = pin->GetOrAddBoolean("bondi", "fill_interior", false);

    // The outer boundary is held fixed at its initial analytic value every step
    // (McKinney et al. 2014, sec. 5.10); the inner boundary uses outflow condition,
    // so we don't register anything for it here.
    auto bound_pkg = pmb->packages.Get<KHARMAPackage>("Boundaries");
    if (pin->GetOrAddBoolean("bondi_rad", "set_outer_bound", true)) {
        pin->SetString("boundaries", "outer_x1", "bondi_rad");
        bound_pkg->KBoundaries[BoundaryFace::outer_x1] =
            SetBondiRad<IndexDomain::outer_x1>;
    }


    return TaskStatus::complete;
}

TaskStatus SetBondiRadImpl(
    std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();

    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    const bool use_rad = pmb->packages.AllPackages().count("RadM1");

    //Initialize u_rad and uvec_rad to anything just so it won't scream.
    GridScalar u_rad;
    GridVector uvec_rad;

    if(use_rad){
        u_rad = rc->Get("prims.u_rad").data;
        uvec_rad = rc->Get("prims.uvec_rad").data;
    }

    const Real mdot = pmb->packages.Get("GRMHD")->Param<Real>("mdot_rad");
    const Real T_out = pmb->packages.Get("GRMHD")->Param<Real>("T_out_rad");
    const Real fp = pmb->packages.Get("GRMHD")->Param<Real>("fp_rad");
    const Real r_out = pmb->packages.Get("GRMHD")->Param<Real>("r_out_rad");
    const Real r_in_bondi = pmb->packages.Get("GRMHD")->Param<Real>("r_in_bondi_rad");
    const bool fill_interior = pmb->packages.Get("GRMHD")->Param<bool>("fill_interior_rad");

    // Gamma is *derived* from fp (eq. 95), not read from <GRMHD> gamma.
    const Real gam = 1. + (1. / 3.) * ((2. + 2. * fp) / (1. + 2. * fp));

    // rho_0 evaluated at the outer radius, to anchor the polytropic T(r) (eq. 94)
    const Real ur_out = -m::sqrt(2. / r_out);
    const Real rho0_out = -mdot / (4. * M_PI * r_out * r_out * ur_out);

    const GRCoordinates& G = pmb->coords;


    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const IndexRange ib = bounds.GetBoundsI(domain);
    const IndexRange jb = bounds.GetBoundsJ(domain);
    const IndexRange kb = bounds.GetBoundsK(domain);

    pmb->par_for("bondi_rad_boundary", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                 KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            GReal Xnative[GR_DIM], Xembed[GR_DIM];
            G.coord(k, j, i, Loci::center, Xnative);
            G.coord_embed(k, j, i, Loci::center, Xembed);
            GReal r = Xembed[1];

            if (r < r_in_bondi) {
                if (fill_interior) {
                    r = r_in_bondi;
                } else {
                    rho(k, j, i) = 0.;
                    u(k, j, i) = 0.;
                    uvec(0, k, j, i) = 0.;
                    uvec(1, k, j, i) = 0.;
                    uvec(2, k, j, i) = 0.;
                    if(use_rad){
                        u_rad(k, j, i) = 0.;
                        uvec_rad(0, k, j, i) = 0.;
                        uvec_rad(1, k, j, i) = 0.;
                        uvec_rad(2, k, j, i) = 0.;
                    }
                    return;
                }
            }

            const Real ur = -m::sqrt(2. / r);
            Real ucon_bl[GR_DIM] = {0, ur, 0, 0};

            GReal gcov_bl[GR_DIM][GR_DIM];
            SphBLCoords(G.coords.get_a()).gcov_embed(Xembed, gcov_bl);

            set_ut(gcov_bl, ucon_bl);

            auto gdet_bl = G.coords.gdet_embed(Xembed);

            // Then transform that 4-vector to KS (or not, if we're using BL base coords)
            Real ucon_base[GR_DIM];
            mpark::get<SphKSCoords>(G.coords.base).vec_from_bl(Xembed, ucon_bl, ucon_base);

            
            //const Real rho0 = -mdot / (4. * M_PI * r * r * ucon_base[1]) * 1./1.5;
            const Real rho0 = -mdot / (4. * M_PI * gdet_bl * ucon_base[1]);

            const Real T = T_out * m::pow(rho0 / rho0_out, gam - 1.);
            const Real u_internal = rho0 * T / (gam - 1.);
            const Real pgas = rho0 * T;

            rho(k, j, i) = rho0;
            u(k, j, i) = u_internal;

            // Convert the radial 4-velocity to the native primitive
            Real ucon_native[GR_DIM];
            G.coords.bl_fourvel_to_native(Xnative, ucon_bl, ucon_native);
            Real gcon[GR_DIM][GR_DIM];
            G.gcon(Loci::center, j, i, gcon);
            Real u_prim[NVEC];
            fourvel_to_prim(gcon, ucon_native, u_prim);
            uvec(0, k, j, i) = u_prim[0];
            uvec(1, k, j, i) = u_prim[1];
            uvec(2, k, j, i) = u_prim[2];

            if(use_rad){
                u_rad(k, j, i) = 3. * fp * pgas;
                uvec_rad(0, k, j, i) = 0.;
                uvec_rad(1, k, j, i) = 0.;
                uvec_rad(2, k, j, i) = 0.;
            }
        });

    return TaskStatus::complete;
}
