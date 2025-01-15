// Load the grid variables up with primitives from an old KHARMA run
#pragma once

#include "decs.hpp"

#include "mesh/mesh.hpp"

// added by Hyerin (10/07/22)
#include "bondi.hpp"
#include "b_flux_ct.hpp"

/**
 * Read the header of an KHARMA HDF5 restart file, and set appropriate parameters
 * Call this before mesh creation!
 */
void ReadKharmaRestartHeader(std::string fname, ParameterInput *pin);

/**
 * Read data from an KHARMA restart file. Does not support >1 meshblock in Parthenon
 * 
 * Returns stop time tf of the original simulation, for e.g. replicating regression tests
 */
TaskStatus ReadKharmaRestart(std::shared_ptr<MeshBlockData<Real>> rc, ParameterInput *pin);

// Hint form resize.hpp
// TODO: (Hyerin) should I do const for x1, x2, x3, var?
KOKKOS_INLINE_FUNCTION void Xtoindex(const GReal XG[GR_DIM],
                                  //Real *x1, Real *x2, Real *x3,
                                   const GridScalar& x1, const GridScalar& x2, const GridScalar& x3,
                                   const hsize_t length[GR_DIM], int& iblock,
                                   int& i, int& j, int& k, GReal del[GR_DIM])
{
    Real dx1, dx2, dx3, dx_sum, dx1_min, dx2_min, dx3_min, dx_sum_min;

    // initialize
    iblock =0;
    i = 0;
    j = 0;
    k = 0;
    dx1_min = (XG[1]-x1(iblock,i))*(XG[1]-x1(iblock,i));
    dx2_min = (XG[2]-x2(iblock,j))*(XG[2]-x2(iblock,j));
    dx3_min = (XG[3]-x3(iblock,k))*(XG[3]-x3(iblock,k));
    dx_sum_min = dx1_min + dx2_min + dx3_min;

    for (int iblocktemp = 0; iblocktemp < length[0]; iblocktemp++) {
        // independently searching for minimum for i,j,k
        for (int itemp = 0; itemp < length[1]; itemp++) {
            dx1 = (XG[1]-x1(iblocktemp,itemp))*(XG[1]-x1(iblocktemp,itemp));
            if (dx1 < dx1_min) {
                dx1_min = dx1;
                i = itemp;
            }
        }
        for (int jtemp = 0; jtemp < length[2]; jtemp++) {
            dx2 = (XG[2]-x2(iblocktemp,jtemp))*(XG[2]-x2(iblocktemp,jtemp));
            if (dx2 < dx2_min) {
                dx2_min = dx2;
                j = jtemp;
            }
        }
        for (int ktemp = 0; ktemp < length[3]; ktemp++) {
            dx3 = (XG[3]-x3(iblocktemp,ktemp))*(XG[3]-x3(iblocktemp,ktemp));
            if (dx3 < dx3_min) {
                dx3_min = dx3;
                k = ktemp;
            }
        }
        dx_sum = (XG[1]-x1(iblocktemp,i))*(XG[1]-x1(iblocktemp,i)) + 
                 (XG[2]-x2(iblocktemp,j))*(XG[2]-x2(iblocktemp,j)) + 
                 (XG[3]-x3(iblocktemp,k))*(XG[3]-x3(iblocktemp,k));
        if (dx_sum < dx_sum_min) {
            dx_sum_min = dx_sum;
            iblock = iblocktemp;
        }
    }

    del[1] = 0.; //(XG[1] - ((i) * dx[1] + startx[1])) / dx[1];
    del[2] = 0.;//(XG[2] - ((j) * dx[2] + startx[2])) / dx[2];
    del[3] = 0.;// (phi   - ((k) * dx[3] + startx[3])) / dx[3];
    if (m::abs(dx1_min / m::pow(XG[1],2.)) > 1.e-8) printf("Xtoindex: dx2 pretty large = %g at r= %g \n", dx1_min, XG[1]);
    if (m::abs(dx2_min) > 1.e-8) printf("Xtoindex: dx2 pretty large = %g at th = %g \n", dx2_min, XG[2]);
    if (m::abs(dx3_min) > 1.e-8) printf("Xtoindex: dx2 pretty large = %g at phi = %g \n", dx3_min, XG[3]);
}

// TOOD(BSP) these can be merged and moved back into the fn body now

KOKKOS_INLINE_FUNCTION void get_prim_restart_kharma(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
                    const Real fx1min, const Real fx1max, const bool should_fill, const bool is_spherical,
                    const Real gam, const Real rs, const Real mdot, const Real ur_frac, const Real uphi, const Real rin_bondi, const hsize_t length[GR_DIM], const hsize_t length_fill[GR_DIM],
                    const GridScalar& x1, const GridScalar& x2, const GridScalar& x3, const GridScalar& rho_file, const GridScalar& u_file, const GridVector& uvec_file,
                    const GridScalar& x1_fill, const GridScalar& x2_fill, const GridScalar& x3_fill, const GridScalar& rho_fill, const GridScalar& u_fill, const GridVector& uvec_fill,
                    const int& k, const int& j, const int& i) 
{
    Real rho = 0, u = 0;
    Real u_prim[NVEC] = {0}; //, B_prim[NVEC];

    GReal X[GR_DIM];
    G.coord(k, j, i, Loci::center, X);
    GReal del[GR_DIM]; // not really needed now since I am doing nearest neighbor interpolation
    int iblocktemp, itemp, jtemp, ktemp;

    // Interpolate the value at this location from the global grid
    if ((!should_fill) && (X[1]<fx1min)) {// if cannot be read from restart file
        // same as Bondi (06/13/23)
        get_prim_bondi(G, true, rs, mdot, gam, ur_frac, uphi, rin_bondi, 0., false, rho, u, u_prim, k, j, i); // TODO(HC) diffinit=true, fill_interior = false for now...

        // printf("Bondi fill location: %g %g %g %g KS: %g %g %g %g\nr: %g T: %g ur: %g\nucon: %g %g %g %g native: %g %g %g %g\nPrims: %g %g %g %g %g\n",
        //         X[0], X[1], X[2], X[3], Xembed[0], Xembed[1], Xembed[2], Xembed[3],
        //         r, T, ur, ucon_bl[0], ucon_bl[1], ucon_bl[2], ucon_bl[3], ucon_native[0], ucon_native[1], ucon_native[2], ucon_native[3],
        //         rho_temp, u_temp, u_prim[0], u_prim[1], u_prim[2]);

   }
    // HyerinTODO: if fname_fill exists and smaller.
    else if ((should_fill) && ((X[1]>fx1max)||(X[1]<fx1min))) { // fill with the fname_fill
        Xtoindex(X, x1_fill, x2_fill, x3_fill, length_fill, iblocktemp, itemp, jtemp, ktemp, del);
        rho = rho_fill(iblocktemp, ktemp, jtemp, itemp);
        u = u_fill(iblocktemp, ktemp, jtemp, itemp);
        VLOOP u_prim[v] = uvec_fill(v, iblocktemp, ktemp, jtemp, itemp);
    }
    else { 
        Xtoindex(X, x1, x2, x3, length, iblocktemp, itemp, jtemp, ktemp, del);
        rho = rho_file(iblocktemp,ktemp,jtemp,itemp);
        u = u_file(iblocktemp,ktemp,jtemp,itemp);
        VLOOP u_prim[v] = uvec_file(v,iblocktemp,ktemp,jtemp,itemp);
        //printf("File fill location: %g %g %g %g new index: %d %d %d from old index: (%d) %d %d %d\n",
        //       X[0], X[1], X[2], X[3], k, j, i, iblocktemp, ktemp, jtemp, itemp);
    }
    // if (u_prim[1] > 1 || u_prim[2] > 1) {
    //     printf("Fill prims: %g %g %g %g %g from bondi,fill,file: %d %d %d\n", rho_temp, u_temp, u_prim[0], u_prim[1], u_prim[2], filled_bondi, filled_fill, filled_file);
    // }
    P(m_p.RHO, k, j, i) = rho;
    P(m_p.UU, k, j, i) = u;
    P(m_p.U1, k, j, i) = u_prim[0]; 
    P(m_p.U2, k, j, i) = u_prim[1];
    P(m_p.U3, k, j, i) = u_prim[2];

}

KOKKOS_INLINE_FUNCTION void get_B_restart_kharma(const GRCoordinates& G, 
                    const Real fx1min, const Real fx1max, const bool should_fill, const Loci loc,
                    const hsize_t length[GR_DIM], const hsize_t length_fill[GR_DIM],
                    const hsize_t lengthf[GR_DIM], const hsize_t lengthf_fill[GR_DIM],
                    const GridScalar& x1, const GridScalar& x2, const GridScalar& x3,
                    const GridScalar& x1f, const GridScalar& x2f, const GridScalar& x3f, const GridVector& B,
                    const GridScalar& x1_fill, const GridScalar& x2_fill, const GridScalar& x3_fill,
                    const GridScalar& x1f_fill, const GridScalar& x2f_fill, const GridScalar& x3f_fill, const GridVector& B_fill, const GridVector& B_save,
                    const int& v, const int& k, const int& j, const int& i) 
{
    Real B_cons[NVEC];
    
    GReal X[GR_DIM];
    G.coord(k, j, i, loc, X);
    GReal del[GR_DIM]; // not really needed now since I am doing nearest neighbor interpolation
    int iblocktemp, itemp, jtemp, ktemp;
    hsize_t length_loc[GR_DIM], length_fill_loc[GR_DIM];
    
    DLOOP1 length_loc[mu] = length[mu];
    DLOOP1 length_fill_loc[mu] = length_fill[mu];
    if (loc != Loci::center) {
        length_loc[dir_of(loc)] = lengthf[dir_of(loc)];
        length_fill_loc[dir_of(loc)] = lengthf_fill[dir_of(loc)];
    }
 
    // Interpolate the value at this location from the global grid
    if ((!should_fill) && (X[1]<fx1min)) {// if cannot be read from restart file
        // Just use the initialization from SeedBField
        B_cons[v] = 0.;
    }
    else if ((should_fill) && ((X[1]>fx1max)||(X[1]<fx1min))) { // fill with the fname_fill
        Xtoindex(X, (loc == Loci::face1)? x1f_fill : x1_fill, 
                    (loc == Loci::face2)? x2f_fill : x2_fill, 
                    (loc == Loci::face3)? x3f_fill : x3_fill, length_fill_loc, iblocktemp, itemp, jtemp, ktemp, del);
        B_cons[v] = B_fill(v, iblocktemp, ktemp, jtemp, itemp);
    }
    else { 
        Xtoindex(X, (loc == Loci::face1)? x1f : x1, 
                    (loc == Loci::face2)? x2f : x2, 
                    (loc == Loci::face3)? x3f : x3, length_loc, iblocktemp, itemp, jtemp, ktemp, del);
        B_cons[v] = B(v, iblocktemp, ktemp, jtemp, itemp);
    }

    B_save(v, k, j, i) = B_cons[v];
}
