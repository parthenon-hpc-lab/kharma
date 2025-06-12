/* 
 *  File: resize_restart.cpp
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

#include "resize_restart.hpp"

#include "b_flux_ct.hpp"
#include "hdf5_utils.h"
#include "kharma_utils.hpp"
#include "interpolation.hpp"
#include "types.hpp"

#include <sys/stat.h>
#include <ctype.h>

// TODO: The iharm3d restart format fails to record several things we must guess:
// 1. Sometimes, even precise domain boundaries in native coordinates
// 2. Which coordinate system was used
// 3. Any coordinate system parameters
// Better to either:
// a. read KHARMA restart files so we can re-grid
// b. use the IL dump format, but in double precision (or even in single w/cleanup)
// Either would be very useful independently

// This exists to simplify some initializer lists below
// This indicates I know that moving from signed->unsigned is dangerous,
// and sign off that these results are positive (they are)
hsize_t static_max(int i, int n) { return static_cast<hsize_t>(m::max(i, n)); }
hsize_t static_min(int i, int n) { return static_cast<hsize_t>(m::min(i, n)); }

void ReadIharmRestartHeader(std::string fname, ParameterInput *pin)
{
    // Read the restart file and set parameters that need to be specified at early loading
    hdf5_open(fname.c_str());

    // Read everything from root
    hdf5_set_directory("/");
    // Print version
    hid_t string_type = hdf5_make_str_type(20);
    char version[20];
    hdf5_read_single_val(version, "version", string_type);
    if (MPIRank0()) {
        std::cout << "Initialized from " << fname << ", file version " << version << std::endl << std::endl;
    }

    // Read what we need from the file, regardless of where we're putting it
    int n1file, n2file, n3file;
    hdf5_read_single_val(&n1file, "n1", H5T_STD_I32LE);
    hdf5_read_single_val(&n2file, "n2", H5T_STD_I32LE);
    hdf5_read_single_val(&n3file, "n3", H5T_STD_I32LE);

    double x1min, x2min, x3min;
    double x1max, x2max, x3max;
    bool use_native_bounds = false;
    // Spherical for guessing
    double Rin, Rout;
    if (hdf5_exists("x1Min")) {
        // If available, read domain boundaries exactly.  This is mostly
        // for re-gridded KHARMA dumps.
        hdf5_read_single_val(&x1min, "x1Min", H5T_IEEE_F64LE);
        hdf5_read_single_val(&x1max, "x1Max", H5T_IEEE_F64LE);
        hdf5_read_single_val(&x2min, "x2Min", H5T_IEEE_F64LE);
        hdf5_read_single_val(&x2max, "x2Max", H5T_IEEE_F64LE);
        hdf5_read_single_val(&x3min, "x3Min", H5T_IEEE_F64LE);
        hdf5_read_single_val(&x3max, "x3Max", H5T_IEEE_F64LE);
        use_native_bounds = true;
    } else if (hdf5_exists("a")) {
        // Only read these if the better versions aren't available
        hdf5_read_single_val(&Rin, "Rin", H5T_IEEE_F64LE);
        hdf5_read_single_val(&Rout, "Rout", H5T_IEEE_F64LE);
    } else {
        throw std::runtime_error("Unknown restart file type!");
    }

    // Anything always necessary for spherical coordinates
    double a, hslope;
    bool file_in_spherical = false;
    if (hdf5_exists("a")) {
        hdf5_read_single_val(&a, "a", H5T_IEEE_F64LE);
        hdf5_read_single_val(&hslope, "hslope", H5T_IEEE_F64LE);
        file_in_spherical = true;
    }

    // Anything else
    double gam, t, dt, tf;
    hdf5_read_single_val(&gam, "gam", H5T_IEEE_F64LE);
    hdf5_read_single_val(&t, "t", H5T_IEEE_F64LE);
    hdf5_read_single_val(&dt, "dt", H5T_IEEE_F64LE);
    hdf5_read_single_val(&tf, "tf", H5T_IEEE_F64LE);

    // Set the number of primitive vars
    // TODO do this better by recording/counting flags in MODEL
    if(hdf5_exists("game")) {
        pin->SetInteger("resize_restart", "nfprim", 10);
    } else {
        pin->SetInteger("resize_restart", "nfprim", 8);
    }

    // End HDF5 reads
    hdf5_close();

    // Record the parameters of the file grid
    // Note the iharm3d-style naming as mnemonic
    pin->SetInteger("resize_restart", "n1tot", n1file);
    pin->SetInteger("resize_restart", "n2tot", n2file);
    pin->SetInteger("resize_restart", "n3tot", n3file);
    if (use_native_bounds) {
        // If available, set the domain boundaries exactly
        pin->SetReal("resize_restart", "startx1", x1min);
        pin->SetReal("resize_restart", "stopx1", x1max);
        pin->SetReal("resize_restart", "startx2", x2min);
        pin->SetReal("resize_restart", "stopx2", x2max);
        pin->SetReal("resize_restart", "startx3", x3min);
        pin->SetReal("resize_restart", "stopx3", x3max);
    } else {
        // Otherwise, guess them
        pin->SetReal("resize_restart", "startx1", m::log(Rin));
        pin->SetReal("resize_restart", "stopx1", m::log(Rout));
        pin->SetReal("resize_restart", "startx2", 0.0);
        pin->SetReal("resize_restart", "stopx2", 1.0);
        pin->SetReal("resize_restart", "startx3", 0.0);
        pin->SetReal("resize_restart", "stopx3", 2*M_PI);
    }

    // If specified, set *our* grid to exactly match the *file's* grid
    if (pin->GetOrAddBoolean("resize_restart", "regrid_only", false)) {
        // This locks the Parthenon mesh size to be zone-for-zone the same as the iharm3d dump file...
        pin->SetInteger("parthenon/mesh", "nx1", n1file);
        pin->SetInteger("parthenon/mesh", "nx2", n2file);
        pin->SetInteger("parthenon/mesh", "nx3", n3file);
        if (pin->GetOrAddBoolean("resize_restart", "one_meshblock", false)) {
            // We can re-split the mesh now by using nearest-neighbor
            // The option is provided to restart without necessarily knowing
            // the mesh size beforehand to build a compatible meshblock
            pin->SetInteger("parthenon/meshblock", "nx1", n1file);
            pin->SetInteger("parthenon/meshblock", "nx2", n2file);
            pin->SetInteger("parthenon/meshblock", "nx3", n3file);
        }
        // ...which of course also means setting our geometry to exactly match.
        if (use_native_bounds) {
            // If available, set the boundaries exactly as read from native coords
            pin->SetReal("parthenon/mesh", "x1min", x1min);
            pin->SetReal("parthenon/mesh", "x1max", x1max);
            pin->SetReal("parthenon/mesh", "x2min", x2min);
            pin->SetReal("parthenon/mesh", "x2max", x2max);
            pin->SetReal("parthenon/mesh", "x3min", x3min);
            pin->SetReal("parthenon/mesh", "x3max", x3max);

            // Going from domain->coords values is better to match everything
            if (file_in_spherical) {
                pin->SetReal("coordinates", "r_in", m::exp(x1min));
                pin->SetReal("coordinates", "r_out", m::exp(x1max));
            }
        } else {
            std::cout << "Guessing geometry when restarting! This is potentially very bad to do!" << std::endl;
            // NOTE: the reason guessing is bad here has to do with old KHARMA versions:
            // input parameters (even geometry) were only stored to 6-digit precision in old KHARMA.
            // This means that r_in and x3max especially were cut off, and only correct to 6 digits.
            // Restarting with more accurate parameters can mess with the B field divergence,
            // so we warn about it.
            pin->SetReal("coordinates", "r_in", Rin);
            pin->SetReal("coordinates", "r_out", Rout);
            // xNmin/max will then be set by kharma.cpp.
        }
    }

    // Set the coordinate system to match the restart file *even if we're resizing*
    // Mapping to new spins/systems is theoretically fine, but we do not want to
    // do this in any applications yet
    if (file_in_spherical) {
        pin->SetReal("coordinates", "a", a);
        pin->SetReal("coordinates", "hslope", hslope);

        // Sadly restarts did not record MKS vs FMKS
        // Guess FMKS if not specified in parameter file
        pin->SetString("coordinates", "base", "spherical_ks");
        if (!pin->DoesParameterExist("coordinates", "transform")) {
            pin->SetString("coordinates", "transform", "funky");
        }
    } else {
        std::cout << "Guessing the restart file is in Cartesian coordinates!" << std::endl;
        // Anything without a BH spin was pretty likely Cartesian
        pin->SetString("coordinates", "base", "cartesian_minkowski");
        pin->SetString("coordinates", "transform", "null");
    }
    // Boundary conditions will be set based on coordinate system in kharma.cpp

    // We should always use the restart's fluid gamma & current time
    pin->SetReal("GRMHD", "gamma", gam);
    pin->SetReal("parthenon/time", "start_time", t);
    // Set the original simulation's end time, if we wanted that
    // Maybe for MHDModes restart test, or keeping to lower res end time
    if (pin->GetOrAddBoolean("resize_restart", "use_tf", false)) {
        pin->SetReal("parthenon/time", "tlim", tf);
    }
    if (pin->GetOrAddBoolean("resize_restart", "use_dt", true)) {
        // Setting dt here is actually for KHARMA,
        // which returns this from EstimateTimestep in step 0
        pin->SetReal("parthenon/time", "dt", dt);
    }
}

TaskStatus ReadIharmRestart(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin)
{
    auto pmb = rc->GetBlockPointer();

    const auto fname = pin->GetString("resize_restart", "fname"); // Require this, don't guess
    const bool regrid_only = pin->GetOrAddBoolean("resize_restart", "regrid_only", false);
    const bool is_spherical = pin->GetBoolean("coordinates", "spherical");

    // Size/domain of the file we're reading *from*.
    const hsize_t nfprim = pin->GetInteger("resize_restart", "nfprim");
    const hsize_t n1tot = pin->GetInteger("resize_restart", "n1tot");
    const hsize_t n2tot = pin->GetInteger("resize_restart", "n2tot");
    const hsize_t n3tot = pin->GetInteger("resize_restart", "n3tot");
    const GReal startx[GR_DIM] = {0,
        pin->GetReal("resize_restart", "startx1"),
        pin->GetReal("resize_restart", "startx2"),
        pin->GetReal("resize_restart", "startx3")};
    const GReal stopx[GR_DIM] = {0,
        pin->GetReal("resize_restart", "stopx1"),
        pin->GetReal("resize_restart", "stopx2"),
        pin->GetReal("resize_restart", "stopx3")};
    const GReal dx[GR_DIM] = {0., (stopx[1] - startx[1])/n1tot,
                                  (stopx[2] - startx[2])/n2tot,
                                  (stopx[3] - startx[3])/n3tot};

    // Sanity checks.  Unlikely to fire but nice to have
    if (regrid_only) {
        // Check the mesh we're declaring matches
        if (pin->GetInteger("parthenon/mesh", "nx1") != n1tot ||
            pin->GetInteger("parthenon/mesh", "nx2") != n2tot ||
            pin->GetInteger("parthenon/mesh", "nx3") != n3tot) {
            printf("Mesh size does not match!\n");
            printf("[%d %d %d] vs [%llu %llu %llu]",
                pin->GetInteger("parthenon/mesh", "nx1"),
                pin->GetInteger("parthenon/mesh", "nx2"),
                pin->GetInteger("parthenon/mesh", "nx3"),
                n1tot, n2tot, n3tot);
        }

        if (!close_to(pin->GetReal("parthenon/mesh", "x1min"), startx[1]) ||
            !close_to(pin->GetReal("parthenon/mesh", "x1max"), stopx[1]) ||
            !close_to(pin->GetReal("parthenon/mesh", "x2min"), startx[2]) ||
            !close_to(pin->GetReal("parthenon/mesh", "x2max"), stopx[2]) ||
            !close_to(pin->GetReal("parthenon/mesh", "x3min"), startx[3]) ||
            !close_to(pin->GetReal("parthenon/mesh", "x3max"), stopx[3])) {
            printf("Mesh shape does not match!\n");
            printf("X1 %g vs %g & %g vs %g\nX2 %g vs %g & %g vs %g\nX3 %g vs %g & %g vs %g",
                pin->GetReal("parthenon/mesh", "x1min"), startx[1],
                pin->GetReal("parthenon/mesh", "x1max"), stopx[1],
                pin->GetReal("parthenon/mesh", "x2min"), startx[2],
                pin->GetReal("parthenon/mesh", "x2max"), stopx[2],
                pin->GetReal("parthenon/mesh", "x3min"), startx[3],
                pin->GetReal("parthenon/mesh", "x3max"), stopx[3]);
        }
    }

    if(MPIRank0()) std::cout << "Reading mesh from file to cache..." << std::endl;

    // In this section we're dealing with two different meshes: the one we're interpolating *from* (the "file" grid)
    // and the one we're interpolating *to* -- the "meshblock."
    // Additionally, in the "file" mesh we must deail with global file locations (no ghost zones, global index, prefixed "g")
    // as well as local file locations (locations in a cache we read to host memory, prefixed "m")

    // Size/domain of the MeshBlock we're reading *to*.
    // Note that we fill all zones, now that we might be interpolating to faces without syncing primitive B
    IndexDomain domain = IndexDomain::entire;
    const IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    const IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    const IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
    const auto& G = pmb->coords;

    // Total file size
    // TODO separate nmprim to stop at 8 prims if we don't need e-
    hsize_t fdims[] = {nfprim, n3tot, n2tot, n1tot};

    // Figure out the subset in global space corresponding to our memory cache
    int gis, gjs, gks, gie, gje, gke;
    if (regrid_only) {
        // For nearest neighbor "interpolation," we don't need any ghost zones
        // Global location of first zone of our new grid
        double X[GR_DIM];
        G.coord(kb.s, jb.s, ib.s, Loci::center, X);
        // Global file coordinate corresponding to that location
        Interpolation::Xtoijk_nearest(X, startx, dx, gis, gjs, gks);
        // Same for the end
        G.coord(kb.e, jb.e, ib.e, Loci::center, X);
        Interpolation::Xtoijk_nearest(X, startx, dx, gie, gje, gke);
    } else {
        // Linear interpolation case: we need ghost zones
        // Global leftmost corner of our grid block
        double tmp[GR_DIM], X[GR_DIM];
        G.coord(kb.s, jb.s, ib.s, Loci::corner, X);
        // Global file coordinate corresponding to that location
        // Note this will be the *left* side already, so we'll never read below this.
        // The values gis,gjs,gks can/will be <0 sometimes
        Interpolation::Xtoijk(X, startx, dx, gis, gjs, gks, tmp);
        // Global rightmost corner, same deal
        G.coord(kb.e+1, jb.e+1, ib.e+1, Loci::corner, X);
        Interpolation::Xtoijk(X, startx, dx, gie, gje, gke, tmp);
        // Include one extra zone in each direction, for right side of linear interp
        gke += 1; gje += 1; gie += 1;
    }

    // Truncate the file read sizes so we don't overrun the file data
    hsize_t fstart[4] = {0, static_max(gks, 0), static_max(gjs, 0), static_max(gis, 0)};
    // Test gXe against last valid index, i.e. nXtot-1
    hsize_t fstop[4] = {nfprim-1, static_min(gke, n3tot-1), static_min(gje, n2tot-1), static_min(gie, n1tot-1)};
    // We add one here to get sizes from indices
    hsize_t fcount[4] = {fstop[0] - fstart[0] + 1,
                         fstop[1] - fstart[1] + 1,
                         fstop[2] - fstart[2] + 1,
                         fstop[3] - fstart[3] + 1};
    // If we overran an index on the left, we need to leave blank rows, potentially filled w/periodic vals
    hsize_t mstart[4] = {0, (gks < 0) ? -gks : 0, (gjs < 0) ? -gjs : 0, (gis < 0) ? -gis : 0};
    // Total memory size is never truncated
    // This calculation produces XxYx2 arrays for 2D sims w/linear interp but that's fine
    hsize_t nmk = gke-gks+1, nmj = gje-gjs+1, nmi = gie-gis+1;
    hsize_t mdims[4] = {nfprim, nmk, nmj, nmi};
    // TODO these should be const but hdf5_read_array yells about it, fix that
    // TODO should yell if any of these fired for nearest-neighbor

    // Allocate the array we'll need
    hsize_t nmblock = nmk * nmj * nmi;
    // TODO this may be float[] if we ever want to read dump files as restarts
    double *ptmp = new double[nfprim*nmblock];

    // Open the file
    hdf5_open(fname.c_str());
    hdf5_set_directory("/");

    // Read the main array
    hdf5_read_array(ptmp, "p", 4, fdims, fstart, fcount, mdims, mstart, H5T_IEEE_F64LE);

    // Do some special reads from elsewhere in the file to fill periodic bounds
    // Note we do NOT fill outflow/reflecting bounds here -- instead, we treat them specially below
    // TODO this could probably be a lot cleaner
    hsize_t fstart_tmp[4], fcount_tmp[4], mstart_tmp[4];
#define RESET_COUNTS DLOOP1 {fstart_tmp[mu] = fstart[mu]; fcount_tmp[mu] = fcount[mu]; mstart_tmp[mu] = mstart[mu];}
    if (gks < 0 && pmb->boundary_flag[BoundaryFace::inner_x3] == BoundaryFlag::periodic) {
        RESET_COUNTS
        // same X1/X2, but take only the globally LAST ranks in X3
        int nrank = 0 - gks;
        fstart_tmp[1] = n3tot - nrank;
        fcount_tmp[1] = nrank;
        // Read it to the FIRST ranks of our array
        mstart_tmp[1] = 0;
        hdf5_read_array(ptmp, "p", 4, fdims, fstart_tmp, fcount_tmp, mdims, mstart_tmp, H5T_IEEE_F64LE);
    }
    if (gke > n3tot-1 && pmb->boundary_flag[BoundaryFace::outer_x3] == BoundaryFlag::periodic) {
        RESET_COUNTS
        // same X1/X2, but take only the globally FIRST ranks in X3
        int nrank = gke - (n3tot-1);
        fstart_tmp[1] = 0;
        fcount_tmp[1] = nrank;
        // Read it to the LAST ranks of our array
        mstart_tmp[1] = mdims[1] - nrank;
        hdf5_read_array(ptmp, "p", 4, fdims, fstart_tmp, fcount_tmp, mdims, mstart_tmp, H5T_IEEE_F64LE);
    }
    if (gjs < 0 && pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::periodic) {
        RESET_COUNTS
        int nrank = 0 - gjs;
        fstart_tmp[2] = n2tot - nrank;
        fcount_tmp[2] = nrank;
        mstart_tmp[2] = 0;
        hdf5_read_array(ptmp, "p", 4, fdims, fstart_tmp, fcount_tmp, mdims, mstart_tmp, H5T_IEEE_F64LE);
    }
    if (gje > n2tot-1 && pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::periodic) {
        RESET_COUNTS
        int nrank = gje - (n2tot-1);
        fstart_tmp[2] = 0;
        fcount_tmp[2] = nrank;
        mstart_tmp[2] = mdims[2] - nrank;
        hdf5_read_array(ptmp, "p", 4, fdims, fstart_tmp, fcount_tmp, mdims, mstart_tmp, H5T_IEEE_F64LE);
    }
    if (gis < 0 && pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::periodic) {
        RESET_COUNTS
        int nrank = 0 - gis;
        fstart_tmp[3] = n1tot - nrank;
        fcount_tmp[3] = nrank;
        mstart_tmp[3] = 0;
        hdf5_read_array(ptmp, "p", 4, fdims, fstart_tmp, fcount_tmp, mdims, mstart_tmp, H5T_IEEE_F64LE);
    }
    if (gie > n1tot-1 && pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::periodic) {
        RESET_COUNTS
        int nrank = gie - (n1tot-1);
        fstart_tmp[3] = 0;
        fcount_tmp[3] = nrank;
        mstart_tmp[3] = mdims[3] - nrank;
        hdf5_read_array(ptmp, "p", 4, fdims, fstart_tmp, fcount_tmp, mdims, mstart_tmp, H5T_IEEE_F64LE);
    }

    hdf5_close();

    if (MPIRank0()) std::cout << "Read!" << std::endl;

    // Get the arrays we'll be writing to
    // TODO this is probably easier AND more flexible if we pack them
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    GridVector B_P = rc->Get("prims.B").data;
    auto rho_host = rho.GetHostMirror();
    auto u_host = u.GetHostMirror();
    auto uvec_host = uvec.GetHostMirror();
    auto B_host = B_P.GetHostMirror();

    // Interpolate on the host side & copy into the mirror Views
    // Nearest-neighbor interpolation is currently only used when grids exactly correspond -- otherwise, linear interpolation is used
    // to minimize the resulting B field divergence.
    if (regrid_only) {
        // TODO Kokkos calls here had problems with CUDA, reintroduce/fix
        // OpenMP here conflicts with Kokkos parallel in some cases, so we're stuck
        for (int k=kb.s; k <= kb.e; ++k) for (int j=jb.s; j <= jb.e; ++j) for (int i=ib.s; i <= ib.e; ++i) {
            GReal X[GR_DIM]; int gk, gj, gi;
            G.coord(k, j, i, Loci::center, X);
            Interpolation::Xtoijk_nearest(X, startx, dx, gi, gj, gk);
            // TODO verify this never reads zones outside the cache
            // Calculate indices inside our cached block
            int mk = gk - gks, mj = gj - gjs, mi = gi - gis;
            // Fill cells of the new block with equivalents in the cached block
            rho_host(k, j, i) = ptmp[0*nmblock + mk*nmj*nmi + mj*nmi + mi];
            u_host(k, j, i)   = ptmp[1*nmblock + mk*nmj*nmi + mj*nmi + mi];
            VLOOP uvec_host(v, k, j, i) = ptmp[(2+v)*nmblock + mk*nmj*nmi + mj*nmi + mi];
            VLOOP B_host(v, k, j, i) = ptmp[(5+v)*nmblock + mk*nmj*nmi + mj*nmi + mi];
        }
    } else {
        // TODO real boundary flags. Repeat on any outflow/reflecting bounds
        const bool repeat_x1i = is_spherical;
        const bool repeat_x1o = is_spherical;
        const bool repeat_x2i = is_spherical;
        const bool repeat_x2o = is_spherical;

        for (int k=kb.s; k <= kb.e; ++k) for (int j=jb.s; j <= jb.e; ++j) for (int i=ib.s; i <= ib.e; ++i) {
            GReal X[GR_DIM], del[GR_DIM]; int gk, gj, gi;
            // Get the zone center location
            G.coord(k, j, i, Loci::center, X);
            // Get global indices
            Interpolation::Xtoijk(X, startx, dx, gi, gj, gk, del);
            // Make any corrections due to global boundaries
            // Currently just repeats the last zone: everywhere beyond last zone center has center's value
            if (repeat_x1i && gi < 0) { gi = 0; del[1] = 0; }
            if (repeat_x1o && gi >= n1tot-1) { gi = n1tot - 1; del[1] = 0; }
            if (repeat_x2i && gj < 0) { gj = 0; del[2] = 0; }
            if (repeat_x2o && gj >= n2tot-1) { gj = n2tot - 1; del[2] = 0; }
            // Calculate indices inside our cached block
            int mk = gk - gks, mj = gj - gjs, mi = gi - gis;
            // Interpolate the value at this location from the cached grid
            rho_host(k, j, i) = Interpolation::linear(mi, mj, mk, nmi, nmj, nmk, del, &(ptmp[0*nmblock]));
            u_host(k, j, i) = Interpolation::linear(mi, mj, mk, nmi, nmj, nmk, del, &(ptmp[1*nmblock]));
            VLOOP uvec_host(v, k, j, i) = Interpolation::linear(mi, mj, mk, nmi, nmj, nmk, del, &(ptmp[(2+v)*nmblock]));
            VLOOP B_host(v, k, j, i) = Interpolation::linear(mi, mj, mk, nmi, nmj, nmk, del, &(ptmp[(5+v)*nmblock]));
        }
    }

    // Deep copy to device
    rho.DeepCopy(rho_host);
    u.DeepCopy(u_host);
    uvec.DeepCopy(uvec_host);
    B_P.DeepCopy(B_host);
    Kokkos::fence();

    // Delete our cache.  Only we ever used it, so we're safe here.
    delete[] ptmp;

    return TaskStatus::complete;
}
