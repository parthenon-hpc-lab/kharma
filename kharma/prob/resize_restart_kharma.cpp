/* 
 *  File: resize_restart_kharma.cpp
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

#include "resize_restart_kharma.hpp"

#include "boundaries.hpp"
#include "hdf5_utils.h"
#include "types.hpp"

#include <outputs/restart_hdf5.hpp>

#include <sys/stat.h>
#include <ctype.h>

// Reads in KHARMA restart file but at a different simulation size

void ReadFillFile(int i, ParameterInput *pin) {
    char str[20];
    auto fname_fill = pin->GetOrAddString("resize_restart", "fname_fill", "none");

    if (!(fname_fill == "none")) {
        auto restartReader = std::make_unique<RestartReaderHDF5>(fname_fill.c_str());

        // Load input stream
        std::unique_ptr<ParameterInput> fpinput;
        fpinput = std::make_unique<ParameterInput>();
        auto inputString = restartReader->GetInputString();
        std::istringstream is(inputString);
        fpinput->LoadFromStream(is);

        Real fnx1 = fpinput->GetInteger("parthenon/mesh", "nx1");
        Real fmbnx1 = fpinput->GetInteger("parthenon/meshblock", "nx1");
        
        restartReader = nullptr;

        sprintf(str, "restart%d_nx1", i);
        pin->SetInteger("parthenon/mesh", str, fnx1);
        pin->SetInteger("parthenon/meshblock", str, fmbnx1);
    }
}

void ReadKharmaRestartHeader(std::string fname, ParameterInput *pin)
{
    bool use_dt = pin->GetOrAddBoolean("resize_restart", "use_dt", true);
    bool use_tf = pin->GetOrAddBoolean("resize_restart", "use_tf", false);

    // Read input from restart file 
    // (from external/parthenon/src/parthenon_manager.cpp)
    auto restartReader = std::make_unique<RestartReaderHDF5>(fname.c_str());

    // Load input stream
    std::unique_ptr<ParameterInput> fpinput;
    fpinput = std::make_unique<ParameterInput>();
    auto inputString = restartReader->GetInputString();
    std::istringstream is(inputString);
    fpinput->LoadFromStream(is);

    // TODO(BSP) is there a way to copy all parameters finput->pin and fine-tune later?
    int fnx1, fnx2, fnx3, fmbnx1, fmbnx2, fmbnx3;
    fnx1 = fpinput->GetInteger("parthenon/mesh", "nx1");
    fnx2 = fpinput->GetInteger("parthenon/mesh", "nx2");
    fnx3 = fpinput->GetInteger("parthenon/mesh", "nx3");
    fmbnx1 = fpinput->GetInteger("parthenon/meshblock", "nx1");
    fmbnx2 = fpinput->GetInteger("parthenon/meshblock", "nx2");
    fmbnx3 = fpinput->GetInteger("parthenon/meshblock", "nx3");
    Real fx1min = fpinput->GetReal("parthenon/mesh", "x1min");
    Real fx1max = fpinput->GetReal("parthenon/mesh", "x1max");
    bool fghostzones = fpinput->GetBoolean("parthenon/output1", "ghost_zones");
    int fnghost = fpinput->GetInteger("parthenon/mesh", "nghost");
    auto fBfield = fpinput->GetOrAddString("b_field", "type", "none");
    if (pin->GetOrAddBoolean("resize_restart", "use_restart_size", false)) {
        // This locks the mesh size to be zone-for-zone the same as the iharm3d dump file
        pin->SetInteger("parthenon/mesh", "nx1", fnx1);
        pin->SetInteger("parthenon/mesh", "nx2", fnx2);
        pin->SetInteger("parthenon/mesh", "nx3", fnx3);
        pin->SetInteger("parthenon/meshblock", "nx1", fmbnx1);
        pin->SetInteger("parthenon/meshblock", "nx2", fmbnx2);
        pin->SetInteger("parthenon/meshblock", "nx3", fmbnx3);
    }
    // Record the old values in any case
    pin->SetInteger("parthenon/mesh", "restart_nx1", fnx1);
    pin->SetInteger("parthenon/mesh", "restart_nx2", fnx2);
    pin->SetInteger("parthenon/mesh", "restart_nx3", fnx3);
    pin->SetInteger("parthenon/meshblock", "restart_nx1", fmbnx1);
    pin->SetInteger("parthenon/meshblock", "restart_nx2", fmbnx2);
    pin->SetInteger("parthenon/meshblock", "restart_nx3", fmbnx3);
    pin->SetReal("parthenon/mesh", "restart_x1min", fx1min);
    pin->SetReal("parthenon/mesh", "restart_x1max", fx1max);
    pin->SetInteger("parthenon/mesh", "restart_nghost", fnghost);
    pin->SetBoolean("parthenon/mesh", "restart_ghostzones", fghostzones);
    pin->SetString("b_field", "type", fBfield); // (12/07/22) Hyerin need to test

    Real gam, tNow, dt, tf;
    gam = fpinput->GetReal("GRMHD", "gamma");
    tNow = restartReader->GetAttr<Real>("Info", "Time");
    dt = restartReader->GetAttr<Real>("Info", "dt");
    tf = fpinput->GetReal("parthenon/time", "tlim");
    int ncycle = restartReader->GetAttr<int>("Info", "NCycle");

    pin->SetReal("GRMHD", "gamma", gam);
    pin->SetReal("parthenon/time", "start_time", tNow);
    if (use_dt) {
        pin->SetReal("parthenon/time", "dt", dt);
    }
    if (use_tf) {
        pin->SetReal("parthenon/time", "tlim", tf);
    }
    pin->SetInteger("parthenon/time", "ncycle", ncycle);
    // TODO NSTEP, next tdump/tlog, etc?

    GReal a = fpinput->GetReal("coordinates", "a");
    pin->SetReal("coordinates", "a", a);
    if (fpinput->DoesParameterExist("coordinates", "hslope")) {
        GReal hslope = fpinput->GetReal("coordinates", "hslope");
        pin->SetReal("coordinates", "hslope", hslope);
    }
    ReadFillFile(1, pin);

    // File closed here when restartReader falls out of scope
}

TaskStatus ReadKharmaRestart(std::shared_ptr<MeshBlockData<Real>> rc, ParameterInput *pin)
{
    auto pmb = rc->GetBlockPointer();

    const hsize_t n1tot = pin->GetInteger("parthenon/mesh", "restart_nx1");
    const hsize_t n2tot = pin->GetInteger("parthenon/mesh", "restart_nx2");
    const hsize_t n3tot = pin->GetInteger("parthenon/mesh", "restart_nx3");
    const hsize_t n1mb = pin->GetInteger("parthenon/meshblock", "restart_nx1");
    const hsize_t n2mb = pin->GetInteger("parthenon/meshblock", "restart_nx2");
    const hsize_t n3mb = pin->GetInteger("parthenon/meshblock", "restart_nx3");
    auto fname = pin->GetString("resize_restart", "fname"); // Require this, don't guess
    auto fname_fill = pin->GetOrAddString("resize_restart", "fname_fill", "none");
    const hsize_t f_n1tot = pin->GetOrAddInteger("parthenon/mesh", "restart1_nx1", -1);
    const hsize_t f_n1mb = pin->GetOrAddInteger("parthenon/meshblock", "restart1_nx1", -1);
    const bool is_spherical = pin->GetBoolean("coordinates", "spherical");
    const Real fx1min = pin->GetReal("parthenon/mesh", "restart_x1min");
    const Real fx1max = pin->GetReal("parthenon/mesh", "restart_x1max");
    const Real mdot = pin->GetOrAddReal("bondi", "mdot", 1.0);
    const Real rs = pin->GetOrAddReal("bondi", "rs", 8.0);
    const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
    const bool fghostzones = pin->GetBoolean("parthenon/mesh", "restart_ghostzones");
    auto b_field_type = pin->GetOrAddString("b_field", "type", "none");
    int verbose = pin->GetOrAddInteger("debug", "verbose", 0);
    const Real ur_frac = pin->GetOrAddReal("bondi", "ur_frac", 1.); 
    const Real uphi = pin->GetOrAddReal("bondi", "uphi", 0.); 

    // Derived parameters
    hsize_t nBlocks = (int) (n1tot*n2tot*n3tot)/(n1mb*n2mb*n3mb);
    hsize_t f_nBlocks = (int) (f_n1tot*n2tot*n3tot)/(f_n1mb*n2mb*n3mb);
    const bool should_fill = !(fname_fill == "none");
    const Real dx1 = (fx1max - fx1min) / n1tot;
    int fnghost = pin->GetReal("parthenon/mesh", "restart_nghost");
    const Real fx1min_ghost = fx1min - fnghost*dx1;
    const Real fx1max_ghost = fx1max + fnghost*dx1;
    const bool include_B = (b_field_type != "none");
    // A placeholder to save the B fields for SeedBField
    GridVector B_Save;
    if (include_B) B_Save = rc->Get("B_Save").data;

    auto& G = pmb->coords;

    // read from a restart file and save it to static GridScalar

    if (!fghostzones) fnghost=0; // reset to 0
    int x3factor=1;
    if (n3tot <= 1) x3factor=0; // if less than 3D, do not add ghosts in x3
    hsize_t length[GR_DIM] = {nBlocks,
                                n1mb+2*fnghost,
                                n2mb+2*fnghost,
                                n3mb+2*fnghost*x3factor}; 
    hsize_t f_length[GR_DIM] = {f_nBlocks,
                                f_n1mb+2*fnghost,
                                n2mb+2*fnghost,
                                n3mb+2*fnghost*x3factor}; 
    const int block_sz = length[0]*length[1]*length[2]*length[3];
    const int f_block_sz = f_length[0]*f_length[1]*f_length[2]*f_length[3];

    if (MPIRank0() && verbose > 0) {
        std::cout << "Reading mesh size " << n1tot << "x" << n2tot << "x" << n3tot <<
                        " block size " << n1mb << "x" << n2mb << "x" << n3mb << std::endl;
        std::cout << "Reading " << length[0] << " meshblocks of total size " <<
                     length[1] << "x" <<  length[2]<< "x" << length[3] << std::endl;
    }
    
    
    // read from file and stored in device Hyerin (10/18/2022)
    GridScalar x1_f_device("x1_f_device", length[0], length[1]); 
    GridScalar x2_f_device("x2_f_device", length[0], length[2]); 
    GridScalar x3_f_device("x3_f_device", length[0], length[3]); 
    GridScalar rho_f_device("rho_f_device", length[0], length[3], length[2], length[1]); 
    GridScalar u_f_device("u_f_device", length[0], length[3], length[2], length[1]); 
    GridVector uvec_f_device("uvec_f_device", NVEC, length[0], length[3], length[2], length[1]); 
    GridVector B_f_device("B_f_device", NVEC, length[0], length[3], length[2], length[1]);
    auto x1_f_host = x1_f_device.GetHostMirror();
    auto x2_f_host = x2_f_device.GetHostMirror();
    auto x3_f_host = x3_f_device.GetHostMirror();
    auto rho_f_host = rho_f_device.GetHostMirror();
    auto u_f_host = u_f_device.GetHostMirror();
    auto uvec_f_host = uvec_f_device.GetHostMirror();
    auto B_f_host = B_f_device.GetHostMirror();
    // Hyerin (09/19/2022) : new attempt to read the file 
    hdf5_open(fname.c_str());
    hdf5_set_directory("/");
    Real *rho_file = new double[block_sz];
    Real *u_file = new double[block_sz];
    Real *uvec_file = new double[NVEC*block_sz];
    Real *B_file = new double[NVEC*block_sz];
    Real *x1_file = new double[length[0]*length[1]];
    Real *x2_file = new double[length[0]*length[2]];
    Real *x3_file = new double[length[0]*length[3]];
    //static hsize_t fdims[] = {length[0], 1, length[3], length[2], length[1],1}; //outdated
    static hsize_t fdims[] = {length[0], length[3], length[2], length[1]};
    //static hsize_t fdims_vec[] = {length[0], length[3], length[2], length[1],3}; //outdated
    static hsize_t fdims_vec[] = {length[0], NVEC, length[3], length[2], length[1]};
    static hsize_t fdims_x1[] = {length[0], length[1]};
    static hsize_t fdims_x2[] = {length[0], length[2]};
    static hsize_t fdims_x3[] = {length[0], length[3]};
    hsize_t fstart[] = {0, 0, 0, 0};
    hsize_t fstart_vec[] = {0, 0, 0, 0, 0};
    hsize_t fstart_x[] = {0, 0};
    hdf5_read_array(rho_file, "prims.rho", 4, fdims, fstart, fdims, fdims, fstart, H5T_IEEE_F64LE);
    hdf5_read_array(u_file, "prims.u", 4, fdims, fstart, fdims, fdims, fstart, H5T_IEEE_F64LE);
    hdf5_read_array(uvec_file, "prims.uvec", 5, fdims_vec, fstart_vec, fdims_vec, fdims_vec, fstart_vec, H5T_IEEE_F64LE);
    //if (include_B) hdf5_read_array(B_file, "prims.B", 5, fdims_vec, fstart_vec, fdims_vec, fdims_vec, fstart_vec, H5T_IEEE_F64LE);
    if (include_B) hdf5_read_array(B_file, "cons.B", 5, fdims_vec, fstart_vec, fdims_vec, fdims_vec, fstart_vec, H5T_IEEE_F64LE);
    hdf5_read_array(x1_file, "VolumeLocations/x", 2, fdims_x1, fstart_x, fdims_x1, fdims_x1, fstart_x, H5T_IEEE_F64LE);
    hdf5_read_array(x2_file, "VolumeLocations/y", 2, fdims_x2, fstart_x, fdims_x2, fdims_x2, fstart_x, H5T_IEEE_F64LE);
    hdf5_read_array(x3_file, "VolumeLocations/z", 2, fdims_x3, fstart_x, fdims_x3, fdims_x3, fstart_x, H5T_IEEE_F64LE);
    hdf5_close();
    
    GridScalar x1_fill_device("x1_fill_device", f_length[0], f_length[1]); 
    GridScalar x2_fill_device("x2_fill_device", f_length[0], f_length[2]); 
    GridScalar x3_fill_device("x3_fill_device", f_length[0], f_length[3]); 
    GridScalar rho_fill_device("rho_fill_device", f_length[0], f_length[3], f_length[2], f_length[1]); 
    GridScalar u_fill_device("u_fill_device", f_length[0], f_length[3], f_length[2], f_length[1]); 
    GridVector uvec_fill_device("uvec_fill_device", NVEC, f_length[0], f_length[3], f_length[2], f_length[1]); 
    GridVector B_fill_device("B_fill_device", NVEC, f_length[0], f_length[3], f_length[2], f_length[1]); 
    auto x1_fill_host = x1_fill_device.GetHostMirror();
    auto x2_fill_host = x2_fill_device.GetHostMirror();
    auto x3_fill_host = x3_fill_device.GetHostMirror();
    auto rho_fill_host = rho_fill_device.GetHostMirror();
    auto u_fill_host = u_fill_device.GetHostMirror();
    auto uvec_fill_host = uvec_fill_device.GetHostMirror();
    auto B_fill_host = B_fill_device.GetHostMirror();
    Real *rho_filefill = new double[f_block_sz];
    Real *u_filefill = new double[f_block_sz];
    Real *uvec_filefill = new double[f_block_sz*NVEC];
    Real *B_filefill = new double[f_block_sz*NVEC];
    Real *x1_filefill = new double[f_length[0]*f_length[1]];
    Real *x2_filefill = new double[f_length[0]*f_length[2]];
    Real *x3_filefill = new double[f_length[0]*f_length[3]];
    static hsize_t fill_fdims[] = {f_length[0], f_length[3], f_length[2], f_length[1]};
    static hsize_t fill_fdims_vec[] = {f_length[0], NVEC, f_length[3], f_length[2], f_length[1]};
    static hsize_t fill_fdims_x1[] = {f_length[0], f_length[1]};
    static hsize_t fill_fdims_x2[] = {f_length[0], f_length[2]};
    static hsize_t fill_fdims_x3[] = {f_length[0], f_length[3]};
    if (should_fill) { 
        hdf5_open(fname_fill.c_str());
        hdf5_set_directory("/");
        hdf5_read_array(rho_filefill, "prims.rho", 4, fill_fdims, fstart, fill_fdims, fill_fdims, fstart, H5T_IEEE_F64LE);
        hdf5_read_array(u_filefill, "prims.u", 4, fill_fdims, fstart, fill_fdims, fill_fdims, fstart, H5T_IEEE_F64LE);
        hdf5_read_array(uvec_filefill, "prims.uvec", 5, fill_fdims_vec, fstart_vec, fill_fdims_vec, fill_fdims_vec, fstart_vec, H5T_IEEE_F64LE);
        if (include_B) hdf5_read_array(B_filefill, "cons.B", 5, fill_fdims_vec, fstart_vec, fill_fdims_vec, fill_fdims_vec, fstart_vec,H5T_IEEE_F64LE);
        hdf5_read_array(x1_filefill, "VolumeLocations/x", 2, fill_fdims_x1, fstart_x, fill_fdims_x1, fill_fdims_x1, fstart_x, H5T_IEEE_F64LE);
        hdf5_read_array(x2_filefill, "VolumeLocations/y", 2, fill_fdims_x2, fstart_x, fill_fdims_x2, fill_fdims_x2, fstart_x, H5T_IEEE_F64LE);
        hdf5_read_array(x3_filefill, "VolumeLocations/z", 2, fill_fdims_x3, fstart_x, fill_fdims_x3, fill_fdims_x3, fstart_x, H5T_IEEE_F64LE);
        hdf5_close();
    }

    // save the grid coordinate values to host array
    for (int iblocktemp = 0; iblocktemp < length[0]; iblocktemp++) {
        for (int itemp = 0; itemp < length[1]; itemp++) {
            x1_f_host(iblocktemp,itemp) = x1_file[length[1]*iblocktemp+itemp];
        }
        for (int jtemp = 0; jtemp < length[2]; jtemp++) {
            x2_f_host(iblocktemp,jtemp) = x2_file[length[2]*iblocktemp+jtemp];
        }
        for (int ktemp = 0; ktemp < length[3]; ktemp++) {
            x3_f_host(iblocktemp,ktemp) = x3_file[length[3]*iblocktemp+ktemp];
        }
    }
    // re-arrange uvec such that it can be read in the VLOOP
    int vector_file_index, scalar_file_index;
    for (int iblocktemp = 0; iblocktemp < length[0]; iblocktemp++) {
        for (int itemp = 0; itemp < length[1]; itemp++) {
            for (int jtemp = 0; jtemp < length[2]; jtemp++) {
                for (int ktemp = 0; ktemp < length[3]; ktemp++) {
                    scalar_file_index = length[1]*(length[2]*(length[3]*iblocktemp+ktemp)+jtemp)+itemp;

                    rho_f_host(iblocktemp,ktemp,jtemp,itemp) = rho_file[scalar_file_index];
                    u_f_host(iblocktemp,ktemp,jtemp,itemp) = u_file[scalar_file_index];
                    for (int ltemp = 0; ltemp < 3; ltemp++) {
                        //vector_file_index = 3*(scalar_file_index)+ltemp; // outdated parthenon phdf5 saving order
                        vector_file_index = length[1]*(length[2]*(length[3]*(NVEC*iblocktemp+ltemp)+ktemp)+jtemp)+itemp;
                        
                        uvec_f_host(ltemp,iblocktemp,ktemp,jtemp,itemp) = uvec_file[vector_file_index];
                        if (include_B) B_f_host(ltemp,iblocktemp,ktemp,jtemp,itemp) = B_file[vector_file_index];
                    }
                }
            }
        }
    }

    if (should_fill) {
        // save the grid coordinate values to host array
        for (int iblocktemp = 0; iblocktemp < f_length[0]; iblocktemp++) {
            for (int itemp = 0; itemp < f_length[1]; itemp++) {
                x1_fill_host(iblocktemp,itemp) = x1_filefill[f_length[1]*iblocktemp+itemp];
            }
            for (int jtemp = 0; jtemp < f_length[2]; jtemp++) {
                x2_fill_host(iblocktemp,jtemp) = x2_filefill[f_length[2]*iblocktemp+jtemp];
            }
            for (int ktemp = 0; ktemp < f_length[3]; ktemp++) {
                x3_fill_host(iblocktemp,ktemp) = x3_filefill[f_length[3]*iblocktemp+ktemp];
            }
        }
        // re-arrange uvec such that it can be read in the VLOOP
        for (int iblocktemp = 0; iblocktemp < f_length[0]; iblocktemp++) {
            for (int itemp = 0; itemp < f_length[1]; itemp++) {
                for (int jtemp = 0; jtemp < f_length[2]; jtemp++) {
                    for (int ktemp = 0; ktemp < f_length[3]; ktemp++) {
                        scalar_file_index = f_length[1]*(f_length[2]*(f_length[3]*iblocktemp+ktemp)+jtemp)+itemp;

                        rho_fill_host(iblocktemp,ktemp,jtemp,itemp) = rho_filefill[scalar_file_index];
                        u_fill_host(iblocktemp,ktemp,jtemp,itemp) = u_filefill[scalar_file_index];
                        for (int ltemp = 0; ltemp < 3; ltemp++) {
                            vector_file_index = f_length[1]*(f_length[2]*(f_length[3]*(NVEC*iblocktemp+ltemp)+ktemp)+jtemp)+itemp;
                            
                            uvec_fill_host(ltemp,iblocktemp,ktemp,jtemp,itemp) = uvec_filefill[vector_file_index];
                            if (include_B) B_fill_host(ltemp,iblocktemp,ktemp,jtemp,itemp) = B_filefill[vector_file_index];
                        }
                    }
                }
            }
        }
    }

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    // Deep copy to device
    x1_f_device.DeepCopy(x1_f_host);
    x2_f_device.DeepCopy(x2_f_host);
    x3_f_device.DeepCopy(x3_f_host);
    rho_f_device.DeepCopy(rho_f_host);
    u_f_device.DeepCopy(u_f_host);
    uvec_f_device.DeepCopy(uvec_f_host);
    if (include_B) B_f_device.DeepCopy(B_f_host);
    if (fname_fill != "none") {
        x1_fill_device.DeepCopy(x1_fill_host);
        x2_fill_device.DeepCopy(x2_fill_host);
        x3_fill_device.DeepCopy(x3_fill_host);
        rho_fill_device.DeepCopy(rho_fill_host);
        u_fill_device.DeepCopy(u_fill_host);
        uvec_fill_device.DeepCopy(uvec_fill_host);
        if (include_B) B_fill_device.DeepCopy(B_fill_host);
    }
    Kokkos::fence();

    PackIndexMap prims_map, cons_map;
    auto P = GRMHD::PackMHDPrims(rc.get(), prims_map);
    auto U = GRMHD::PackMHDCons(rc.get(), cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    // Device-side interpolate & copy into the mirror array
    if (MPIRank0() && verbose > 0) {
        std::cout << "Initializing KHARMA restart.  Filling " << fx1min_ghost << " to " << fx1max_ghost << " from " << fname
                    << " and the rest from " << fname_fill << std::endl;
        std::cout << "Vacuum gam: " << gam << " mdot: " << mdot << " rs: " << rs << std::endl;
    }

    // Read to the entire meshblock -- we'll set the Dirichlet boundaries based on the
    // ghost zone data we read here.
    auto domain = IndexDomain::entire;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    pmb->par_for("copy_restart_state_kharma", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            get_prim_restart_kharma(G, P, m_p,
                fx1min_ghost, fx1max_ghost, should_fill, is_spherical, gam, rs, mdot, ur_frac, uphi, length, f_length,
                x1_f_device, x2_f_device, x3_f_device, rho_f_device, u_f_device, uvec_f_device,
                x1_fill_device, x2_fill_device, x3_fill_device, rho_fill_device, u_fill_device, uvec_fill_device,
                k, j, i);
            if (include_B) {
                get_B_restart_kharma(G, fx1min_ghost, fx1max_ghost, should_fill, length, f_length,
                    x1_f_device, x2_f_device, x3_f_device, B_f_device,
                    x1_fill_device, x2_fill_device, x3_fill_device, B_fill_device, B_Save,
                    k, j, i);
            }
        }
    );

    return TaskStatus::complete;
}
