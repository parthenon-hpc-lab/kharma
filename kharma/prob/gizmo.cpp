/*
 *  File: bondi.cpp
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

#include "gizmo.hpp"

#include "floors.hpp"
#include "flux_functions.hpp"

/**
 * Initialization of domain from output of cosmological simulation code GIZMO
 * Note this requires
 */
TaskStatus InitializeGIZMO(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin)
{
    auto pmb = rc->GetBlockPointer();

    const Real mdot = pin->GetOrAddReal("bondi", "mdot", 1.0);
    const Real rs = pin->GetOrAddReal("bondi", "rs", 8.0);

    // Set the innermost radius to apply the initialization
    const Real a = pin->GetReal("coordinates", "a");
    const Real rin_default = 1 + m::sqrt(1 - a * a) + 0.1;
    const Real rin_init = pin->GetOrAddReal("gizmo", "r_in", rin_default);

    auto datfn = pin->GetOrAddString("gizmo", "datfn", "none");

    if (!(pmb->packages.Get("GRMHD")->AllParams().hasKey("gizmo_dat")))
        pmb->packages.Get("GRMHD")->AddParam<std::string>("gizmo_dat", datfn);
    if (!(pmb->packages.Get("GRMHD")->AllParams().hasKey("rin_init")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("rin_init", rin_init);

    // Set the interior domain to the analytic solution to begin
    // This tests that PostInitialize will correctly fill ghost zones with the boundary we
    // set
    SetGIZMO(rc, IndexDomain::interior);

    return TaskStatus::complete;
}

TaskStatus SetGIZMO(
    std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();

    // std::cerr << "GIZMO on domain: " << BoundaryName(domain) << std::endl;
    //  Don't apply GIZMO initialization to X1 boundaries
    if (domain == IndexDomain::outer_x1 || domain == IndexDomain::inner_x1) {
        return TaskStatus::complete;
    }

    PackIndexMap prims_map, cons_map;
    auto P = GRMHD::PackMHDPrims(rc.get(), prims_map);
    auto U = GRMHD::PackMHDCons(rc.get(), cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    const Real mdot = pmb->packages.Get("GRMHD")->Param<Real>("mdot");
    const Real rs = pmb->packages.Get("GRMHD")->Param<Real>("rs");
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb->packages);

    auto datfn = pmb->packages.Get("GRMHD")->Param<std::string>("gizmo_dat");
    auto rin_init = pmb->packages.Get("GRMHD")->Param<Real>("rin_init");

    // Just the X1 right boundary
    GRCoordinates G = pmb->coords;
    CoordinateEmbedding cs = G.coords;

    // Set the Bondi conditions wherever we're asked
    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

    const IndexRange ib = bounds.GetBoundsI(domain);
    const IndexRange jb = bounds.GetBoundsJ(domain);
    const IndexRange kb = bounds.GetBoundsK(domain);

    // GIZMO shell
    // Read the gizmo data file
    FILE* fptr = fopen(datfn.c_str(), "r");
    const int datlen = 100000;
    Real* rarr = new double[datlen];
    Real* rhoarr = new double[datlen];
    Real* Tarr = new double[datlen];
    Real* vrarr = new double[datlen];
    Real* Mencarr = new double[datlen];
    int length = 0, itemp = 0;
    while (fscanf(fptr, "%lf %lf %lf %lf %lf\n", &(rarr[itemp]), &(rhoarr[itemp]),
               &(Tarr[itemp]), &(vrarr[itemp]), &(Mencarr[itemp])) ==
           5) { // assign the read value to variable, and enter it in array
        itemp++;
    }
    fclose(fptr);
    length = itemp;

    GridVector r_device("r_device", length);
    GridVector rho_device("rho_device", length);
    GridVector T_device("T_device", length);
    GridVector vr_device("vr_device", length);
    auto r_host = r_device.GetHostMirror();
    auto rho_host = rho_device.GetHostMirror();
    auto T_host = T_device.GetHostMirror();
    auto vr_host = vr_device.GetHostMirror();
    for (itemp = 0; itemp < length; itemp++) {
        r_host(itemp) = rarr[itemp];
        rho_host(itemp) = rhoarr[itemp];
        T_host(itemp) = Tarr[itemp];
        vr_host(itemp) = vrarr[itemp];
    }
    r_device.DeepCopy(r_host);
    rho_device.DeepCopy(rho_host);
    T_device.DeepCopy(T_host);
    vr_device.DeepCopy(vr_host);

    Kokkos::fence();

    pmb->par_for("gizmo_shell", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
        {
            // same vacuum conditions at rin_init
            GReal Xshell[GR_DIM] = {0, rin_init, 0, 0};
            int i_sh;
            GReal del_sh;
            XtoindexGIZMO(Xshell, r_device, length, i_sh, del_sh);
            Real vacuum_rho, vacuum_u_over_rho;
            vacuum_rho = rho_device(i_sh) * (1. - del_sh) + rho_device(i_sh + 1) * del_sh;
            vacuum_u_over_rho =
                (T_device(i_sh) * (1. - del_sh) + T_device(i_sh + 1) * del_sh) /
                (gam - 1.);

            get_prim_gizmo_shell(G, cs, P, m_p, gam, rin_init, rs, vacuum_rho,
                vacuum_u_over_rho, r_device, rho_device, T_device, vr_device, length, k,
                j, i);
        });

    return TaskStatus::complete;
}
