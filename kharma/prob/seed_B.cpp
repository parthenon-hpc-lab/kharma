/*
 *  File: seed_B.cpp
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
#include "seed_B.hpp"

#include "b_ct.hpp"
#include "b_flux_ct.hpp"

#include "boundaries.hpp"
#include "coordinate_utils.hpp"
#include "domain.hpp"
#include "flux.hpp"
#include "fm_torus.hpp"
#include "chakrabarti_torus.hpp"
#include "grmhd_functions.hpp"

using namespace parthenon;

/**
 * Perform a Parthenon MPI reduction.
 * Should only be used in initialization code, as it
 * is very not fast.
 */
template<typename T>
inline T MPIReduce_once(T f, MPI_Op O)
{
    MPIBarrier();
    parthenon::AllReduce<T> reduction;
    reduction.val = f;
    reduction.StartReduce(O);
    // Wait on results
    while (reduction.CheckReduce() == parthenon::TaskStatus::incomplete);
    // TODO catch errors?
    MPIBarrier();
    return reduction.val;
}

// Shorter names for the reductions we use here
Real MaxBsq(MeshData<Real>* md)
{
    return Reductions::DomainReduction<Reductions::Var::bsq, UserHistoryOperation::max,
        Real>(md);
}
Real MaxPressure(MeshData<Real>* md)
{
    return Reductions::DomainReduction<Reductions::Var::gas_pressure,
        UserHistoryOperation::max, Real>(md);
}
Real MinBeta(MeshData<Real>* md)
{
    return Reductions::DomainReduction<Reductions::Var::beta, UserHistoryOperation::min,
        Real>(md);
}

template<BSeedType Seed>
TaskStatus SeedBFieldType(MeshBlockData<Real>* rc, ParameterInput* pin,
    IndexDomain domain = IndexDomain::entire)
{
    auto pmb = rc->GetBlockPointer();
    auto pkgs = pmb->packages.AllPackages();

    // Fields
    GridScalar rho = rc->Get("prims.rho").data;
    const auto& G = pmb->coords;

    // Parameters
    std::string b_field_type = pin->GetString("b_field", "type");
    auto prob = pin->GetString("parthenon/job", "problem_id");
    bool is_torus = (prob == "torus");
    // If a torus, what kind?
    std::string torus_type = "";
    if (is_torus) {
        torus_type = pin->GetString("torus", "type");
    }
    bool is_fm = false, is_chakrabarti = false;
    if (torus_type == "fishbone_moncrief") {
        is_fm = true;
    } else if (torus_type == "chakrabarti") {
        is_chakrabarti = true;
    }

    auto fname_fill = pin->GetOrAddString("resize_restart", "fname_fill", "none");
    const bool should_fill = !(fname_fill == "none");
    Real fx1min, fx1max, dx1, fx1min_ghost;
    int n1tot, fnghost;
    if (prob == "resize_restart_kharma") {
        fx1min = pin->GetReal("parthenon/mesh", "restart_x1min");
        fx1max = pin->GetReal("parthenon/mesh", "restart_x1max");
        fnghost = pin->GetReal("parthenon/mesh", "restart_nghost");
        n1tot = pin->GetInteger("parthenon/mesh", "restart_nx1");
        dx1 = (fx1max - fx1min) / n1tot;
        fx1min_ghost = fx1min - fnghost * dx1;
    }

    // Indices
    // TODO handle filling faces with domain < entire more gracefully
    IndexRange3 b = KDomain::GetRange(rc, domain);
    int ndim = pmb->pmy_mesh->ndim;

    // Shortcut to field values for easy fields
    if constexpr (Seed == BSeedType::constant || Seed == BSeedType::monopole ||
                  Seed == BSeedType::split_monopole_const ||
                  Seed == BSeedType::orszag_tang || Seed == BSeedType::wave ||
                  Seed == BSeedType::shock_tube) {
        // All custom B fields should set what they need of these.
        // We take the same names, but they may mean different things to the
        // particular init function, check seed_B.hpp
        const Real B10 = pin->GetOrAddReal("b_field", "B10", 0.);
        const Real B20 = pin->GetOrAddReal("b_field", "B20", 0.);
        const Real B30 = pin->GetOrAddReal("b_field", "B30", 0.);
        const Real k1 = pin->GetOrAddReal("b_field", "k1", 0.);
        const Real k2 = pin->GetOrAddReal("b_field", "k2", 0.);
        const Real k3 = pin->GetOrAddReal("b_field", "k3", 0.);
        const Real phase = pin->GetOrAddReal("b_field", "phase", 0.);
        const Real amp_B1 = pin->GetOrAddReal("b_field", "amp_B1", 0.);
        const Real amp_B2 = pin->GetOrAddReal("b_field", "amp_B2", 0.);
        const Real amp_B3 = pin->GetOrAddReal("b_field", "amp_B3", 0.);
        const Real amp2_B1 = pin->GetOrAddReal("b_field", "amp2_B1", 0.);
        const Real amp2_B2 = pin->GetOrAddReal("b_field", "amp2_B2", 0.);
        const Real amp2_B3 = pin->GetOrAddReal("b_field", "amp2_B3", 0.);

        if (pkgs.count("B_CT")) {
            auto B_Uf = rc->PackVariables(std::vector<std::string>{"cons.fB"});
            // Fill at 3 different locations
            // Avoid overstepping even as we fill *every face*
            IndexRange3 b1 = KDomain::GetRange(rc, domain, F1);
            pmb->par_for("B_field_B1", b1.ks, b1.ke, b1.js, b1.je, b1.is, b1.ie,
                         KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
                {
                    GReal Xembed[GR_DIM];
                    double null1, null2;
                    // TODO handle calling Seed() mid-run and adding field
                    G.coord_embed(k, j, i, Loci::face1, Xembed);
                    GReal gdet = G.gdet(Loci::face1, j, i);
                    double B_Pf1 = B10;
                    seed_b<Seed>(Xembed, gdet, k1, k2, k3, phase, amp_B1, amp_B2, amp_B3,
                        amp2_B1, amp2_B2, amp2_B3, B_Pf1, null1, null2);
                    B_Uf(F1, 0, k, j, i) = B_Pf1 * gdet;
                });
            IndexRange3 b2 = KDomain::GetRange(rc, domain, F2);
            pmb->par_for("B_field_B2", b2.ks, b2.ke, b2.js, b2.je, b2.is, b2.ie,
                         KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
                {
                    GReal Xembed[GR_DIM];
                    double null1, null2;
                    G.coord_embed(k, j, i, Loci::face2, Xembed);
                    GReal gdet = G.gdet(Loci::face2, j, i);
                    double B_Pf2 = B20;
                    seed_b<Seed>(Xembed, gdet, k1, k2, k3, phase, amp_B1, amp_B2, amp_B3,
                        amp2_B1, amp2_B2, amp2_B3, null1, B_Pf2, null2);
                    B_Uf(F2, 0, k, j, i) = B_Pf2 * gdet;
                });
            IndexRange3 b3 = KDomain::GetRange(rc, domain, F3);
            pmb->par_for("B_field_B2", b3.ks, b3.ke, b3.js, b3.je, b3.is, b3.ie,
                         KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
                {
                    GReal Xembed[GR_DIM];
                    double null1, null2;
                    G.coord_embed(k, j, i, Loci::face3, Xembed);
                    GReal gdet = G.gdet(Loci::face3, j, i);
                    double B_Pf3 = B30;
                    seed_b<Seed>(Xembed, gdet, k1, k2, k3, phase, amp_B1, amp_B2, amp_B3,
                        amp2_B1, amp2_B2, amp2_B3, null1, null2, B_Pf3);
                    B_Uf(F3, 0, k, j, i) = B_Pf3 * gdet;
                });
            // Update primitive variables
            B_CT::BlockUtoP(rc, domain);
        } else if (pkgs.count("B_FluxCT")) {
            GridVector B_P = rc->Get("prims.B").data;
            pmb->par_for("B_field_B", b.ks, b.ke, b.js, b.je, b.is, b.ie,
                         KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
                {
                    GReal Xembed[GR_DIM];
                    G.coord_embed(k, j, i, Loci::center, Xembed);
                    const GReal gdet = G.gdet(Loci::center, j, i);
                    B_P(V1, k, j, i) = B10;
                    B_P(V2, k, j, i) = B20;
                    B_P(V3, k, j, i) = B30;
                    seed_b<Seed>(Xembed, gdet, k1, k2, k3, phase, amp_B1, amp_B2, amp_B3,
                        amp2_B1, amp2_B2, amp2_B3, B_P(V1, k, j, i), B_P(V2, k, j, i),
                        B_P(V3, k, j, i));
                });
            // We still need to update conserved flux values, but then we're done
            B_FluxCT::BlockPtoU(rc, domain);
        } // TODO B_CD!!
        return TaskStatus::complete;
    } else { // Seed with vector potential A otherwise
        // Require and load what we need if necessary
        Real A0 = pin->GetOrAddReal("b_field", "A0", 0.);
        Real min_A = pin->GetOrAddReal("b_field", "min_A", 0.2);
        // Init-specific loads
        Real a, rin, rmax, gam, kappa, rho_norm, arg1, n, rs, rb;
        Real tilt = 0; // Needs to be initialized
        // Chakrabarti-specific variables
        const Real rho_max = pin->GetOrAddReal("torus", "rho_max", 1.0);
        Real gm1, lnh_in, lnh_peak, pgas_over_rho_peak, rho_peak;
        GReal cc, nn; 
        Real potential_rho_pow, potential_falloff, potential_r_pow;
        switch (Seed) {
            case BSeedType::sane:
            case BSeedType::mad:
            case BSeedType::mad_quadrupole:
            case BSeedType::r3s3:
            case BSeedType::r5s5:
            case BSeedType::gaussian:
                // Torus parameters
                rin = pin->GetReal("torus", "rin");
                rmax = pin->GetReal("torus", "rmax");
                kappa = pin->GetReal("torus", "kappa");
                tilt = pin->GetReal("torus", "tilt") / 180. * M_PI;
                // Other things we need only for torus evaluation
                gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
                rho_norm = pmb->packages.Get("GRMHD")->Param<Real>("rho_norm");
                a = G.coords.get_a();
                break;
            case BSeedType::vertical_chakrabarti:
                // A separate case for the vertical field initialized for the Chakrabarti torus
                // Fluid parameters
                gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
                gm1 = gam - 1.;
                // Field init parameters
                potential_rho_pow = pin->GetOrAddReal("b_field", "potential_rho_pow", 1.0);
                potential_falloff = pin->GetOrAddReal("b_field", "potential_falloff", 0.0);
                potential_r_pow   = pin->GetOrAddReal("b_field", "potential_r_pow", 0.0);
                // Torus parameters
                rin  = pin->GetReal("torus", "rin");
                rmax = pin->GetReal("torus", "rmax");
                tilt = pin->GetReal("torus", "tilt") / 180. * M_PI;
                // Spacetime geometry parameters
                a = G.coords.get_a();
                // Now compute relevant parameters
                cn_calc(a, rin, rmax, &cc, &nn);
                lnh_in             = lnh_calc(a, rin, rin, 1.0, cc, nn);
                lnh_peak           = lnh_calc(a, rin, rmax, 1.0, cc, nn) - lnh_in;
                pgas_over_rho_peak = gm1/gam * (m::exp(lnh_peak) - 1.0);
                rho_peak           = m::pow(pgas_over_rho_peak, 1.0 / gm1) / rho_max;
                break;
            case BSeedType::orszag_tang_a:
                A0 = pin->GetReal("orszag_tang", "tscale");
                arg1 = pin->GetReal("orszag_tang", "phase");
                break;
            case BSeedType::r1s2:
                gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
                n = 1. / (gam - 1.);
                rs = pin->GetOrAddReal("bondi", "rs", m::sqrt(1e5));
                if (m::abs(n - 1.5) < 0.01)
                    rb = rs * rs * 80. / (27. * gam);
                else
                    rb = (4 * (n + 1)) / (2 * (n + 3) - 9) * rs;
                break;
            default:
                break;
        }

        // For all other fields...
        // Find the magnetic vector potential.  In X3 symmetry only A_phi is non-zero,
        // But for tilted conditions we must keep track of all components
        // TODO(CEP) Make the vector potential a proper edge-centered field, sync it
        // before B calc
        IndexRange3 be = KDomain::GetRange(rc, domain, E3);
        IndexSize3 sz = KDomain::GetBlockSize(rc);
        ParArrayND<double> A("A", NVEC, sz.n3 + 1, sz.n2 + 1, sz.n1 + 1);
        pmb->par_for("B_field_A", be.ks, be.ke, be.js, be.je, be.is, be.ie,
            KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
            {
                GReal Xnative[GR_DIM];
                GReal Xembed[GR_DIM], Xmidplane[GR_DIM];
                G.coord(k, j, i, Loci::corner, Xnative);
                G.coord_embed(k, j, i, Loci::corner, Xembed);
                // What are our corresponding "midplane" values for evaluating the
                // function?
                rotate_polar(Xembed, tilt, Xmidplane);
                const GReal r = Xmidplane[1], th = Xmidplane[2];

                // Trigonometric values
                const GReal sth = sin(th);
                const GReal cth = cos(th);

                // In case we need zone sizes
                const GReal dxc[GR_DIM] = {0., G.Dxc<1>(i), G.Dxc<2>(j), G.Dxc<3>(k)};

                // This is written under the assumption re-computed rho is more accurate
                // than a bunch of averaging in a meaningful way.  Just use the average if
                // not.
                Real rho_av;
                bool in_torus = false;
                if (is_torus) {
                    if (is_fm) {
                        // Find rho at corner directly for torii
                        rho_av = fm_torus_rho(a, rin, rmax, gam, kappa, r, th) / rho_norm;
                    } else if (is_chakrabarti) {
                        // Find rho
                        const Real lnh = lnh_calc(a, rin, r, sth, cc, nn);
                        if (lnh >= 0.0) {
                            in_torus = true;
                            Real pg_over_rho = gm1 / gam * (m::exp(lnh) - 1.0);
                            rho_av = m::pow(pg_over_rho, 1. / gm1) / rho_peak;
                        }
                    }
                } else {
                    // Use averages for anything else
                    // Avoid overstepping array bounds (but allow overstepping domain
                    // bounds)
                    const int ii = clip((uint)i, (uint)1, sz.n1 - 1);
                    const int jj = clip((uint)j, (uint)1, sz.n2 - 1);
                    const int kk = clip((uint)k, (uint)1, sz.n3 - 1);
                    if (ndim > 2) {
                        rho_av =
                            (rho(kk, jj, ii) + rho(kk, jj, ii - 1) + rho(kk, jj - 1, ii) +
                                rho(kk, jj - 1, ii - 1) + rho(kk - 1, jj, ii) +
                                rho(kk - 1, jj, ii - 1) + rho(kk - 1, jj - 1, ii) +
                                rho(kk - 1, jj - 1, ii - 1)) /
                            8;
                    } else {
                        rho_av = (rho(kk, jj, ii) + rho(kk, jj, ii - 1) +
                                     rho(kk, jj - 1, ii) + rho(kk, jj - 1, ii - 1)) /
                                 4;
                    }
                }

                Real Aphi = seed_a<Seed>(Xmidplane, dxc, rho_av, rin, min_A, A0, arg1, in_torus, rho_max,\
                                        potential_rho_pow, potential_falloff, potential_r_pow);
                // Real Aphi =
                //     seed_a<Seed>(Xmidplane, dxc, rho_av, rin, min_A, A0, arg1, rb);

                if (tilt != 0.0) {
                    // This is *covariant* A_mu of an untilted disk
                    const double A_untilt_lower[GR_DIM] = {0., 0., 0., Aphi};
                    // Raise to contravariant vector, since rotate_polar_vec will need
                    // that. Note we have to do this in the midplane! The coord_to_native
                    // calculation involves an iterative solve for MKS/FMKS
                    GReal Xnative_midplane[GR_DIM] = {0},
                          gcon_midplane[GR_DIM][GR_DIM] = {0};
                    G.coords.coord_to_native(Xmidplane, Xnative_midplane);
                    G.coords.gcon_native(Xnative_midplane, gcon_midplane);
                    double A_untilt[GR_DIM] = {0};
                    DLOOP2
                        A_untilt[mu] += gcon_midplane[mu][nu] * A_untilt_lower[nu];

                    // Then rotate
                    double A_tilt[GR_DIM] = {0};
                    double A_untilt_embed[GR_DIM] = {0}, A_tilt_embed[GR_DIM] = {0};
                    G.coords.con_vec_to_embed(Xnative_midplane, A_untilt, A_untilt_embed);
                    rotate_polar_vec(
                        Xmidplane, A_untilt_embed, -tilt, Xembed, A_tilt_embed);
                    G.coords.con_vec_to_native(Xnative, A_tilt_embed, A_tilt);

                    // Lower the result as we need curl(A_mu).  Done at local zone.
                    double A_tilt_lower[GR_DIM] = {0}, gcov[GR_DIM][GR_DIM] = {0};
                    G.coords.gcov_native(Xnative, gcov);
                    DLOOP2
                        A_tilt_lower[mu] += gcov[mu][nu] * A_tilt[nu];
                    VLOOP
                        A(v, k, j, i) = A_tilt_lower[1 + v];
                } else {
                    // Some problems rely on a very accurate A->B, which the rotation
                    // lacks. So, we preserve exact values in the no-tilt case.
                    A(V3, k, j, i) = Aphi;
                }
            });

        if (pkgs.count("B_CT")) {
            auto B_Uf = rc->PackVariables(std::vector<std::string>{"cons.fB"});
            // This is why we make A 1 zone larger than "entire":
            // we need this stencil-2 op over the whole domain
            IndexRange3 bB = KDomain::GetRange(rc, domain, 0, 1);
            if (ndim > 2) {
                B_CT::EdgeCurl<F1, 3>(rc, A, B_Uf, domain);
                B_CT::EdgeCurl<F2, 3>(rc, A, B_Uf, domain);
                B_CT::EdgeCurl<F3, 3>(rc, A, B_Uf, domain);
            } else if (ndim > 1) {
                B_CT::EdgeCurl<F1, 2>(rc, A, B_Uf, domain);
                B_CT::EdgeCurl<F2, 2>(rc, A, B_Uf, domain);
                B_CT::EdgeCurl<F3, 2>(rc, A, B_Uf, domain);
            } else {
                throw std::runtime_error("Must initialize 1D field directly!");
            }
            B_CT::BlockUtoP(rc, domain);
            // std::cout << "Block divB: " << B_CT::BlockMaxDivB(rc) << std::endl;
        } else if (pkgs.count("B_FluxCT")) {
            // Calculate B-field.  Curl can be run all together since
            // all directions areover all cells
            GridVector B_U = rc->Get("cons.B").data;
            IndexRange3 bl = KDomain::GetRange(rc, domain);
            if (ndim > 2) {
                pmb->par_for("B_field_B_3D", bl.ks, bl.ke, bl.js, bl.je, bl.is, bl.ie,
                             KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
                    {
                        B_FluxCT::averaged_curl_3D(G, A, B_U, k, j, i);
                    });
            } else if (ndim > 1) {
                pmb->par_for("B_field_B_2D", bl.ks, bl.ke, bl.js, bl.je, bl.is, bl.ie,
                             KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
                    {
                        B_FluxCT::averaged_curl_2D(G, A, B_U, k, j, i);
                    });
            } else {
                throw std::runtime_error("Must initialize 1D field directly!");
            }

            if (prob == "resize_restart_kharma") {
                GridVector B_Save = rc->Get("B_Save").data;
                // Hyerin (12/19/22) copy over data after initialization
                pmb->par_for("B_field_B_3D", bl.ks, bl.ke, bl.js, bl.je, bl.is, bl.ie,
                    KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
                    {
                        GReal X[GR_DIM];
                        G.coord(k, j, i, Loci::center, X);

                        if ((!should_fill) &&
                            (X[1] <
                                fx1min_ghost)) { // if cannot be read from restart file
                            // do nothing. just use the initialization from SeedBField
                        } else {
                            VLOOP
                                B_U(v, k, j, i) = B_Save(v, k, j, i);
                        }
                    });
            }
            // Finally, make sure we initialize the primitive field too
            B_FluxCT::BlockUtoP(rc, domain);
        }

        return TaskStatus::complete;
    }
}

TaskStatus SeedBField(MeshData<Real>* md, ParameterInput* pin)
{
    Flag("SeedBField");
    std::string b_field_type = pin->GetString("b_field", "type");
    auto pmesh = md->GetMeshPointer();
    const int verbose = pmesh->packages.Get("Globals")->Param<int>("verbose");

    if (MPIRank0() && verbose) {
        std::cout << "Seeding B field with type " << b_field_type << std::endl;
    }

    TaskStatus status = TaskStatus::incomplete;
    for (int i = 0; i < md->NumBlocks(); i++) {
        auto* rc = md->GetBlockData(i).get();

        // I could make this a map or something,
        // but this is the only place I decode it.
        // TODO could also save it to a package...
        // TODO accumulate TaskStatus properly?
        if (b_field_type == "constant") {
            status = SeedBFieldType<BSeedType::constant>(rc, pin);
        } else if (b_field_type == "monopole") {
            status = SeedBFieldType<BSeedType::monopole>(rc, pin);
        } else if (b_field_type ==
                   "monopole_cube") { // Legacy name for the correct monopole init
            status = SeedBFieldType<BSeedType::monopole>(rc, pin);
        } else if (b_field_type == "sane") {
            status = SeedBFieldType<BSeedType::sane>(rc, pin);
        } else if (b_field_type == "mad") {
            status = SeedBFieldType<BSeedType::mad>(rc, pin);
        } else if (b_field_type == "mad_quadrupole") {
            status = SeedBFieldType<BSeedType::mad_quadrupole>(rc, pin);
        } else if (b_field_type == "r3s3") {
            status = SeedBFieldType<BSeedType::r3s3>(rc, pin);
        } else if (b_field_type == "steep" || b_field_type == "r5s5") {
            status = SeedBFieldType<BSeedType::r5s5>(rc, pin);
        } else if (b_field_type == "gaussian") {
            status = SeedBFieldType<BSeedType::gaussian>(rc, pin);
        } else if (b_field_type == "bz_monopole") {
            status = SeedBFieldType<BSeedType::bz_monopole>(rc, pin);
        } else if (b_field_type == "split_monopole") {
            status = SeedBFieldType<BSeedType::split_monopole>(rc, pin);
        } else if (b_field_type == "split_monopole_const") {
            status = SeedBFieldType<BSeedType::split_monopole_const>(rc, pin);
        } else if (b_field_type == "vertical") {
            status = SeedBFieldType<BSeedType::vertical>(rc, pin);
        } else if (b_field_type == "vertical_chakrabarti") {
            status = SeedBFieldType<BSeedType::vertical_chakrabarti>(rc, pin);
        } else if (b_field_type == "r1s2") {
            status = SeedBFieldType<BSeedType::r1s2>(rc, pin);
        } else if (b_field_type == "orszag_tang") {
            status = SeedBFieldType<BSeedType::orszag_tang>(rc, pin);
        } else if (b_field_type == "orszag_tang_a") {
            status = SeedBFieldType<BSeedType::orszag_tang_a>(rc, pin);
        } else if (b_field_type == "wave") {
            status = SeedBFieldType<BSeedType::wave>(rc, pin);
        } else if (b_field_type == "shock_tube") {
            status = SeedBFieldType<BSeedType::shock_tube>(rc, pin);
        } else {
            throw std::invalid_argument(
                "Magnetic field seed type not supported: " + b_field_type);
        }
    }

    EndFlag();
    return status;
}

TaskStatus NormalizeBField(MeshData<Real>* md, ParameterInput* pin)
{
    Flag("NormBField");
    // Check which solver we'll be using
    auto pmesh = md->GetMeshPointer();
    const int verbose = pmesh->packages.Get("Globals")->Param<int>("verbose");

    // Default to the general literature beta_min of 100.
    // As noted above, by default this uses the definition max(P)/max(P_B)!
    Real desired_beta_min = pin->GetOrAddReal("b_field", "beta_min", 100.);

    // "Legacy" is the much more common normalization:
    // It's the ratio of max values over the domain i.e. max(P) / max(P_B),
    // not necessarily a local min(beta)
    Real beta_calc_legacy = pin->GetOrAddBoolean("b_field", "legacy_norm", true);

    // Calculate current beta_min value
    Real bsq_max, p_max, beta_min;
    if (beta_calc_legacy) {
        bsq_max = MPIReduce_once(MaxBsq(md), MPI_MAX);
        p_max = MPIReduce_once(MaxPressure(md), MPI_MAX);
        beta_min = p_max / (0.5 * bsq_max);
    } else {
        beta_min = MPIReduce_once(MinBeta(md), MPI_MIN);
    }
    Real norm = m::sqrt(beta_min / desired_beta_min);

    if (MPIRank0() && verbose > 0) {
        if (beta_calc_legacy) {
            std::cout << "B^2 max pre-norm: " << bsq_max << std::endl;
            std::cout << "Pressure max pre-norm: " << p_max << std::endl;
        }
        std::cout << "Beta min pre-norm: " << beta_min << std::endl;
        std::cout << "Normalizing by: " << norm << std::endl;
    }

    // Then normalize B by sqrt(beta/beta_min)
    if (beta_min > 0) {
        if (pmesh->packages.AllPackages().count("B_CT")) {
            KHARMADriver::ScaleFace(std::vector<std::string>{"cons.fB"}, md, norm);
            B_CT::MeshUtoP(md, IndexDomain::entire);
        } else {
            KHARMADriver::Scale(std::vector<std::string>{"prims.B", "cons.B"}, md, norm);
        }
    } // else yell?

    // Measure again to check
    if (verbose > 0) {
        Real bsq_max, p_max, beta_min;
        if (beta_calc_legacy) {
            bsq_max = MPIReduce_once(MaxBsq(md), MPI_MAX);
            p_max = MPIReduce_once(MaxPressure(md), MPI_MAX);
            beta_min = p_max / (0.5 * bsq_max);
        } else {
            beta_min = MPIReduce_once(MinBeta(md), MPI_MIN);
        }
        if (MPIRank0()) {
            if (beta_calc_legacy) {
                std::cout << "B^2 max post-norm: " << bsq_max << std::endl;
                std::cout << "Pressure max post-norm: " << p_max << std::endl;
            }
            std::cout << "Beta min post-norm: " << beta_min << std::endl;
        }
    }

    EndFlag(); // NormBField
    return TaskStatus::complete;
}
