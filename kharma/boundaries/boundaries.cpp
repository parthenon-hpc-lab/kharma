/*
 *  File: boundaries.cpp
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
#include "boundaries.hpp"

#include "bondi.hpp"
#include "decs.hpp"
#include "domain.hpp"
#include "kharma.hpp"
#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"
#include "reductions.hpp"
#include "types.hpp"

#include "b_ct.hpp"
#include "b_flux_ct.hpp"
#include "flux.hpp"

// Parthenon's boundaries
#include <bvals/boundary_conditions.hpp>

// Very bad definition. Still necessary, though
// TODO get rid of them eventually
#define PLOOP for(int ip=0; ip < nvar; ++ip)

std::shared_ptr<KHARMAPackage> KBoundaries::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t> &packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Boundaries");
    Params &params = pkg->AllParams();

    // OPTIONS FOR SPECIFIC BOUNDARIES
    bool spherical = pin->GetBoolean("coordinates", "spherical");
    // Global check inflow sets inner/outer X1 by default
    bool check_inflow_global = pin->GetOrAddBoolean("boundaries", "check_inflow", spherical);

    // Option to excise a bit at the poles when calculating fluxes
    bool excise_polar_flux = pin->GetOrAddBoolean("boundaries", "excise_polar_flux", false);
    params.Add("excise_polar_flux", excise_polar_flux);
    if (excise_polar_flux) { // These options are *completely* incompatible
        pin->SetBoolean("boundaries", "zero_polar_flux", false);
    }
    // Otherwise, those fluxes should be zero
    bool zero_polar_flux = pin->GetOrAddBoolean("boundaries", "zero_polar_flux", spherical);
    params.Add("zero_polar_flux", zero_polar_flux);

    // Apply physical boundaries to conserved GRMHD variables rho u^r, T^mu_nu
    // Probably inadvisable?
    bool domain_bounds_on_conserved = pin->GetOrAddBoolean("boundaries", "domain_bounds_on_conserved", false);
    params.Add("domain_bounds_on_conserved", domain_bounds_on_conserved);

    // Fix the X1/X2 corner by replacing the reflecting condition with the inflow
    // Never use this if not in spherical coordinates
    // Activates by default only with reflecting X2/outflow X1 and interior boundary inside EH
    // TODO(BSP) may also be specific to Funky MKS coords with zero_point==startx1
    bool fix_corner = false;
    if (spherical) {
        bool correct_bounds =
            (pin->GetString("boundaries", "inner_x2") == "reflecting" &&
             pin->GetString("boundaries", "outer_x2") == "reflecting" &&
             pin->GetString("boundaries", "inner_x1") == "outflow");
        bool inside_eh = pin->GetBoolean("coordinates", "domain_intersects_eh");
        fix_corner = pin->GetOrAddBoolean("boundaries", "fix_corner", correct_bounds && inside_eh);
        // Allow overriding with specific name
        fix_corner = pin->GetOrAddBoolean("boundaries", "fix_corner_inner", fix_corner);
    }
    params.Add("fix_corner_inner", fix_corner);
    params.Add("fix_corner_outer", pin->GetOrAddBoolean("boundaries", "fix_corner_outer", false));

    // We can't use GetVariablesByFlag yet, so ask the packages
    // These flags get anything that needs a physical boundary during the run
    using FC = Metadata::FlagCollection;
    FC ghost_vars = FC({Metadata::FillGhost, Metadata::Conserved})
                + FC({Metadata::FillGhost, Metadata::GetUserFlag("Primitive")})
                - FC({Metadata::GetUserFlag("StartupOnly")});
    int nvar = KHARMA::PackDimension(packages.get(), ghost_vars);
    // Face-centered fields: some duplicate stuff, leaving it separate for now
    FC ghost_vars_f = FC({Metadata::FillGhost, Metadata::Face})
                - FC({Metadata::GetUserFlag("StartupOnly")});
    int nvar_f = 3 * m::max(KHARMA::PackDimension(packages.get(), ghost_vars_f), 1);

    // TODO encapsulate this
    Metadata m_x1, m_x2, m_x3, m_x1_f, m_x2_f, m_x3_f;
    {
        // We also don't know the mesh size, since it's not constructed.  We infer.
        const int ng = pin->GetInteger("parthenon/mesh", "nghost");
        const int nx1 = pin->GetInteger("parthenon/meshblock", "nx1");
        const int n1 = nx1 + 2 * ng;
        const int nx2 = pin->GetInteger("parthenon/meshblock", "nx2");
        const int n2 = (nx2 == 1) ? nx2 : nx2 + 2 * ng;
        const int nx3 = pin->GetInteger("parthenon/meshblock", "nx3");
        const int n3 = (nx3 == 1) ? nx3 : nx3 + 2 * ng;

        // These are declared *backward* from how they will be indexed
        std::vector<int> s_x1({ng, n2, n3, nvar});
        std::vector<int> s_x2({n1, ng, n3, nvar});
        std::vector<int> s_x3({n1, n2, ng, nvar});
        // Dirichlet conditions must be restored when restarting!
        m_x1 = Metadata({Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Restart}, s_x1);
        m_x2 = Metadata({Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Restart}, s_x2);
        m_x3 = Metadata({Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Restart}, s_x3);

        if (nvar_f > 0) {
            // Face mesh sizes
            const int ng_f = pin->GetInteger("parthenon/mesh", "nghost") + 1;
            const int nx1 = pin->GetInteger("parthenon/meshblock", "nx1");
            const int n1_f = (nx2 == 1) ? nx1 : nx1 + 2 * ng + 1;
            const int nx2 = pin->GetInteger("parthenon/meshblock", "nx2");
            const int n2_f = (nx2 == 1) ? nx2 : nx2 + 2 * ng + 1;
            const int nx3 = pin->GetInteger("parthenon/meshblock", "nx3");
            const int n3_f = (nx3 == 1) ? nx3 : nx3 + 2 * ng + 1;

            // These are declared *backward* from how they will be indexed
            std::vector<int> s_x1_f({ng_f, n2_f, n3_f, nvar_f});
            std::vector<int> s_x2_f({n1_f, ng_f, n3_f, nvar_f});
            std::vector<int> s_x3_f({n1_f, n2_f, ng_f, nvar_f});
            // Note these are *NOT* face variables, they cannot be indexed by TopologicalElement.
            // Instead, we make them a normal vector and use int(TE) % 3
            m_x1_f = Metadata({Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Restart}, s_x1_f);
            m_x2_f = Metadata({Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Restart}, s_x2_f);
            m_x3_f = Metadata({Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Restart}, s_x3_f);
        }
    }

    // Set options for each boundary
    for (int i = 0; i < BOUNDARY_NFACES; i++) {
        const auto bface = (BoundaryFace) i;
        const auto bdomain = BoundaryDomain(bface);
        const auto bname = BoundaryName(bface);
        const auto bdir = BoundaryDirection(bface);
        const auto binner = BoundaryIsInner(bface);
        // Get the boundary type we specified in kharma
        auto btype = pin->GetString("boundaries", bname);
        params.Add(bname, btype);

        // OPTIONS FOR ANY BOUNDARY

        // Prevent inflow at boundaries.
        // This is two separate checks, but default to enabling/disabling together for X1 and not elsewhere
        bool check_inflow = pin->GetOrAddBoolean("boundaries", "check_inflow_" + bname, check_inflow_global && bdir == X1DIR);
        params.Add("check_inflow_" + bname, check_inflow);

        // Ensure fluxes through the zero-size face at the pole are zero
        bool zero_flux = pin->GetOrAddBoolean("boundaries", "zero_flux_" + bname, zero_polar_flux && bdir == X2DIR);
        params.Add("zero_flux_" + bname, zero_flux);
        // OR allow them via faux-excision
        bool excise_flux = pin->GetOrAddBoolean("boundaries", "excise_flux_" + bname, excise_polar_flux && bdir == X2DIR);
        params.Add("excise_flux_" + bname, excise_flux);

        // Allow specifically dP to outflow in otherwise Dirichlet conditions
        // Only used for viscous_bondi problem
        bool outflow_EMHD = pin->GetOrAddBoolean("boundaries", "outflow_EMHD_" + bname, false);
        params.Add("outflow_EMHD_" + bname, outflow_EMHD);

        // Options specific to face-centered B fields, which require a lot of care at boundaries
        if (packages->AllPackages().count("B_CT")) {
            // Invert X2 face values to reflect across polar boundary
            bool invert_F2 = pin->GetOrAddBoolean("boundaries", "reflect_face_vector_" + bname, (btype == "reflecting"));
            params.Add("reflect_face_vector_"+bname, invert_F2);
            // If you'll have field loops exiting the domain, outflow conditions need to be cleaned so as not to
            // introduce divergence to the first physical zone.
            bool clean_face_B = pin->GetOrAddBoolean("boundaries", "clean_face_B_" + bname, (btype == "outflow"));
            params.Add("clean_face_B_"+bname, clean_face_B);
            // Forcibly reconnect field loops that get trapped around the polar boundary
            bool reconnect_B3 = pin->GetOrAddBoolean("boundaries", "reconnect_B3_" + bname, false);
            params.Add("reconnect_B3_"+bname, reconnect_B3);

            // Special EMF averaging.  Allows B3 to "slip" around the pole
            // Also useful to allow coherent motion even with Dirichlet boundaries, for e.g. multizone
            bool average_EMF = pin->GetOrAddBoolean("boundaries", "average_EMF_" + bname, (btype == "transmitting"));
            params.Add("average_EMF_"+bname, average_EMF);
            // Otherwise, always zero EMFs to prevent B field escaping the domain in polar/dirichlet bounds
            // Default for dirichlet conditions unless averaging is set manually
            bool zero_EMF = pin->GetOrAddBoolean("boundaries", "zero_EMF_" + bname, (btype == "reflecting" ||
                                                                                    (btype == "dirichlet" && !average_EMF)));
            params.Add("zero_EMF_"+bname, zero_EMF);
        }
        // Advect together/cancel velocity or angular momentum "loops" around the pole, similar to B3 above
        // Probably not needed anymore, now polar boundary conditions are fixed.  cancel_U3 does not conserve angular momentum.
        bool cancel_U3 = pin->GetOrAddBoolean("boundaries", "cancel_U3_" + bname, false);
        params.Add("cancel_U3_"+bname, cancel_U3);
        bool cancel_T3 = pin->GetOrAddBoolean("boundaries", "cancel_T3_" + bname, false);
        params.Add("cancel_T3_"+bname, cancel_T3);

        // String manip to get the Parthenon boundary name, e.g., "ox1_bc"
        auto bname_parthenon = bname.substr(0, 1) + "x" + bname.substr(7, 8) + "_bc";
        // Parthenon implements periodic conditions
        // For the rest, they should call our default wrapper, which we register in main()
        if (btype == "periodic") {
            pin->SetString("parthenon/mesh", bname_parthenon, "periodic");
        } else {
            pin->SetString("parthenon/mesh", bname_parthenon, "user");

            // Register the actual boundaries with the package, which our wrapper will use
            // when called via Parthenon's "user" conditions
            if (btype == "dirichlet") {
                // Dirichlet boundaries: allocate
                pkg->AddField("Boundaries." + bname, (bdir == X1DIR) ? m_x1 : ((bdir == X2DIR) ? m_x2 : m_x3));
                if (nvar_f > 0) {
                    pkg->AddField("Boundaries.f." + bname, (bdir == X1DIR) ? m_x1_f : ((bdir == X2DIR) ? m_x2_f : m_x3_f));
                }
                switch (bface) {
                case BoundaryFace::inner_x1:
                    pkg->KBoundaries[bface] = KBoundaries::Dirichlet<BoundaryFace::inner_x1>;
                    break;
                case BoundaryFace::outer_x1:
                    pkg->KBoundaries[bface] = KBoundaries::Dirichlet<BoundaryFace::outer_x1>;
                    break;
                case BoundaryFace::inner_x2:
                    pkg->KBoundaries[bface] = KBoundaries::Dirichlet<BoundaryFace::inner_x2>;
                    break;
                case BoundaryFace::outer_x2:
                    pkg->KBoundaries[bface] = KBoundaries::Dirichlet<BoundaryFace::outer_x2>;
                    break;
                case BoundaryFace::inner_x3:
                    pkg->KBoundaries[bface] = KBoundaries::Dirichlet<BoundaryFace::inner_x3>;
                    break;
                case BoundaryFace::outer_x3:
                    pkg->KBoundaries[bface] = KBoundaries::Dirichlet<BoundaryFace::outer_x3>;
                    break;
                default:
                    break;
                }
            } else if (btype == "reflecting") {
                switch (bface) {
                case BoundaryFace::inner_x1:
                    pkg->KBoundaries[bface] = BoundaryFunction::ReflectInnerX1;
                    break;
                case BoundaryFace::outer_x1:
                    pkg->KBoundaries[bface] = BoundaryFunction::ReflectOuterX1;
                    break;
                case BoundaryFace::inner_x2:
                    pkg->KBoundaries[bface] = BoundaryFunction::ReflectInnerX2;
                    break;
                case BoundaryFace::outer_x2:
                    pkg->KBoundaries[bface] = BoundaryFunction::ReflectOuterX2;
                    break;
                case BoundaryFace::inner_x3:
                    pkg->KBoundaries[bface] = BoundaryFunction::ReflectInnerX3;
                    break;
                case BoundaryFace::outer_x3:
                    pkg->KBoundaries[bface] = BoundaryFunction::ReflectOuterX3;
                    break;
                default:
                    break;
                }
            } else if (btype == "transmitting") {
                switch (bface) {
                case BoundaryFace::inner_x1:
                    pkg->KBoundaries[bface] = KBoundaries::OneBlockTransmit<BoundaryFace::inner_x1>;
                    break;
                case BoundaryFace::outer_x1:
                    pkg->KBoundaries[bface] = KBoundaries::OneBlockTransmit<BoundaryFace::outer_x1>;
                    break;
                case BoundaryFace::inner_x2:
                    pkg->KBoundaries[bface] = KBoundaries::OneBlockTransmit<BoundaryFace::inner_x2>;
                    break;
                case BoundaryFace::outer_x2:
                    pkg->KBoundaries[bface] = KBoundaries::OneBlockTransmit<BoundaryFace::outer_x2>;
                    break;
                case BoundaryFace::inner_x3:
                    pkg->KBoundaries[bface] = KBoundaries::OneBlockTransmit<BoundaryFace::inner_x3>;
                    break;
                case BoundaryFace::outer_x3:
                    pkg->KBoundaries[bface] = KBoundaries::OneBlockTransmit<BoundaryFace::outer_x3>;
                    break;
                default:
                    break;
                }
                if (pin->GetString("coordinates", "transform") == "fmks" || pin->GetString("coordinates", "transform") == "funky")
                    throw std::runtime_error("Transmitting polar boundary conditions require coordinates symmetric about theta=0!");
                // TODO also check for wedge simulations x3<2pi
            } else if (btype == "outflow") {
                switch (bface) {
                case BoundaryFace::inner_x1:
                    pkg->KBoundaries[bface] = BoundaryFunction::OutflowInnerX1;
                    break;
                case BoundaryFace::outer_x1:
                    pkg->KBoundaries[bface] = BoundaryFunction::OutflowOuterX1;
                    break;
                case BoundaryFace::inner_x2:
                    pkg->KBoundaries[bface] = BoundaryFunction::OutflowInnerX2;
                    break;
                case BoundaryFace::outer_x2:
                    pkg->KBoundaries[bface] = BoundaryFunction::OutflowOuterX2;
                    break;
                case BoundaryFace::inner_x3:
                    pkg->KBoundaries[bface] = BoundaryFunction::OutflowInnerX3;
                    break;
                case BoundaryFace::outer_x3:
                    pkg->KBoundaries[bface] = BoundaryFunction::OutflowOuterX3;
                    break;
                default:
                    break;
                }
            } else if (btype == "bondi") {
                // Boundaries will need these to be recorded into a 'Params'
                AddBondiParameters(pin, *packages);
                switch (bface) {
                case BoundaryFace::inner_x1:
                    pkg->KBoundaries[bface] = SetBondi<IndexDomain::inner_x1>;
                    break;
                case BoundaryFace::outer_x1:
                    pkg->KBoundaries[bface] = SetBondi<IndexDomain::outer_x1>;
                    break;
                case BoundaryFace::inner_x2:
                    pkg->KBoundaries[bface] = SetBondi<IndexDomain::inner_x2>;
                    break;
                case BoundaryFace::outer_x2:
                    pkg->KBoundaries[bface] = SetBondi<IndexDomain::outer_x2>;
                    break;
                case BoundaryFace::inner_x3:
                    pkg->KBoundaries[bface] = SetBondi<IndexDomain::inner_x3>;
                    break;
                case BoundaryFace::outer_x3:
                    pkg->KBoundaries[bface] = SetBondi<IndexDomain::outer_x3>;
                    break;
                default:
                    break;
                }
            } else {
                throw std::runtime_error("Unknown boundary type: "+btype);
            }
        }
    }

    // If we've split in the phi direction or we're running in 2D
    if (pin->GetInteger("parthenon/mesh", "nx3") != pin->GetInteger("parthenon/meshblock", "nx3") ||
        pin->GetInteger("parthenon/mesh", "nx3") == 1) {
        if (pin->GetString("boundaries", "inner_x3") == "transmitting" || pin->GetString("boundaries", "outer_x3") == "transmitting")
            throw std::runtime_error("Transmitting polar boundary conditions require 3D with one block in x3!");
        if (params.Get<bool>("cancel_U3_inner_x3") || params.Get<bool>("cancel_U3_outer_x3") ||
            params.Get<bool>("cancel_T3_inner_x3") || params.Get<bool>("cancel_T3_outer_x3"))
            throw std::runtime_error("Polar cancellations require 3D with one block in x3!");
        if (packages->AllPackages().count("B_CT") &&
            (params.Get<bool>("reconnect_B3_inner_x3") || params.Get<bool>("reconnect_B3_outer_x3")))
            throw std::runtime_error("Polar reconnections require 3D with one block in x3!");
    }

    // Callbacks
    // Fix flux
    pkg->FixFlux = KBoundaries::FixFlux;
    // Source term (only needed for excise_flux)
    pkg->AddSource = KBoundaries::AddSource;
    return pkg;
}

void KBoundaries::ApplyBoundary(std::shared_ptr<MeshBlockData<Real>> &rc, IndexDomain domain, bool coarse)
{
    Flag("ApplyBoundary"); // this is not a callback, flag for ourselves
    // KHARMA has to do some extra tasks in addition to just applying the usual
    // boundary conditions.  Therefore, we "wrap" Parthenon's (or our own)
    // boundary functions with this one.

    auto pmb = rc->GetBlockPointer();
    auto pkg = pmb->packages.Get<KHARMAPackage>("Boundaries");
    auto& params = pkg->AllParams();

    // TODO canonize this as a function. Prints all variables in the current MBD/MD object,
    // which can now be smaller than everything.
    // std::cout << rc->GetVariableVector().size() << std::endl;
    // for (auto &var : rc->GetVariableVector()) {
    //     std::cout << var->label() << " ";
    // }
    // std::cout << std::endl;

    const auto bface = BoundaryFaceOf(domain);
    const auto bname = BoundaryName(bface);
    const auto btype_name = params.Get<std::string>(bname);
    const auto bdir = BoundaryDirection(bface);
    const bool binner = BoundaryIsInner(bface);

    // We get called over a lot of different packs depending on doing physical boundaries, EMF boundaries,
    // boundaries during solves/GMG, and so on.  Check once whether this is a "normal" boundary condition
    // with the GRMHD variables
    // TODO redesign boundary functions as per-package callbacks which take a pack+map
    // TODO probably retire PackMHDPrims, it's not more useful than just packing on the flag.
    PackIndexMap dummy_map;
    bool full_grmhd_boundary = GRMHD::PackMHDPrims(rc.get(), dummy_map).GetDim(4) > 0;

    // Averaging ops on *physical* cells must be done before computing boundaries
    // We should do a PreBoundaries callback...
    if (pmb->packages.AllPackages().count("B_CT")) {
        auto bfpack = rc->PackVariables({Metadata::Face, Metadata::FillGhost, Metadata::GetUserFlag("B_CT")});
        if (params.Get<bool>("reconnect_B3_" + bname) && bfpack.GetDim(4) > 0) {
            Flag("ReconnectFaceB_"+bname);
            B_CT::ReconnectBoundaryB3(rc.get(), domain, bfpack, coarse);
            EndFlag();
        }
    }
    if (pmb->packages.AllPackages().count("GRMHD")) {
        if (params.Get<bool>("cancel_U3_" + bname) && full_grmhd_boundary) {
            GRMHD::CancelBoundaryU3(rc.get(), domain, coarse);
        }
        if (params.Get<bool>("cancel_T3_" + bname) && full_grmhd_boundary) {
            GRMHD::CancelBoundaryT3(rc.get(), domain, coarse);
        }
    }

    // Always call through to the registered boundary function
    Flag("Apply "+bname+" boundary: "+btype_name);
    pkg->KBoundaries[bface](rc, coarse);
    EndFlag();

    // Then a bunch of common boundary "touchups"
    // Nothing below is designed, nor necessary, for coarse buffers
    if (coarse) {
        EndFlag();
        return;
    }

    // Fixes and special cases for face/edge-centered variabes in B_CT
    if (pmb->packages.AllPackages().count("B_CT")) {
        // Delegate EMF boundaries to the B_CT package
        // Only until per-variable boundaries available in Parthenon
        // Warning: Even though the EMFs are sync'd separately,
        // they still can sneak into "real" boundary exchanges,
        // so we can't assume their presence means they are alone
        auto& emfpack = rc->PackVariables(std::vector<std::string>{"B_CT.emf"});
        if (emfpack.GetDim(4) > 0) {
            if (params.Get<bool>("zero_EMF_" + bname)) {
                Flag("ZeroEMF_"+bname);
                B_CT::ZeroBoundaryEMF(rc.get(), domain, emfpack, coarse);
                EndFlag();
            }
            if (params.Get<bool>("average_EMF_" + bname)) {
                Flag("AverageEMF_"+bname);
                B_CT::AverageBoundaryEMF(rc.get(), domain, emfpack, coarse);
                EndFlag();
            }
        }

        // Correct Parthenon's reflecting conditions on the corresponding face
        // Note these are REFLECTING SPECIFIC, not suitable for the similar op w/transmitting
        // TODO move this to Parthenon.  Move out of this case if we gain non-B_CT face variables
        auto fpack = rc->PackVariables({Metadata::Face, Metadata::FillGhost, Metadata::GetUserFlag("SplitVector")});
        if (params.Get<bool>("reflect_face_vector_" + bname) && fpack.GetDim(4) > 0) {
            Flag("ReflectFace_"+bname);
            const TopologicalElement face = FaceOf(bdir);
            auto b = KDomain::GetBoundaryRange(rc, domain, face, coarse);
            // Zero the last physical face, otherwise invert.
            auto i_f = (binner) ? b.ie : b.is;
            auto j_f = (binner) ? b.je : b.js;
            auto k_f = (binner) ? b.ke : b.ks;
            pmb->par_for(
                "reflect_face_vector_" + bname, 0, fpack.GetDim(4)-1, b.ks, b.ke, b.js, b.je, b.is, b.ie,
                KOKKOS_LAMBDA (const int &v, const int &k, const int &j, const int &i) {
                    const int kk = (bdir == 3) ? k_f - (k - k_f) : k;
                    const int jj = (bdir == 2) ? j_f - (j - j_f) : j;
                    const int ii = (bdir == 1) ? i_f - (i - i_f) : i;
                    fpack(face, v, k, j, i) = ((bdir == 1 && i == i_f) ||
                                            (bdir == 2 && j == j_f) ||
                                            (bdir == 3 && k == k_f)) ? 0. : -fpack(face, v, kk, jj, ii);
                }
            );
            EndFlag();
        }

        // Correct orthogonal B field component to eliminate divergence in last rank
        // and ghosts. Used for outflow conditions when field lines will exit domain
        auto bfpack = rc->PackVariables({Metadata::Face, Metadata::FillGhost, Metadata::GetUserFlag("B_CT")});
        if (params.Get<bool>("clean_face_B_" + bname) && bfpack.GetDim(4) > 0) {
            Flag("CleanFaceB_"+bname);
            B_CT::DestructiveBoundaryClean(rc.get(), domain, bfpack, coarse);
            EndFlag();
        }
    }

    // This function, ApplyBoundary, is called in 2 places we might not expect:
    // 1. Syncing only the EMF during runs with CT
    // 2. Syncing boundaries while solving for B field
    // The above operations are general to these cases
    // But, anything beyond this point isn't needed for those cases & may crash
    if (!full_grmhd_boundary) {
        EndFlag();
        return;
    }

    // Prevent inflow of material by changing fluid speeds,
    // anywhere we've specified.
    if (params.Get<bool>("check_inflow_" + bname)) {
        Flag("CheckInflow_"+bname);
        CheckInflow(rc, domain, coarse);
        EndFlag();
    }

    // Allow specifically dP to outflow in otherwise Dirichlet conditions
    // Only used for viscous_bondi problem, should be moved in there somehow
    if (params.Get<bool>("outflow_EMHD_" + bname)) {
        Flag("OutflowEMHD_"+bname);
        auto EMHDg = rc->PackVariables({Metadata::GetUserFlag("EMHDVar"), Metadata::FillGhost});
        const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
        const auto &range = (bdir == 1) ? bounds.GetBoundsI(IndexDomain::interior)
                                : (bdir == 2 ? bounds.GetBoundsJ(IndexDomain::interior)
                                    : bounds.GetBoundsK(IndexDomain::interior));
        const int ref = binner ? range.s : range.e;
        pmb->par_for_bndry(
            "outflow_EMHD", IndexRange{0,EMHDg.GetDim(4)-1}, domain, CC, coarse,
            KOKKOS_LAMBDA (const int &v, const int &k, const int &j, const int &i) {
                EMHDg(v, k, j, i) = EMHDg(v, (bdir == 3) ? ref : k, (bdir == 2) ? ref : j, (bdir == 1) ? ref : i);
            }
        );
        EndFlag();
    }

    /*
    * KHARMA is very particular about corner boundaries.
    * In particular, we apply the outflow boundary over ALL X2 & X3.
    * Then we apply the polar bound only where outflow is not applied,
    * and periodic bounds only where neither other bound applies.
    * The latter is accomplished regardless of Parthenon's definitions,
    * since these functions are run after Parthenon's MPI boundary syncs &
    * replace whatever they've done.
    * However, the former must be added after the X2 boundary call,
    * replacing the reflecting conditions in the X1/X2 corner (or in 3D, edge)
    * with outflow conditions based on the updated ghost cells.
    */
    if (bdir == X2DIR) {
        // TODO test more carefully whether this is still needed for face-centered B...

        // If we're on the interior edge, re-apply that edge for our block by calling
        // exactly the same function that Parthenon does.  This ensures we're applying
        // the same thing, just emulating calling it after X2.
        if (params.Get<bool>("fix_corner_inner")) {
            if (pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user) {
                Flag("FixCorner");
                ApplyBoundary(rc, IndexDomain::inner_x1, coarse);
                EndFlag();
            }
        }
        if (params.Get<bool>("fix_corner_outer")) {
            if (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::user) {
                Flag("FixCorner");
                ApplyBoundary(rc, IndexDomain::outer_x1, coarse);
                EndFlag();
            }
        }
    }

    bool sync_prims = pmb->packages.Get("Driver")->Param<bool>("sync_prims");
    // There are two modes of operation here:
    if (sync_prims) {
        // 1. Exchange/prolongate/restrict PRIMITIVE variables: (ImEx driver)
        //    Primitive variables (except B field!) are marked FillGhost
        //    Explicitly run UtoP on B field, then PtoU on everything
        // TODO there should be a set of B field wrappers that dispatch this
        auto pkgs = pmb->packages.AllPackages();
        if (pkgs.count("B_CT")) {
            B_CT::BlockUtoP(rc.get(), domain, coarse);
        } else {
            B_FluxCT::BlockUtoP(rc.get(), domain, coarse);
        }
        Flux::BlockPtoU(rc.get(), domain, coarse);
    } else {
        // 2. Exchange/prolongate/restrict CONSERVED variables: (KHARMA driver)
        //    Conserved variables are marked FillGhost, plus FLUID PRIMITIVES.
        if (!params.Get<bool>("domain_bounds_on_conserved")) {
            // To apply primitive boundaries to GRMHD, we run PtoU on that ONLY,
            // and UtoP on EVERYTHING ELSE
            Packages::BoundaryPtoUElseUtoP(rc.get(), domain, coarse);
        } else {
            // If we want to apply boundaries to conserved vars, just run UtoP on EVERYTHING
            Packages::BoundaryUtoP(rc.get(), domain, coarse);
        }
    }

    EndFlag();
}

void KBoundaries::CheckInflow(std::shared_ptr<MeshBlockData<Real>> &rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();
    const auto &G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    PackIndexMap prims_map;
    auto P = GRMHD::PackMHDPrims(rc.get(), prims_map, coarse);
    const VarMap m_p(prims_map, false);

    // Inflow check
    // Iterate over all boundary domain zones w/p=0
    pmb->par_for_bndry(
        "check_inflow", IndexRange{0, 0}, domain, CC, coarse,
        KOKKOS_LAMBDA(const int &p, const int &k, const int &j, const int &i) {
            KBoundaries::check_inflow(G, P, domain, m_p.U1, k, j, i);
        }
    );
}

TaskStatus KBoundaries::FixFlux(MeshData<Real> *md)
{
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    auto& params = pmb0->packages.Get("Boundaries")->AllParams();

    // Fluxes are defined at faces, so there is one more valid flux than
    // valid cell in the face direction.  That is, e.g. F1 is valid on
    // an (N1+1)xN2xN3 grid, F2 on N1x(N2+1)xN3, etc.
    // These functions do *not* need an extra row outside the domain,
    // like B_FluxCT::ZeroBoundaryFlux does.
    const int ndim = pmesh->ndim;
    // One-zone halo for fluxes
    const IndexRange3 bi = KDomain::GetRange(md, IndexDomain::interior);
    const IndexRange3 b1 = KDomain::GetRange(md, IndexDomain::interior, -1, 1);

    for (auto &pmb : pmesh->block_list) {
        auto &rc = pmb->meshblock_data.Get(md->StageName());

        for (int i = 0; i < BOUNDARY_NFACES; i++) {
            BoundaryFace bface = (BoundaryFace)i;
            auto bname = BoundaryName(bface);
            const auto bdir = BoundaryDirection(bface);
            const auto binner = BoundaryIsInner(bface);

            if (bdir > ndim) continue;

            const IndexRange3 bf = KDomain::GetRange(rc, IndexDomain::interior, FaceOf(bdir));

            // Fluxes are needed in 1-zone halo for FluxCT
            IndexRange3 b = b1;
            // Range for inner_x1 bounds is first face only, etc.
            if (bdir == 1) {
                b.is = b.ie = (binner) ? bf.is : bf.ie;
            } else if (bdir == 2) {
                b.js = b.je = (binner) ? bf.js : bf.je;
            } else {
                b.ks = b.ke = (binner) ? bf.ks : bf.ke;
            }

            PackIndexMap cons_map;
            auto &F = rc->PackVariablesAndFluxes({Metadata::WithFluxes}, cons_map);

            // If we should check inflow on this face...
            if (params.Get<bool>("check_inflow_" + bname)) {
                const int m_rho = cons_map["cons.rho"].first;
                // ...and if this face of the block corresponds to a global boundary...
                if (pmb->boundary_flag[bface] == BoundaryFlag::user) {
                    if (binner) {
                        pmb->par_for(
                            "zero_inflow_flux_" + bname, b.ks, b.ke, b.js, b.je, b.is, b.ie,
                            KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                                F.flux(bdir, m_rho, k, j, i) = m::min(F.flux(bdir, m_rho, k, j, i), 0.);
                            }
                        );
                    } else {
                        pmb->par_for(
                            "zero_inflow_flux_" + bname, b.ks, b.ke, b.js, b.je, b.is, b.ie,
                            KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                                F.flux(bdir, m_rho, k, j, i) = m::max(F.flux(bdir, m_rho, k, j, i), 0.);
                            }
                        );
                    }
                }
            }

            // If we should zero flux through this face...
            if (params.Get<bool>("zero_flux_" + bname)) {
                // ...and if this face of the block corresponds to a global boundary...
                if (pmb->boundary_flag[bface] == BoundaryFlag::user) {
                    pmb->par_for(
                        "zero_flux_" + bname, 0, F.GetDim(4) - 1, b.ks, b.ke, b.js, b.je, b.is, b.ie,
                        KOKKOS_LAMBDA(const int &p, const int &k, const int &j, const int &i) {
                            F.flux(bdir, p, k, j, i) = 0.;
                        }
                    );
                }
            }

            // If we should replace fluxes with excised versions...
            if (params.Get<bool>("excise_flux_" + bname)) {
                // ...and if this face of the block corresponds to a global boundary...
                if (pmb->boundary_flag[bface] == BoundaryFlag::user) {
                    if (bdir != 2) throw std::runtime_error("Excised polar fluxes only fully implemented in X2!");

                    // Pack w/B to match indices with the `Flux.X` below
                    // We won't *update* B field though
                    auto &F = rc->PackVariablesAndFluxes({Metadata::WithFluxes}, cons_map);

                    // Going to need the primitive vars
                    PackIndexMap prims_map;
                    std::vector<MetadataFlag> prims_flags = {Metadata::GetUserFlag("Primitive"), Metadata::Cell};
                    const auto& P_all = rc->PackVariables(prims_flags, prims_map);
                    const VarMap m_u(cons_map, true), m_p(prims_map, false);

                    // And a ton of other stuff
                    // But we're modifying the live temporaries, and eventually fluxes, here
                    const auto& Pl_all = rc->PackVariables(std::vector<std::string>{"Flux.Pl"});
                    const auto& Pr_all = rc->PackVariables(std::vector<std::string>{"Flux.Pr"});
                    const auto& Ul_all = rc->PackVariables(std::vector<std::string>{"Flux.Ul"});
                    const auto& Ur_all = rc->PackVariables(std::vector<std::string>{"Flux.Ur"});
                    const auto& Fl_all = rc->PackVariables(std::vector<std::string>{"Flux.Fl"});
                    const auto& Fr_all = rc->PackVariables(std::vector<std::string>{"Flux.Fr"});
                    // We recalculate sound speeds when we update fluxes
                    const auto& cmax  = rc->PackVariables(std::vector<std::string>{"Flux.cmax"});
                    const auto& cmin  = rc->PackVariables(std::vector<std::string>{"Flux.cmin"});

                    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb->packages);
                    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
                    const auto& G = pmb->coords;
                    const int nvar = F.GetDim(4);
                    // Loci "outer" and "inner" refer to right and left.
                    // At right edge, our "outer" toward domain is Loci "inner" as in left half
                    const Loci loc = (binner) ? Loci::outer_half : Loci::inner_half;

                    const IndexRange3 bi = KDomain::GetRange(rc, IndexDomain::interior, CC);
                    // Cell center of our two which is a physical zone
                    const int j_cell = (binner) ? b.js : b.js - 1;

                    // Replace fluxes through the pole (would be zero) with fluxes through
                    // the middle of the cell. Should be general, remember this has 1-zone halo!
                    pmb->par_for(
                        "excise_flux_" + bname, b.ks, b.ke, b.js, b.je, b.is, b.ie,
                        KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                            // Face i,j,k borders cell with same index and 1 left with index:
                            int kk = (bdir == 3) ? k - 1 : k;
                            int jj = (bdir == 2) ? j - 1 : j;
                            int ii = (bdir == 1) ? i - 1 : i;

                            // "Reconstruct" at cell midplanes: equivalent to donor-cell
                            PLOOP {
                                Pl_all(ip, k, j, i) = P_all(ip, kk, jj, ii);
                                Pr_all(ip, k, j, i) = P_all(ip, k, j, i);
                            }

                            FourVectors Dtmp;
                            // Left
                            GRMHD::calc_4vecs(G, Pl_all, m_p, k, j, i, Loci::center, Dtmp);
                            Flux::prim_to_flux(G, Pl_all, m_p, Dtmp, emhd_params, gam, k, j, i, 0, Ul_all, m_u, Loci::center);
                            Flux::prim_to_flux(G, Pl_all, m_p, Dtmp, emhd_params, gam, k, j, i, bdir, Fl_all, m_u, Loci::center);
                            // Magnetosonic speeds
                            Real cmaxL, cminL;
                            Flux::vchar_global(G, Pl_all, m_p, Dtmp, gam, emhd_params, k, j, i, Loci::center, bdir, cmaxL, cminL);
                            // Record speeds
                            cmax(bdir-1, k, j, i) = m::max(0., cmaxL);
                            cmin(bdir-1, k, j, i) = m::min(0., cminL);

                            // Right
                            GRMHD::calc_4vecs(G, Pr_all, m_p, k, j, i, Loci::center, Dtmp);
                            Flux::prim_to_flux(G, Pr_all, m_p, Dtmp, emhd_params, gam, k, j, i, 0, Ur_all, m_u, Loci::center);
                            Flux::prim_to_flux(G, Pr_all, m_p, Dtmp, emhd_params, gam, k, j, i, bdir, Fr_all, m_u, Loci::center);
                            // Magnetosonic speeds
                            Real cmaxR, cminR;
                            Flux::vchar_global(G, Pr_all, m_p, Dtmp, gam, emhd_params, k, j, i, Loci::center, bdir, cmaxR, cminR);

                            // Reset cmax/cmin based on our flux
                            cmax(bdir-1, k, j, i) =  m::max(cmax(bdir-1, k, j, i), cmaxR);
                            cmin(bdir-1, k, j, i) = -m::min(cmin(bdir-1, k, j, i), cminR);

                            // Use LLF flux
                            PLOOP {
                                if (ip != m_u.B1 && ip != m_u.B2 && ip != m_u.B3)
                                    F.flux(bdir, ip, k, j, i) = Flux::llf(Fl_all(ip, k, j, i), Fr_all(ip, k, j, i),
                                                                    cmax(bdir-1, k, j, i), cmin(bdir-1, k, j, i),
                                                                    Ul_all(ip, k, j, i), Ur_all(ip, k, j, i));
                            }
                        }
                    );
                    // Then average to make absolutely sure fluxes match
                    // TODO only for X2 bound currently!
                    // Must pay attention that only physical zones are touched: no averaging w/ghosts!
                    const int Nk3p = (bi.ke - bi.ks + 1);
                    const int Nk3p2 = Nk3p/2;
                    const int ksp = bi.ks;
                    // Run over X1 *interior* on the X2 face, for half the *interior* X3 range
                    pmb->par_for(
                        "average_excised_flux_" + bname, 0, F.GetDim(4)-1, bi.ks, bi.ks + Nk3p2 - 1, b.js, b.je, bi.is, bi.ie,
                        KOKKOS_LAMBDA(const int &v, const int &k, const int &j, const int &i) {
                            const int ki = ((k - ksp + Nk3p2) % Nk3p) + ksp;
                            Real avg = 0.;
                            if (v == m_u.B1 || v == m_u.B2 || v == m_u.B3) {
                                // Don't balance B, it's just zero
                                return;
                            } else if (v == m_u.U2 || v == m_u.U3) {
                                // Flux direction reversed, but *coordinate also reverses*
                                avg = (F.flux(bdir, v, k, j, i) + F.flux(bdir, v, ki, j, i)) / 2;
                                F.flux(bdir, v, ki, j, i) = avg;
                            } else {
                                // Only the flux direction reverses
                                avg = (F.flux(bdir, v, k, j, i) - F.flux(bdir, v, ki, j, i)) / 2;
                                F.flux(bdir, v, ki, j, i) = -avg;
                            }
                            F.flux(bdir, v, k, j, i)  = avg;
                        }
                    );

                    // Now, interpolate primitive variables to the outer half-cell, and
                    // compute fluxes through the half-faces normally
                    const int n1 = pmb0->cellbounds.ncellsi(IndexDomain::entire);
                    const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
                    const size_t var_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(nvar, n1);
                    const size_t recon_scratch_bytes = 2 * var_size_in_bytes;

                    for (auto dir : OrthogonalDirs(bdir)) {
                        // Interpolate the last row to the outer half-cell (once, before below loop over dirs)
                        // We won't be needing the cell-centered P again (right?)
                        // This is a local update since j_cell != jn
                        const int jn = (binner) ? j_cell + 1 : j_cell - 1;
                        pmb->par_for(
                            "excise_flux_" + bname, 0, nvar-1, b.ks, b.ke, b.is, b.ie,
                            KOKKOS_LAMBDA(const int &ip, const int &k, const int &i) {
                                // Ul is just a scratch!  We need the half-cell values now but we will still
                                // need the full-cell-centered prims later
                                // TODO real temporary of plane-size only!!
                                Ul_all(ip, k, j_cell, i) = 0.75 * P_all(ip, k, j_cell, i) + 0.25 * P_all(ip, k, jn, i);
                            }
                        );

                        // Reconstruct using the outer-half values
                        parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "excise_flux_" + bname + "_recon", pmb0->exec_space,
                            recon_scratch_bytes, scratch_level, b.ks, b.ke, j_cell, j_cell,
                            KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& k, const int& j) {
                                ScratchPad2D<Real> Pl_s(member.team_scratch(scratch_level), nvar, n1);
                                ScratchPad2D<Real> Pr_s(member.team_scratch(scratch_level), nvar, n1);

                                // Chosen to not require additional floors.  Could bump order and floor/fallback like in GetFlux
                                // Remember Ul here is just the half-cell centered primitive vars
                                if (dir == X1DIR) {
                                    KReconstruction::ReconstructRow<KReconstruction::Type::linear_mc, X1DIR>(member, Ul_all, k, j, b.is, b.ie, Pl_s, Pr_s);
                                } else if (dir == X2DIR) {
                                    KReconstruction::ReconstructRow<KReconstruction::Type::linear_mc, X2DIR>(member, Ul_all, k, j, b.is, b.ie, Pl_s, Pr_s);
                                } else if (dir == X3DIR) {
                                    KReconstruction::ReconstructRow<KReconstruction::Type::linear_mc, X3DIR>(member, Ul_all, k, j, b.is, b.ie, Pl_s, Pr_s);
                                }
                                member.team_barrier();

                                // Assign local temporaries back to global arrays for the next kernel
                                parthenon::par_for_inner(member, b.is, b.ie,
                                    [&](const int& i) {
                                        PLOOP {
                                            Pl_all(ip, k, j, i) = Pl_s(ip, i);
                                            Pr_all(ip, k, j, i) = Pr_s(ip, i);
                                        }
                                    }
                                );
                            }
                        );

                        // Compute new fluxes
                        pmb->par_for(
                            "excise_flux_" + bname, b.ks, b.ke, j_cell, j_cell, b.is, b.ie,
                            KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                                FourVectors Dtmp;
                                // Left
                                GRMHD::calc_4vecs(G, Pl_all, m_p, k, j, i, loc, Dtmp);
                                Flux::prim_to_flux(G, Pl_all, m_p, Dtmp, emhd_params, gam, k, j, i, 0, Ul_all, m_u, loc);
                                Flux::prim_to_flux(G, Pl_all, m_p, Dtmp, emhd_params, gam, k, j, i, dir, Fl_all, m_u, loc);
                                // Magnetosonic speeds
                                Real cmaxL, cminL;
                                Flux::vchar_global(G, Pl_all, m_p, Dtmp, gam, emhd_params, k, j, i, loc, dir, cmaxL, cminL);
                                // Record speeds
                                cmax(dir-1, k, j, i) = m::max(0., cmaxL);
                                cmin(dir-1, k, j, i) = m::min(0., cminL);

                                // Right
                                GRMHD::calc_4vecs(G, Pr_all, m_p, k, j, i, loc, Dtmp);
                                Flux::prim_to_flux(G, Pr_all, m_p, Dtmp, emhd_params, gam, k, j, i, 0, Ur_all, m_u, loc);
                                Flux::prim_to_flux(G, Pr_all, m_p, Dtmp, emhd_params, gam, k, j, i, dir, Fr_all, m_u, loc);
                                // Magnetosonic speeds
                                Real cmaxR, cminR;
                                Flux::vchar_global(G, Pr_all, m_p, Dtmp, gam, emhd_params, k, j, i, loc, dir, cmaxR, cminR);

                                // Reset cmax/cmin based on our flux
                                cmax(dir-1, k, j, i) =  m::max(cmax(dir-1, k, j, i), cmaxR);
                                cmin(dir-1, k, j, i) = -m::min(cmin(dir-1, k, j, i), cminR);

                                // Use LLF flux
                                PLOOP {
                                    if (ip != m_u.B1 && ip != m_u.B2 && ip != m_u.B3)
                                        F.flux(dir, ip, k, j, i) = Flux::llf(Fl_all(ip, k, j, i), Fr_all(ip, k, j, i),
                                                                        cmax(dir-1, k, j, i), cmin(dir-1, k, j, i),
                                                                        Ul_all(ip, k, j, i), Ur_all(ip, k, j, i)) * 0.5;
                                }
                            }
                        );
                    }
                }
            }
        }
    }

    return TaskStatus::complete;
}

void KBoundaries::AddSource(MeshData<Real> *md, MeshData<Real> *mdudt, IndexDomain domain)
{
    // Note we're ignoring "domain," we just add the "source" where it's needed next to the pole
    auto pmesh = mdudt->GetMeshPointer();
    auto& params = pmesh->packages.Get<KHARMAPackage>("Boundaries")->AllParams();
    for (int i=0; i < mdudt->NumBlocks(); ++i) {
        auto &rc = mdudt->GetBlockData(i);
        auto pmb = rc->GetBlockPointer();
        for (int i = 0; i < BOUNDARY_NFACES; i++) {
            BoundaryFace bface = (BoundaryFace)i;
            auto bname = KBoundaries::BoundaryName(bface);
            const auto bdir = KBoundaries::BoundaryDirection(bface);
            const auto binner = KBoundaries::BoundaryIsInner(bface);
            const auto bdomain = KBoundaries::BoundaryDomain(bface);

            if (bdir > pmesh->ndim) continue;

            // If we should replace fluxes with excised versions...
            if (params.Get<bool>("excise_flux_" + bname)) {
                // ...and if this face of the block corresponds to a global boundary...
                if (pmb->boundary_flag[bface] == BoundaryFlag::user) {
                    if (bdir != 2) throw std::runtime_error("Excised polar fluxes only fully implemented in X2!");

                    const IndexRange3 bi = KDomain::GetRange(rc, IndexDomain::interior);

                    // Interior only! We're about to sync anyway
                    IndexRange3 b = bi;
                    // Range is last physical cell-center around the pole
                    if (bdir == 1) {
                        b.is = b.ie = (binner) ? bi.is : bi.ie;
                    } else if (bdir == 2) {
                        b.js = b.je = (binner) ? bi.js : bi.je;
                    } else {
                        b.ks = b.ke = (binner) ? bi.ks : bi.ke;
                    }

                    // The magnetic field is probably defined at faces; even if it's defined in cells,
                    // we shouldn't be monkeying with it.  We just do not adjust it here.
                    auto &dUdt = rc->PackVariables({Metadata::GetUserFlag("HD"), Metadata::WithFluxes});
                    const auto& G = pmb->coords;
                    const Loci loc = (binner) ? Loci::outer_half : Loci::inner_half;

                    pmb->par_for(
                        "normalize_excised_flux_" + bname, 0, dUdt.GetDim(4)-1, b.ks, b.ke, b.js, b.je, b.is, b.ie,
                        KOKKOS_LAMBDA(const int &v, const int &k, const int &j, const int &i) {
                            // Factor of 2 because cell is half-size in fluxdiv
                            // gdet factors move conserved vars at outer cell to the center
                            dUdt(v, k, j, i) *= 2 * G.gdet(Loci::center, j, i) / G.gdet(loc, j, i);
                        }
                    );

                }
            }
        }
    }
}
