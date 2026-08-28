/*
 *  File: entropy.cpp
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
#include "entropy.hpp"

#include "decs.hpp"
#include "domain.hpp"
#include "floors.hpp"
#include "flux.hpp"
#include "kharma_driver.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

using namespace parthenon;

namespace Entropy
{

std::shared_ptr<KHARMAPackage> Initialize(
    ParameterInput* pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Entropy");
    Params& params = pkg->AllParams();

    // Whether to also track an "idealized" entropy: the value Ktot would have if the
    // fluid had only ever been passively advected, with no dissipation.
    bool advect_entropy = pin->GetOrAddBoolean("entropy", "advect_entropy", false);
    params.Add("advect_entropy", advect_entropy);

    // Whether to set Ktot (& Ktot_adv) to their default values -- the entropy implied by
    // the problem's initial rho, u -- at startup.  Problems which set these values
    // themselves should disable this, or override the parameter during their own
    // initialization.
    bool init_to_default = pin->GetOrAddBoolean("entropy", "init_to_default", true);
    params.Add("init_to_default", init_to_default, true);

    Metadata::AddUserFlag("TrackEntropy");
    // Ktot is the fluid entropy, recomputed at the end of each sub-step from the other
    // fluid primitives.
    std::vector<MetadataFlag> flags_entropy = {Metadata::Cell,
        Metadata::GetUserFlag("Explicit"), Metadata::GetUserFlag("TrackEntropy")};

    auto& driver = packages->Get("Driver")->AllParams();
    auto flags_prim = driver.Get<std::vector<MetadataFlag>>("prim_flags");
    flags_prim.insert(flags_prim.end(), flags_entropy.begin(), flags_entropy.end());
    auto flags_cons = driver.Get<std::vector<MetadataFlag>>("cons_flags");
    flags_cons.insert(flags_cons.end(), flags_entropy.begin(), flags_entropy.end());

    // Total (real) entropy of the fluid
    pkg->AddField("cons.Ktot", flags_cons);
    pkg->AddField("prims.Ktot", flags_prim);

    // Idealized (advected, no-dissipation) entropy, if requested.  Nice to have.
    if (advect_entropy) {
        pkg->AddField("cons.Ktot_adv", flags_cons);
        pkg->AddField("prims.Ktot_adv", flags_prim);
    }

    pkg->BlockUtoP = Entropy::BlockUtoP;
    pkg->BoundaryUtoP = Entropy::BlockUtoP;

    return pkg;
}

TaskStatus InitEntropy(MeshBlockData<Real>* rc, ParameterInput* pin)
{
    Flag("InitEntropy");
    auto pmb = rc->GetBlockPointer();

    // Don't initialize if we've already done so e.g. in the Hubble problem
    if (!pmb->packages.Get("Entropy")->Param<bool>("init_to_default")) {
        EndFlag();
        return TaskStatus::complete;
    }

    // Covers prims.Ktot and, if enabled, prims.Ktot_adv -- both start from the same,
    // real initial entropy implied by the problem's initial rho, u.
    auto& k_P = rc->PackVariables(
        {Metadata::GetUserFlag("TrackEntropy"), Metadata::GetUserFlag("Primitive")});
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    pmb->par_for("init_entropy", 0, k_P.GetDim(4) - 1, ks, ke, js, je, is, ie,
                 KOKKOS_LAMBDA(const int& p, const int& k, const int& j, const int& i)
        {
            k_P(p, k, j, i) = Entropy::CalcEntropy(rho(k, j, i), u(k, j, i), gam);
        });

    EndFlag();
    return TaskStatus::complete;
}

void BlockUtoP(MeshBlockData<Real>* rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();

    // No need for a "map" here, we just want everything that fits these criteria.
    auto& k_P = rc->PackVariables(
        {Metadata::GetUserFlag("TrackEntropy"), Metadata::GetUserFlag("Primitive")});
    auto& k_U = rc->PackVariables(
        {Metadata::GetUserFlag("TrackEntropy"), Metadata::Conserved});
    // And then the local density
    GridScalar rho_U = rc->Get("cons.rho").data;

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    int is = bounds.is(domain), ie = bounds.ie(domain);
    int js = bounds.js(domain), je = bounds.je(domain);
    int ks = bounds.ks(domain), ke = bounds.ke(domain);
    pmb->par_for("UtoP_entropy", 0, k_P.GetDim(4) - 1, ks, ke, js, je, is, ie,
                 KOKKOS_LAMBDA(const int& p, const int& k, const int& j, const int& i)
        {
            k_P(p, k, j, i) = k_U(p, k, j, i) / rho_U(k, j, i);
        });
}

TaskStatus ApplyEntropyUpdate(MeshBlockData<Real>* rc)
{
    Flag("ApplyEntropyUpdate");
    auto pmb = rc->GetBlockPointer();

    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridScalar ktot = rc->Get("prims.Ktot").data;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    // Must be applied over the entire domain, ghost zones included: this needs to stay
    // consistent with the rest of the fluid state, which has already been updated there.
    const IndexRange ib = rc->GetBoundsI(IndexDomain::entire);
    const IndexRange jb = rc->GetBoundsJ(IndexDomain::entire);
    const IndexRange kb = rc->GetBoundsK(IndexDomain::entire);
    pmb->par_for("update_entropy", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            // Reset Ktot to the real, dissipation-included entropy implied by this step's
            // final rho, u. This is the baseline the *next* step's pure advection will
            // be compared against, to isolate that step's dissipation in turn.
            // Note Ktot_adv (if tracked) is deliberately left untouched: it should evolve
            // via pure advection for the whole run, with no such reset.
            ktot(k, j, i) = Entropy::CalcEntropy(rho(k, j, i), u(k, j, i), gam);
        });

    EndFlag();
    return TaskStatus::complete;
}

void ApplyFloors(MeshBlockData<Real>* mbd, IndexDomain domain)
{
    auto pmb = mbd->GetBlockPointer();
    auto packages = pmb->packages;

    PackIndexMap prims_map;
    auto P = mbd->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    const VarMap m_p(prims_map, false);

    auto fflag = mbd->PackVariables(std::vector<std::string>{"fflag"}, prims_map);

    const auto& G = pmb->coords;

    const Floors::Prescription floors =
        packages.Get("Floors")->Param<Floors::Prescription>("prescription");
    const Floors::Prescription floors_inner =
        packages.Get("Floors")->Param<Floors::Prescription>("prescription_inner");

    const IndexRange3 b = KDomain::GetRange(mbd, domain);
    pmb->par_for("apply_entropy_floors", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA(const int& k, const int& j, const int& i)
        {
            Real ktot_max;
            if (floors.radius_dependent_floors && G.coords.is_spherical() &&
                G.r(k, j, i) < floors.floors_switch_r) {
                ktot_max = floors_inner.ktot_max;
            } else {
                ktot_max = floors.ktot_max;
            }

            if (P(m_p.KTOT, k, j, i) > ktot_max) {
                fflag(0, k, j, i) = Floors::FFlag::KTOT | (int)fflag(0, k, j, i);
                P(m_p.KTOT, k, j, i) = ktot_max;
            }

            // TODO(CEP) restore Ressler adjustment option
            // Ressler adjusts KTOT & KEL to conserve u whenever adjusting rho
            // but does *not* recommend adjusting them when u hits floors/ceilings.
            // This is in contrast to ebhlight, which heats electrons before applying
            // *any* floors, and resets KTOT during floor application without touching
            // KEL.  Note the KEL half of this would now belong in the Electrons package.
        });
    Flux::BlockPtoU(mbd, domain);
}

} // namespace Entropy
