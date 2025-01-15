/*
 *  File: inverter.cpp
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
#include "inverter.hpp"

// inverter.hpp includes the template and instantiations in the correct order

#include "domain.hpp"
#include "flux.hpp"
#include "reductions.hpp"

int Inverter::CountPFlags(MeshData<Real> *md)
{
    return Reductions::CountFlags(md, "pflag", Inverter::status_names, IndexDomain::interior, false)[0];
}

std::shared_ptr<KHARMAPackage> Inverter::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Inverter");
    Params &params = pkg->AllParams();

    // Inversion scheme.  Could be separate packages but they do share a lot,
    // and could share more e.g. inline floor applications
    std::vector<std::string> allowed_inverter_names = {"none", "onedw", "kastaun"};
    std::string inverter_name = pin->GetOrAddString("inverter", "type", "kastaun", allowed_inverter_names);
    bool use_kastaun = false;
    if (inverter_name == "onedw") {
        params.Add("inverter_type", Type::onedw);
    } else if (inverter_name == "kastaun") {
        params.Add("inverter_type", Type::kastaun);
        use_kastaun = true;
    } else if (inverter_name == "none") {
        params.Add("inverter_type", Type::none);
    }

    // Solver options
    // Any other Noble et al. implemented for fun should use lower tol/iter count, see Noble+06
    Real err_tol = pin->GetOrAddReal("inverter", "err_tol", (use_kastaun) ? 1e-12 : 1e-8);
    params.Add("err_tol", err_tol);
    int iter_max = pin->GetOrAddInteger("inverter", "iter_max", (use_kastaun) ? 25 : 8);
    params.Add("iter_max", iter_max);

    // Floor options
    // Use a custom block for inverter floors to allow customization.  Not sure anyone *wants* that but...
    if (!pin->DoesBlockExist("inverter_floors")) {
        params.Add("inverter_prescription", Floors::MakePrescription(pin, "floors"));
            if (pin->DoesBlockExist("floors_inner"))
                params.Add("inverter_prescription_inner", Floors::MakePrescriptionInner(pin, Floors::MakePrescription(pin, "floors"), "floors_inner"));
            else
                params.Add("inverter_prescription_inner", Floors::MakePrescriptionInner(pin, Floors::MakePrescription(pin, "floors"), "floors"));
    } else {
        params.Add("inverter_prescription", Floors::MakePrescription(pin, "inverter_floors"));
        params.Add("inverter_prescription_inner", Floors::MakePrescriptionInner(pin, Floors::MakePrescription(pin, "inverter_floors"), "inverter_floors"));
    }

    // Fixup options
    bool fix_average_neighbors = pin->GetOrAddBoolean("inverter", "fix_average_neighbors", !use_kastaun);
    params.Add("fix_average_neighbors", fix_average_neighbors);
    // Fix by replacing with floors, uvec=0. Usually a fallback for no neighbors,
    // but also used if Kastaun hits max_iter
    bool fix_atmosphere = pin->GetOrAddBoolean("inverter", "fix_atmosphere", true);
    params.Add("fix_atmosphere", fix_atmosphere);
    // TODO add version attempting to recover from entropy, stuff like that

    // Flag denoting UtoP inversion failures
    // Needs boundary sync if the fixup code will use neighbors, and if
    // we're syncing prims and fixing up after
    bool sync_prims = packages->Get("Driver")->Param<bool>("sync_prims");
    Metadata m;
    if (sync_prims && fix_average_neighbors) {
        m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost});
    } else {
        m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    }
    pkg->AddField("pflag", m);

    // When not using floors, we need to declare fflag for ourselves
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::Overridable});
    pkg->AddField("fflag", m);

    // This package may be loaded even when evolving implicitly, e.g. for FOFC
    // Only register our callbacks if they're needed for explicit evolution or a guess
    if (!pin->GetBoolean("GRMHD", "implicit") || pin->GetBoolean("emhd", "ideal_guess")) {
        // We exist basically to do this
        pkg->BlockUtoP = Inverter::BlockUtoP;
        // We want to run U->P on most boundaries when we're synchronizing conserved variables
        pkg->BoundaryUtoP = Inverter::BlockUtoP;
        // However, we apply domain boundaries to primitives.
        // Registering this additional function conveys that to the callers in `Packages` and `Boundaries`
        pkg->DomainBoundaryPtoU = Flux::BlockPtoUMHD;
    }

    pkg->PostStepDiagnosticsMesh = Inverter::PostStepDiagnostics;

    // List (vector) of HistoryOutputVars that will all be enrolled as output variables
    parthenon::HstVar_list hst_vars = {};
    // Count total floors as a history item
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, CountPFlags, "PFlags"));
    // TODO entries for each individual flag?
    // add callbacks for HST output to the Params struct, identified by the `hist_param_key`
    pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

    return pkg;
}

/**
 * Internal inversion fn, templated on inverter type.  Calls through to templated u_to_p
 * This is called with the correct template argument from BlockUtoP
 */
template<Inverter::Type inverter>
inline void BlockPerformInversion(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();

    PackIndexMap prims_map, cons_map;
    auto U = GRMHD::PackMHDCons(rc, cons_map);
    auto P = GRMHD::PackHDPrims(rc, prims_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    auto fflag = rc->PackVariables(std::vector<std::string>{"fflag"});
    auto pflag = rc->PackVariables(std::vector<std::string>{"pflag"});

    if (U.GetDim(4) == 0 || pflag.GetDim(4) == 0)
        return;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    auto &pars = pmb->packages.Get("Inverter")->AllParams();
    const Real err_tol = pars.Get<Real>("err_tol");
    const int iter_max = pars.Get<int>("iter_max");
    const Floors::Prescription inverter_floors       = pars.Get<Floors::Prescription>("inverter_prescription");
    const Floors::Prescription inverter_floors_inner = pars.Get<Floors::Prescription>("inverter_prescription_inner");
    const bool radius_dependent_floors = inverter_floors.radius_dependent_floors;

    const auto& G = pmb->coords;

    // Get the primitives from our conserved versions
    // Notice by default, we recover variables for only the physical (interior or interior-ghost)
    // zones!  These are the only ones which are filled at our point in the step
    const IndexRange3 b = (domain == IndexDomain::entire)
                          ? KDomain::GetPhysicalRange(rc) : KDomain::GetRange(rc, domain, coarse);
    pmb->par_for("U_to_P", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            const Floors::Prescription& myfloors = (inverter_floors.radius_dependent_floors
                                            && G.coords.is_spherical()
                                            && G.r(k, j, i) < inverter_floors.floors_switch_r) ?
                                            inverter_floors_inner : inverter_floors;
            int pflagl = Inverter::u_to_p<inverter>(G, U, m_u, gam, k, j, i, P, m_p, Loci::center,
                                                    myfloors, iter_max, err_tol);
            pflag(0, k, j, i) = pflagl % Floors::FFlag::MINIMUM;
            int fflagl = (pflagl / Floors::FFlag::MINIMUM) * Floors::FFlag::MINIMUM;
            fflag(0, k, j, i) = fflagl;
            // Generally after inversion we manipulate P and call this ourselves
            // Enable this if that doesn't stay true
            // if (fflagl) {
            //     // If we applied a floor during recovery, update the cons
            //     GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u);
            // }
        }
    );
}

void Inverter::BlockUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    // This only chooses an implementation.  See BlockPerformInversion and implementations e.g. onedw.hpp
    auto& type = rc->GetBlockPointer()->packages.Get("Inverter")->Param<Type>("inverter_type");
    switch(type) {
    case Type::onedw:
        BlockPerformInversion<Type::onedw>(rc, domain, coarse);
        break;
    case Type::kastaun:
        BlockPerformInversion<Type::kastaun>(rc, domain, coarse);
        break;
    case Type::none:
        break;
    }
    // This is dangerous since there are many blocks/packs and we need one reduction. For later.
    //Reductions::StartFlagReduce(md, "pflag", Inverter::status_names, IndexDomain::interior, false, 1);
}

TaskStatus Inverter::PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md)
{
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Options
    const auto& pars = pmesh->packages.Get("Globals")->AllParams();
    const int flag_verbose = pars.Get<int>("flag_verbose");

    // Debugging/diagnostic info about inversion flags
    // TODO grab the total and die on too many
    if (flag_verbose >= 1) {
        // TODO this should move into UtoP when everything goes MeshData
        Reductions::StartFlagReduce(md, "pflag", Inverter::status_names, IndexDomain::interior, false, 1);
        Reductions::CheckFlagReduceAndPrintHits(md, "pflag", Inverter::status_names, IndexDomain::interior, false, 1);

        // If we're the only floors, print those too
        if (!pmesh->packages.AllPackages().count("Floors")) {
            Reductions::StartFlagReduce(md, "fflag", Floors::FFlag::flag_names, IndexDomain::interior, true, 0);
            // Debugging/diagnostic info about floors
            Reductions::CheckFlagReduceAndPrintHits(md, "fflag", Floors::FFlag::flag_names, IndexDomain::interior, true, 0);
        }
    }

    return TaskStatus::complete;
}
