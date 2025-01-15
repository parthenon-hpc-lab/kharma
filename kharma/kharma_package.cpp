/*
 *  File: kharma_package.cpp
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
#include "kharma_package.hpp"

#include "types.hpp"

// TODO clearly this needs a better concept of ordering.
// probably this means something that returns an ordered list of packages
// for the given operation, based on... declared dependencies?
// it could also use full meshification & return codes

TaskStatus Packages::FixFlux(MeshData<Real> *md)
{
    Flag("FixFlux");
    auto kpackages = md->GetMeshPointer()->packages.AllPackagesOfType<KHARMAPackage>();
    for (auto kpackage : kpackages) {
        if (kpackage.second->FixFlux != nullptr) {
            Flag("FixFlux_"+kpackage.first);
            kpackage.second->FixFlux(md);
            EndFlag();
        }
    }
    EndFlag();
    return TaskStatus::complete;
}

TaskStatus Packages::BlockUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    Flag("BlockUtoP");
    // Apply UtoP from B_CT first, as this fills cons.B at cell centers
    auto pmb = rc->GetBlockPointer();
    auto kpackages = rc->GetBlockPointer()->packages.AllPackagesOfType<KHARMAPackage>();
    if (kpackages.count("B_CT")) {
        KHARMAPackage *pkpackage = pmb->packages.Get<KHARMAPackage>("B_CT");
        if (pkpackage->BlockUtoP != nullptr) {
            Flag("BlockUtoP_B_CT");
            pkpackage->BlockUtoP(rc, domain, coarse);
            EndFlag();
        }
    }
    // Then GRMHD, as some packages require GRMHD prims in place for U->P
    if (kpackages.count("Inverter")) {
        KHARMAPackage *pkpackage = pmb->packages.Get<KHARMAPackage>("Inverter");
        if (pkpackage->BlockUtoP != nullptr) {
            Flag("BlockUtoP_Inverter");
            pkpackage->BlockUtoP(rc, domain, coarse);
            EndFlag();
        }
    }
    for (auto kpackage : kpackages) {
        if (kpackage.second->BlockUtoP != nullptr && kpackage.first != "B_CT" && kpackage.first != "Inverter") {
            Flag("BlockUtoP_"+kpackage.first);
            kpackage.second->BlockUtoP(rc, domain, coarse);
            EndFlag();
        }
    }
    EndFlag();
    return TaskStatus::complete;
}
TaskStatus Packages::MeshUtoP(MeshData<Real> *md, IndexDomain domain, bool coarse)
{
    // TODO TODO prefer MeshUtoP implementations and fall back
    Flag("MeshUtoP");
    for (int i=0; i < md->NumBlocks(); ++i)
        BlockUtoP(md->GetBlockData(i).get(), domain, coarse);
    EndFlag();
    return TaskStatus::complete;
}

TaskStatus Packages::BoundaryUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    Flag("BoundaryUtoP");
    auto pmb = rc->GetBlockPointer();
    auto kpackages = rc->GetBlockPointer()->packages.AllPackagesOfType<KHARMAPackage>();
    if (kpackages.count("Inverter")) {
        KHARMAPackage *pkpackage = pmb->packages.Get<KHARMAPackage>("Inverter");
        if (pkpackage->BoundaryUtoP != nullptr) {
            Flag("BoundaryUtoP_Inverter");
            pkpackage->BoundaryUtoP(rc, domain, coarse);
            EndFlag();
        }
    }
    for (auto kpackage : kpackages) {
        if (kpackage.second->BoundaryUtoP != nullptr && kpackage.first != "Inverter") {
            Flag("BoundaryUtoP_"+kpackage.first);
            kpackage.second->BoundaryUtoP(rc, domain, coarse);
            EndFlag();
        }
    }
    EndFlag();
    return TaskStatus::complete;
}

TaskStatus Packages::BoundaryPtoUElseUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    Flag("DomainBoundaryLockstep");
    auto pmb = rc->GetBlockPointer();
    auto kpackages = rc->GetBlockPointer()->packages.AllPackagesOfType<KHARMAPackage>();
    // Some downstream UtoP rely on GRMHD prims, some cons
    if (kpackages.count("Inverter")) {
        KHARMAPackage *pkpackage = pmb->packages.Get<KHARMAPackage>("Inverter");
        if (pkpackage->DomainBoundaryPtoU != nullptr) {
            Flag("DomainBoundaryPtoU_Inverter");
            pkpackage->DomainBoundaryPtoU(rc, domain, coarse);
            EndFlag();
        } else if (pkpackage->BoundaryUtoP != nullptr) { // This won't be called
            Flag("DomainBoundaryUtoP_Inverter");
            pkpackage->BoundaryUtoP(rc, domain, coarse);
            EndFlag();
        }
    }
    for (auto kpackage : kpackages) {
        if (kpackage.second->DomainBoundaryPtoU != nullptr && kpackage.first != "Inverter") {
            Flag("DomainBoundaryPtoU_"+kpackage.first);
            kpackage.second->DomainBoundaryPtoU(rc, domain, coarse);
            EndFlag();
        } else if (kpackage.second->BoundaryUtoP != nullptr && kpackage.first != "Inverter") {
            Flag("DomainBoundaryUtoP_"+kpackage.first);
            kpackage.second->BoundaryUtoP(rc, domain, coarse);
            EndFlag();
        }
    }
    EndFlag();
    return TaskStatus::complete;
}

TaskStatus Packages::AddSource(MeshData<Real> *md, MeshData<Real> *mdudt, IndexDomain domain)
{
    Flag("AddSource");
    auto pmesh = md->GetMeshPointer();
    auto kpackages = pmesh->packages.AllPackagesOfType<KHARMAPackage>();
    if (kpackages.count("Boundaries")) {
        KHARMAPackage *pkpackage = pmesh->packages.Get<KHARMAPackage>("Boundaries");
        if (pkpackage->AddSource != nullptr) {
            Flag("AddSource_Boundaries");
            pkpackage->AddSource(md, mdudt, domain);
            EndFlag();
        }
    }
    for (auto kpackage : kpackages) {
        if (kpackage.second->AddSource != nullptr && kpackage.first != "Boundaries") {
            Flag("AddSource_"+kpackage.first);
            kpackage.second->AddSource(md, mdudt, domain);
            EndFlag();
        }
    }
    EndFlag();
    return TaskStatus::complete;
}

TaskStatus Packages::MeshApplyPrimSource(MeshData<Real> *md)
{
    Flag("MeshApplyPrimSource");
    for (int i=0; i < md->NumBlocks(); ++i) {
        auto rc = md->GetBlockData(i).get();
        auto kpackages = rc->GetBlockPointer()->packages.AllPackagesOfType<KHARMAPackage>();
        for (auto kpackage : kpackages) {
            if (kpackage.second->BlockApplyPrimSource != nullptr) {
                kpackage.second->BlockApplyPrimSource(rc);
            }
        }
    }
    EndFlag();
    return TaskStatus::complete;
}

TaskStatus Packages::MeshApplyFloors(MeshData<Real> *md, IndexDomain domain)
{
    Flag("MeshApplyFloors");

    // Apply the version from "Floors" package first
    auto pmesh = md->GetMeshPointer();
    auto pkgs = pmesh->packages.AllPackages();
    if (pkgs.count("Floors")) {
        KHARMAPackage *pkpackage = pmesh->packages.Get<KHARMAPackage>("Floors");
        if (pkpackage->MeshApplyFloors != nullptr) {
            Flag("MeshApplyFloors_Floors");
            pkpackage->MeshApplyFloors(md, domain);
            EndFlag();
        }
    }
    // Then everything else i.e. block versions
    // TODO(BSP) allow Mesh versions and fallback
    for (int i=0; i < md->NumBlocks(); ++i) {
        auto mbd = md->GetBlockData(i).get();
        auto pmb = mbd->GetBlockPointer();
        auto kpackages = pmb->packages.AllPackagesOfType<KHARMAPackage>();
        for (auto kpackage : kpackages) {
            if (kpackage.first != "Floors") {
                if (kpackage.second->BlockApplyFloors != nullptr) {
                    Flag("BlockApplyFloors_"+kpackage.first);
                    kpackage.second->BlockApplyFloors(mbd, domain);
                    EndFlag();
                }
            }
        }
    }
    EndFlag();
    return TaskStatus::complete;
}

// GENERAL CALLBACKS
// TODO this will need to be mesh'd too
void Packages::UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin)
{
    Flag("UserWorkBeforeOutput");
    auto kpackages = pmb->packages.AllPackagesOfType<KHARMAPackage>();
    for (auto kpackage : kpackages) {
        if (kpackage.second->BlockUserWorkBeforeOutput != nullptr) {
            Flag("UserWorkBeforeOutput_"+kpackage.first);
            kpackage.second->BlockUserWorkBeforeOutput(pmb, pin);
            EndFlag();
        }
    }
    EndFlag();
}

void Packages::PreStepWork(Mesh *pmesh, ParameterInput *pin, const SimTime &tm)
{
    Flag("PreStepWork");
    auto kpackages = pmesh->packages.AllPackagesOfType<KHARMAPackage>();
    for (auto kpackage : kpackages) {
        if (kpackage.second->PreStepWork != nullptr) {
            Flag("PreStepWork_"+kpackage.first);
            kpackage.second->PreStepWork(pmesh, pin, tm);
            EndFlag();
        }
    }
    EndFlag();
}

void Packages::PostStepWork(Mesh *pmesh, ParameterInput *pin, const SimTime &tm)
{
    Flag("PostStepWork");
    auto kpackages = pmesh->packages.AllPackagesOfType<KHARMAPackage>();
    for (auto kpackage : kpackages) {
        if (kpackage.second->PostStepWork != nullptr) {
            Flag("PostStepWork_"+kpackage.first);
            kpackage.second->PostStepWork(pmesh, pin, tm);
            EndFlag();
        }
    }
    EndFlag();
}

void Packages::PostStepDiagnostics(Mesh *pmesh, ParameterInput *pin, const SimTime &tm)
{
    // Parthenon's version of this has a bug, but I would probably subclass it anyway.
    // very useful to have a single per-step spot to control any routine print statements
    Flag("PostStepDiagnostics");
    const auto& md = pmesh->mesh_data.Get().get();
    if (md->NumBlocks() > 0) {
        for (auto &package : pmesh->packages.AllPackages()) {
            if (package.second->PostStepDiagnosticsMesh != nullptr) {
                Flag("PostStepDiagnostics_"+package.first);
                package.second->PostStepDiagnosticsMesh(tm, md);
                EndFlag();
            }
        }
    }
    EndFlag();
}

void Packages::PostExecute(Mesh *pmesh, ParameterInput *pin, const SimTime &tm)
{
    Flag("KHARMAPostExecute");
    auto kpackages = pmesh->packages.AllPackagesOfType<KHARMAPackage>();
    for (auto kpackage : kpackages) {
        if (kpackage.second->PostExecute != nullptr) {
            Flag("PostExecute_"+kpackage.first);
            kpackage.second->PostExecute(pmesh, pin, tm);
            EndFlag();
        }
    }
    EndFlag();
}
