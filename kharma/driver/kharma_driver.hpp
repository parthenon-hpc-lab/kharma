/* 
 *  File: kharma_driver.hpp
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
#pragma once

#include "decs.hpp"
#include "domain.hpp"
#include "types.hpp"

#include "flux/reconstruction.hpp"

using namespace parthenon;

// See Initialize()
enum class DriverType{kharma, imex, simple};

/**
 * This is the "Driver" class for KHARMA.
 * A Driver object orchestrates everything that has to be done to a mesh to constitute a step.
 * This means handling RK2/4/predictor-corrector stepping
 * 
 * Somewhat confusingly, but very conveniently, it is also a package; therefore, it defines
 * a static member function Initialize(), which returns a StateDescriptor.
 * Many things in that list are referenced by other packages dependent on this one.
 * 
 * Documentation on this thing: https://github.com/AFD-Illinois/kharma/wiki/The-Driver
 */
class KHARMADriver : public MultiStageDriver {
    public:
        KHARMADriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm) : MultiStageDriver(pin, app_in, pm) {}

        static std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

        // Eliminate Parthenon's print statements when starting up the driver, we have a bunch of our own
        void PreExecute() override { timer_main.reset(); }

        // Also override the timestep calculation, so we can start moving options etc out of GRMHD package
        void SetGlobalTimeStep() override;

        // And the PostExecute, so we can add a package callback here
        void PostExecute(DriverStatus status) override;

        /**
         * Make a TaskCollection according to which step type was chosen (kharma, imex, simple).
         * This represents all tasks to be done in this sub-step (i.e., one TaskList per
         * stage of the integrator)
         */
        TaskCollection MakeTaskCollection(BlockList_t &blocks, int stage) override;

        /**
         * The default step, synchronizing conserved variables and then recovering primitive variables in the ghost zones.
         * See https://github.com/AFD-Illinois/kharma/wiki/The-Driver#kharma-step
         */
        TaskCollection MakeDefaultTaskCollection(BlockList_t &blocks, int stage);

        /**
         * This step syncs primitive variables and treats them as fundamental
         * This accommodates semi-implicit stepping, allowing evolving theories with implicit source terms such as extended MHD
         * See https://github.com/AFD-Illinois/kharma/wiki/The-Driver#imex-step
         */
        TaskCollection MakeImExTaskCollection(BlockList_t &blocks, int stage);

        /**
         * A simple step for experimentation/new implementations.  Does NOT support Face-CT, or much of anything optional.
         * See https://github.com/AFD-Illinois/kharma/wiki/The-Driver#simple-step
         */
        TaskCollection MakeSimpleTaskCollection(BlockList_t &blocks, int stage);

        // BUNDLES
        // The different drivers share substantially similar portions of the full task list, which we gather into
        // single functions here
        /**
         * Add the flux calculations in each direction.  Since the flux functions are templated on which
         * reconstruction is being used, this amounts to a lot of shared lines.
         */
        static TaskID AddFluxCalculations(TaskID& t_start, TaskList& tl, MeshData<Real> *md);

        /**
         * Add first-order flux corrections.  This is split out because it needs an additional MeshData object for the "guess"
         * TODO(BSP) Maybe a less-cowardly approach to adding MeshDatas lets me shove this into AddFluxCalculations
         */
        TaskID AddFOFC(TaskID& t_start, TaskList& tl, MeshData<Real> *md,
                             MeshData<Real> *md_full_step_init, MeshData<Real> *md_sub_step_init,
                             MeshData<Real> *guess_src, MeshData<Real> *guess, int stage);
        /**
         * This function updates a state md_update with the results of an explicit source term calculation
         * placed in md_flux_src.  It includes initialization/RK factors and so requires full- and sub-step
         * initial states too.
         */
        TaskID AddStateUpdate(TaskID& t_start, TaskList& tl, MeshData<Real> *md_full_step_init, MeshData<Real> *md_sub_step_init,
                                MeshData<Real> *md_flux_src, MeshData<Real> *md_update, std::vector<MetadataFlag> flags,
                                bool update_face, int stage);

        /**
         * This function updates a state md_update with the results of an explicit source term calculation
         * placed in md_flux_src. It is similar to `AddStateUpdate` but applies only to variables marked
         * with the `IdealGuess` flag.
         */
        TaskID AddStateUpdateIdealGuess(TaskID& t_start, TaskList& tl, MeshData<Real> *md_full_step_init, MeshData<Real> *md_sub_step_init,
                                MeshData<Real> *md_flux_src, MeshData<Real> *md_update, std::vector<MetadataFlag> flags,
                                bool update_face, int stage);

        /**
         * Add a synchronization retion to an existing TaskCollection tc.
         * Since the region is self-contained, does not return a TaskID
         */
        void AddFullSyncRegion(TaskCollection& tc, std::shared_ptr<MeshData<Real>> &md);

        /**
         * Add just the synchronization step to a task list tl, dependent upon taskID t_start, syncing mesh mc1
         * 
         * This sequence is used identically in several places, so it makes sense
         * to define once and use elsewhere.
         */
        static TaskID AddBoundarySync(const TaskID t_start, TaskList &tl, std::shared_ptr<MeshData<Real>> &md);

        /**
         * Single call to sync all boundary conditions (MPI/internal and domain/physical boundaries)
         * Used anytime boundary sync is needed outside the usual loop of steps.
         * 
         * Only use this during the run if you're debugging!
         */
        static TaskStatus SyncAllBounds(std::shared_ptr<MeshData<Real>> &md);

        // TODO swapped versions of these
        /**
         * Copy variables matching 'flags' from 'source' to 'dest'.
         * Mostly makes things easier to read.
         */
        template<typename T>
        static TaskStatus Copy(std::vector<MetadataFlag> flags, T* source, T* dest)
        {
            return Update::WeightedSumData<std::vector<MetadataFlag>, T>(flags, source, source, 1., 0., dest);
        }

        template<typename MDType>
        static TaskStatus WeightedSumDataFace(const std::vector<MDType> &flags, MeshData<Real> *in1, MeshData<Real> *in2, const Real w1, const Real w2,
                                MeshData<Real> *out)
        {
            Kokkos::Profiling::pushRegion("Task_WeightedSumDataFace");
            const auto &x = in1->PackVariables(flags);
            const auto &y = in2->PackVariables(flags);
            const auto &z = out->PackVariables(flags);
            parthenon::par_for(
                DEFAULT_LOOP_PATTERN, "WeightedSumDataFace", DevExecSpace(), 0, x.GetDim(5) - 1, 0,
                x.GetDim(4) - 1, 0, x.GetDim(3) - 1, 0, x.GetDim(2) - 1, 0, x.GetDim(1) - 1,
                KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
                    // TOOD(someone) This is potentially dangerous and/or not intended behavior
                    // as we still may want to update (or populate) z if any of those vars are
                    // not allocated yet.
                    if (x.IsAllocated(b, l) && y.IsAllocated(b, l) && z.IsAllocated(b, l)) {
                        z(b, F1, l, k, j, i) = w1 * x(b, F1, l, k, j, i) + w2 * y(b, F1, l, k, j, i);
                        z(b, F2, l, k, j, i) = w1 * x(b, F2, l, k, j, i) + w2 * y(b, F2, l, k, j, i);
                        z(b, F3, l, k, j, i) = w1 * x(b, F3, l, k, j, i) + w2 * y(b, F3, l, k, j, i);
                    }
                });
            Kokkos::Profiling::popRegion(); // Task_WeightedSumDataFace
            return TaskStatus::complete;
        }

        /**
         * Scale a variable by 'norm'.
         * Mostly makes things easier to read.
         */
        static TaskStatus Scale(std::vector<std::string> vars,  MeshData<Real>* source, Real norm)
        {
            return Update::WeightedSumData<std::vector<std::string>, MeshData<Real>>(vars, source, source, norm, 0., source);
        }
        static TaskStatus ScaleFace(std::vector<std::string> vars,  MeshData<Real>* source, Real norm)
        {
            return WeightedSumDataFace(vars, source, source, norm, 0., source);
        }

        /**
         * Replace Parthenon's `FluxDivergence` with a more adaptable version: pack on a customizable set
         * of flags, optionally include a halo around physical zones (useful for predicting bad zones in FOFC)
         */
        static TaskStatus FluxDivergence(MeshData<Real> *in_obj, MeshData<Real> *dudt_obj,
                                  std::vector<MetadataFlag> flags = {Metadata::WithFluxes, Metadata::Cell},
                                  int halo=0)
        {
            const auto &vin = in_obj->PackVariablesAndFluxes(flags);
            auto dudt = dudt_obj->PackVariables(flags);

            const IndexRange3 b = KDomain::GetRange(in_obj, IndexDomain::interior, -halo, halo);

            const int ndim = vin.GetNdim();
            parthenon::par_for(
                DEFAULT_LOOP_PATTERN, "FluxDivergenceMesh", DevExecSpace(), 0, vin.GetDim(5) - 1, 0,
                vin.GetDim(4) - 1, b.ks, b.ke, b.js, b.je, b.is, b.ie,
                KOKKOS_LAMBDA(const int m, const int l, const int k, const int j, const int i) {
                    if (dudt.IsAllocated(m, l) && vin.IsAllocated(m, l)) {
                        const auto &coords = vin.GetCoords(m);
                        const auto &v = vin(m);
                        dudt(m, l, k, j, i) = Update::FluxDivHelper(l, k, j, i, ndim, coords, v);
                    }
                }
            );

            return TaskStatus::complete;
        }

};
