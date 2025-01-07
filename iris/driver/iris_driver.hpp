/* 
 *  File: iris_driver.hpp
 *  
 *  BSD 3-Clause License
 *  
 *  Copyright (c) 2025, Iris contributors
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
#include "types.hpp"

using namespace parthenon;

/**
 * 
 */
class IrisDriver : public Driver {
    public:
        IrisDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm) : Driver(pin, app_in, pm) {
            InitializeOutputs();
        }

        static std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

        // Eliminate Parthenon's print statements when starting up the driver, we have a bunch of our own
        void PreExecute() override { timer_main.reset(); }

        // Wrapper for all things the invocation should do
        DriverStatus Execute() override;

        // The tasks required to make an image
        template <typename T>
        TaskCollection MakeTaskCollection(T &blocks);

        // And the PostExecute, so we can add a package callback here
        void PostExecute(DriverStatus status) override;
    
        static double elapsed_output() { return timer_output.seconds(); }

    protected:
        static Kokkos::Timer timer_output;
};
