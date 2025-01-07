/* 
 *  File: main.cpp
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

// Iris headers
#include "iris.hpp"
#include "iris_driver.hpp"
#include "version.hpp"

// KHARMA headers
#include "decs.hpp"
#include "types.hpp"

// Parthenon headers
#include <parthenon/parthenon.hpp>

// Local headers
#include <fstream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

// Print warnings about configuration
#if DEBUG
#warning "Compiling with debug"

// Stacktrace on sigint. Amazingly useful
#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
void print_backtrace(int sig) {
  void *array[100];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 100);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}
#endif

using namespace parthenon;

/**
 * Main function for Iris.  Parses parameters, prints basics,
 * constructs the `Mesh`, calls the Driver.
 */
int main(int argc, char *argv[])
{
    ParthenonManager pman;

    // A couple of callbacks are Iris-wide single functions
    pman.app_input->ProcessPackages = Iris::ProcessPackages;
    // pman.app_input->ProblemGenerator = Iris::ProblemGenerator;
    // Ideally rays will never hit the coordinate singularity, but we
    // register reflecting conditions so Parthenon doesn't yell
    pman.app_input->RegisterDefaultReflectingBoundaryConditions();

    // Initialize Parthenon for MPI (also Kokkos, parses command line, etc.)
    Flag("ParthenonInit");
    auto manager_status = pman.ParthenonInitEnv(argc, argv);
    EndFlag();

    if(MPIRank0()) {
        // Always print the version header, because it's fun
        // TODO(BSP) proper banner w/refs, names
        const std::string &version = Iris::Version::GIT_VERSION;
        const std::string &branch = Iris::Version::GIT_REFSPEC;
        const std::string &sha1 = Iris::Version::GIT_SHA1;
        std::cout << std::endl;
        std::cout << "Starting Iris imager, version " << version << std::endl;
        std::cout << "Branch " << branch << ", commit hash: " << sha1 << std::endl;
        std::cout << std::endl;
        std::cout << "Iris is released under the BSD 3-clause license." << std::endl;
        std::cout << "Source code is available at https://github.com/AFD-Illinois/kharma/" << std::endl;
        std::cout << std::endl;
    }

    // Check the Parthenon init return code, initialize packages/mesh
    Flag("InitPackagesAndMesh");
    if (manager_status == ParthenonStatus::complete) {
        pman.ParthenonFinalize();
        return 0;
    }
    if (manager_status == ParthenonStatus::error) {
        pman.ParthenonFinalize();
        return 1;
    }
    auto pin = pman.pinput.get(); // All parameters in the input file or command line
    // InitPackagesEtc calls ProcessPackages, then constructs the Mesh
    pman.ParthenonInitPackagesAndMesh();
    // Now pull out the mesh and app_input as well for below
    auto pmesh = pman.pmesh.get(); // The mesh, with list of blocks & locations, size, etc
    auto papp = pman.app_input.get(); // The list of callback functions specified above
    EndFlag();

#if DEBUG
    // Replace Parthenon signal handlers with something that just prints a backtrace
    signal(SIGINT, print_backtrace);
    signal(SIGTERM, print_backtrace);
    signal(SIGSEGV, print_backtrace);
#endif

    // Iris is simpler than KHARMA: we just read pin here instead of using more packages
    const int &verbose = pin->GetInteger("debug", "verbose");
    if(MPIRank0() && verbose > 1) {
        // Write all parameters etc. to console if we should be especially wordy
        // Printed above the rest to stay out of the way
        if (verbose > 2) {
            // This dumps the full Kokkos config, useful for double-checking
            // that the compile did what we wanted
            parthenon::ShowConfig();
            pin->ParameterDump(std::cout);
        }

        // Print a list of variables as Parthenon used to (still does by default)
        std::cout << "Variables in use:\n" << *(pmesh->resolved_packages) << std::endl;

        // Print a list of all loaded packages.  Surprisingly useful for debugging init logic
        std::cout << "Packages in use: " << std::endl;
        for (auto package : pmesh->packages.AllPackages()) {
            std::cout << package.first << std::endl;
        }
        std::cout << std::endl;

        // Print the number of meshblocks and ranks in use
        std::cout << "Running with " << pmesh->nbtotal << " total meshblocks, " << MPINumRanks() << " MPI ranks." << std::endl;
        std::cout << "Blocks on rank " << MPIRank() << ": " << pmesh->block_list.size() << "\n" << std::endl;
    }
    // If very verbose, print # meshblocks on *every* rank, not just rank 0
    if (verbose > 2) {
        //MPIBarrier();
        if (MPIRank() > 0)
            std::cout << "Blocks on rank " << MPIRank() << ": " << pmesh->block_list.size() << "\n" << std::endl;
    }

    // Begin code block to ensure driver is cleaned up
    {
        // Pull out things we need to give the driver
        auto pin = pman.pinput.get(); // All parameters in the input file or command line

        //MPIBarrier();
        IrisDriver driver(pin, papp, pmesh);

        // Then execute the driver. This is a Parthenon function inherited by our KHARMADriver object,
        // which will call MakeTaskCollection, then execute the tasks on the mesh for each portion
        // of each step until a stop criterion is reached.
        Flag("driver.Execute");
        //MPIBarrier();
        auto driver_status = driver.Execute();
        EndFlag();
    }

    // Parthenon cleanup includes Kokkos, MPI
    Flag("ParthenonFinalize");
    pman.ParthenonFinalize();
    EndFlag();

    return 0;
}
