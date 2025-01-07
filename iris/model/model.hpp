/* 
 *  File: model.hpp
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

// Iris headers
#include "coefficients.hpp"
#include "file.hpp"
#include "ipolarray.hpp"
#include "tetrads.hpp"

// KHARMA headers
#include "decs.hpp"
#include "types.hpp"

using namespace parthenon;

/**
 */
namespace Model {

class StopCondition {
    public:
        // Whether to stop on first midplane cross
        bool midplane;
        // Maximum radius to integrate normally
        GReal x1max_geo, x1min_geo;
        // Midplane location
        GReal x2mid;

        StopCondition(CoordinateEmbedding coords, ParameterInput *pin) {
            midplane = pin->GetString("model", "type") == "thin_disk";
            double default_out = pin->DoesParameterExist("coordinates", "r_out") ?
                                 m::min(pin->GetReal("coordinates", "r_out"), 100.) :
                                 100.;
            x1max_geo = coords.r_to_native(pin->GetOrAddReal("model", "rmax_geo", default_out));
            x1min_geo = coords.r_to_native(pin->GetOrAddReal("model", "rmin_geo",
                                                             coords.get_horizon() + 0.1));
            x2mid = (coords.stopx(2) - coords.startx(2)) / 2;
        }
};

enum class StopFlag{none, r_inner, r_outer, thin_disk, meshblock};

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus SetStokesThindisk(MeshBlock *pmb);

/* condition for stopping the backwards-in-lambda
   integration of the photon geodesic */
KOKKOS_INLINE_FUNCTION bool stop_backward_integration(const GReal &x1i, const GReal &x2i,
                                                    const GReal &x3i, const GReal &x1f,
                                                    const GReal &x2f, const GReal &x3f,
                                                    const GReal &Kcon1f,
                                                    StopCondition stop_condition, int &stopflag)
{
    // Radial conditions:
    // Stop either when crossing the maximum outward, or the EH at all
    if ((x1i < stop_condition.x1max_geo && x1f > stop_condition.x1max_geo) ||
        (x1f > stop_condition.x1max_geo && Kcon1f < 0.)) {
        stopflag = (int) StopFlag::r_outer;
        return true;
    }
    if (x1f < stop_condition.x1min_geo) {
        stopflag = (int) StopFlag::r_inner;
        return true;
    }

    // Midplane condition: stop if crossing in either direction
    if (stop_condition.midplane && (
            (x2i < stop_condition.x2mid && x2f > stop_condition.x2mid) ||
            (x2i > stop_condition.x2mid && x2f < stop_condition.x2mid))) {
        stopflag = (int) StopFlag::thin_disk;
        return true;
    }

    return false;
}

KOKKOS_INLINE_FUNCTION void process_radiation(const GRCoordinates &G,
                                              Kokkos::complex<double> N_coord[GR_DIM][GR_DIM],
                                              double X[GR_DIM], double Kcon[GR_DIM],
                                              const VariablePack<Real> &P, const VarMap &m_p,
                                              const double &RHO_unit, const double &B_unit,
                                              const double &dlam, const bool zero_emission)
{
    const auto &coords = G.coords;
    // Tetrad frame
    double Ucon[GR_DIM], Ucov[GR_DIM], Bcon[GR_DIM], Bcov[GR_DIM];
    File::get_fourvectors(G, X, P, m_p, Ucon, Ucov, Bcon, Bcov);
    double Gcov[GR_DIM][GR_DIM];
    coords.gcov_native(X, Gcov);
    double Econ[GR_DIM][GR_DIM], Ecov[GR_DIM][GR_DIM];
    make_plasma_tetrad(Ucon, Kcon, Bcon, Gcov, Econ, Ecov);
    // N to tetrad
    Kokkos::complex<double> N_tetrad[GR_DIM][GR_DIM];
    N_to_tetrad(N_coord, Ecov, N_tetrad);
    // Tetrad to Stokes
    double SI, SQ, SU, SV;
    N_to_stokes(N_tetrad, SI, SQ, SU, SV);
    //printf("Initial Stokes: %g %g %g %g\n", SI, SQ, SU, SV);
    // Find j,a,r
    FitParams params;
    File::get_params(G, X, Kcon, Ucon, Ucov, Bcon, Bcov, RHO_unit, B_unit, P, m_p, params);
    //params.print();
    double jI, jQ, jU, jV, aI, aQ, aU, aV, rQ, rU, rV;
    if (zero_emission) {
        jI = 0; jQ = 0; jU = 0; jV = 0;
    }
    Coefficients::jar_calc(params, jI, jQ, jU, jV, aI, aQ, aU, aV, rQ, rU, rV);
    //printf("Coeffs: %g %g %g %g/%g %g %g %g/%g %g %g\n", jI, jQ, jU, jV, aI, aQ, aU, aV, rQ, rU, rV);
    // Evolve Stokes
    evolve_stokes(SI, SQ, SU, SV, jI, jQ, jU, jV, aI, aQ, aU, aV, rQ, rU, rV, dlam);
    //printf("New Stokes: %g %g %g %g\n", SI, SQ, SU, SV);
    // Stokes to tetrad
    stokes_to_N(SI, SQ, SU, SV, N_tetrad);
    // Tetrad to coord
    N_to_coord(N_tetrad, Econ, N_coord);
}

}
