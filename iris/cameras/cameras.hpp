/* 
 *  File: cameras.hpp
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
#include "units.hpp"

// KHARMA headers
#include "decs.hpp"
#include "types.hpp"

using namespace parthenon;

class Camera {
    public:
        int nx, ny;
        // KS coordinate location
        double r, th, phi;
        // Native coordinate location
        double X[GR_DIM], K[GR_DIM]; // X defined for pinhole, K for planar
        // FOV in r_g, radian, from Earth
        double dx, dy, fovx, fovy, fovx_dsource, fovy_dsource;
        // Distance
        double dsource;
        // Scale from intensity to flux in Jy
        double scale;
        // Other parameters passed directly to geodesic creation
        double xoff, yoff, rotcam, frequency;
        // Convention for Stokes Q & U.  0 == normal, EVPA East of North
        int qu_conv;

        Camera(CoordinateEmbedding &coords, ParameterInput *pin, std::string block) {
            // Size, image
            nx = pin->GetOrAddInteger(block, "nx", 160);
            ny = pin->GetOrAddInteger(block, "ny", 160);

            // Location/FOV
            r = pin->GetReal(block, "r");
            // Convert to radian
            th = pin->GetReal(block, "theta") / 180. * M_PI;
            phi = pin->GetReal(block, "phi") / 180. * M_PI;
            double Xembed[GR_DIM] = {0., r, th, phi};
            coords.coord_to_native(Xembed, X);

            dsource = pin->GetOrAddReal(block, "dsource", DM87_PC) * PC;

            // set DX/DY using fov_dsource if possible, otherwise DX, otherwise old default
            // TODO accept fovx/fovy directly?  TODO arctan?
            // TODO check for square pixels at the end?
            double L_unit = pin->GetReal("model", "L_unit");
            double fov_to_d = dsource / L_unit / MUAS_PER_RAD;
            if (pin->DoesParameterExist(block, "fovx_dsource")) { // FOV was specified
                fovx_dsource = pin->GetReal(block, "fovx_dsource");
                if (pin->DoesParameterExist(block, "fovy_dsource")) {
                    fovy_dsource = pin->GetReal(block, "fovy_dsource");
                } else {
                    fovy_dsource = fovx_dsource;
                }
                dx = fovx_dsource * fov_to_d;
                dy = fovy_dsource * fov_to_d;
            } else if (pin->DoesParameterExist(block, "dx")) {
                dx = pin->GetReal(block, "dx");
                if (pin->DoesParameterExist(block, "dy")) {
                    dy = pin->GetReal(block, "dy");
                } else {
                    dy = dx;
                }
                fovx_dsource = dx / fov_to_d;
                fovy_dsource = dy / fov_to_d;

            } else {
                std::cerr << "No FOV was specified for camera " << block
                          << ". Using default 160muas!" << std::endl;
                fovx_dsource = 160.;
                fovy_dsource = 160.;
                dx = fovx_dsource * fov_to_d;
                dy = fovy_dsource * fov_to_d;
            }
            // Set the *camera* fov values, in radian.
            fovx = dx / r;
            fovy = dy / r;
            // And the scale intensity->flux
            scale = (dx * L_unit / nx) * (dy * L_unit / ny) / (dsource * dsource) / JY;

            xoff = pin->GetOrAddReal(block, "xoff", 0.0);
            yoff = pin->GetOrAddReal(block, "yoff", 0.0);
            rotcam = pin->GetOrAddReal(block, "rotcam", 0.0);
            frequency = pin->GetOrAddReal(block, "frequency", 2.3e11);
            qu_conv = pin->GetOrAddReal(block, "qu_conv", 0);
        }

        void print() {
            printf("Xcam[] = %e %e %e %e\n", X[0], X[1], X[2], X[3]);
            printf("intensity [cgs] to flux per pixel [Jy] conversion: %g\n", scale);
            printf("Dsource: %g [cm]\n", dsource);
            printf("Dsource: %g [kpc]\n", dsource/(1.e3*PC));
            printf("FOVx, FOVy: %g %g [GM/c^2]\n", dx, dy);
            printf("FOVx, FOVy: %g %g [rad]\n", fovx_dsource / MUAS_PER_RAD, fovy_dsource / MUAS_PER_RAD);
            printf("FOVx, FOVy: %g %g [muas]\n", fovx_dsource, fovy_dsource);
            printf("Resolution: %dx%d\n", nx, ny);
        }
};

/**
 * Parthenon package for handling cameras together:
 * 1. generates lists of X,K for tracing all pixels of all cameras, with no distinction
 * 2. splits pixel lists out into images, calculate observer frame Stokes params
 * 
 */
namespace Cameras {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

TaskStatus InitGeodesics(MeshBlock* pmb);

TaskStatus WriteCameras(MeshBlock* pmb);

};
