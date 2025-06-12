/* 
 *  File: ismr.hpp
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
#include "types.hpp"

#include <parthenon/parthenon.hpp>

/**
 * This package implements internal static mesh de-refinement by averaging variables next to the coordinate
 * poles, creating effective "larger zones" and allowing increases to the timestep, without any modification
 * to the existing data structures or block layout.
 * Currently it is limited to spherical coordinates: use Parthenon's block-based refinement otherwise.
 * Also note that the averaging operation corresponds to a first-order refinement/derefinement, possibly
 * compromising overall second-order convergence in the very near-polar region.
 * 
 * The operator averages variables only in the phi-direction nearing the pole, mapping 1:2 zones
 * for each of several levels.  In this way, zones near the pole can be much wider in phi without losing
 * resolution in r, theta.
 * 
 * Idea is taken from Matthew Liska/H-AMR (Liska+ 2022)
 * Implementation credit Hyerin Cho, please cite Cho+ 2024b (in prep) with description and first
 * use of this implementation.
 */
namespace ISMR {

/**
 * Initialize ISMR parameters
 */
std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

/**
 * Derefinement operation for fluid/cell-centered variables 
 */
TaskStatus DerefinePoles(MeshData<Real> *md);

}
