/*
 *  File: variables.hpp
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

#include "pack/default_names.hpp"

namespace rays {

PAR_SWARMVAR(parthenon::Real, swarm, t);
PAR_SWARMVAR(parthenon::Real, swarm, k);
PAR_SWARMVAR(parthenon::Real, rays, path_len);
PAR_SWARMVAR(int, rays, camera_id);
PAR_SWARMVAR(int, rays, camera_i);
PAR_SWARMVAR(int, rays, camera_j);
PAR_SWARMVAR(int, rays, nstep_geo);
PAR_SWARMVAR(int, rays, nstep_rad);
PAR_SWARMVAR(int, rays, stop_flag);
PAR_SWARMVAR(int, rays, at_camera);
PAR_SWARMVAR(parthenon::Real, rays, xpath);
PAR_SWARMVAR(parthenon::Real, rays, kpath);
PAR_SWARMVAR(parthenon::Real, rays, dlpath);
PAR_SWARMVAR(parthenon::Real, rays, I);
PAR_SWARMVAR(parthenon::Real, rays, Nr);
PAR_SWARMVAR(parthenon::Real, rays, Ni);

}
