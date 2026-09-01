/*
 *  File: gaussian.cpp
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

#include "gaussian.hpp"
#include "problem.hpp"

#include <cmath>
#include <random>

float normalRand()
{
    // TODO this can definitely be Kokkosified
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0, 1};
    return d(gen);
}

#if USE_FFTW

#include "fftw3.h"

void create_grf(int Nx1, int Nx2, double lx1, double lx2, double* dv1, double* dv2)
{
    double dkx1 = 2 * M_PI / lx1;
    double dkx2 = 2 * M_PI / lx2;
    double Dx1 = lx1 / Nx1;
    double Dx2 = lx2 / Nx2;

    double kx1max = 2 * M_PI / (2 * Dx1);
    double kx2max = 2 * M_PI / (2 * Dx2);
    double k_peak = 4 * M_PI / lx1;

    fftw_complex *dvkx1, *dvkx2;
    dvkx1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx1 * Nx2);
    dvkx2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx1 * Nx2);
#pragma omp parallel for simd collapse(2)
    for (size_t i = 0; i < Nx1; i++) {
        for (size_t j = 0; j < Nx2; j++) {
            double retx1 = i * dkx1;
            double retx2 = j * dkx2;
            if (retx1 > kx1max) retx1 = retx1 - 2 * kx1max;
            if (retx2 > kx2max) retx2 = retx2 - 2 * kx2max;
            double curr_k_magn = pow(pow(retx1, 2) + pow(retx2, 2), 0.5);

            double pwr_spct = pow(curr_k_magn, 6) * exp(-8 * curr_k_magn / k_peak);
            if (curr_k_magn != 0) {
                retx1 /= curr_k_magn;
                retx2 /= curr_k_magn;
            }

            double noisy_dvkx1_real = pwr_spct * normalRand();
            double noisy_dvkx1_imag = pwr_spct * normalRand();
            double noisy_dvkx2_real = pwr_spct * normalRand();
            double noisy_dvkx2_imag = pwr_spct * normalRand();

            // real part of kx, using real part of dot product, and the kx component.
            // second line is imag part
            double adj_dvkx1_real =
                (retx1 * noisy_dvkx1_real + retx2 * noisy_dvkx2_real) * retx1;
            double adj_dvkx1_imag =
                (retx1 * noisy_dvkx1_imag + retx2 * noisy_dvkx2_imag) * retx1;
            double adj_dvkx2_real =
                (retx1 * noisy_dvkx1_real + retx2 * noisy_dvkx2_real) * retx2;
            double adj_dvkx2_imag =
                (retx1 * noisy_dvkx1_imag + retx2 * noisy_dvkx2_imag) * retx2;

            dvkx1[i * Nx1 + j][0] = noisy_dvkx1_real - adj_dvkx1_real;
            dvkx1[i * Nx1 + j][1] = noisy_dvkx1_imag - adj_dvkx1_imag;
            dvkx2[i * Nx1 + j][0] = noisy_dvkx2_real - adj_dvkx2_real;
            dvkx2[i * Nx1 + j][1] = noisy_dvkx2_imag - adj_dvkx2_imag;
        }
    }

    fftw_plan p_x1, p_x2;
    p_x1 = fftw_plan_dft_2d(
        Nx1, Nx2, dvkx1, dvkx1, FFTW_BACKWARD, FFTW_ESTIMATE); // in-place
    p_x2 = fftw_plan_dft_2d(Nx1, Nx2, dvkx2, dvkx2, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p_x1); // look for threads documentation
    fftw_execute(p_x2);

    fftw_destroy_plan(p_x1);
    fftw_destroy_plan(p_x2);
#pragma omp parallel for simd collapse(2)
    for (size_t i = 0; i < Nx1; i++) {
        for (size_t j = 0; j < Nx2; j++) {
            dv1[i * Nx1 + j] = dvkx1[i * Nx1 + j][0];
            dv2[i * Nx1 + j] = dvkx2[i * Nx1 + j][0];
        }
    }
    fftw_free(dvkx1);
    fftw_free(dvkx2);
}

#else

void create_grf(int Nx1, int Nx2, double lx1, double lx2, double* dv1, double* dv2)
{
    throw std::runtime_error("Attempted to use an FFT to generate a Gaussian random "
                             "field, but KHARMA was compiled without FFT support!");
}
#endif
