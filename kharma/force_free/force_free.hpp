/* 
 *  File: force_free.hpp
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
#include "grmhd_functions.hpp"
#include "invert_template.hpp"
#include "types.hpp"

/**
 * Force-free and hybrid force-free electrodynamics, stolen from Chael (2024) and koral,
 * electromagnetic theory from McKinney (2006)
 */
namespace Force_Free {

// Theory/governing equations for determining velocity parallel to B field
enum class ParallelTheory{zero=5, cold=4, entropy=3, entropy_mhd=2, hot_mhd=1};
// Type of internal energy recovery
enum class Etype{entropy=0, hot};

#define FF_CONSTANTS_LEN 6
// Presuming I don't want this. Remove when sure
//#define NOLOGINS

// Denote inversion failures (pflags)
// This enum should grow to cover any inversion algorithm
enum class FFFlag{none=0, success, neg_input, max_iter, bad_damp, bad_w, neg_rho, neg_u, neg_rhou};

static const std::map<int, std::string> flag_names = {
    {(int) FFFlag::success, "Successful Force-Free"},
    {(int) FFFlag::neg_input, "Negative input"},
    {(int) FFFlag::max_iter, "Hit max iter"},
    {(int) FFFlag::bad_damp, "Unsuccessful damping"},
    {(int) FFFlag::bad_w, "Invalid solved W"},
    {(int) FFFlag::neg_rho, "Negative rho"},
    {(int) FFFlag::neg_u, "Negative U"},
    {(int) FFFlag::neg_rhou, "Negative rho & U"}
};

int CountFFFlags(MeshData<Real> *md);

/**
 * Declare new velocities
 */
std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

/**
 * TODO
 */
TaskStatus BlockUtoP(MeshBlockData<Real> *md, IndexDomain domain, bool coarse=false);
inline TaskStatus MeshUtoP(MeshData<Real> *md, IndexDomain domain, bool coarse) {
    Flag("MeshUtoP");
    for (int i=0; i < md->NumBlocks(); ++i)
        BlockUtoP(md->GetBlockData(i).get(), domain, coarse);
    EndFlag();
    return TaskStatus::complete;
}

/**
 * TODO
 */
TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md);

/**
 * Apply ceiling to force-free parallel velocity
 */
void ApplyFloors(MeshBlockData<Real> *mbd, IndexDomain domain);

// Functions optimized by the U2P solver below,
// reflecting hot gas or entropy conservation/adiabatic limit
template<Etype etype>
KOKKOS_INLINE_FUNCTION void f_u2p_parallel(const Real W, const Real cons[FF_CONSTANTS_LEN],
                                            Real &f, Real &df, Real &err);

template<>
KOKKOS_INLINE_FUNCTION void f_u2p_parallel<Etype::hot>(const Real W, const Real cons[FF_CONSTANTS_LEN],
                                Real &f, Real &df, Real &err)
{

  // constants
  const Real D = cons[0];
  const Real Y = cons[1];
  const Real Z = cons[2];
  const Real gammam2_perp = cons[3];
  const Real afac = cons[4];

  const Real vparsq = SQR(Y)/SQR(W);
  const Real gammam2 = gammam2_perp - vparsq;
  const Real gammam1 = m::sqrt(gammam2);

  const Real rho = D * gammam1;
  const Real pgas = afac * W * gammam2 - afac*rho;

  const Real drhodW = D * vparsq/(W*gammam1);
  const Real dpgasdW = afac * (gammam2_perp + vparsq - drhodW);
  
  // root function
  f = Z + W - pgas;
  // error
  err = m::abs(f) / (m::abs(pgas) + m::abs(W) + m::abs(Z));
  // derivitive
  df = 1. - dpgasdW;
}

template<>
KOKKOS_INLINE_FUNCTION void f_u2p_parallel<Etype::entropy>(const Real W, const Real cons[FF_CONSTANTS_LEN],
                                    Real &f, Real &df, Real &err)
{
    // constants
    const Real D = cons[0];
    const Real Y = cons[1];
    //const Real Z = cons[2];
    const Real gammam2_perp = cons[3];
    const Real afac = cons[4];
    const Real Sc = cons[5];

    const Real vparsq = SQR(Y)/SQR(W);
    const Real gammam2 = gammam2_perp - vparsq; //1/gamma^2
    const Real gammam1 = m::sqrt(gammam2);
    const Real rho = D * gammam1;
    const Real pgas = afac * (W * gammam2 - rho);

    // TODO pass gamma in
    const Real pgamma = 1. / (1. - afac); //=GAMMA; (should be consistent);
    
    const Real drhodW = D*vparsq/(W*gammam1);
    const Real dpgasdW = afac*(gammam2_perp + vparsq - drhodW);

#ifdef NOLOGINS
    // specific entropy (without leading factor of rho)
    const Real scalc = pgas / m::pow(rho, pgamma) / (pgamma - 1.);
    const Real dscalcdrho = -pgamma * pgas / m::pow(rho,pgamma + 1.) / (pgamma - 1.);
    const Real  dscalcdpgas = 1. / m::pow(rho, pgamma) / (pgamma - 1.);
#else
    const Real indexn = 1. / (pgamma-1.);
    const Real scalc = m::log(m::pow(pgas, indexn) / m::pow(rho, indexn+1.));
    const Real dscalcdrho = -1. * (indexn + 1.) / rho;
    const Real dscalcdpgas = indexn / pgas;
#endif

    // root function
    f = scalc * D - Sc; 
    // error
    err = m::abs(f) / (m::abs(scalc * D) + m::abs(Sc));
    df = D * (dscalcdrho * drhodW + dscalcdpgas * dpgasdW);
}

/**
 * Solver for parallel momentum when setting GRMHD primitive variables
 * based on FF solution conserved variables.
 */
template<Etype etype>
KOKKOS_INLINE_FUNCTION FFFlag u2p_solver_ff_parallel(Real &W, Real cons[6])
{
    Real conv = 1e-8;
    int max_iter = 100;
    Real damp_increase_fac = 2.;
    int damp_max_iter = 500;

    // TODO pass the constants rather than magic array
    // TODO make above vars into parameters
    // TODO can we Kastaun here?
    Real D = cons[0];
    Real Y = cons[1];
    Real Z = cons[2];
    Real gammam2_perp = cons[3];
    Real afac = cons[4];
    Real Sc = cons[5];

    // if(verbose > 1) {
    //     printf("In parallel entropy solver\n");
    //     printf("%e %e %e %e %e %e\n",D,Y,Z,gammam2_perp,afac,Sc);
    // }

    Real f0 = 0., dfdW = 0.;
    Real err; // TODO Set to std::max or something?
    
    // Make sure that W is large enough so that v^2 < 1 :
    // Test if initial guess is out of bounds and damp if so
    int i_increase = 0;
    do {
        f_u2p_parallel<etype>(W, cons, f0, dfdW, err);

        Real vpar2guess = SQR(Y)/SQR(W);
        if (((gammam2_perp - vpar2guess) < 0
        || !isfinite(f0) || !isfinite(dfdW))
        && (i_increase < damp_max_iter)) {
            // if(verbose>1) {
            //     printf("error in init W : %e -> %e (%e %e %e)\n",W,damp_increase_fac*W,
            //         gammam2_perp-vpar2guess,f0,dfdW);
            //     printf("D %e Y %e Z %e gammam2 %e a %e Sc %e\n",D,Y,Z,gammam2_perp,afac,Sc);
            //         getch();
            // }
            W *= damp_increase_fac;
            i_increase++;
            continue;
        } else {
            break;
        }
    } while(1);
    
    if (i_increase >= damp_max_iter) {
        // if(verbose>1) {
        //     printf("failed to find initial W in parallel solver\n");
        //     printf("W : %e->%e | (%e %e %e)\n",Wguess,W,gammam2_perp-vpar2guess,f0,dfdW);
        // }
        return FFFlag::bad_damp;
    }

    f_u2p_parallel<etype>(W, cons, f0, dfdW, err);
    // if(verbose>1) printf("\ninitial W:%e\n",W);
    // if(verbose>1) printf("initial f:%e\n",f0);
    // if(verbose>1) printf("initial err:%e\n\n",err);

    // 1d Newton solver
    int iter = 0;
    do {
        Real Wprev=W;
        iter++;

        f_u2p_parallel<etype>(W, cons, f0, dfdW, err);
        
        //if(verbose>1) printf("%d parallel solver: %e %e %e %e\n",iter,W,f0,dfdW,err);

        // get out of fixed point
        if(dfdW == 0.) {
            W *= 1.1;
            continue;
        }
        
        Real Wnew = W - f0 / dfdW;
        Real dumpfac = 1.;
        
        // test if goes out of bounds and damp solution if so
        int idump = 0;
        do {
            Real f0tmp = 0., dfdWtmp = 0., errtmp = 0.;
            f_u2p_parallel<etype>(W, cons, f0tmp, dfdWtmp, errtmp);
        
            Real vpar2guess = SQR(Y) / SQR(Wnew);
            if(((gammam2_perp - vpar2guess) < 0
                || !isfinite(f0tmp) || !isfinite(dfdWtmp) || !isfinite(dfdWtmp))
                && (idump < damp_max_iter)) {
                idump++;
                dumpfac /= 2.;
                Wnew = W - dumpfac * f0 / dfdW;
                continue;
            } else {
                break;
            }
        } while(1);
        
        if(idump >= damp_max_iter) {
            //if(verbose>0) printf("damped unsuccessfuly in parallel solver\n");
            //getchar();
            return FFFlag::bad_damp;
        }
        
        W = Wnew;
        
        // if(fabs(W) > BIG) {
        //   if(verbose>0) printf("W has gone out of bounds in parallel solver\n");
        //   //getchar();
        //   return -202;
        // }
        
        if (err < conv ||
            (m::abs((W - Wprev) / Wprev) < conv && err < m::sqrt(conv))) {
            break;
        }
    } while(iter < max_iter);
  
    if(iter >= max_iter) {
        // if(verbose>0) {
        //     printf("iter exceeded in parallel u2p_solver \n");
        //     printf("W %e f0 %e dfdW %e\n",W,f0,dfdW);
        //     printf("D %e Y %e Z %e gm2 %e a %e Sc %e\n",D,Y,Z,gammam2_perp,afac,Sc);
        // }
        return FFFlag::max_iter;
    }
  
    if(!isfinite(W))
    {
        // if(verbose > 0) printf("nan/inf W in parallel u2p_solver: \n");
        return FFFlag::bad_w;
    }

    if((gammam2_perp - SQR(Y)/SQR(W)) < 0) {
        //if(verbose > 0) printf("final W out of bounds in parallel solver\n");
        return FFFlag::bad_w;
    }

    // if(verbose > 1) {
    //     printf("final W %e\n",W);
    //     f_u2p_parallel<etype>(W, cons, f0, dfdW, err);
    //     printf("final f %e\n",f0);
    //     printf("final err %e\n\n",err);
    // }
    return FFFlag::success;
}

}
