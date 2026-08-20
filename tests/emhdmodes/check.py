import numpy as np
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

import pyharm

if __name__=='__main__':
    outputdir = './'

    NVAR = 10
    VARS = ['rho', 'u', 'u1', 'u2', 'u3', 'B1', 'B2', 'B3', 'q', 'dP']
    RES = [int(r) for r in sys.argv[1].split(",")]
    LONG = sys.argv[2]
    SHORT = sys.argv[3]


    # problem params
    var0 = np.zeros(NVAR)
    var0[0] = 1.
    var0[1] = 2.
    var0[5] = 0.1
    var0[6] = 0.3

    # L1 initialization
    L1  = np.zeros([len(RES), NVAR])
    fit = np.zeros([len(RES), NVAR])

    # perturbation (for 2D EMHD wave)
    dvar_cos = np.zeros(NVAR)
    dvar_cos[0] = -0.518522524082246
    dvar_cos[1] = 0.5516170736393813
    dvar_cos[2] = 0.008463122479547856
    dvar_cos[3] = -0.16175466371870734
    dvar_cos[5] = -0.05973794979640743
    dvar_cos[6] = 0.02986897489820372
    dvar_cos[8] = 0.5233486841539436
    dvar_cos[9] = 0.2909106062057657
    dvar_sin = np.zeros(NVAR)
    dvar_sin[0] = 0.1792647678001878
    dvar_sin[2] = -0.011862022608466367
    dvar_sin[3] = 0.034828080823603294
    dvar_sin[5] = 0.03351707506150924
    dvar_sin[6] = -0.016758537530754618
    dvar_sin[8] = -0.04767672501939603
    dvar_sin[9] = -0.02159452055336572

    # loop over RES
    for r in range(len(RES)):
        # load data
        dump = pyharm.load_dump("emhd_2d_"+str(RES[r])+"_end_"+SHORT+".phdf")


        params = dump.params
        amp = float(params['amp'])
        k1  = 2*np.pi # TODO record
        k2  = 4*np.pi

        real_omega  = params['omega_real']
        imag_omega  = params['omega_imag']
        higher_order_terms = params['higher_order_terms']
        conduction_alpha = params['conduction_alpha']
        viscosity_alpha = params['viscosity_alpha']
        gam = params['gam']
        t = dump['t']

        grid = dump.grid
        cos_phi = np.cos(k1*grid['x'] + k2*grid['y'] + imag_omega*t)
        sin_phi = np.sin(k1*grid['x'] + k2*grid['y'] + imag_omega*t)

        # compute analytic result
        var_analytic  = []
        for i in range(NVAR):
            var_analytic.append(var0[i] + ((amp*cos_phi*dvar_cos[i]) + (amp*sin_phi*dvar_sin[i])) * np.exp(real_omega*t))
        var_analytic = np.asarray(var_analytic)

        var_numerical = dump['prims']

        if higher_order_terms:
            print("Higher order terms enabled")
            Theta = (gam - 1.) * dump['UU'] / dump['RHO']
            cs2   = gam * (gam - 1.) * dump['UU'] / (dump['RHO'] + (gam * dump['UU']) )

            var_numerical[8,Ellipsis] *= np.sqrt(conduction_alpha * cs2 * dump['RHO'] * Theta**2)
            var_numerical[9,Ellipsis] *= np.sqrt(viscosity_alpha * cs2 * dump['RHO'] * Theta)

        for n in range(NVAR):
            L1[r,n] = np.mean(np.fabs(var_numerical[n] - var_analytic[n]))

    # MEASURE CONVERGENCE
    L1 = np.array(L1)
    powerfits = [0.,]*NVAR
    fail = 0
    for k in range(NVAR):
        if not (dvar_cos[k] == 0 and dvar_sin[k] == 0):
            powerfits[k] = np.polyfit(np.log(RES), np.log(L1[:,k]), 1)[0]
            print("Power fit {}: {} {}".format(VARS[k], powerfits[k], L1[:,k]))
            if powerfits[k] > -1.6 or powerfits[k] < -3.3:
                # Everything *should* converge at ~2, but we relax the reqt due to known behavior:
                # 1. B field in WENO seems to lag, at ~1.7
                # 2. Problems run under linear/MC seem to converge ~2.5 in most variables
                # 3. EMHD modes with ideal guess has ~3 convergence for rho
                fail = 1

    # plot
    colors = ['indigo', 'goldenrod', 'darkgreen', 'crimson', 'xkcd:blue', 'xkcd:magenta', 'green', 'xkcd:yellowgreen', 'xkcd:teal', 'xkcd:olive']
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)

    fig.suptitle(LONG)

    # loop over prims
    tracker = 0
    for n in range(NVAR):
        if abs((dvar_cos[n] != 0) or abs(dvar_sin[n] != 0)):
            ax.loglog(RES, L1[:,n], color=colors[n], marker='o', label=pyharm.pretty(VARS[n]))
            tracker += 1

    ax.loglog([RES[0], RES[-1]], 100*amp*np.asarray([float(RES[0]), float(RES[-1])])**(-2), color='k', linestyle='dashed', label='$N^{-2}$')
    plt.xscale('log', base=2)
    ax.legend()
    plt.savefig(os.path.join(outputdir, "emhd_linear_mode_convergence_"+SHORT+".png"), dpi=300)

    exit(fail)
