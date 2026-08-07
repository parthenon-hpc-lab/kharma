import numpy as np
import os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyharm

# Reference solutions for all four PLUTO-paper radiative M1 shock tube
# tests (Melon Fuksman & Mignone 2019
REF_DIR = "ref_data"
REF_VARS = ['rho', 'pg', 'u1', 'Erf', 'Fx']


def load_reference(test_num):
    d = os.path.join(REF_DIR, "test{}".format(test_num))
    x_ref = np.loadtxt(os.path.join(d, "shock_soln_xCoords.txt"))
    ref = {}
    for v in REF_VARS:
        ref[v] = np.loadtxt(os.path.join(d, "shock_soln_{}.txt".format(v)))
    return x_ref, ref


def read_shock_dump(fname):
    dump = pyharm.load_dump(fname)

    gamma = dump['gam']
    x = dump['x'][:, 0, 0]
    rho_a = dump['rho'][:, 0, 0]
    pg_a = (gamma - 1.) * dump['u'][:, 0, 0]
    u1_a = dump['ucon'][1, :, 0, 0]
    erad_a = dump['u_rad'][:, 0, 0]
    u1rad_a = dump['uvec_rad'][0, :, 0, 0]

    o = np.argsort(x)
    x, rho_a, pg_a, u1_a, erad_a, u1rad_a = \
        x[o], rho_a[o], pg_a[o], u1_a[o], erad_a[o], u1rad_a[o]

    u0_rad = np.sqrt(1. + u1rad_a**2)
    u0_gas = np.sqrt(1. + u1_a**2)

    R00 = (4. / 3.) * erad_a * u0_rad**2 - (1. / 3.) * erad_a
    R01 = (4. / 3.) * erad_a * u0_rad * u1rad_a
    R11 = (4. / 3.) * erad_a * u1rad_a**2 + (1. / 3.) * erad_a

    fx_a = R01 * u0_gas - R11 * u1_a - (R00 * u0_gas**2 - 2 * R01 * u0_gas * u1_a + R11 * u1_a**2) * u1_a

    return x, rho_a, pg_a, u1_a, erad_a, u1rad_a, fx_a


if __name__ == '__main__':
    plotsdir = sys.argv[1]
    filesdir = sys.argv[2]
    test_num = int(sys.argv[3])
    resolutions = [int(r) for r in sys.argv[4].split(',')]
    resolutions = np.array(resolutions)

    x_ref, ref = load_reference(test_num)

    L1 = {v: [] for v in REF_VARS}

    for res in resolutions:
        x, rho, pg, u1, erf, u1rad, fx = read_shock_dump(
            os.path.join(filesdir, 'shock_test{}.out0.final.res{:d}.phdf'.format(test_num, res)))
        test_vals = {'rho': rho, 'pg': pg, 'u1': u1, 'Erf': erf, 'Fx': fx}

        for v in REF_VARS:
            # Interpolate the fine (3200-zone) reference onto this trial's
            # (coarser) grid.
            ref_interp = np.interp(x, x_ref, ref[v])
            L1[v].append(np.mean(np.abs(test_vals[v] - ref_interp)))

        plt.figure(figsize=(6, 6))
        plt.plot(x, rho, '.-', ms=2, label="res={}".format(res))
        plt.plot(x_ref, ref['rho'], 'k--', lw=1, label="reference (res=3200)")
        plt.xlabel('x'); plt.ylabel(r'$\rho$')
        plt.legend()
        plt.savefig(os.path.join(plotsdir, "rad_shocktube_test{}_rho_{}.png".format(test_num, res)))
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.plot(x, fx, '.-', ms=2, label="res={}".format(res))
        plt.plot(x_ref, ref['Fx'], 'k--', lw=1, label="reference (res=3200)")
        plt.xlabel('x'); plt.ylabel(r'$\tilde{F}^x$ (gas-comoving frame)')
        plt.legend()
        plt.savefig(os.path.join(plotsdir, "rad_shocktube_test{}_Fx_{}.png".format(test_num, res)))
        plt.close()

    fail = 0
    powerfits = {}
    for v in REF_VARS:
        L1[v] = np.array(L1[v])
        powerfits[v] = np.polyfit(np.log(resolutions), np.log(L1[v]), 1)[0]
        print("test{} {} Powerfit: {} L1: {}".format(test_num, v, powerfits[v], L1[v]))
        # These bounds were chosen heuristically.
        if powerfits[v] < -1.5 or powerfits[v] > -0.5:
            fail = 1

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    colors = {'rho': 'darkblue', 'pg': 'orange', 'u1': 'darkgreen', 'Erf': 'crimson', 'Fx': 'purple'}
    for v in REF_VARS:
        ax.plot(resolutions, L1[v], color=colors[v], marker='^', markersize=8, label=v)
    mid = len(resolutions) // 2
    amp = L1['rho'][mid] * float(resolutions[mid])
    ax.plot([resolutions[0], resolutions[-1]],
            amp * np.asarray([resolutions[0], resolutions[-1]], dtype=float)**(-1.0),
            color='k', linestyle='dashed', label='$N^{-1}$')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Resolution')
    plt.ylabel('L1 Norm (vs. 3200-zone reference)')
    plt.title("Shocktube test {}".format(test_num))
    plt.legend()
    plt.savefig(os.path.join(plotsdir, 'rad_shocktube_test{}_convergence.png'.format(test_num)), dpi=200)
    plt.close()

    exit(fail)
