import numpy as np
import os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import pyharm

# Reference solutions for all four PLUTO-paper radiative M1 shock tube
# tests (Melon Fuksman & Mignone 2019
REF_DIR = "./ref_data"
REF_VARS = ['rho', 'pg', 'u1', 'Erf', 'Fx']


def load_reference(test_num):
    fname = os.path.join(REF_DIR, "shocktube{}.final.phdf".format(test_num))
    x_ref, rho, pg, u1, erf, u1rad, fx = read_shock_dump(fname)
    return x_ref, {'rho': rho, 'pg': pg, 'u1': u1, 'Erf': erf, 'Fx': fx}


def read_shock_dump(fname):
    dump = pyharm.load_dump(fname)

    gamma = dump['gam']
    x = dump['x'][:, 0, 0]
    rho_a = dump['rho'][:, 0, 0]
    pg_a = (gamma - 1.) * dump['u'][:, 0, 0]
    u1_a = dump['ucon'][1, :, 0, 0]
    erad_a = dump['prims.u_rad'][:, 0, 0]
    with h5py.File(fname, "r") as f:
        uvec_rad = f["prims.uvec_rad"][:]

    # Concatenate meshblocks along x1
    uvec_rad = np.concatenate(uvec_rad, axis=-1)
    u1rad_a = uvec_rad[0, 0, 0, :]

    o = np.argsort(x)
    x, rho_a, pg_a, u1_a, erad_a, u1rad_a = \
        x[o], rho_a[o], pg_a[o], u1_a[o], erad_a[o], u1rad_a[o]

    u0_rad = np.sqrt(1. + u1rad_a**2)

    fx_a = u1rad_a / u0_rad

    return x, rho_a, pg_a, u1_a, erad_a, u1rad_a, fx_a


if __name__ == '__main__':
    plotsdir = sys.argv[1]
    filesdir = sys.argv[2]
    test_num = sys.argv[3]
    resolutions = [int(r) for r in sys.argv[4].split(',')]
    resolutions = np.array(resolutions)

    x_ref, ref = load_reference(test_num)
    ref_res = len(x_ref)

    L1 = {v: [] for v in REF_VARS}

    for res in resolutions:
        x, rho, pg, u1, erf, u1rad, fx = read_shock_dump(
            os.path.join(filesdir, 'shock_test{}.out0.final.res{:d}.phdf'.format(test_num, res)))
        test_vals = {'rho': rho, 'pg': pg, 'u1': u1, 'Erf': erf, 'Fx': fx}

        for v in REF_VARS:
            # Interpolate the reference onto this trial's
            # (coarser) grid.
            ref_interp = np.interp(x, x_ref, ref[v])
            L1[v].append(np.mean(np.abs(test_vals[v] - ref_interp)))

        fig, axs = plt.subplots(len(REF_VARS), 1, figsize=(7, 12), sharex=True)
        panel_labels = {'rho': r'$\rho$', 'pg': r'$p_g$', 'u1': r'$u^1$',
                         'Erf': r'$E_{rf}$', 'Fx': r'$u^x_{rad}/u^t_{rad}$'}
        for ax, v in zip(axs, REF_VARS):
            ax.plot(x, test_vals[v], '.-', ms=2, label="res={}".format(res))
            ax.plot(x_ref, ref[v], 'k--', lw=1, label="reference (res={})".format(ref_res))
            ax.set_ylabel(panel_labels[v], fontsize=14)
        axs[0].legend()
        axs[-1].set_xlabel('x')
        fig.suptitle("Shocktube test {} (res={})".format(test_num, res))
        fig.tight_layout()
        plt.savefig(os.path.join(plotsdir, "rad_shocktube_test{}_{}.png".format(test_num, res)))
        plt.close(fig)

    fail = 0
    powerfits = {}
    fit_mask = resolutions != ref_res
    for v in REF_VARS:
        L1[v] = np.maximum(np.array(L1[v]), 1e-300)
        powerfits[v] = np.polyfit(np.log(resolutions[fit_mask]), np.log(L1[v][fit_mask]), 1)[0]
        print("test{} {} Powerfit: {} L1: {}".format(test_num, v, powerfits[v], L1[v]))
        if powerfits[v] > -0.90:
            fail = 1
        if test_num == '4a' and powerfits[v] < -0.5:
            fail = 0

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
    plt.ylabel('L1 Norm (vs. res={} reference)'.format(ref_res))
    plt.title("Shocktube test {}".format(test_num))
    plt.legend()
    plt.savefig(os.path.join(plotsdir, 'rad_shocktube_test{}_convergence.png'.format(test_num)), dpi=200)
    plt.close()

    exit(fail)
