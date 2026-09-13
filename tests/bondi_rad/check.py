import numpy as np
import os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import pyharm

c = 2.99792458e10
mp = 1.672621777e-24  
kb = 1.3806488e-16 
mu = 1.0

REF_DIR = "./ref_data"
REF_VARS = ['rho', 'T', 'Erf', 'Fr']


def load_reference(test_num):
    fname = os.path.join(REF_DIR, "bondi_rad{}_512.phdf".format(test_num))
    x_ref, rho, T, Erf, Fr, u1 = read_bondi_dump(fname)
    return x_ref, {'rho': rho, 'T': T, 'Erf': Erf, 'Fr': Fr}


def radiation_fluid_frame(gcov, gcon, ucon_gas, ucov_gas, Erf, uvec_rad, r):
    N1 = Erf.shape[0]
    qsq = (gcov[:, 1, 1] * uvec_rad[0]**2 + gcov[:, 2, 2] * uvec_rad[1]**2 +
           gcov[:, 3, 3] * uvec_rad[2]**2 +
           2. * (gcov[:, 1, 2] * uvec_rad[0] * uvec_rad[1] +
                 gcov[:, 1, 3] * uvec_rad[0] * uvec_rad[2] +
                 gcov[:, 2, 3] * uvec_rad[1] * uvec_rad[2]))
    Gamma_rad = np.sqrt(1. + qsq)
    alpha = 1. / np.sqrt(-gcon[:, 0, 0])

    u_rad_con = np.zeros((4, N1))
    u_rad_con[0] = Gamma_rad / alpha
    for mu in range(1, 4):
        u_rad_con[mu] = uvec_rad[mu - 1] - Gamma_rad * alpha * gcon[:, 0, mu]

    R_munu = (4. / 3. * Erf[None, None, :] *
              u_rad_con[:, None, :] * u_rad_con[None, :, :] +
              1. / 3. * Erf[None, None, :] * gcon.transpose(1, 2, 0))

    E_fluid = np.einsum('ij...,i...,j...->...', R_munu, ucov_gas, ucov_gas)
    F_con = -np.einsum('ij...,j...->i...', R_munu, ucov_gas) - E_fluid[None, :] * ucon_gas
    F_r_fluid = F_con[1] * r
    return E_fluid, F_r_fluid


def read_bondi_dump(fname):
    dump = pyharm.load_dump(fname)

    g = dump.grid
    Xall = g.coord_all()
    N1 = dump['rho'].shape[0]
    gcov = np.array([g.coords.gcov(Xall[:, i, 0, 0]) for i in range(N1)])
    gcon = np.array([g.coords.gcon(Xall[:, i, 0, 0]) for i in range(N1)])

    temp_scale = mu * mp * c**2 / kb

    x = dump['r'][:, 0, 0]
    rho_a = dump['rho'][:, 0, 0]
    T_a = dump['T'][:, 0, 0] * temp_scale
    u1_a = dump['u^r'][:, 0, 0]

    ucon_gas = dump['ucon'][:, :, 0, 0]
    ucov_gas = dump['ucov'][:, :, 0, 0]
    Erf_gasframe = dump['prims.u_rad'][:, 0, 0]

    with h5py.File(fname, "r") as f:
        uvec_rad_raw = f["prims.uvec_rad"][:]
    uvec_rad_raw = np.concatenate(uvec_rad_raw, axis=-1)
    uvec_rad = uvec_rad_raw[:, 0, 0, :]

    Erad_fluid, Fr_fluid = radiation_fluid_frame(
        gcov, gcon, ucon_gas, ucov_gas, Erf_gasframe, uvec_rad, x)

    o = np.argsort(x)
    x, rho_a, T_a, u1_a, Erad_fluid, Fr_fluid = \
        x[o], rho_a[o], T_a[o], u1_a[o], Erad_fluid[o], Fr_fluid[o]

    return x, rho_a, T_a, Erad_fluid, Fr_fluid, u1_a


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
        x, rho, T, Erf, Fr, u1 = read_bondi_dump(
            os.path.join(filesdir, 'bondi_rad{}.out0.final.res{:d}.phdf'.format(test_num, res)))
        test_vals = {'rho': rho, 'T': T, 'Erf': Erf, 'Fr': Fr, 'u1': u1}

        mask = x <= 1e4
        for v in REF_VARS:
            ref_interp = np.interp(x, x_ref, ref[v])
            L1[v].append(np.mean(np.abs(test_vals[v][mask] - ref_interp[mask])))

        fig, axs = plt.subplots(len(REF_VARS), 1, figsize=(7, 12), sharex=True)
        panel_labels = {'rho': r'$\rho$', 'T': r'$T_{\rm gas}$', 'Erf': r'$E_{rf}$',
                         'Fr': r'$\tilde{F}^r$', 'u1': r'$u^r$'}
        for ax, v in zip(axs, REF_VARS):
            ax.plot(x, np.abs(test_vals[v]), '.-', ms=2, label="res={}".format(res))
            ax.plot(x_ref, np.abs(ref[v]), 'k--', lw=1, label="reference (res={})".format(ref_res))
            ax.set_ylabel(panel_labels[v], fontsize=14)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(right=1e4)
        axs[0].legend()
        axs[-1].set_xlabel('r')
        fig.suptitle("Bondi rad test {} (res={})".format(test_num, res))
        fig.tight_layout()
        plt.savefig(os.path.join(plotsdir, "bondi_rad_test{}_{}.png".format(test_num, res)))
        plt.close(fig)

    fail = 0
    powerfits = {}
    fit_mask = resolutions != ref_res
    for v in REF_VARS:
        L1[v] = np.maximum(np.array(L1[v]), 1e-300)
        powerfits[v] = np.polyfit(np.log(resolutions[fit_mask]), np.log(L1[v][fit_mask]), 1)[0]
        print("test{} {} Powerfit: {} L1: {}".format(test_num, v, powerfits[v], L1[v]))
        if powerfits[v] > -1.20:
            fail = 1

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    colors = {'rho': 'darkblue', 'T': 'orange', 'Erf': 'crimson', 'Fr': 'purple', 'u1': 'darkgreen'}
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
    plt.title("Bondi rad test {}".format(test_num))
    plt.legend()
    plt.savefig(os.path.join(plotsdir, 'bondi_rad_test{}_convergence.png'.format(test_num)), dpi=200)
    plt.close()

    exit(fail)
