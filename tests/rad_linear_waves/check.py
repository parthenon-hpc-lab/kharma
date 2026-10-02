import numpy as np
import os, sys, re
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import h5py
import pyharm

WAVETYPES = ['sonic', 'slow', 'fast']
REGIMES = ['puremhd', 'thin', 'thick']
REGIME_LABEL = {'puremhd': 'pure', 'thin': 'thin', 'thick': 'thick'}
EXPECTED_SLOPE = {'puremhd': -2.0, 'thin': -2.0, 'thick': -1.0}
SLOPE_TOL = 0.3


def get_param(input_text, key, cast=float):
    m = re.search(r'^{}\s*=\s*([^\s#]+)'.format(key), input_text, re.M)
    if m is None:
        raise KeyError("Couldn't find '{}' in embedded <mhdmodes> params".format(key))
    return cast(m.group(1))


def read_linear(fname):
    with h5py.File(fname, 'r') as f:
        input_text = f['Input'].attrs['File']
        if isinstance(input_text, bytes):
            input_text = input_text.decode('utf-8')

    rho0 = get_param(input_text, 'rho0')
    k1 = get_param(input_text, 'k1')
    k2 = get_param(input_text, 'k2')
    k3 = get_param(input_text, 'k3')
    phase0 = get_param(input_text, 'phase')
    drho = get_param(input_text, 'drho_real') + 1j * get_param(input_text, 'drho_imag')
    omega = get_param(input_text, 'omega_real') + 1j * get_param(input_text, 'omega_imag')

    return {'rho0': rho0, 'k': np.array([k1, k2, k3]), 'phase': phase0,
            'drho': drho, 'omega': omega}


def analytic_rho(params, x, t):
    k1, k2, k3 = params['k']
    phase_full = params['omega'].real * t - (k1 * x + params['phase'])
    damping = np.exp(-params['omega'].imag * t)
    return params['rho0'] + damping * (params['drho'] * np.exp(1j * phase_full)).real


def convergence_for_mode(pattern):
    fnames = sorted(
        glob(pattern),
        key=lambda x: int(re.search(r"mhd_modes_(\d+)", x).group(1))
    )
    if not fnames:
        raise FileNotFoundError("No dumps matched pattern: {}".format(pattern))

    Ns, L1s = [], []
    for fname in fnames:
        dump = pyharm.load_dump(fname)
        params = read_linear(fname)

        rho_num = dump['rho'][:, 0, 0]
        x = dump['x'][:, 0, 0]
        N = rho_num.shape[0]

        rho_a = analytic_rho(params, x, dump['t'])
        L1 = np.sum(np.abs(rho_num - rho_a)) / N

        Ns.append(N)
        L1s.append(L1)

    return np.array(Ns), np.array(L1s)


if __name__ == '__main__':
    plotsdir = sys.argv[1]
    filesdir = sys.argv[2]

    results = {}
    for regime in REGIMES:
        rlabel = REGIME_LABEL[regime]
        for wavetype in WAVETYPES:
            label = wavetype if regime == 'puremhd' else "{}_{}".format(wavetype, regime)
            dirname = "{}_{}".format(rlabel, wavetype)
            pattern = os.path.join(
                filesdir, dirname, "mhd_modes_*_{}_{}.phdf".format(rlabel, wavetype))
            N_arr, L1_arr = convergence_for_mode(pattern)
            results[label] = (regime, wavetype, N_arr, L1_arr)

    fail = 0
    for label, (regime, wavetype, N_arr, L1_arr) in results.items():
        L1_safe = np.maximum(L1_arr, 1e-300)
        powerfit = np.polyfit(np.log(N_arr), np.log(L1_safe), 1)[0]
        expected = EXPECTED_SLOPE[regime]
        print("{} powerfit: {} L1: {}".format(label, powerfit, list(L1_arr)))
        if not (expected - SLOPE_TOL <= powerfit <= expected + SLOPE_TOL):
            fail = 1

    PANELS = [
        ("No radiation", 'puremhd'),
        ("Optically thin", 'thin'),
        ("Optically thick", 'thick'),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharey=True)

    for ax, (title, regime) in zip(axes, PANELS):
        panel_results = {label: (N_arr, L1_arr)
                          for label, (r, w, N_arr, L1_arr) in results.items() if r == regime}

        for label, (N_arr, L1_arr) in panel_results.items():
            ax.loglog(N_arr, L1_arr, 'o-', label=label, ms=4)

        if panel_results:
            N_ref, L1_ref = next(iter(panel_results.values()))
            ax.loglog(N_ref, 2 * L1_ref[0] * (N_ref / N_ref[0]) ** (-1),
                       'k--', alpha=0.5)
            ax.loglog(N_ref, 5e-1 * L1_ref[0] * (N_ref / N_ref[0]) ** (-2),
                       'k:', alpha=0.5)

        ax.set_xlabel(r'$N$', fontsize=14)
        ax.set_title(title)

        ax.set_xticks([64, 128, 256, 512])
        ax.set_xticklabels(['64', '128', '256', '512'])
        ax.xaxis.set_minor_formatter(NullFormatter())

        ax.grid(True, which='both', alpha=0.3)

    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles[:3], ['sonic', 'fast', 'slow'],
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01))

    axes[0].set_ylabel(r'L$_1$ norm', fontsize=12)
    axes[1].set_ylabel(r'L$_1$ norm', fontsize=12)
    axes[2].set_ylabel(r'L$_1$ norm', fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(os.path.join(plotsdir, "radmhdmodes_convergence.png"), dpi=300,
                bbox_inches='tight')
    plt.close(fig)

    exit(fail)
