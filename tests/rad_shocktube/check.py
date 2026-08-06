import numpy as np
import os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyharm

from numpy.polynomial import Chebyshev as C

# Reference solution for "test 4" (radiation-pressure-dominated wave) of
# Melon Fuksman & Mignone 2019 run at 800 zones, which we cross-checked
# against PLUTO's own M1 solver output directly (RMS agreement ~0.05-0.1%
# over the whole domain).

#There's probably a better way to do this for the future CI test, when integrated.
DOMAIN = [-19.975, 19.975]

RHO_COEF = [
    2.1806458524e+00, 1.6228695187e+00, 2.5964222552e-01, -3.9917934199e-01, -2.0676486051e-01,
    1.1406129295e-01, 1.3951113230e-01, -2.8304630711e-03, -7.6544675742e-02, -3.2135806731e-02,
    2.9809785255e-02, 3.2000732052e-02, -2.6565911415e-03, -2.0120274881e-02, -8.1942726451e-03,
    8.2924885512e-03, 9.0413150252e-03, -7.0420806903e-04, -5.8380939796e-03, -2.5002937313e-03,
    2.3601293495e-03, 2.7748092368e-03, -1.0833009722e-04, -1.7749457271e-03, -8.2600800488e-04,
    6.7551044261e-04, 8.8379094762e-04, 1.8974427208e-05, -5.5445829972e-04, -2.8497568495e-04,
    1.9410200693e-04, 2.8495426144e-04, 2.8044044041e-05, -1.7488056488e-04, -1.0227613868e-04,
    5.6631915211e-05, 9.1850083469e-05, 1.6627645120e-05, -5.4051001942e-05, -3.8708012754e-05,
    1.6955446039e-05, 2.9591707848e-05, 7.2848109314e-06, -1.6197274858e-05, -1.4850766499e-05,
    3.9329631606e-06, 1.0430615153e-05, 1.2722552020e-06, -4.1686062963e-06, -7.1455523780e-06,
    5.4415551917e-07,
]

U1_COEF = [
    4.4179327017e-01, -3.1015615914e-01, -2.4037892826e-03, 8.2298079822e-02, 4.2053049491e-03,
    -3.0894903591e-02, -5.4557035778e-03, 1.0442886299e-02, 5.1617946647e-03, -2.4961125531e-03,
    -3.5763197341e-03, -8.5547186971e-05, 1.7372679519e-03, 6.8758037776e-04, -4.8010597092e-04,
    -6.3667619888e-04, -5.2281505012e-05, 3.7150188248e-04, 1.6166545824e-04, -1.0871123799e-04,
    -1.3751464632e-04, -2.9891066388e-05, 8.7537847619e-05, 5.1278500453e-05, -3.0522379775e-05,
    -3.1971509918e-05, -1.0742439493e-05, 1.8251527091e-05, 1.9640827284e-05, -9.2311496644e-06,
    -9.3178418806e-06, -1.5891688984e-06, 1.9550050832e-06, 7.3856532982e-06, -1.7213012563e-06,
    -4.3069502618e-06, 9.4456707112e-07, -4.0373502417e-07, 1.8564688797e-06, 8.7729111890e-07,
    -2.4521244593e-06, 7.2019399905e-07, 1.1309239653e-07, -1.6271724854e-07, 9.8942893818e-07,
    -8.6771549592e-07, -2.3334907221e-07, 7.8875950780e-07, -7.2160486097e-07, 7.1186870286e-07,
    -2.4759872343e-07,
]

ERF_COEF = [
    7.5825717292e-01, 6.8526243022e-01, -2.8465613244e-02, -1.6926157531e-01, 8.9843360559e-03,
    5.6548634919e-02, 7.1086083422e-03, -1.8563005208e-02, -1.1785464113e-02, 6.6826883215e-03,
    7.9329339280e-03, -1.8598378603e-03, -3.0124763171e-03, -1.0388398529e-03, 8.6489281879e-04,
    1.7675955267e-03, -5.0615229893e-04, -8.4972007749e-04, 1.1111802225e-04, 5.6665741137e-06,
    3.9372445880e-04, 3.5161080636e-05, -3.8949665349e-04, 1.3126322997e-04, 2.0511195566e-05,
    -9.1020846271e-06, 1.2759789957e-04, -1.5692297916e-04, 7.0099127634e-06, 8.3380555005e-05,
    -7.8396037224e-05, 6.5508380442e-05, -1.7130544394e-05, -5.5383820708e-05, 7.2815561789e-05,
    -3.6401438390e-05, -4.2896091436e-06, 3.6970535002e-05, -4.8108451926e-05, 2.6647769421e-05,
    1.0733574337e-05, -3.1868115964e-05, 3.3448899052e-05, -1.5309013507e-05, -7.5045970668e-06,
    2.5255893940e-05, -2.0822754776e-05, 7.8469249526e-06, 1.2721503568e-05, -1.8710933676e-05,
    2.0963151255e-05,
]

FX_COEF = [
    -4.2540863250e-02, 6.8557107401e-03, 7.5191219594e-02, -1.5635238532e-02, -5.1792616494e-02,
    1.4224612028e-02, 2.7935350102e-02, -6.1767887359e-03, -1.2536093443e-02, -9.9434624880e-04,
    5.6779616132e-03, 3.2942182040e-03, -2.7799228146e-03, -2.2125760388e-03, 7.5501551053e-04,
    8.8265879384e-04, 5.4556224639e-04, -4.5761271117e-04, -6.7808098464e-04, 3.0466483907e-04,
    2.1790557885e-04, 2.6448416566e-05, 1.6444006231e-05, -2.4384344569e-04, 5.6270883522e-05,
    1.3447226777e-04, -6.9132197812e-05, 4.2103268771e-05, -4.7396831432e-05, -4.1504135109e-05,
    8.9014429304e-05, -3.7189612308e-05, -6.3575759930e-06, 2.5522642229e-05, -4.6399762742e-05,
    3.4564150354e-05, 6.3117875171e-06, -3.0415697898e-05, 2.8972108769e-05, -1.7285939952e-05,
    -4.8612467309e-06, 2.3179916421e-05, -2.4188629436e-05, 8.9954038366e-06, 5.5057933436e-06,
    -1.6628951658e-05, 1.4632557649e-05, -5.5332204194e-06, -9.3321573978e-06, 1.2912482259e-05,
    -1.4103073172e-05,
]

REFERENCE = {
    'rho': C(RHO_COEF, domain=DOMAIN),
    'u1': C(U1_COEF, domain=DOMAIN),
    'Erf': C(ERF_COEF, domain=DOMAIN),
    'Fx': C(FX_COEF, domain=DOMAIN),
}


def read_shock_dump(fname):
    dump = pyharm.load_dump(fname)

    gamma = dump['gam']
    x = dump['x'][:, 0, 0]
    rho_a = dump['rho'][:, 0, 0]
    prs_a = (gamma - 1.) * dump['u'][:, 0, 0]
    u1_a = dump['ucon'][1, :, 0, 0]
    erad_a = dump['u_rad'][:, 0, 0]
    u1rad_a = dump['uvec_rad'][0, :, 0, 0]

    o = np.argsort(x)
    x, rho_a, prs_a, u1_a, erad_a, u1rad_a = \
        x[o], rho_a[o], prs_a[o], u1_a[o], erad_a[o], u1rad_a[o]

    u0_rad = np.sqrt(1. + u1rad_a**2)
    u0_gas = np.sqrt(1. + u1_a**2)

    R00 = (4. / 3.) * erad_a * u0_rad**2 - (1. / 3.) * erad_a
    R01 = (4. / 3.) * erad_a * u0_rad * u1rad_a
    R11 = (4. / 3.) * erad_a * u1rad_a**2 + (1. / 3.) * erad_a

    fx_a = R01 * u0_gas - R11 * u1_a - (R00 * u0_gas**2 - 2 * R01 * u0_gas * u1_a + R11 * u1_a**2) * u1_a

    return x, rho_a, prs_a, u1_a, erad_a, u1rad_a, fx_a


if __name__ == '__main__':
    plotsdir = sys.argv[1]
    filesdir = sys.argv[2]
    resolutions = [int(r) for r in sys.argv[3].split(',')]
    resolutions = np.array(resolutions)

    VARS = ['rho', 'u1', 'Erf', 'Fx']

    L1 = {v: [] for v in VARS}

    for res in resolutions:
        x, rho, prs, u1, erf, u1rad, fx = read_shock_dump(
            os.path.join(filesdir, 'shock.out0.final.res{:d}.phdf'.format(res)))
        test_vars = {'rho': rho, 'u1': u1, 'Erf': erf, 'Fx': fx}

        for v in VARS:
            ref = REFERENCE[v](x)
            L1[v].append(np.mean(np.abs(test_vars[v] - ref)))

        plt.figure(figsize=(6, 6))
        plt.plot(x, rho, '.-', ms=2, label="res={}".format(res))
        xx = np.linspace(DOMAIN[0], DOMAIN[1], 2000)
        plt.plot(xx, REFERENCE['rho'](xx), 'k--', lw=1, label="reference fit")
        plt.xlabel('x'); plt.ylabel(r'$\rho$')
        plt.legend()
        plt.savefig(os.path.join(plotsdir, "rad_shocktube_rho_{}.png".format(res)))
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.plot(x, fx, '.-', ms=2, label="res={}".format(res))
        plt.plot(xx, REFERENCE['Fx'](xx), 'k--', lw=1, label="reference fit")
        plt.xlabel('x'); plt.ylabel(r'$\tilde{F}^x$ (fluid frame)')
        plt.legend()
        plt.savefig(os.path.join(plotsdir, "rad_shocktube_Fx_{}.png".format(res)))
        plt.close()

    # The 800-zone point is the same resolution the reference fit was
    # built from, so treat 800 separately as a floor/sanity check that the run agrees with the fit
    # about as well as the fit agrees with itself.
    fit_mask = resolutions < 800

    fail = 0
    powerfits = {}
    for v in VARS:
        L1[v] = np.array(L1[v])
        powerfits[v] = np.polyfit(np.log(resolutions[fit_mask]), np.log(L1[v][fit_mask]), 1)[0]
        print("{} Powerfit: {} L1: {}".format(v, powerfits[v], L1[v]))
        # These bounds were chosen heuristically.
        if powerfits[v] < -1.7 or powerfits[v] > -1.1:
            fail = 1
        # The 800-zone run should land close to the fit's own residual.
        if 800 in resolutions and L1[v][list(resolutions).index(800)] > 1e-3:
            fail = 1

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    colors = {'rho': 'darkblue', 'u1': 'darkgreen', 'Erf': 'crimson', 'Fx': 'purple'}
    for v in VARS:
        ax.plot(resolutions[fit_mask], L1[v][fit_mask], color=colors[v], marker='^', markersize=8, label=v)
    res_fit = resolutions[fit_mask]
    mid = len(res_fit) // 2
    amp = L1['rho'][fit_mask][mid] * float(res_fit[mid])**0.5
    ax.plot([res_fit[0], res_fit[-1]],
            amp * np.asarray([res_fit[0], res_fit[-1]], dtype=float)**(-1.),
            color='k', linestyle='dashed', label='$N^{-1}$')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Resolution', fontsize = 16)
    plt.ylabel('L1 Norm (vs. reference fit)', fontsize = 16)
    plt.legend()
    plt.savefig(os.path.join(plotsdir, 'rad_shocktube_convergence.png'), dpi=200)
    plt.close()

    exit(fail)
