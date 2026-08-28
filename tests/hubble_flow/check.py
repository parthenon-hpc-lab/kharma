#!/usr/bin/env python3
"""Check the Hubble flow test against Ressler+ 2015 sec. 4.1.1.

Usage: check.py res1 res2 ...
Reads hubble.out0.final.res<N>.phdf, prints L1 errors, and requires the
convergence to be 2nd order.

The heating in this problem is what's being tested: the analytic solution is only
maintained if the source term is time-centred consistently with the integrator.
Note the electron entropy error bottoms out around v^2/c^2 ~ 1e-6 (the analytic
solution is non-relativistic), so don't push the resolution much past 512.
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyharm

# Must match pars/electrons/hubble.par
GAM = 1.666667
GAME = 1.333333
MACH = 1.0
V0 = 1e-3

RHO0 = (MACH / V0) * np.sqrt(GAM * (GAM - 1))
UG0 = (V0 / MACH) / np.sqrt(GAM * (GAM - 1))
# Not free: eq. 40 only solves the electron entropy equation for this ratio
UE0 = (GAM - 2) / (GAME - 2) * UG0

VARS = ["vx", "rho", "u", "kappa_e"]
# L1 errors should converge at this order, allowing some slack for a 4-point fit
EXPECTED_ORDER = 1.8


def analytic(x, t, cooling=True):
    """Ressler+ '15 eqns 37, 39 & 40.  Without the source term, u and Kel are instead
    set by entropy conservation."""
    a = 1 + V0 * t
    ones = np.ones_like(x)
    return {
        "vx": V0 * x / a,
        "rho": RHO0 / a * ones,
        "u": UG0 / (a**2 if cooling else a**GAM) * ones,
        "kappa_e": (GAME - 1) * UE0 / RHO0**GAME
        * (a ** (GAME - 2) if cooling else 1.0) * ones,
    }


def numeric(f):
    return {
        "vx": f["uvec"][0, :, 0, 0],
        "rho": f["rho"][:, 0, 0],
        "u": f["u"][:, 0, 0],
        "kappa_e": f["Kel_Constant"][:, 0, 0],
    }


def compare(fname, cooling=True, plot=False):
    """L1 errors of one dump, normalized by max|analytic| (vx passes through zero)."""
    f = pyharm.load_dump(fname)
    t, x = f["t"], f["x"][:, 0, 0]
    num, ana = numeric(f), analytic(x, t, cooling)

    if plot:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        fig.suptitle("{} (t = {:.4g}, n1 = {})".format(fname, t, f["n1"]))

    errs = {}
    for n, name in enumerate(VARS):
        a, b = num[name], ana[name]
        norm = np.max(np.abs(b))
        errs[name] = np.mean(np.abs(a - b)) / norm
        if plot:
            axes[0, n].plot(x, a, label="KHARMA")
            axes[0, n].plot(x, b, "--", label="analytic")
            axes[0, n].set_title(name)
            axes[0, n].legend()
            axes[1, n].plot(x, (a - b) / norm)
            axes[1, n].set_title(
                "{} rel. error (L1 = {:.3g})".format(name, errs[name]))
            axes[1, n].set_xlabel("x")

    if plot:
        plt.tight_layout()
        plt.savefig(fname.replace(".phdf", ".png"))
        plt.close(fig)

    return errs


def main(resolutions, cooling=True):
    errs = {name: [] for name in VARS}
    for res in resolutions:
        fname = "hubble.out0.final.res{}.phdf".format(res)
        e = compare(fname, cooling, plot=(res == resolutions[-1]))
        print("n1 = {:>5d}:".format(res)
              + "".join("  {} {:.3e}".format(n, e[n]) for n in VARS))
        for name in VARS:
            errs[name].append(e[name])

    fail = 0
    fig, ax = plt.subplots(figsize=(6, 5))
    res = np.array(resolutions, dtype=float)
    for name in VARS:
        err = np.array(errs[name])
        # Least-squares fit of log(err) vs log(n1)
        order = -np.polyfit(np.log(res), np.log(err), 1)[0]
        status = "PASS" if order >= EXPECTED_ORDER else "FAIL"
        if order < EXPECTED_ORDER:
            fail = 1
        print("{:>10s}: convergence order {:.2f}  {}".format(name, order, status))
        ax.loglog(res, err, marker="o", label="{} ({:.2f})".format(name, order))

    ax.loglog(res, errs["u"][0] * (res / res[0]) ** -2.0, "k--", label="2nd order")
    ax.set_xlabel("n1")
    ax.set_ylabel("L1 error")
    ax.legend()
    plt.tight_layout()
    plt.savefig("hubble_convergence.png")

    return fail


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    sys.exit(main([int(a) for a in args], cooling=("--nocooling" not in sys.argv)))
