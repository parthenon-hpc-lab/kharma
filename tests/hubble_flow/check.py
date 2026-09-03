#!/usr/bin/env python3
"""Check the Hubble flow test against Ressler+ 2015 sec. 4.1.1.

Usage: check.py res1,res2,... tag [tag ...]
  tags are "cooling" or "nocooling"; dumps are read from hubble_<tag>.res<N>.phdf

"cooling" is the actual test of the paper: an explicit heating term is added to the
energy equation, and the solution is only maintained if that source is time-centred
consistently with the integrator.  "nocooling" removes the source entirely (the flow
is then adiabatic, u ~ rho^gam and the electron entropy is constant), which isolates
transport and the boundaries -- so a failure there is *not* a heating problem.

Both should converge at 2nd order.  Note the analytic solution is non-relativistic, so
it is only correct to O(v^2/c^2) ~ 1e-6 -- the measured errors stop falling once they
reach that, so don't push the resolution much past 512.
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
COLORS = {"vx": "C0", "rho": "C1", "u": "C2", "kappa_e": "C3"}
# L1 errors should converge at this order (~2)
EXPECTED_ORDER = 1.8


def analytic(x, t, cooling):
    """Ressler+ '15 eqns 37, 39 & 40.  Without the source term u and Kel are instead
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


def compare(fname, cooling, plot_to=None):
    """L1 errors of one dump, normalized by max|analytic| (vx passes through zero)."""
    f = pyharm.load_dump(fname)
    t, x = f["t"], f["x"][:, 0, 0]
    num, ana = numeric(f), analytic(x, t, cooling)

    if plot_to is not None:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        fig.suptitle("{}   (t = {:.4g}, n1 = {}, {})".format(
            fname, t, f["n1"], "cooling" if cooling else "no cooling"))

    errs = {}
    for n, name in enumerate(VARS):
        a, b = num[name], ana[name]
        norm = np.max(np.abs(b))
        errs[name] = np.mean(np.abs(a - b)) / norm
        if plot_to is not None:
            axes[0, n].plot(x, a, color=COLORS[name], label="KHARMA")
            axes[0, n].plot(x, b, "k--", lw=1, label="analytic")
            axes[0, n].set_title(name)
            axes[0, n].legend()
            # Scale to the analytic solution, not to the difference.
            lo, hi = np.min(b), np.max(b)
            span = max(hi - lo, 0.5 * max(abs(hi), abs(lo)))
            axes[0, n].set_ylim(lo - 0.25 * span, hi + 0.25 * span)
            axes[0, n].ticklabel_format(axis="y", useOffset=False)
            axes[1, n].plot(x, (a - b) / norm, color=COLORS[name])
            axes[1, n].axhline(0.0, color="k", lw=0.5)
            axes[1, n].set_title("{} rel. error (L1 = {:.3g})".format(name, errs[name]))
            axes[1, n].set_xlabel("x")

    if plot_to is not None:
        plt.tight_layout()
        plt.savefig(plot_to)
        plt.close(fig)

    return errs


def check_case(tag, resolutions):
    """Errors and fitted convergence order for one case.  Returns (errs, orders, fail)."""
    cooling = tag == "cooling"
    print("=== {} ===".format("With cooling (Ressler+ '15 eq. 38-40)" if cooling
                              else "No cooling (adiabatic, no source term)"))

    errs = {name: [] for name in VARS}
    for res in resolutions:
        fname = "hubble_{}.res{}.phdf".format(tag, res)
        plot_to = ("hubble_{}.png".format(tag) if res == resolutions[-1] else None)
        e = compare(fname, cooling, plot_to)
        print("n1 = {:>5d}:".format(res)
              + "".join("  {} {:.3e}".format(n, e[n]) for n in VARS))
        for name in VARS:
            errs[name].append(e[name])

    fail = 0
    orders = {}
    res = np.array(resolutions, dtype=float)
    for name in VARS:
        err = np.array(errs[name])
        orders[name] = -np.polyfit(np.log(res), np.log(err), 1)[0]
        ok = orders[name] >= EXPECTED_ORDER
        fail |= (not ok)
        print("{:>10s}: convergence order {:.2f}  {}".format(
            name, orders[name], "PASS" if ok else "FAIL"))

    return errs, orders, fail


def main(resolutions, tags):
    res = np.array(resolutions, dtype=float)
    fig, axes = plt.subplots(1, len(tags), figsize=(6 * len(tags), 5), squeeze=False)

    fail = 0
    for n, tag in enumerate(tags):
        errs, orders, f = check_case(tag, resolutions)
        fail |= f
        print()

        ax = axes[0, n]
        for name in VARS:
            ax.loglog(res, errs[name], marker="o", color=COLORS[name],
                      label="{} ({:.2f})".format(name, orders[name]))
        ref = errs["u"][0] * (res / res[0]) ** -2.0
        ax.loglog(res, ref, "k--", lw=1, label="2nd order")
        ax.set_title("cooling" if tag == "cooling" else "no cooling")
        ax.set_xticks(res)
        ax.set_xticklabels([str(r) for r in resolutions])
        ax.set_xticks([], minor=True)
        ax.set_xlabel("n1")
        ax.set_ylabel("L1 error")
        ax.legend(fontsize="small")

    plt.tight_layout()
    plt.savefig("hubble_convergence.png")
    return fail


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main([int(r) for r in sys.argv[1].split(",")], sys.argv[2:]))
