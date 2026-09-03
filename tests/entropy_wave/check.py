#!/usr/bin/env python3
"""Check the 2D advected entropy wave.  This is a test of Ktot_adv.

Ktot_adv is the Noble+ 2009 (arXiv:0808.3140) entropy S = p/rho^(gam-1), an entropy
*density* obeying their eq. 19, div_mu(S u^mu) = 0.  Ktot, which Electrons consume, is
the per-mass entropy p/rho^gam.  The two differ by a factor of rho, so the meaningful
comparison is Ktot vs Ktot_adv/rho -- see the diagnostic below.

CAVEAT: this problem has a uniform 4-velocity, hence div_mu(u^mu) = 0, and in that case
the density and per-mass advection laws coincide.  So this test pins down that Ktot_adv
is transported accurately, but it can NOT distinguish the two normalizations -- it would
pass either way.  The discriminating case is a compressible flow: in Hubble (nocooling)
K is constant along worldlines while S = rho*K falls as 1/(1+v0*t).

Usage: check.py res1,res2,...
  dumps are read from entropy_wave.init.res<N>.phdf and entropy_wave.res<N>.phdf

The problem runs for exactly one period, so the analytic solution at t=tlim is the
initial condition -- and the initial condition is set analytically at cell centers, so
the t=0 dump *is* the analytic solution.  Errors are measured directly against it.

Only Ktot_adv gets a convergence study, because it is the only thing here whose accuracy
is an open question:

  - Ktot is *not* convergence-tested.  ApplyEntropyUpdate resets it from rho,u at the end
    of every sub-step over the entire domain, so at dump time it is an algebraic function
    of rho and u in that same dump.  Its "convergence order" would just restate the
    convergence of rho and u, which other tests cover.  What is worth asserting is the
    identity itself, K = (gam-1) u / rho^gam, which holds to round-off at every
    resolution and fails loudly if the update ever stops covering the whole domain.

The convergence study is backed by two exact checks that hold at every resolution, to
round-off: the Ktot identity above, and conservation of the mean of Ktot_adv (a plain
mean, not mass-weighted, since Ktot_adv is a density).

Writes entropy_wave_convergence.png.  For maps of the fields, see make_plots.py.
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyharm

# Convergence order to require of Ktot_adv.
MIN_ORDER = 2.0
# Ktot is re-established from rho,u every sub-step, so this is a discrete identity rather
# than an approximation; the measured deviation is ~1e-16.
IDENTITY_TOL = 1e-12
# sum(cons.Ktot_adv) is exactly conserved: it is advected by a pure flux divergence and
# the domain is periodic.
CONS_TOL = 1e-10


def l1(a, b):
    """Relative L1 error of a against b."""
    return np.abs(a - b).mean() / np.abs(b).mean()


def plot_convergence(res_list, err_adv, fname="entropy_wave_convergence.png"):
    res = np.array(res_list, dtype=float)
    e = np.array(err_adv)
    order = -np.polyfit(np.log(res), np.log(e), 1)[0]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(res, e, marker="s", color="C3", lw=2,
            label=r"$K_{{\rm adv}}$  (order {:.2f})".format(order))
    ax.plot(res, e[0] * (res / res[0]) ** -2.0, "k--", lw=1, label=r"$N^{-2}$")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(res[0] / np.sqrt(2.0), res[-1] * np.sqrt(2.0))
    ax.set_xticks(res)
    ax.set_xticklabels([str(int(r)) for r in res])
    ax.set_xlabel("N  (grid is N x N)")
    ax.set_ylabel("relative L1 error at t = one period")
    ax.set_title(r"Advected entropy wave: $K_{\rm adv}$ convergence")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print("Wrote {}".format(fname))


def main():
    res_list = [int(r) for r in sys.argv[1].split(",")]

    err_adv = []
    for res in res_list:
        i = pyharm.load_dump("entropy_wave.init.res{}.phdf".format(res))
        f = pyharm.load_dump("entropy_wave.res{}.phdf".format(res))
        gam = f["gam"]

        # Ktot is the entropy of the current rho,u.  ApplyEntropyUpdate re-establishes
        # this over the entire domain every sub-step, so a failure here means the update
        # stopped covering some zones -- not that the scheme is inaccurate.
        ident = (gam - 1.0) * f["u"] * f["rho"] ** -gam
        k_err = np.abs(f["Ktot"] - ident).max() / np.abs(f["Ktot"]).mean()
        if k_err > IDENTITY_TOL:
            print("FAIL: Ktot != (gam-1)u/rho^gam at {}^2 (max rel. dev {:.3e} > {:.0e}); "
                  "ApplyEntropyUpdate is not covering the whole domain.".format(
                      res, k_err, IDENTITY_TOL))
            return 2

        # The mean of Ktot_adv is conserved by the discretisation.  A stale or
        # uninitialised ghost zone shows up here as a fixed, resolution-independent
        # offset rather than as a convergence failure.
        # NB Ktot_adv is a density, conserved as sqrt(-g) u^t Ktot_adv, so the plain 
        # mean is right only because gdet = 1 and u^t is uniform here.  In curved 
        # space, or with a non-uniform Lorentz factor, weight by gdet*u^t.
        a0, a1 = i["Ktot_adv"], f["Ktot_adv"]
        drift = a1.mean() / a0.mean() - 1.0
        if abs(drift) > CONS_TOL:
            print("FAIL: <Ktot_adv> drifted by {:+.3e} at {}^2 "
                  "(tolerance {:.0e}); Ktot_adv is not being advected "
                  "conservatively.".format(drift, res, CONS_TOL))
            return 3

        err_adv.append(l1(a1, a0))
        # Diagnostic: the accumulated dissipation. Ktot is per mass and Ktot_adv is a
        # density, so the comparable quantity is Ktot - Ktot_adv/rho.  Zero is the
        # correct answer here, and its convergence is implied by the error above;
        # reported because it is what Electrons consume.
        diss = (np.abs(f["Ktot"] - f["Ktot_adv"] / f["rho"]).mean()
                / np.abs(i["Ktot"]).mean())

        print("{:4d}^2:  Ktot_adv err {:.3e}   [cons drift {:+.1e}, Ktot identity "
              "{:.1e}, diag: Ktot-Ktot_adv/rho {:.3e}]".format(
                  res, err_adv[-1], drift, k_err, diss))

    order = -np.polyfit(np.log(res_list), np.log(err_adv), 1)[0]
    ok = order >= MIN_ORDER
    print(" Ktot_adv: convergence order {:.2f}  {}".format(
        order, "PASS" if ok else "FAIL"))

    plot_convergence(res_list, err_adv)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
