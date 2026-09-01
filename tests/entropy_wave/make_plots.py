#!/usr/bin/env python3
"""Maps of rho and Ktot_adv for the first and last dumps of the entropy wave.

Usage: make_plots.py [res ...]
  Defaults to every resolution with a dump pair present.

Writes entropy_wave_fields_res<N>.png, one row per variable:

    initial (t=0)   |   final (t=one period)   |   final - initial

The run lasts exactly one period, so the exact solution at t=tlim is the initial
condition.  The third column is therefore not just a "change" -- it is the error, and
its structure is informative: advection error in a smooth wave shows up as a smooth
phase/amplitude pattern following the wave, whereas a bad ghost zone or a botched
block-to-block exchange shows up localized at a domain or meshblock edge.
"""

import glob
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyharm

# (key, display name, colormap) for the fields we map
FIELDS = [
    ("rho", r"$\rho$", "viridis"),
    ("Ktot_adv", r"$K_{\rm adv}$", "magma"),
]


def available():
    """Resolutions with both an init and a final dump present, ascending."""
    res = []
    for fn in glob.glob("entropy_wave.init.res*.phdf"):
        m = re.search(r"res(\d+)\.phdf$", fn)
        if m and glob.glob("entropy_wave.res{}.phdf".format(m.group(1))):
            res.append(int(m.group(1)))
    return sorted(res)


def plot_res(res):
    i = pyharm.load_dump("entropy_wave.init.res{}.phdf".format(res))
    f = pyharm.load_dump("entropy_wave.res{}.phdf".format(res))

    # pyharm assembles the meshblocks for us; drop the degenerate x3 axis
    x, y = i["x"][:, :, 0], i["y"][:, :, 0]

    fig, axes = plt.subplots(len(FIELDS), 3, figsize=(15, 4.6 * len(FIELDS)))
    axes = np.atleast_2d(axes)
    fig.suptitle("Advected entropy wave, {res}x{res}   "
                 "(t = 0 and t = {t:.4g}, one full period)".format(res=res, t=f["t"]))

    for row, (key, label, cmap) in enumerate(FIELDS):
        a, b = i[key][:, :, 0], f[key][:, :, 0]
        diff = b - a

        # Share one color scale between the two states so the eye compares them, not
        # their individual normalizations.
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        # Symmetric scale for the difference, so zero is the neutral color
        dmax = np.abs(diff).max()
        dmax = dmax if dmax > 0 else 1.0

        panels = [
            (a, "{} initial".format(label), cmap, dict(vmin=lo, vmax=hi)),
            (b, "{} final".format(label), cmap, dict(vmin=lo, vmax=hi)),
            (diff, "{} final - initial  (= error)".format(label), "RdBu_r",
             dict(vmin=-dmax, vmax=dmax)),
        ]
        for col, (data, title, cm, norm) in enumerate(panels):
            ax = axes[row, col]
            im = ax.pcolormesh(x, y, data, cmap=cm, shading="auto", **norm)
            ax.set_title(title, fontsize=11)
            ax.set_aspect("equal")
            ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        print("  {:>9s}: initial [{:.4f}, {:.4f}]  final [{:.4f}, {:.4f}]  "
              "max|error| {:.3e}".format(key, a.min(), a.max(), b.min(), b.max(),
                                         np.abs(diff).max()))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fname = "entropy_wave_fields_res{}.png".format(res)
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print("Wrote {}".format(fname))


def main():
    res_list = [int(r) for r in sys.argv[1:]] if len(sys.argv) > 1 else available()
    if not res_list:
        print("No entropy_wave dump pairs found -- run ./run.sh first.")
        return 1
    for res in res_list:
        print("{}^2:".format(res))
        plot_res(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
