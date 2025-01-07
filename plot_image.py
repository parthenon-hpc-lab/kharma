#!/usr/bin/env python3

import os, sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

#varname = sys.argv[1]
fname = sys.argv[1]

with h5py.File(fname, 'r') as dfile:
    for varname in ["unpol", "I", "Q", "U", "V", "pathlen", "nstep", "stopflag"]:
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        arrname = "camera_"+varname
        max_abs = np.max(np.abs(dfile[arrname][0]))
        img = ax.imshow(dfile[arrname][0].T, origin='lower', cmap='RdBu_r', vmax=max_abs, vmin=-max_abs)
        plt.colorbar(img)
        fig.savefig(f"image_{varname}.png")
        fig.clear()
