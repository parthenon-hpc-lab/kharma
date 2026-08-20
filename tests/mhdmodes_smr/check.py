#!/usr/bin/env python

# MHD linear modes convergence plots
import os,sys
from matplotlib.colors import LightSource
import numpy as np
import matplotlib.pyplot as plt

import pyharm
import pyharm.plots as pplt

RES = [int(x) for x in sys.argv[1].split(",")]
LONG = sys.argv[2]
SHORT = sys.argv[3]
if len(sys.argv) > 4:
    DIM = sys.argv[4]
else:
    DIM = "3d"
if len(sys.argv) > 5:
    DIR = int(sys.argv[5])
else:
    DIR = 0

print(DIR)

NVAR = 8
VARS = ['rho', 'u', 'u1', 'u2', 'u3', 'B1', 'B2', 'B3']

amp = 1.e-4
k1 = 2.*np.pi
k2 = 2.*np.pi
if DIM == "3d" and DIR == 0:
    k3 = 2.*np.pi
else:
    k3 = 0
var0 = np.zeros(NVAR)

# Background
var0[0] = 1.
var0[1] = 1.
# Magnetic field
var0[5] = 1.
var0[6] = 0.
var0[7] = 0.

L1 = []

# EIGENMODES: 3D
dvar = np.zeros(NVAR)
if DIM == "3d" and DIR == 0:
    if "entropy" in SHORT:
        dvar[0] = 1.
    if "slow" in SHORT:
        dvar[0] = 0.556500332363
        dvar[1] = 0.742000443151
        dvar[2] = -0.282334999306
        dvar[3] = 0.0367010491491
        dvar[4] = 0.0367010491491
        dvar[5] = -0.195509141461
        dvar[6] = 0.0977545707307
        dvar[7] = 0.0977545707307
    if "alfven" in SHORT:
        dvar[3] = -0.339683110243
        dvar[4] = 0.339683110243
        dvar[6] = 0.620173672946
        dvar[7] = -0.620173672946
    if "fast" in SHORT:
        dvar[0] = 0.481846076323
        dvar[1] = 0.642461435098
        dvar[2] = -0.0832240462505
        dvar[3] = -0.224080007379
        dvar[4] = -0.224080007379
        dvar[5] = 0.406380545676
        dvar[6] = -0.203190272838
        dvar[7] = -0.203190272838
else:
    # EIGENMODES: 2D
    # We only *convergence check* dir = 3 i.e. X1/X2 plane runs
    # Other directions are useful for diagnosis but won't fail if 3D runs don't
    if "entropy" in SHORT:
        dvar[0] = 1.
    if "slow" in SHORT:
        dvar[0] = 0.558104461559
        dvar[1] = 0.744139282078
        dvar[2] = -0.277124827421
        dvar[3] = 0.0630348927707
        dvar[5] = -0.164323721928
        dvar[6] = 0.164323721928
    if "alfven" in SHORT:
        dvar[4] = 0.480384461415
        dvar[7] = 0.877058019307
    if "fast" in SHORT:
        dvar[0] = 0.476395427447
        dvar[1] = 0.635193903263
        dvar[2] = -0.102965815319
        dvar[3] = -0.316873207561
        dvar[5] = 0.359559114174
        dvar[6] = -0.359559114174

dvar *= amp

for m, res in enumerate(RES):
    # TODO compare the analytic solution rather than cheating like this
    # pyharm needs a 2o accurate interpolation, or else preserve original blocks
    start = pyharm.load_dump("mhd_{}_{}_{}_start.phdf".format(DIM, SHORT, res))
    dump = pyharm.load_dump("mhd_{}_{}_{}_end.phdf".format(DIM, SHORT, res))

    X1 = dump['x']
    X2 = dump['y']
    X3 = dump['z']

    dvar_code = []
    dvar_code.append(dump['RHO'] - start['RHO'])
    dvar_code.append(dump['UU'] - start['UU'])
    dvar_code.append(dump['U1'] - start['U1'])
    dvar_code.append(dump['U2'] - start['U2'])
    dvar_code.append(dump['U3'] - start['U3'])
    try:
        dvar_code.append(dump['B1'] - start['B1'])
        dvar_code.append(dump['B2'] - start['B2'])
        dvar_code.append(dump['B3'] - start['B3'])
    except IOError:
        NVAR = 5

    dvar_sol = []
    L1.append([])
    for k in range(NVAR):
      #dvar_sol.append(np.real(dvar[k])*np.cos(k1*X1 + k2*X2 + k3*X3))
      L1[m].append(np.mean(np.fabs(dvar_code[k])))

      fig = plt.figure(figsize=(5,5))
      ax = fig.add_subplot(1,1,1)
      pplt.plot_xz(ax, dump, dvar_code[k], native=True, window=[0,1,0,1])
      plt.savefig("compare_{}_{}_{}_{}.png".format(VARS[k], DIM, SHORT, res))
      

# MEASURE CONVERGENCE
L1 = np.array(L1)
powerfits = [0.,]*NVAR
fail = 0
for k in range(NVAR):
    if abs(dvar[k]) != 0.:
        powerfits[k] = np.polyfit(np.log(RES), np.log(L1[:,k]), 1)[0]

        print("Power fit {}: {} {}".format(VARS[k], powerfits[k], L1[:,k]))
        # These bounds were chosen heuristically: fast u2/u3 converge fast
        if powerfits[k] > -1.9 or ("entropy" not in SHORT and powerfits[k] < -2.1):
            # Allow entropy wave to converge fast, otherwise everything is ~2
            fail = 1

# MAKE PLOTS
fig = plt.figure(figsize=(5,5))

ax = fig.add_subplot(1,1,1)
for k in range(NVAR):
    if abs(dvar[k]) != 0.:
        ax.plot(RES, L1[:,k], marker='s', label=VARS[k])

norm = L1[0,0]*RES[0]*RES[0]
if norm < 1e-4:
    norm = L1[0,3]*RES[0]*RES[0]
xmin = RES[0]/2.
xmax = RES[-1]*2.
ax.plot([xmin, xmax], norm*np.asarray([xmin, xmax])**-2., color='k', linestyle='--', label='N^-2')

plt.xscale('log', base=2); plt.yscale('log')
plt.xlim([RES[0]/np.sqrt(2.), RES[-1]*np.sqrt(2.)])
plt.xlabel('N'); plt.ylabel('L1')
plt.title("{}".format(LONG))
plt.legend(loc=1)
plt.savefig("convergence_modes_{}_{}.png".format(DIM,SHORT))

exit(fail)
