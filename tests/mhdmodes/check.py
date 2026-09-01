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

# USE DUMPS IN FOLDERS OF GIVEN FORMAT
for m, res in enumerate(RES):
    dump = pyharm.load_dump("mhd_{}_{}_{}_end.phdf".format(DIM, SHORT, res))

    X1 = dump['X1']
    X2 = dump['X2']
    X3 = dump['X3']

    dvar_code = []
    dvar_code.append(dump['RHO'] - var0[0])
    dvar_code.append(dump['UU'] - var0[1])
    dvar_code.append(dump['U1'] - var0[2])
    dvar_code.append(dump['U2'] - var0[3])
    dvar_code.append(dump['U3'] - var0[4])
    try:
        dvar_code.append(dump['B1'] - var0[5])
        dvar_code.append(dump['B2'] - var0[6])
        dvar_code.append(dump['B3'] - var0[7])
    except IOError:
        NVAR = 5

    dvar_sol = []
    L1.append([])
    for k in range(NVAR):
      dvar_sol.append(np.real(dvar[k])*np.cos(k1*X1 + k2*X2 + k3*X3))
      L1[m].append(np.mean(np.fabs(dvar_code[k] - dvar_sol[k])))

# MEASURE CONVERGENCE
L1 = np.array(L1)
powerfits = [0.,]*NVAR
fail = 0
for k in range(NVAR):
    if abs(dvar[k]) != 0.:
        powerfits[k] = np.polyfit(np.log(RES), np.log(L1[:,k]), 1)[0]

        print("Power fit {}: {} {}".format(VARS[k], powerfits[k], L1[:,k]))
        # These bounds were chosen heuristically as tight as possible
        # entropy wave converges with spatial order, up to ~5, so we only check >=2
        # TODO linear_vl and donor_cell special cases
        if "entropy" in SHORT:
            if powerfits[k] > -1.94:
                fail = 1
        elif "kharma" in SHORT:
            # TODO there's an issue in the CI setup causing the KHARMA driver test
            # (NOT the other runs using that driver!) to converge at -1.9
            # I can't reproduce it on other machines or GPUs; YMMV
            if VARS[k] == 'u2' or VARS[k] == 'u3':
                if powerfits[k] > -1.75 or powerfits[k] < -2.3:
                    fail = 1
            else:
                if powerfits[k] > -1.85 or powerfits[k] < -2.15:
                    fail = 1
        else:
            # Most runs all variables are between 1.95-2.05
            # But w/Face-CT and upwinding, u2/u3 converge at 2.3, rho/u/u1 at 2.15
            # TODO special-case those and tighten the general bounds
            if VARS[k] == 'u2' or VARS[k] == 'u3':
                if powerfits[k] > -1.94 or powerfits[k] < -2.3:
                    fail = 1
            else:
                if powerfits[k] > -1.94 or powerfits[k] < -2.15:
                    fail = 1

# MAKE PLOTS
fig = plt.figure(figsize=(3,3))

ax = fig.add_subplot(1,1,1)
for k in range(NVAR):
    if abs(dvar[k]) != 0.:
        ax.plot(RES, L1[:,k], marker='s', label=VARS[k])

if "alfven" in SHORT:
    norm = L1[0,4]*RES[0]*RES[0]
else:
    norm = L1[0,0]*RES[0]*RES[0]

xmin = RES[0]/2.
xmax = RES[-1]*2.
ax.plot([xmin, xmax], norm*np.asarray([xmin, xmax])**-2., color='k', linestyle='--', label='N^-2')

plt.xscale('log', base=2); plt.yscale('log')
plt.xlim([RES[0]/np.sqrt(2.), RES[-1]*np.sqrt(2.)])
plt.xlabel('N'); plt.ylabel('L1')
#plt.title("{}".format(LONG))
plt.legend(loc=1)
plt.subplots_adjust(left=0.25, bottom=0.18, top=0.97, right=0.97)
plt.savefig("convergence_modes_{}_{}.png".format(DIM,SHORT), dpi=300)

exit(fail)
