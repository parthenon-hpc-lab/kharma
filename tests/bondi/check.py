#!/usr/bin/env python

# Bondi problem convergence plots
# TODO could use the analytic solution here for extra rigor

import os,sys
import numpy as np
import matplotlib.pyplot as plt

import pyharm

RES = [int(x) for x in sys.argv[1].split(",")]
LONG = sys.argv[2]
SHORT = sys.argv[3]
VARS = ('RHO', 'UU', 'U1')

L1 = {}

# 2d
for res in RES:
    start = pyharm.load_dump("bondi_2d_{}_start_{}.phdf".format(res, SHORT))
    end = pyharm.load_dump("bondi_2d_{}_end_{}.phdf".format(res, SHORT))
    params = start.params

    # Start from at least outside the outer BL coord singularity
    # Usually the test itself will start from r=3M and avoid this
    imin = 0
    while start['r1d'][imin] < (1 + np.sqrt(1 + start['a']**2) + 0.2):
        imin += 1

    for var in VARS:
        if not var in L1:
            L1[var] = []

        var0 = np.mean(start[var][imin:,:,:], axis=1)
        var1 = np.mean(end[var][imin:,:,:], axis=1)
        L1[var].append(np.mean(np.fabs(var1 - var0)))

        if var == 'RHO':
            r = start['r1d'][imin:]
            fig = plt.figure(figsize=(5,5))
            plt.loglog(r, var0, label='Initial')
            plt.loglog(r, var1, label='Final')
            plt.xlabel('r'); plt.ylabel('rho')
            plt.title("Bondi test stability, {}".format(LONG))
            plt.legend()
            plt.savefig("bondi_compare_{}_{}.png".format(res, SHORT))

# MEASURE CONVERGENCE
fail = 0
for var in VARS:
    L1[var] = np.array(L1[var])
    powerfit = np.polyfit(np.log(RES), np.log(L1[var]), 1)[0]
    print("{} Powerfit: {} L1: {}".format(var, powerfit, L1[var]))
    if powerfit < -2.3 or powerfit > -1.85:
        fail = 1

# MAKE PLOTS
fig = plt.figure(figsize=(5,5))

for var in VARS:
    plt.plot(RES, L1[var], marker='s', label=var)

# Guideline at N^-2
# Key the guideline from the middle point
amp = L1['RHO'][len(RES)//2]*RES[len(RES)//2]**2
xmin = RES[0]/2.
xmax = RES[-1]*2.
plt.plot([xmin, xmax], amp*np.asarray([xmin, xmax])**-2., color='k', linestyle='--', label='N^-2')

plt.xscale('log', base=2); plt.yscale('log')
plt.xlim([RES[0]/np.sqrt(2.), RES[-1]*np.sqrt(2.)])
plt.xlabel('N'); plt.ylabel('L1')
plt.title("Bondi test convergence, {}".format(LONG))
plt.legend(loc=1)
plt.savefig("convergence_bondi_{}.png".format(SHORT))

exit(fail)
