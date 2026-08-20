
import numpy as np
import h5py
import matplotlib.pyplot as plt

import pyharm

f = pyharm.load_dump("hubble.out0.final.phdf")

gam  = 1.666667
game = 1.333333

mach = 1
v0 = 1e-3
fel0 = 1.0
t = f['t']

rho0 = (mach/v0) * np.sqrt(gam*(gam-1))
ug0  = (v0/mach) / np.sqrt(gam*(gam-1))
u0 = fel0 * ug0

# Total number of cells
x = np.linspace(0.0, 1.0, f['Kel_Constant'].shape[0])

fig, ax = plt.subplots(2,2, figsize=(10,10))
ax[0, 0].plot(x,f['uvec'][0,:,0,0])
ax[0, 0].plot(x, v0*x / (1 + v0 * t))
ax[0, 0].set_title("vx")

ax[0, 1].plot(x,f['rho'][:,0,0])
ax[0, 1].plot(x, rho0 / (1 + v0 * t) * np.ones_like(x))
ax[0, 1].set_title("rho")

ax[1, 0].plot(x,f['u'][:,0,0])
ax[1, 0].plot(x, ug0 / (1 + v0 * t)**2 * np.ones_like(x))
ax[1, 0].set_title("u")

kap = (gam - 2) * (game - 1) / (game - 2) * u0 / rho0**game * (1 + v0 * t)**(game - 2)
ax[1, 1].plot(x, f['Kel_Constant'][:,0,0])
ax[1, 1].plot(x, kap*np.ones_like(x))
ax[1, 1].set_title("kappa_e")

plt.savefig("hubble.png")
