import numpy as np
import os, sys, h5py, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

from pyharm.grid import make_some_grid

if __name__=='__main__':
	outputdir = './'
	kharmadir = '../../'
	RES = [int(r) for r in sys.argv[1].split(",")]
	
	CONDUCTION = 1
	if CONDUCTION: 
		NVAR = 5
		VARS = ['rho', 'u', 'u1', 'q', 'deltaP']
	else: 
		NVAR = 4
		VARS = ['rho', 'u', 'u1', 'deltaP']

	L1_norm = np.zeros([len(RES), NVAR])

	for r, res in enumerate(RES):

		# load analytic result
		rho_analytic	 = np.loadtxt(os.path.join(outputdir, 'shock_soln_{}_default'.format(res), 'shock_soln_rho.txt'))
		u_analytic		 = np.loadtxt(os.path.join(outputdir, 'shock_soln_{}_default'.format(res), 'shock_soln_u.txt'))
		u1_analytic		 = np.loadtxt(os.path.join(outputdir, 'shock_soln_{}_default'.format(res), 'shock_soln_u1.txt'))
		if CONDUCTION:
			q_analytic   = np.loadtxt(os.path.join(outputdir, 'shock_soln_{}_default'.format(res), 'shock_soln_q.txt'))
		dP_analytic    = np.loadtxt(os.path.join(outputdir, 'shock_soln_{}_default'.format(res), 'shock_soln_dP.txt'))
		x_analytic     = np.loadtxt(os.path.join(outputdir, 'shock_soln_{}_default'.format(res), 'shock_soln_xCoords.txt'))

		# load code data
		dfile = h5py.File('emhd_1d_{}_end.h5'.format(res), 'r')

		rho       = np.squeeze(dfile['prims'][Ellipsis,0][()])
		u         = np.squeeze(dfile['prims'][Ellipsis,1][()])
		u1        = np.squeeze(dfile['prims'][Ellipsis,2][()])
		if CONDUCTION:
				q_tilde   = np.squeeze(dfile['prims'][Ellipsis,8][()])
		dP_tilde  = np.squeeze(dfile['prims'][Ellipsis,9][()])

		t   = dfile['t'][()]
		gam = dfile['header/gam'][()]
		hot = dfile['header/higher_order_terms'][()]
		higher_order_terms = hot.decode('UTF-8') if isinstance(hot, bytes) else ("TRUE" if bool(hot) else "FALSE")
		conduction_alpha = dfile['header/conduction_alpha'][()]
		viscosity_alpha  = dfile['header/viscosity_alpha'][()]
		dfile.close()

		# compute q, dP
		P   = (gam - 1.) * u
		Theta = P / rho
		cs2 = (gam * P) / (rho + (gam * u))

		tau = 0.1
		chi_emhd = conduction_alpha * cs2 * tau
		nu_emhd  = viscosity_alpha * cs2 * tau

		if higher_order_terms=="TRUE":
			if CONDUCTION:
				q  = q_tilde * np.sqrt(chi_emhd * rho * Theta**2 / tau)
			dP = dP_tilde * np.sqrt(nu_emhd * rho * Theta / tau)
		else :
			if CONDUCTION:
				q  = q_tilde
			dP = dP_tilde
    
		# compute L1 norm
		L1_norm[r,0] = np.mean(np.fabs(rho-rho_analytic))
		L1_norm[r,1] = np.mean(np.fabs(u-u_analytic))
		L1_norm[r,2] = np.mean(np.fabs(u1-u1_analytic))
		if CONDUCTION:
			L1_norm[r,3] = np.mean(np.fabs(q-q_analytic))
			L1_norm[r,4] = np.mean(np.fabs(dP-dP_analytic))
		else:
			L1_norm[r,3] = np.mean(np.fabs(dP-dP_analytic))

	print("L1 norms vs resolution (%s):" % ",".join(VARS))
	for r_i, res in enumerate(RES):
		print(res, " ".join("%.3e" % v for v in L1_norm[r_i]))

	# plotting parameters
	mpl.rcParams['figure.dpi'] = 300
	mpl.rcParams['savefig.dpi'] = 300
	mpl.rcParams['figure.autolayout'] = True
	mpl.rcParams['axes.titlesize'] = 16
	mpl.rcParams['axes.labelsize'] = 14
	mpl.rcParams['xtick.labelsize'] = 12
	mpl.rcParams['ytick.labelsize'] = 12
	mpl.rcParams['axes.xmargin'] = 0.02
	mpl.rcParams['axes.ymargin'] = 0.02
	mpl.rcParams['legend.fontsize'] = 'medium'
	colors = ['indigo', 'goldenrod', 'darkgreen', 'crimson', 'xkcd:blue']


	# plot
	plt.close()
	fig = plt.figure(figsize=(6,6))
	ax = fig.add_subplot(1,1,1)

	# loop over prims
	tracker = 0
	for n in range(NVAR):
		color = colors[tracker]
		ax.loglog(RES, L1_norm[:,n], color=color, marker='o', label=VARS[n])
		tracker+=1

	ax.loglog([RES[0], RES[-1]], 10*np.asarray([float(RES[0]), float(RES[-1])])**(-2), color='k', linestyle='dashed', label='$N^{-2}$')
	#ax.loglog([RES[0], RES[-1]], 0.1*np.asarray([float(RES[0]), float(RES[-1])])**(-1), color='k', linestyle='dotted', label='$N^{-1}$')
	plt.xscale('log', base=2)
	ax.set_xlabel('Resolution')
	ax.set_ylabel('L1 norm')
	ax.legend()
	plt.savefig(os.path.join(outputdir,'emhd_shock_test_convergence_0.5M.png'), dpi=300)
