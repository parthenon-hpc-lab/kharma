#!/usr/bin/env python

# This runs a "multi-zone" KHARMA sequence
# See --help

import os
import sys
import click
import glob
import subprocess
import pickle

import numpy as np
import h5py

import pyharm

def format_args(args):
    """Format a dict in var=val format for Parthenon"""
    arg_list = []
    for key in args.keys():
        arg_list += [key+"={}".format(args[key]).lower()]
    return arg_list


def calc_runtime(r_out, r_b):
    """r/v where v=sqrt(v_ff**2+c_s**2)"""
    #return r_out/np.sqrt(1./r_out + 1./r_b)
    return np.power(min(r_out,r_b),3./2)

def data_dir(n):
    """Data directory naming scheme"""
    return "{:05d}".format(n)

def calc_nx1(kwargs, r_out=None, r_in=None):#(given_nx1, nzones):
    """adjust to a new nx1 for a larger annulus to have effectively the same resolution as other annuli """
    if r_out is None: r_out = kwargs['r_out']
    if r_in is None: r_in = kwargs['r_in']
    nzones_plus_one = int(np.log(r_out/r_in)/np.log(kwargs['base'])) # equal to (nzones+1)
    given_nx1 = kwargs['nx1']
    nx1 = int((given_nx1/2.)*(nzones_plus_one))
    return nx1

@click.command()
# Run parameters
@click.option('--nx1', default=64, help="1-Run radial resolution")
@click.option('--nx2', default=64, help="1-Run theta resolution")
@click.option('--nx3', default=64, help="1-Run phi resolution")
@click.option('--nx1_mb', default=64, help="1-Run radial block resolution")
@click.option('--nx2_mb', default=32, help="1-Run theta block resolution")
@click.option('--nx3_mb', default=32, help="1-Run phi block resolution")
@click.option('--nzones', default=8, help="Total number of zones (annuli)")
@click.option('--base', default=8, help="Exponent base for annulus sizes")
@click.option('--nruns', default=300, help="Total number of runs to perform")
@click.option('--spin', default=0.0, help="BH spin")
@click.option('--bz', default=0.0, help="B field Z component. Zero for no field")
@click.option('--cfl', default=0.9, help="Courant condition fraction.  Defaults to 0.5 in B field")
@click.option('--tlim', default=None, help="Enforce a specific tlim for every run (for testing)")
@click.option('--tmax', default=None, help="Maximum time in units of Bondi time")
@click.option('--nlim', default=float(5e4), help="Consistent max number of steps for each run")
@click.option('--r_b', default=1.e5, help="Bondi radius. None chooses based on nzones")
@click.option('--jitter', default=0.0, help="Proportional jitter to apply to starting state. Default 10% w/B field")
# Flags and options
@click.option('--kharma_bin', default="kharma.cuda", help="Name (not path) of KHARMA binary to run")
@click.option('--kharma_args', default="", help="Arguments for KHARMA run.sh")
@click.option('--short_t_out', is_flag=True, help="Use shorter outermost annulus")
@click.option('--long_t_in', is_flag=True, help="Use longer time for innermost annulus")
@click.option('--restart', is_flag=True, help="Restart from most recent run parameters")
@click.option('--parfile', default=None, help="Parameter filename")
@click.option('--gizmo', is_flag=True, help="Start from GIZMO data")
@click.option('--gizmo_fname', default="../gizmo_data.txt", help="Filename of GIZMO data")
@click.option('--ext_g', is_flag=True, help="Include external gravity")
# Don't use this
@click.option('--start_time', default=0.0, help="Starting time. Only use if you know what you're doing.")
@click.option('--onezone', is_flag=True, help="Run onezone instead.")
@click.option('--lin_recon', is_flag=True, help="Use linear reconstruction instead of weno.")
@click.option('--combine_out_ann', is_flag=True, help="Combine outer annuli larger than Bondi radius.")
@click.option('--move_rin', is_flag=True, help="Move r_in instead of switching btw same sized annuli.")
@click.option('--gamma_max', default=10, help="Gamma_max floor.")
@click.option('--gamma', default=5./3, help="adiabatic index.")
@click.option('--rhomin', default=1e-6, help="rho min geom.")
@click.option('--umin', default=1e-8, help="u min geom.")
@click.option('--btype', default="r1s2", help="b field type")
@click.option('--coord', default=None, help="coordinate system")
@click.option('--df', is_flag=True, help="Use drift frame instead of normal when applying floors.")
def run_multizone(**kwargs):
    """This script runs a "multi-zone" KHARMA sequence.
    The idea is to divide a large domain (~1e8M radius) into several "zones,"
    then evolve them one at a time while keeping the others constant.
    This allows recovering long-term steady-state behavior quickly, by evolving each
    "zone" on its own timescale.
    Each run takes the final state of the last run, expands the domain inward or outward, and
    evolves the resulting domain/state.
    
    This mode now supports magnetic fields, arbitrary overlaps and coordinates, and other niceties.
    """
    # We're kept in a script subdirectory in kharma/
    mz_dir = os.path.dirname(os.path.realpath(__file__))
    # parent
    kharma_dir = mz_dir+"/../../.."
    # Get our name from the working dir
    run_name = os.getcwd().split("/")[-1]

    # Assign initial arguments, based on either:
    # 1. Loading last-started run when restarting
    # 2. Computing arguments from kwargs if beginning fresh
    if kwargs['restart']:
        # Crude, but I need to know what was passed to override on restore
        kwargs_save = {}
        for arg in [a.replace("-","").split("=")[0] for a in sys.argv[1:] if "-" in a]:
            kwargs_save[arg] = kwargs[arg]
        restart_file = open('restart.p', 'rb')
        kwargs = {**kwargs, **pickle.load(restart_file)}
        args = pickle.load(restart_file)
        restart_file.close()
        if kwargs['onezone']: 
            # if onezone, just inherit everything from restart.p except nruns
            kwargs['nruns'] = kwargs_save['nruns']
            update_args(kwargs['start_run'], kwargs, args)
            kwargs['start_run'] += 1
        else:
            for arg in kwargs_save.keys():
                if 'nlim' not in arg: # can change nlim from previous run
                    kwargs[arg] = kwargs_save[arg]
        args['parthenon/time/nlim'] = kwargs['nlim']
    else:
        # First run arguments
        base = kwargs['base']
        args = {}
        args['parthenon/job/problem_id'] = "bondi"
        args['resize_restart/base'] = base
        args['resize_restart/nzone'] = kwargs['nzones']
        args['resize_restart/iteration'] = 1
        kwargs['start_run'] = 0

        turn_around = kwargs['nzones'] - 1
        args['coordinates/r_out'] = base**(turn_around+2)
        if kwargs['onezone']:
            args['coordinates/r_in'] = 1
        else:
            args['coordinates/r_in'] = base**turn_around
        # Initialize half-vacuum, unless it's the first GIZMO run
        if kwargs['gizmo']:
            args['bondi/r_shell'] = 3e6 #args['coordinates/r_in']
        else:
            args['bondi/r_shell'] = base**(turn_around+2)/2.

        # bondi & vacuum parameters
        # TODO derive these from r_b or gizmo
        if args['coordinates/r_out'] < 1e5 and kwargs['bz']>1e-4: #kwargs['nzones'] == 3 or kwargs['nzones'] == 6:
            kwargs['r_b'] = 256
            logrho = -4.13354231
            log_u_over_rho = -2.57960521
        elif kwargs['nzones'] == 4:
            kwargs['r_b'] = 256
            logrho = -4.200592800419657
            log_u_over_rho = -2.62430556
        elif kwargs['gizmo']:
            #kwargs['r_b'] = 1e5
            logrho = -7.80243572
            log_u_over_rho = -5.34068635
        else:
            kwargs['r_b'] = 1e5
            logrho = -8.2014518
            log_u_over_rho = -5.2915149
        args['bondi/vacuum_logrho'] = logrho
        args['bondi/vacuum_log_u_over_rho'] = log_u_over_rho
        if abs(kwargs['gamma']- 5./3.)<1e-2:
            # only when gamma=5/3, rb=rs^2
            args['bondi/rs'] = np.sqrt(float(kwargs['r_b']))
        else:
            n = 1./(kwargs['gamma']-1)
            args['bondi/rs'] = (2*(n+3)-9)/(4*(n+1))*float(kwargs['r_b'])
        args['bondi/ur_frac'] = 0

        # B field additions
        if kwargs['bz'] != 0.0:
            # Set a field to initialize with 
            args['b_field/type'] = kwargs["btype"] #"r1s2" #"vertical"
            args['b_field/solver'] = "flux_ct"
            args['b_field/bz'] = kwargs['bz']
            # Compress coordinates to save time
            if kwargs['coord'] is not None:
                args['coordinates/transform'] = kwargs['coord']
                args['coordinates/lin_frac'] = 0.7
            elif kwargs['nx2'] >= 128 and not kwargs['onezone']:
                args['coordinates/transform'] = "fmks"
                args['coordinates/mks_smooth'] = 0.
                args['coordinates/poly_xt'] = 0.8
                args['coordinates/poly_alpha'] = 16
            else:
                args['coordinates/transform'] = "mks"
                args['coordinates/hslope'] = 0.3
            # Enable the floors
            args['floors/disable_floors'] = False
            args['floors/gamma_max'] = kwargs['gamma_max']
            if kwargs['df']:
                args['floors/frame'] = 'drift'
            # And modify a bunch of defaults
            # Assume we will always want jitter if we have B unless a 2D problem
            if kwargs['jitter'] == 0.0 and kwargs['nx3']>1 :
                kwargs['jitter'] = 0.1
            # Lower the cfl condition in B field
            args['GRMHD/cfl'] = 0.5
            if kwargs['lin_recon']:
                args['GRMHD/reconstruction'] = "linear_vl"
            else:
                # use weno5
                args['GRMHD/reconstruction'] = "weno5"
        args['GRMHD/gamma'] = kwargs["gamma"]
        args['floors/rho_min_geom'] = kwargs['rhomin']
        args['floors/u_min_geom'] = kwargs['umin']

        # Parameters directly from defaults/cmd
        args['perturbation/u_jitter'] = kwargs['jitter']
        args['GRMHD/cfl'] = kwargs['cfl']
        args['coordinates/a'] = kwargs['spin']
        args['coordinates/ext_g'] = kwargs['ext_g']
        args['bondi/use_gizmo'] = kwargs['gizmo']
        args['gizmo_shell/datfn'] = kwargs['gizmo_fname']
        args['parthenon/time/nlim'] = kwargs['nlim']

        # effective nzones (Hyerin 07/27/23)
        if (kwargs['combine_out_ann'] or kwargs['move_rin']) and not kwargs['onezone']:
            # think what's the smallest annulus where the logarithmic middle radius is larger than r_b 
            # (i.e. 8^n > 1e5 for base=8 r_b=1e5 where n is the nth smallest annulus)
            kwargs['nzones_eff'] = int(np.ceil(np.log(kwargs['r_b'])/np.log(kwargs['base'])))
            args['coordinates/r_in'] = base**(kwargs['nzones_eff']-1)
            if kwargs['base'] < 2: # this means that the second smallest annulu's r_in is inside the horizon
                args['coordinates/r_in'] = base**(kwargs['nzones_eff'])
        else:
            kwargs['nzones_eff'] = kwargs['nzones']
        args['resize_restart/nzone_eff'] = kwargs['nzones_eff']

        # Mesh size
        #if kwargs['onezone'] or kwargs['combine_out_ann'] or kwargs['move_rin']:
        args['parthenon/mesh/nx1'] = calc_nx1(kwargs,args['coordinates/r_out'],args['coordinates/r_in'])#kwargs['nzones'])#int((kwargs['nx1']/2.)*(kwargs['nzones']+1))
            #args['parthenon/mesh/nx1'] = calc_nx1(kwargs['nx1'],kwargs['nzones']-kwargs['nzones_eff']+1)
            #int((kwargs['nx1']/2.)*((kwargs['nzones']-kwargs['nzones_eff']+1)+1)) # number of zones for last annulus is nzones-nzones_eff+1
        #else:
            #args['parthenon/mesh/nx1'] = kwargs['nx1']
        args['parthenon/mesh/nx2'] = kwargs['nx2']
        args['parthenon/mesh/nx3'] = kwargs['nx3']
        args['parthenon/meshblock/nx1'] = args['parthenon/mesh/nx1']
        args['parthenon/meshblock/nx2'] = kwargs['nx2_mb']
        args['parthenon/meshblock/nx3'] = kwargs['nx3_mb']


    # Any derived parameters once we've loaded args/kwargs
    # Default parameters are in mz_dir
    if kwargs['parfile'] is None:
        kwargs['parfile'] = mz_dir+"/multizone.par"

    stop = False
    # Iterate, starting with the default args and updating as we go
    for run_num in np.arange(kwargs['start_run'], kwargs['nruns']):
        # run times for each annulus
        r_out = args['coordinates/r_out']
        r_b = float(kwargs['r_b'])
        base = args['resize_restart/base']
        #outermost_zone = 2 * (kwargs['nzones'] - 1)
        if kwargs['tlim'] is None:
            # Calculate free-fall time
            #if kwargs['short_t_out'] and run_num % outermost_zone == 0:
            #    runtime = calc_runtime(r_out/base, r_b)
            #    print("SHORT_T_OUT @ RUN # {}: r_out={:.4g}, but next largest annulus r_out={:.4g} used for the runtime".format(run_num, r_out, r_out/base))
            #else:
            if not kwargs['move_rin']: runtime = calc_runtime(r_out, r_b)
            else: runtime = calc_runtime(args['coordinates/r_in']*base**2,r_b)
            # B field runs use half this
            if kwargs['bz'] != 0.0:
                runtime /= np.power(base,3./2)*2 # half of free-fall time at the log middle radius
            if args['coordinates/r_out'] >= base**(kwargs['nzones']+1):
                # double the runtime for the outermost annulus
                runtime *= 2 
            if args['coordinates/r_in']<2:
                runtime *= 2 # double the runtime for innermost annulus
                if kwargs['long_t_in']:
                    print("LONG_T_IN @ RUN # {}: using longer runtime".format(run_num))
                    runtime *= 5 # 5 tff at the log middle radius
        else:
            runtime = float(kwargs['tlim'])

        tlim = kwargs['start_time'] + runtime
        if kwargs['onezone']: tlim = runtime
        if kwargs['tmax'] is None: tlim_max = 500.*np.power(r_b,3./2.) #1200
        else: 
            tlim_max = float(kwargs['tmax']) * np.power(r_b,3./2.)
            print(tlim_max)
        if tlim > tlim_max:
            stop = True
        args['parthenon/time/tlim'] = tlim #min(kwargs['start_time'] + runtime,10.*np.power(r_b,3./2))

        # Output timing (TODO make options)
        if kwargs['onezone']:
            runtime = calc_runtime(r_out, r_b)
        args['parthenon/output0/dt'] = max((runtime/10.), 1e-7) 
        args['parthenon/output1/dt'] = max((runtime/5.), 1e-7) #
        args['parthenon/output2/dt'] = runtime/10 #0.

        # Start any future run from this point
        kwargs['start_run'] = run_num

        # Now that we've determined all parameters, save them as used
        restart_file = open('restart.p', 'wb')
        pickle.dump(kwargs, restart_file)
        pickle.dump(args, restart_file)
        restart_file.close()
        # And print them
        print(run_name+": iter {}, run {} : radius {:.4g} to {:.4g}, time {:.4g} to {:.4g}".format(
                args['resize_restart/iteration'], run_num,
                args['coordinates/r_in'], args['coordinates/r_out'],
                kwargs['start_time'], args['parthenon/time/tlim']))

        ddir = data_dir(run_num)
        os.makedirs(ddir, exist_ok=True)
        fout = open(ddir+"/kharma.log", "w")
        if kwargs['kharma_bin'] not in ["", "kharma.cuda"]:
            kharma_bin_arg = ["-b", kwargs['kharma_bin']]
        else:
            kharma_bin_arg = []
        ret_obj = subprocess.run([kharma_dir+"/run.sh"] + kharma_bin_arg +
                      ["-i", kwargs['parfile'], "-d", ddir] + format_args(args),
                      stdout=fout, stderr=subprocess.STDOUT)
        fout.close()

        # Don't continue (& save restart data, etc) if KHARMA returned error
        if ret_obj.returncode != 0:
            print("KHARMA returned error: {}. Exiting.".format(ret_obj.returncode))
            exit(-1)
        if stop:
            print("tlim max reached!")
            break

        # Update parameters for the next pass
        # This updates both kwargs (start_time) and args (coordinates, dt, iteration #, fnames)
        update_args(run_num, kwargs, args)


def update_args(run_num, kwargs, args):
    # Update the dictionary of args to prepare for the *next* run (run_num+1).
    # Called after the first run is finished, and after each run subsequently

    # We'll always be restarting after the first run
    args['parthenon/job/problem_id']="resize_restart_kharma"

    # Filename to restart from
    fname_dir = data_dir(run_num)
    if kwargs['onezone']: fname = sorted(glob.glob(fname_dir+"/*.rhdf"))[-1]
    else: fname=glob.glob(fname_dir+"/*final.rhdf")[0]
    # Get start_time, ncycle, dt from previous run
    kwargs['start_time'] = pyharm.io.get_dump_time(fname)
    d = pyharm.load_dump(fname)
    iteration  = d['iteration']
    last_r_out = d['r_out']
    last_r_in = d['r_in']
    del d
    # TODO read all of Params/Info in pyharm
    f = h5py.File(fname, 'r')
    dt_last = f['Params'].attrs['Globals/dt_last']
    f.close()

    # Increment iteration count when we just finished the outermost zone
    if run_num > 0 and run_num % (kwargs['nzones_eff'] - 1) == 0:
        iteration += 1
    args['resize_restart/iteration'] = iteration

    # Are we moving inward?
    out_to_in=(-1)**(1+iteration) # if iteration odd, out_to_in=1, if even, out_to_in=-1
    # if out_to_in > 0:
    #   print("Moving inward:")
    # else:
    #   print("Moving outward:")

    # Choose timestep and radii for the next run: smaller/larger as we step in/out
    if not kwargs['onezone']:
        args['parthenon/time/dt'] = max(dt_last * kwargs['base']**(-3./2.*out_to_in) / 4, 1e-5)
        if out_to_in > 0:
            if not kwargs['move_rin']: args['coordinates/r_out'] = last_r_in * kwargs['base'] #last_r_out / kwargs['base']
            args['coordinates/r_in'] = last_r_in / kwargs['base']
        else:
            if not kwargs['move_rin']: args['coordinates/r_out'] = last_r_out * kwargs['base']
            args['coordinates/r_in'] = last_r_in * kwargs['base']
        
        if kwargs['combine_out_ann'] and args['coordinates/r_in']>= kwargs['base']**(kwargs['nzones_eff']-(kwargs['base']>2)):
            # if the next simulation is at the largest annulus,
            # make r_out and nx1 larger
            # if base < 2, the largest r_in is base^nzones_eff. if not, base^nzones_eff-1
            args['coordinates/r_out'] = kwargs['base']**(kwargs['nzones']+1)
            #args['parthenon/mesh/nx1'] = larger_ann_nx1(kwargs['nx1'],kwargs['nzones']-kwargs['nzones_eff']+1)
        args['parthenon/mesh/nx1'] = calc_nx1(kwargs,args['coordinates/r_out'],args['coordinates/r_in'])
        #else:
            #args['parthenon/mesh/nx1'] = kwargs['nx1'] # given nx1
        args['parthenon/meshblock/nx1'] = args['parthenon/mesh/nx1']


    # Get filename to fill in the rest that fname doesn't cover
    if run_num + 1 < kwargs['nzones_eff']:
        fname_fill = "none"
    else:
        # TODO explain why this number is correct
        fname_fill_dir = data_dir(2 * (iteration - 1) * (kwargs['nzones_eff'] - 1) - (run_num + 1))
        fname_fill = glob.glob(fname_fill_dir+"/*final.rhdf")[0]
        args['perturbation/u_jitter'] = 0. # jitter is turned off when not initializing.
    args['resize_restart/fname'] = fname
    args['resize_restart/fname_fill'] = fname_fill

if __name__=="__main__":
  run_multizone()
