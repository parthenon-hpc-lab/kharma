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
@click.option('--nruns', default=3000, help="Total number of runs to perform")
@click.option('--spin', default=0.0, help="BH spin")
@click.option('--bz', default=0.0, help="B field Z component. Zero for no field")
@click.option('--tlim', default=None, help="Enforce a specific tlim for every run (for testing)")
@click.option('--nlim', default=-1, help="Consistent max number of steps for each run")
@click.option('--r_b', default=1.e5, help="Bondi radius. None chooses based on nzones")
@click.option('--jitter', default=0.0, help="Proportional jitter to apply to starting state. Default 10% w/B field")
# Flags and options
@click.option('--kharma_bin', default="kharma.cuda", help="Name (not path) of KHARMA binary to run")
@click.option('--kharma_args', default="", help="Arguments for KHARMA run.sh")
@click.option('--short_t_out', is_flag=True, help="Use shorter outermost annulus")
@click.option('--restart', is_flag=True, help="Restart from most recent run parameters")
@click.option('--parfile', default=None, help="Parameter filename")
@click.option('--gizmo', is_flag=True, help="Start from GIZMO data")
@click.option('--gizmo_fname', default="../gizmo_data.txt", help="Filename of GIZMO data")
@click.option('--ext_g', is_flag=True, help="Include external gravity")
# Don't use this
@click.option('--start_time', default=0.0, help="Starting time. Only use if you know what you're doing.")
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
        args['coordinates/r_in'] = base**turn_around
        # Initialize half-vacuum, unless it's the first GIZMO run
        if kwargs['gizmo']:
            args['bondi/r_shell'] = args['coordinates/r_in']
        else:
            args['bondi/r_shell'] = base**(turn_around+2)/2.

        # bondi & vacuum parameters
        # TODO derive these from r_b or gizmo
        if kwargs['nzones'] == 3 or kwargs['nzones'] == 6:
            kwargs['r_b'] = 256
            logrho = -4.13354231
            log_u_over_rho = -2.57960521
        elif kwargs['nzones'] == 4:
            kwargs['r_b'] = 256
            logrho = -4.200592800419657
            log_u_over_rho = -2.62430556
        elif kwargs['gizmo']:
            kwargs['r_b'] = 1e5
            logrho = -7.80243572
            log_u_over_rho = -5.34068635
        else:
            kwargs['r_b'] = 1e5
            logrho = -8.2014518
            log_u_over_rho = -5.2915149
        args['bondi/vacuum_logrho'] = logrho
        args['bondi/vacuum_log_u_over_rho'] = log_u_over_rho
        args['bondi/rs'] = np.sqrt(float(kwargs['r_b']))

        # B field additions
        if kwargs['bz'] != 0.0:
            # Set a field to initialize with 
            args['b_field/type'] = "r1s2" #"vertical"
            args['b_field/solver'] = "flux_ct"
            args['b_field/bz'] = kwargs['bz']
            # Compress coordinates to save time
            args['coordinates/transform'] = "mks"
            args['coordinates/hslope'] = 0.3
            # Enable the floors
            args['floors/disable_floors'] = False
            # And modify a bunch of defaults
            # Assume we will always want jitter if we have B unless a 2D problem
            if kwargs['jitter'] == 0.0 and kwargs['nx3']>1 :
                kwargs['jitter'] = 0.1
            # Lower the cfl condition in B field
            args['GRMHD/cfl'] = 0.5

        # Parameters directly from defaults/cmd
        args['perturbation/u_jitter'] = kwargs['jitter']
        args['coordinates/a'] = kwargs['spin']
        args['coordinates/ext_g'] = kwargs['ext_g']
        args['bondi/use_gizmo'] = kwargs['gizmo']
        args['gizmo_shell/datfn'] = kwargs['gizmo_fname']
        args['parthenon/time/nlim'] = kwargs['nlim']
        # Mesh size
        args['parthenon/mesh/nx1'] = kwargs['nx1']
        args['parthenon/mesh/nx2'] = kwargs['nx2']
        args['parthenon/mesh/nx3'] = kwargs['nx3']
        args['parthenon/meshblock/nx1'] = kwargs['nx1_mb']
        args['parthenon/meshblock/nx2'] = kwargs['nx2_mb']
        args['parthenon/meshblock/nx3'] = kwargs['nx3_mb']

    # Any derived parameters once we've loaded args/kwargs
    # Default parameters are in mz_dir
    if kwargs['parfile'] is None:
        kwargs['parfile'] = mz_dir+"/multizone.par"

    # Iterate, starting with the default args and updating as we go
    for run_num in np.arange(kwargs['start_run'], kwargs['nruns']):
        # run times for each annulus
        r_out = args['coordinates/r_out']
        r_b = float(kwargs['r_b'])
        base = args['resize_restart/base']
        outermost_zone = 2 * (kwargs['nzones'] - 1)
        if kwargs['tlim'] is None:
            # Calculate free-fall time
            if kwargs['short_t_out'] and run_num % outermost_zone == 0:
                runtime = calc_runtime(r_out/base, r_b)
                print("SHORT_T_OUT @ RUN # {}: r_out={:.4g}, but next largest annulus r_out={:.4g} used for the runtime".format(run_num, r_out, r_out/base))
            else:
                runtime = calc_runtime(r_out, r_b)
            # B field runs use half this
            if kwargs['bz'] != 0.0:
                runtime /= 10 # 2
        else:
            runtime = float(kwargs['tlim'])
        args['parthenon/time/tlim'] = kwargs['start_time'] + runtime

        # Output timing (TODO make options)
        args['parthenon/output0/dt'] = max(int(runtime/10.), 1)
        args['parthenon/output1/dt'] = max(int(runtime/5.), 1)
        args['parthenon/output2/dt'] = runtime/100.

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

        # Update parameters for the next pass
        # This updates both kwargs (start_time) and args (coordinates, dt, iteration #, fnames)
        update_args(run_num, kwargs, args)


def update_args(run_num, kwargs, args):
    # Update the dictionary of args to prepare for the *next* run (run_num+1).
    # Called after the first run is finished, and after each run subsequently

    # We'll always be restarting after the first run
    args['parthenon/job/problem_id']="resize_restart_kharma"

    # Filename to restart from
    fname_dir = "{:05d}".format(run_num)
    fname=glob.glob(fname_dir+"/*final.rhdf")[0]
    # Get start_time, ncycle, dt from previous run
    start_time = pyharm.io.get_dump_time(fname)
    kwargs['start_time'] = start_time
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
    outermost = False # about to simulate the outermost annulus
    innermost = False # go back to innermost annulus
    if run_num > 0 and int(np.log(last_r_in)/np.log(kwargs["base"])) == kwargs['nzones']-1:
        iteration += 1
        innermost = True
    if int(np.log(last_r_in)/np.log(kwargs['base'])) == kwargs['nzones']-2:
        outermost = True
    args['resize_restart/iteration'] = iteration

    # Are we moving inward?
    out_to_in=-1 # if iteration odd, out_to_in=1, if even, out_to_in=-1

    # Choose timestep and radii for the next run: smaller/larger as we step in/out
    if innermost:
        fname_dir = "{:05d}".format(run_num-(kwargs['nzones']-2))
        fname=glob.glob(fname_dir+"/*final.rhdf")[0]

        args['coordinates/r_out'] = np.power(kwargs['base'],2)
        args['coordinates/r_in'] = np.power(kwargs['base'],0)
        fname_fill_dir = data_dir(run_num - (kwargs['nzones']-1))
        fname_fill = glob.glob(fname_fill_dir+"/*final.rhdf")[0]

        f = h5py.File(fname_fill, 'r')
        args['parthenon/time/start_time'] = start_time
        args['resize_restart/use_ti'] = False
        args['parthenon/time/dt'] = f['Params'].attrs['Globals/dt_last']
        f.close()
    else:
        args['resize_restart/use_ti'] = True
        args['parthenon/time/dt'] = max(dt_last * kwargs['base']**(-3./2.*out_to_in) / 4, 1e-5)
        if out_to_in > 0:
            args['coordinates/r_out'] = last_r_out / kwargs['base']
            args['coordinates/r_in'] = last_r_in / kwargs['base']
        else:
            args['coordinates/r_out'] = last_r_out * kwargs['base']
            args['coordinates/r_in'] = last_r_in * kwargs['base']

        # Get filename to fill in the rest that fname doesn't cover
        if run_num + 1 < kwargs['nzones']:
            fname_fill = "none"
        else:
            # TODO explain why this number is correct
            #fname_fill_dir = data_dir(2 * (iteration - 1) * (kwargs['nzones'] - 1) - (run_num + 1))
            if outermost:
                fname_fill_dir = data_dir(run_num-(kwargs['nzones']-1))
            else:
                fname_fill_dir = data_dir(run_num-(kwargs['nzones']-2))
            fname_fill = glob.glob(fname_fill_dir+"/*final.rhdf")[0]
    args['resize_restart/fname'] = fname
    args['resize_restart/fname_fill'] = fname_fill

if __name__=="__main__":
  run_multizone()
