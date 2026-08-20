import numpy as np
import matplotlib.pyplot as plt
import glob
import pyharm
from pyharm.plots.plot_dumps import plot_xz
import os #
import pdb

from make_plot import calc_Mdot, r_average, matplotlib_settings
from bondi_analytic import define_globals

def get_zone_num(dump):
    zone_num = int(np.log(dump["r_in"])/np.log(int(dump["base"])))
    return zone_num

def calc_eta(dump): # efficiency (Narayan et al. 2022)
    Pout = pyharm.shell_sum(dump, 'FE_norho', at_i=None) # FE_norho = -T^1_0 - rho*u^1
    return Pout/calc_Mdot(dump)

def calc_Edot(dump):
    return -pyharm.shell_sum(dump, 'FE')

def calc_Edot_EM(dump):
    return -pyharm.shell_sum(dump, 'FE_EM')

def calc_Edot_Fl(dump):
    return -pyharm.shell_sum(dump, 'FE_Fl')

def calc_Edot_EN(dump):
    return -pyharm.shell_sum(dump, 'FE_EN')

def calc_Edot_KE(dump):
    return -pyharm.shell_sum(dump, 'FE_PAKE')

def find_edge(dirtag):
    dirs=sorted(glob.glob("../data/"+dirtag+"/*[0-9][0-9][0-9][0-9][0-9]/"))

    # find the edge thru a backward search
    for i, dr in enumerate(dirs[::-1]):
        fname=sorted(glob.glob(dr+"*.rhdf"))[-1]
        dump = pyharm.load_dump(fname,ghost_zones=False)
        zone_num = get_zone_num(dump)
        if zone_num == 0 or zone_num == dump["nzone"]-1:
            edge_run = len(dirs)-1-i
            try:
                edge_iter = int(dump["iteration"])
            except:
                edge_iter = int(np.maximum(np.ceil(edge_run/(dump["nzone"]-1)),1) )
            break
    print(dirtag+" edge run: ", edge_run, " edge_iter ", edge_iter)
    return dirs, dump, edge_run, edge_iter

def plot_Mdot_eta(dirtag, ax_passed=None, color='k', lw=1, marker=None, label=None, show_Bondi=False, show_divisions=False, avg=False):
    if ax_passed is None:
        fig,ax = plt.subplots(1,2,figsize=(16,6))
    else:
        ax = ax_passed

    dirs, dump, edge_run, edge_iter = find_edge(dirtag)
    
    # run backwards nzone times from edge_run
    radii = np.array([])
    Mdot = np.array([])
    Edot = np.array([])
    Edot_EM = np.array([])
    Edot_Fl = np.array([])
    Edot_EN = np.array([])
    Edot_KE = np.array([])
    eta = np.array([])
    eta_EM = np.array([])
    eta_Fl = np.array([])
    eta_EN = np.array([])
    eta_KE = np.array([])
    eta_adv = np.array([])
    eta_conv = np.array([])
    n_zones = dump["nzone"]
    n_radii = len(dump["r1d"])

    # if averaging
    if avg:
        #FM_sum=[None]*n_zones # sum of rho*u^1 # this doesn't make a difference!
        Mdot_zones=[None]*n_zones
        eta_zones=[None]*n_zones
        eta_EM_zones=[None]*n_zones
        eta_Fl_zones=[None]*n_zones
        Edot_zones=[None]*n_zones
        Edot_EM_zones=[None]*n_zones
        Edot_Fl_zones=[None]*n_zones
        Edot_EN_zones=[None]*n_zones
        Edot_KE_zones=[None]*n_zones
        FM_zones=[None]*n_zones
        Be_nob_zones=[None]*n_zones
        num_sum=[0]*n_zones
        r_zones=[None]*n_zones
        gdet_zones=[None]*n_zones
        
        iteration = 2# edge_iter//2  # 1000 #
        for i in range(iteration*(n_zones-1)+1):
            files=sorted(glob.glob(dirs[edge_run-i]+"/*.phdf")) # HYERIN TEST rhdf -> phdf
            for file_ in files[len(files)//100:]:#2:]:  # only add last half
                dump = pyharm.load_dump(file_,ghost_zones=False)
                zone = get_zone_num(dump)
                if Mdot_zones[zone] is None: Mdot_zones[zone] = calc_Mdot(dump) #1+ r_average(dump, dump["ucov"][0]) #
                else: Mdot_zones[zone] +=  calc_Mdot(dump)#1+r_average(dump, dump["ucov"][0]) #
                #if eta_zones[zone] is None: eta_zones[zone] = calc_eta(dump)
                #else: eta_zones[zone] += calc_eta(dump)
                if Edot_zones[zone] is None: Edot_zones[zone] = calc_Edot(dump)
                else: Edot_zones[zone] += calc_Edot(dump)
                if Edot_EM_zones[zone] is None: Edot_EM_zones[zone] = calc_Edot_EM(dump)
                else: Edot_EM_zones[zone] += calc_Edot_EM(dump)
                if Edot_Fl_zones[zone] is None: Edot_Fl_zones[zone] = calc_Edot_Fl(dump)
                else: Edot_Fl_zones[zone] += calc_Edot_Fl(dump)
                if Edot_EN_zones[zone] is None: Edot_EN_zones[zone] = calc_Edot_EN(dump)
                else: Edot_EN_zones[zone] += calc_Edot_EN(dump)
                if Edot_KE_zones[zone] is None: Edot_KE_zones[zone] = calc_Edot_KE(dump)
                else: Edot_KE_zones[zone] += calc_Edot_KE(dump)
                if FM_zones[zone] is None: FM_zones[zone] = dump["FM"]
                else: FM_zones[zone] += dump["FM"]
                if Be_nob_zones[zone] is None: Be_nob_zones[zone] = dump["Be_nob"]
                else: Be_nob_zones[zone] += dump["Be_nob"]
                num_sum[zone]+=1
                if r_zones[zone] is None:
                    r_zones[zone] = dump["r1d"]
                if gdet_zones[zone] is None:
                    gdet_zones[zone] = dump["gdet"]
        print(num_sum)
        for zone in range(n_zones):
            #Mdot_zones[zone] = -np.squeeze(np.sum(FM_sum[zone]/num_sum[zone] * gdet_zones[zone] * dump['dx2'] * dump['dx3'],axis=(1,2)))
            Mdot_zones[zone] /= num_sum[zone]
            #eta_zones[zone] /= num_sum[zone]
            Edot_zones[zone] /= num_sum[zone]
            Edot_EM_zones[zone] /= num_sum[zone]
            Edot_Fl_zones[zone] /= num_sum[zone]
            Edot_EN_zones[zone] /= num_sum[zone]
            Edot_KE_zones[zone] /= num_sum[zone]
            FM_zones[zone] /= num_sum[zone]
            Be_nob_zones[zone] /= num_sum[zone]
            mask = np.full(n_radii, True, dtype=bool)
            if zone < n_zones-1:
                mask[-int(n_radii/4):] = False
            if zone > 0:
                mask[:int(n_radii/4)] = False
            radii = np.concatenate([radii,r_zones[zone][mask]])
            Mdot = np.concatenate([Mdot,Mdot_zones[zone][mask]])
            eta = np.concatenate([eta,(Mdot_zones[zone]-Edot_zones[zone])[mask]])#/Mdot_zones[zone]
            eta_EM = np.concatenate([eta_EM,(-Edot_EM_zones[zone])[mask]])#/Mdot_zones[zone]
            eta_Fl = np.concatenate([eta_Fl,(Mdot_zones[zone]-Edot_Fl_zones[zone])[mask]])#/Mdot_zones[zone]
            eta_EN = np.concatenate([eta_EN,(-Edot_EN_zones[zone])[mask]])#/Mdot_zones[zone]
            eta_KE = np.concatenate([eta_KE,(-Edot_KE_zones[zone])[mask]])#/Mdot_zones[zone]
            # convection 
            Edot_adv = np.squeeze(np.sum(Be_nob_zones[zone]*FM_zones[zone] * gdet_zones[zone] * dump['dx2'] * dump['dx3'],axis=(1,2)))
            Edot_conv = Mdot_zones[zone]-Edot_Fl_zones[zone]-Edot_adv
            eta_conv = np.concatenate([eta_conv,(Edot_conv)[mask]])#/Mdot_zones[zone]
            eta_adv = np.concatenate([eta_adv,(Edot_adv)[mask]])#/Mdot_zones[zone]
            #Edot = np.concatenate([Edot,Edot_zones[zone][mask]])
    else:
        for i in range(n_zones):
            fname=sorted(glob.glob(dirs[edge_run-i]+"/*.rhdf"))[-1]
            dump = pyharm.load_dump(fname,ghost_zones=False)
            zone = get_zone_num(dump)

            # trimming edges
            r1d = dump["r1d"]
            mask = np.full(n_radii, True, dtype=bool)
            if zone < n_zones-1:
                mask[-int(n_radii/4):] = False
            if zone > 0:
                mask[:int(n_radii/4)] = False

            radii = np.concatenate([radii,r1d[mask]])
            Mdot = np.concatenate([Mdot,calc_Mdot(dump)[mask]])
            eta = np.concatenate([eta,(1.-calc_Edot(dump)/calc_Mdot(dump))[mask]])
        
    order = np.argsort(radii)   
    radii = radii[order]
    Mdot = Mdot[order]
    #if avg:
    #    Edot = Edot[order]
    #    eta = 1.-Edot/Mdot[0]
    #else:
    eta = eta[order]
    eta_EM = eta_EM[order]
    eta_Fl = eta_Fl[order]
    eta_EN = eta_EN[order]
    eta_KE = eta_KE[order]
    eta_conv = eta_conv[order]
    eta_adv = eta_adv[order]

    ax[0].plot(radii, Mdot, color=color, marker=marker, lw=lw, label=label)
    ax[0].plot(radii, -Mdot, color=color, ls=':', marker=marker, lw=lw)
    ax[1].plot(radii, eta, color=color, marker=marker, lw=lw)
    ax[1].plot(radii, -eta, color=color, ls=':', marker=marker, lw=lw)
    if 1:
        ax[1].plot(radii, eta_EM, color='r', marker=marker, lw=lw,label='EM')
        ax[1].plot(radii, -eta_EM, color='r', ls=':', marker=marker, lw=lw)
        ax[1].plot(radii, eta_Fl, color='b', marker=marker, lw=lw,label='Fl')
        ax[1].plot(radii, -eta_Fl, color='b', ls=':', marker=marker, lw=lw)
        #ax[1].plot(radii, eta_EN, color='c', marker=marker, lw=lw,label='EN')
        #ax[1].plot(radii, -eta_EN, color='c', ls=':', marker=marker, lw=lw)
        #ax[1].plot(radii, eta_KE, color='m', marker=marker, lw=lw,label='KE')
        #ax[1].plot(radii, -eta_KE, color='m', ls=':', marker=marker, lw=lw)
        ax[1].plot(radii, eta_conv, color='orange', marker=marker, lw=lw,label='conv')
        ax[1].plot(radii, -eta_conv, color='orange', ls=':', marker=marker, lw=lw)
        ax[1].plot(radii, eta_adv, color='pink', marker=marker, lw=lw,label='adv')
        ax[1].plot(radii, -eta_adv, color='pink', ls=':', marker=marker, lw=lw)
        ax[1].legend()
    ax[0].legend()

    # show Bondi
    if show_Bondi:
        _,C1,_,_=define_globals(dump["rs"])
        ax[0].plot(radii,C1*4.*np.pi*np.ones(np.shape(radii)),color='grey', ls=':', alpha=.7, lw=8, zorder=-100)
    for ax_each in ax:

        # show annuli divisions
        if show_divisions:
            divisions = [8**i for i in range(n_zones+2)]
            for div in divisions:
                ax_each.axvline(div, alpha=0.2, color='grey', lw=1)

            ax_each.axvline(dump["rs"]**2, color='red', lw=1, alpha=0.2)
        
        # formatting
        ax_each.set_xscale('log'); ax_each.set_yscale('log')
        ax_each.set_xlim(left=2)
        ax_each.set_xlabel(pyharm.pretty('r'))

    ax[0].set_ylabel(pyharm.pretty('Mdot'))
    ax[1].set_ylabel(r'$\eta\dot{M}$')#r'$1-\dot{E}/\dot{M}$')
    ax[1].set_ylim(bottom=1e-10)

    if ax_passed is None:
        plt.savefig("./plots/plot_Mdot.png",bbox_inches='tight')
    else:
        return ax

def plot_shell_summed(ax,dump,x,var,mask=None,color='k',lw=5,already_summed=False):
    if not already_summed: var = np.squeeze(np.sum(var * dump['gdet'] * dump['dx2'] * dump['dx3'],axis=(1,2)))
    ax.plot(x[mask], var[mask],color=color,lw=lw)
    ax.plot(x[mask], -var[mask],color=color,lw=lw, ls=':')
    return ax


def compare_FE_slice(dirtag,avg=True):
    plt.rcParams.update({'font.size': 60})
    fig, ax = plt.subplots(4,3,figsize=(90,60),sharex=True, sharey='row', gridspec_kw={'height_ratios': [1, 4, 4, 1]})
    
    dirs, dump, edge_run, edge_iter = find_edge(dirtag)
    
    # run backwards nzone times from edge_run
    radii = np.array([])
    FE = np.array([])
    FE_Fl = np.array([])
    FE_EN = np.array([])
    FE_KE = np.array([])
    FE_adv = np.array([])
    FE_conv = np.array([])
    n_zones = dump["nzone"]
    r_out = np.power(int(dump["base"]),n_zones+1)
    if n_zones < 8:
        vmin=-1e-2
    else: vmin=-1e-3
    if "onezone" in dirtag or "oz" in dirtag:
        n_zones = 1
        r_out = dump["r_out"]
    n_radii = len(dump["r1d"])

    FM_zones=[None]*n_zones
    FE_zones=[None]*n_zones
    FE_Fl_zones=[None]*n_zones
    FE_EN_zones=[None]*n_zones
    FE_KE_zones=[None]*n_zones
    FE_adv_zones=[None]*n_zones
    FE_conv_zones=[None]*n_zones
    Be_nob_zones=[None]*n_zones
    num_sum=[0]*n_zones
    r_zones=[None]*n_zones
    dump_zones=[None]*n_zones
    
    iteration = 2000 #edge_iter//2  # 
    for i in range(iteration*(n_zones-1)+1):
        files=sorted(glob.glob(dirs[edge_run-i]+"/*.phdf")) # HYERIN TEST rhdf -> phdf
        for file_ in files[len(files)//2:]:  # only add last half
            dump = pyharm.load_dump(file_,ghost_zones=False)
            zone = get_zone_num(dump)
            if FM_zones[zone] is None: FM_zones[zone] = dump["FM"]
            else: FM_zones[zone] += dump["FM"]
            if FE_zones[zone] is None: FE_zones[zone] = dump["FE_norho"]
            else: FE_zones[zone] += dump["FE_norho"]
            if FE_Fl_zones[zone] is None: FE_Fl_zones[zone] = dump["FE_Fl"]-dump["rho"]*dump['ucon'][1]
            else: FE_Fl_zones[zone] += dump["FE_Fl"]-dump["rho"]*dump['ucon'][1]
            if FE_EN_zones[zone] is None: FE_EN_zones[zone] = dump["FE_EN"]
            else: FE_EN_zones[zone] += dump["FE_EN"]
            if FE_KE_zones[zone] is None: FE_KE_zones[zone] = dump["FE_PAKE"]
            else: FE_KE_zones[zone] += dump["FE_PAKE"]
            if Be_nob_zones[zone] is None: Be_nob_zones[zone] = dump["Be_nob"]
            else: Be_nob_zones[zone] += dump["Be_nob"]
            num_sum[zone]+=1
            if r_zones[zone] is None:
                r_zones[zone] = dump["r1d"]
            if dump_zones[zone] is None:
                dump_zones[zone] = dump
    print(num_sum)
    for zone in range(n_zones):
        FE_zones[zone] /= num_sum[zone]
        FE_Fl_zones[zone] /= num_sum[zone]
        FE_EN_zones[zone] /= num_sum[zone]
        FE_KE_zones[zone] /= num_sum[zone]
        Be_nob_zones[zone] /= num_sum[zone]
        FM_zones[zone] /= num_sum[zone]
        FE_adv_zones[zone] = Be_nob_zones[zone]*FM_zones[zone] # advection
        FE_conv_zones[zone] = FE_Fl_zones[zone]-FE_adv_zones[zone] # convection

        # masking
        mask = np.full(n_radii, True, dtype=bool)
        if zone < n_zones-1:
            mask[-int(n_radii/4):] = False
        if zone > 0:
            mask[:int(n_radii/4)] = False
        dump = dump_zones[zone]

        # plot 
        window = (np.log(2), np.log(r_out), 0,1)
        vmax=-vmin; lw= 4
        for i in [0,3]: ax[i,0].set_yscale('log'); ax[i,0].set_ylim([1e-6,-vmin])
        Edot_net = np.squeeze(np.sum(FE_zones[zone] * dump['gdet'] * dump['dx2'] * dump['dx3'],axis=(1,2)))
        x = np.log(r_zones[zone])
        for ax_each in np.array([ax[0],ax[3]]).reshape(-1): plot_shell_summed(ax_each,dump,x,Edot_net,mask=mask, color='k',already_summed=True)
        plot_shell_summed(ax[0,1],dump,x,FE_EN_zones[zone],mask=mask, color='b')
        plot_shell_summed(ax[0,2],dump,x,FE_KE_zones[zone],mask=mask, color='b')
        plot_shell_summed(ax[3,0],dump,x,FE_Fl_zones[zone],mask=mask, color='b')
        plot_shell_summed(ax[3,1],dump,x,FE_adv_zones[zone],mask=mask, color='b')
        plot_shell_summed(ax[3,2],dump,x,FE_conv_zones[zone],mask=mask, color='b')
        plot_xz(ax[1,0], dump, FE_zones[zone]     * dump["gdet"] , native=True, symlog=True, vmin=vmin,vmax=vmax, cbar=(zone==0),shading='flat',window=window,average=avg, mask=mask)#
        plot_xz(ax[2,0], dump, FE_Fl_zones[zone]  * dump["gdet"] , native=True, symlog=True, vmin=vmin,vmax=vmax, cbar=0,shading='flat',window=window,average=avg, mask=mask)
        plot_xz(ax[1,1], dump, FE_EN_zones[zone]  * dump["gdet"] , native=True, symlog=True, vmin=vmin,vmax=vmax, cbar=0,shading='flat',window=window,average=avg, mask=mask)
        plot_xz(ax[1,2], dump, FE_KE_zones[zone]  * dump["gdet"] , native=True, symlog=True, vmin=vmin,vmax=vmax, cbar=0,shading='flat',window=window,average=avg, mask=mask)
        plot_xz(ax[2,1], dump, FE_adv_zones[zone] * dump["gdet"] , native=True, symlog=True, vmin=vmin,vmax=vmax, cbar=0,shading='flat',window=window,average=avg, mask=mask)
        plot_xz(ax[2,2], dump, FE_conv_zones[zone]* dump["gdet"] , native=True, symlog=True, vmin=vmin,vmax=vmax, cbar=0,shading='flat',window=window,average=avg, mask=mask)

    
    ax[1,0].set_title('FE_tot')
    ax[2,0].set_title('FE_Fl')
    ax[1,1].set_title('FE_EN')
    ax[1,2].set_title('FE_KE')
    ax[2,1].set_title('FE_adv')
    ax[2,2].set_title('FE_conv')
    fig.tight_layout()
    fig.suptitle(dirtag[7:]+" runs: {}-{}".format(edge_run-iteration*(n_zones-1),edge_run), y=1.02)
    save_path="./plots/"+dirtag
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    savefig_name='/FE_slice.png'
    plt.savefig(save_path+savefig_name,bbox_inches='tight')
    plt.savefig("./plots/"+savefig_name,bbox_inches='tight') # make a copy here as well
    
def FE_adv_slice(dirtag,avg=True):
    matplotlib_settings()
    fig, ax = plt.subplots(1,3,figsize=(24,6),sharex=True, sharey=True)
    
    dirs, dump, edge_run, edge_iter = find_edge(dirtag)
    
    # run backwards nzone times from edge_run
    radii = np.array([])
    FE_adv = np.array([])
    n_zones = dump["nzone"]
    r_out = np.power(int(dump["base"]),n_zones+1)
    if n_zones < 8:
        vmin=-1e-2
    else: vmin=-1e-3
    if "onezone" in dirtag or "oz" in dirtag:
        n_zones = 1
        r_out = dump["r_out"]
    n_radii = len(dump["r1d"])

    FM_zones=[None]*n_zones
    FE_adv_zones=[None]*n_zones
    Be_nob_zones=[None]*n_zones
    num_sum=[0]*n_zones
    r_zones=[None]*n_zones
    dump_zones=[None]*n_zones
    
    iteration = 200 #edge_iter//2  # 
    for i in range(iteration*(n_zones-1)+1):
        files=sorted(glob.glob(dirs[edge_run-i]+"/*.phdf")) # HYERIN TEST rhdf -> phdf
        for file_ in files[len(files)//2:]:  # only add last half
            dump = pyharm.load_dump(file_,ghost_zones=False)
            zone = get_zone_num(dump)
            if FM_zones[zone] is None: FM_zones[zone] = dump["FM"]
            else: FM_zones[zone] += dump["FM"]
            if Be_nob_zones[zone] is None: Be_nob_zones[zone] = dump["Be_nob"]
            else: Be_nob_zones[zone] += dump["Be_nob"]
            num_sum[zone]+=1
            if r_zones[zone] is None:
                r_zones[zone] = dump["r1d"]
            if dump_zones[zone] is None:
                dump_zones[zone] = dump
    print(num_sum)
    for zone in range(n_zones):
        Be_nob_zones[zone] /= num_sum[zone]
        FM_zones[zone] /= num_sum[zone]
        FE_adv_zones[zone] = Be_nob_zones[zone]*FM_zones[zone] # advection

        # masking
        mask = np.full(n_radii, True, dtype=bool)
        if zone < n_zones-1:
            mask[-int(n_radii/4):] = False
        if zone > 0:
            mask[:int(n_radii/4)] = False
        dump = dump_zones[zone]

        # plot 
        window = (np.log(2), np.log(r_out), 0,1)
        vmax=-vmin; lw= 4
        x = np.log(r_zones[zone])
        plot_xz(ax[0], dump, FE_adv_zones[zone] * dump["gdet"] , native=True, symlog=True, vmin=vmin,vmax=vmax, cbar=0,shading='flat',window=window,average=avg, mask=mask)
        plot_xz(ax[1], dump, Be_nob_zones[zone] , native=True, symlog=True, vmin=vmin,vmax=vmax, cbar=0,shading='flat',window=window,average=avg, mask=mask)
        plot_xz(ax[2], dump, FM_zones[zone] * dump["gdet"] , native=True, symlog=True, vmin=vmin,vmax=vmax, cbar=0,shading='flat',window=window,average=avg, mask=mask)

    
    ax[0].set_title('FE_adv')
    ax[1].set_title('Be_nob')
    ax[2].set_title(r'$\rho u^r \sqrt{-g}$')
    fig.tight_layout()
    fig.suptitle(dirtag[7:]+" runs: {}-{}".format(edge_run-iteration*(n_zones-1),edge_run), y=1.02)
    save_path="./plots/"+dirtag
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    savefig_name='/FE_adv.png'
    plt.savefig(save_path+savefig_name,bbox_inches='tight')
    plt.savefig("./plots/"+savefig_name,bbox_inches='tight') # make a copy here as well

def _main():
    fig, ax = plt.subplots(1,2,figsize=(16,6))

    args={'show_Bondi': True, 'color':'k', 'show_divisions':False, 'avg':False}

    ### rB=1e5 runs
    # hydro
    #dirtag="bondi_multizone_030723_bondi_128^3"
    #dirtag="bondi_multizone_052423_bondi_64^3_n8_noshock"
    #dirtag="bondi_multizone_041823_bondi_n8b8"
    #dirtag="bondi_multizone_052223_bondi_128^3_n8_nox3split"
    #dirtag="bondi_multizone_022823_bondi_new_coord_noffp"
    #dirtag="062623_hd_ur0"
    #args['label']='HD'
    #ax = plot_Mdot_eta(dirtag, ax, **args)

    # 32^3
    dirtag="071323_beta01_64"
    args['avg']=True
    args['show_Bondi'] = False
    args['show_divisions'] = True
    args['color'] = 'g'
    args['label'] = (dirtag.replace('bondi_multizone_',''))[7:]
    #ax = plot_Mdot_eta(dirtag, ax, **args)
    #compare_FE_slice(dirtag)
    FE_adv_slice(dirtag)
    #dirtag="061323_diffinit_better"
    #ax = plot_Mdot_eta(dirtag, ax, **args)

    """
    # 96^3
    dirtag="production_runs/bondi_bz2e-8_1e8_96"
    args['color'] = 'r'
    args['label']='96^3'
    ax = plot_Mdot_eta(dirtag, ax, **args)
    """

    for ax_each in ax:
        ax_each.set_xlim(right=8**9*1.2)
    """

    ### rB=256 runs
    # hydro
    dirtag="bondi_multizone_050123_bflux0_0_64^3"
    args['label']='HD'
    ax = plot_Mdot_eta(dirtag, ax, lw=4, **args)

    # 32^3
    dirtag="bondi_multizone_042723_bflux0_1e-4_32^3"
    args['avg']=True
    args['show_Bondi'] = False
    args['color'] = 'g'
    args['label']='32^3'
    ax = plot_Mdot_eta(dirtag, ax, **args)

    # 64^3
    dirtag="bondi_multizone_042723_bflux0_1e-4_64^3"
    args['color'] = 'b'
    args['label']='64^3'
    ax = plot_Mdot_eta(dirtag, ax, **args)
    #dirtag="060123_bflux0_n4_64^3"
    #args['marker']='.'
    #ax = plot_Mdot_eta(dirtag, ax, **args)

    # 96^3
    dirtag="bondi_multizone_050123_bflux0_1e-4_96^3"
    args['color'] = 'r'
    args['label']='96^3'
    args['marker']=None
    ax = plot_Mdot_eta(dirtag, ax, **args)

    # 128^3
    dirtag="bondi_multizone_050523_bflux0_1e-4_128^3_n3_noshort"
    args['color'] = 'm'
    args['label']='128^3'
    ax = plot_Mdot_eta(dirtag, ax, **args)

    # weak field
    args['show_divisions'] = True
    args['color'] = 'pink'
    args['marker'] = None
    args['label']='weak'
    dirtag="bondi_multizone_050423_bflux0_1e-8_2d_n4"
    ax = plot_Mdot_eta(dirtag, ax, avg=False, **args)
    for ax_each in ax:
        ax_each.set_xlim(right=8**5*1.2)
    """


    #plt.savefig('./plots/plot_Mdot.png',bbox_inches='tight')

if __name__ == "__main__":
    _main()
