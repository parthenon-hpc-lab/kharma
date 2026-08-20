import numpy as np
import pickle
import pdb
import pyharm
import h5py

ref_rst="/n/holylfs05/LABS/bhi/Users/hyerincho/grmhd/data/restart.p"

def parse_by_block(string):
  params = {}
  for block in string.split("<"):
    if ">" in block:
      blockname=block.split(">")[0]
      for line in block.split("\n"):
        if "=" in line:
          ls = [token.strip().strip('()') for token in line.split("#")[0].split("=") if token != ""]
          try:
            if "." in ls[-1]:
              params[blockname+"/"+ls[0]] = float(ls[-1])
            else:
              params[blockname+"/"+ls[0]] = int(ls[-1])
          except ValueError:
            params[blockname+"/"+ls[0]] = ls[-1]
  return params

def read_rst(rst):
  restart_file = open(rst, 'rb')
  kwargs = pickle.load(restart_file)
  args = pickle.load(restart_file)
  restart_file.close()
  return kwargs, args

def create_rst_from_dump(fname,rst_out):
  # copy from a restart.p file
  kwargs,args = read_rst(ref_rst)

  # modify from last output dump
  #dump = pyharm.load_dump("00001/resize_restart_kharma.out0.00000.phdf")
  f = h5py.File(fname,'r')
  par_string = f['Input'].attrs['File']
  params = parse_by_block(par_string)

  for arg in args.keys():
    if arg in params:
      args[arg] = params[arg]
    else:
      print("WARNING! "+arg +" can't be found")

  for pair in (('nx1','parthenon/mesh/nx1'), ('nx2','parthenon/mesh/nx2'), ('nx3','parthenon/mesh/nx3'), ('nx1_mb','parthenon/meshblock/nx1'), ('nx2_mb','parthenon/meshblock/nx2'), ('nx3_mb','parthenon/meshblock/nx3'), ('nzones','resize_restart/nzone'), ('base','resize_restart/base'), ('nlim','parthenon/time/nlim'), ('bz', 'b_field/bz'), ('start_time','parthenon/time/start_time')):
    kwargs[pair[0]] = params[pair[1]]

  # things that were missed
  kwargs['r_b'] = np.power(float(args['bondi/rs']),2.)
  kwargs['start_run'] = int(fname.split('/')[0].split('_')[-1])
  if "bondi_multizone" in args["resize_restart/fname"]:
    args["resize_restart/fname"] = '/'.join(args["resize_restart/fname"].split("/")[-2:]).replace("bondi_multizone_","")
  if "bondi_multizone" in args["resize_restart/fname_fill"]:
    args["resize_restart/fname_fill"] = '/'.join(args["resize_restart/fname_fill"].split("/")[-2:]).replace("bondi_multizone_","")

  # write a new one
  restart_file = open(rst_out,'wb')
  pickle.dump(kwargs, restart_file)
  pickle.dump(args, restart_file)
  restart_file.close()

def update_rst(rst_out):
  # copy from a reference file
  kwargs,args = read_rst(ref_rst)

  kwargs["parfile"]=kwargs["parfile"].replace("/n/holylfs05/LABS/bhi/Users/hyerincho/grmhd/kharma_fork","/u/hcho1/kharma")
    
  # write a new one
  restart_file = open(rst_out,'wb')
  pickle.dump(kwargs, restart_file)
  pickle.dump(args, restart_file)
  restart_file.close()

def compare_rst(rst):
  kw1,a1=read_rst(rst)
  kw2,a2=read_rst(ref_rst)
  
  if kw1.keys()==kw2.keys():
    for key in kw1.keys():
      if kw1[key] != kw2[key]:
        print("kwargs["+key+"] different: ",kw1[key],kw2[key])
  else:
    print("WARNING: kwargs have different keys")

  if a1.keys()==a2.keys():
    for key in a1.keys():
      if a1[key] != a2[key]:
        print("args["+key+"] different: ",a1[key],a2[key])
  else:
    print("WARNING: args have different keys")

def _main():
  fname = "bondi_multizone_00299/resize_restart_kharma.out0.00000.phdf"
  rst_out = "restart.p"
  #create_rst_from_dump(fname,rst_out)
  #compare_rst(rst_out)
  #kw,a=read_rst(rst_out)
  #print(kw)

if __name__=="__main__":
  _main()
