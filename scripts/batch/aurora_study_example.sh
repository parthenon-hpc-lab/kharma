#!/bin/bash -l
#PBS -A agn_spin
#PBS -N KHARMA
#PBS -l select=8
#PBS -l walltime=168:00:00
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -l place=scatter
#PBS -q capacity
#PBS -j oe
#PBS -m bea
#PBS -M cprather@fas.harvard.edu

# OPTIONS
NRANKS=12
KHARMA_DIR=$HOME/Code/kharma

# What the actual hell, PBS.  Just start in the submit dir
cd ${PBS_O_WORKDIR}

# List nodes, because PBS also doesn't do that

cat $PBS_NODEFILE

# Now grab an array of them, we need to launch 1 job/node and
# this is the easy way to do it.
readarray -t nodes < $PBS_NODEFILE

# 12 is hard-coded in run.sh right now
NNODES=`wc -l < $PBS_NODEFILE`

# run.sh chooses this wrong
#export OMP_NUM_THREADS=4

BASE=$PWD
i=0
for flux in mad sane; do
for bmin in 0.1 1 10 100; do
  mkdir -p ${flux}_beta${bmin}
  cd ${flux}_beta${bmin}
  export MYNODEFILE=$PWD/nodes
  echo ${nodes[$i]} > $MYNODEFILE
  export MPI_EXTRA_ARGS="--hostfile=$MYNODEFILE"
  $KHARMA_DIR/run.sh -n $NRANKS -t 167:50:00 -i ../resupermad.par b_field/type=${flux} b_field/beta_min=${bmin} &
  cd $BASE
  i=$((i + 1))
done
done

# EXTREMELY IMPORTANT
wait
