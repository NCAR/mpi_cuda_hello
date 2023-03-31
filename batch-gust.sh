#!/bin/bash -l
#PBS -A SCSG0001
#PBS -q main
#PBS -l select=2:ncpus=64:mpiprocs=4:ngpus=4
#PBS -N mpi_cuda_hw
#PBS -j oe
#PBS -l walltime=01:10:00

module purge
module load ncarenv/23.03 craype/2.7.20 nvhpc/23.1 cuda/11.7.1 cray-mpich/8.1.25 ncarcompilers/0.8.0

mpiexec -n 8 -ppn 4 get_local_rank ./hello > hello.out
