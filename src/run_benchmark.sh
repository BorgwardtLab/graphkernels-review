#!/bin/bash
#
# run_benchmark.sh: basic job running script for clusters with LSF support. Given
# a data set name and an algorithm, will create a kernel matrix, and do
# the training of said matrix.
#
# This script is intended to run graph kernels on benchmarked datasets that 
# have predefined train/test splits.
#
# All results will be stored in `results_run`.
#
# Usage:
#   run_benchmark.sh DATA ALGORITHM [MAX_ITERATIONS]
#
# Parameters:
#   DATA: Essentially, just a folder name in `data`. All files are taken
#   from there.
#
#   ALGORITHM: An abbreviation of one of the algorithms to run. One of
#   these days, I will document all of them in the repo.
#
#   MAX_ITERATIONS: An optional argument indicating the maximum number
#   of iterations for the SVM.
#
#
# Requirements:
#   - The data (.pickle files) files are stored under /cluster/scratch in the 
#     user's directory. The user needs to store the training/testing data in 
#     subfolders called `train` and `test`.
#
# NOTE: 
#   The kernel matrices will (to prevent quota issues) also be stored on scratch

USER=$(whoami)

python3 ../src/create_kernel_matrices_benchmark.py -a $2 -o /cluster/scratch/$USER/matrices/$1 -train /cluster/scratch/$USER/$1/train/*.pickle -test /cluster/scratch/$USER/$1/test/*.pickle 

if [ -n "$3" ]; then
  python3 ../src/train_benchmark.py /cluster/scratch/$USER/matrices/$1/$2.npz -n $1 -o ../results/$1_$2.json -I $3
else
  python3 ../src/train_benchmark.py /cluster/scratch/$USER/matrices/$1/$2.npz -n $1 -o ../results/$1_$2.json
fi
