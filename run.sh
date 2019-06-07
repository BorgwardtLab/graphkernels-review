#!/bin/sh
#
# run.sh: basic job running script for clusters with LSF support. Given
# a data set name and an algorithm, will create a kernel matrix, and do
# the training of said matrix.
#
# All results will be stored in `results_run`.
#
# Usage:
#   run.sh DATA ALGORITHM
#
# Parameters:
#   DATA: Essentially, just a folder name in `data`. All files are taken
#   from there.
#
#   ALGORITHM: An abbreviation of one of the algorithms to run. One of
#   these days, I will document all of them in the repo.

python3 ../src/create_kernel_matrices.py -o ./matrices/$1 ./data/$1/*.pickle
