#!/usr/bin/env zsh
#
# create_mlg_matrices.sh: supporting script to create MLG kernel
# matrices, i.e. Multi-Scale Laplacian Graph Kernel matrices. It
# follows the parameter results from the original paper to write
# a set of kernel matrices for different parameters to a file. I
# took all parameters from
#
#   The Multiscale Laplacian Graph Kernel
#   Risi Kondor and Horace Pan
#
#   https://arxiv.org/pdf/1603.06186.pdf

ETA_GAMMA_GRID=(0.01 0.1 1)
RADIUS_GRID=(1 2 3 4)
LEVEL_GRID=(1 2 3 4)

# This is taken from the original `sample.sh` script from the GitHub
# repository of the project.
NUM_THREADS=32
GROW=1

# Get parameters from command-line and prepare all paths
NAME=$1
DATA=../data/$1/$1_A.txt
FEATURES=../data/$1/$1_N.txt
SAVE_PATH=../matrices/$1/MLG

# TODO: make configurable?
BIN="$HOME/Projects/MLGkernel/MLGkernel/runMLG"

for ETA in $ETA_GAMMA_GRID; do
  for GAMMA in $ETA_GAMMA_GRID; do
    for R in $RADIUS_GRID; do
      for L in $LEVEL_GRID; do
        echo "$ETA, $GAMMA, $R, $L"
        $BIN -d $DATA -f $FEATURES -s ${SAVE_PATH}_${ETA}_${GAMMA}_${R}_${L}.txt -e $ETA -g $GAMMA -r $R -l $L -t $NUM_THREADS -m $GROW
      done
    done
  done
done
