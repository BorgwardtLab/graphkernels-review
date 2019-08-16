#!/usr/bin/env bash
#
# create_mp_matrix.sh: supporting script to create a kernel matrix for
# the message passing graph kernel. It follows the implementation from
# the following paper:
#
#   Message Passing Graph Kernels
#   Giannis Nikolentzos, Michalis Vazirgiannis
#
#   https://arxiv.org/pdf/1808.02510
#
# Kernel matrices are stored in a single file.

# Get parameters from command-line and prepare all paths
NAME=$1
ROOT=../raw
OUTPUT=../matrices/$1

# TODO: make configurable?
BIN="$HOME/Projects/message_passing_graph_kernels/MPGK_AA.py"

OUT="$OUTPUT/MP.npz"

if [ -f "$OUT" ]; then
  echo "Output matrix $OUT already exists. Not creating a new job."
else
  bsub -W 23:59 -R "rusage[mem=256000]" "python $BIN -a -l $ROOT -o $OUTPUT $NAME"
fi
