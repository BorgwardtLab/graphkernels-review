#!/bin/bash
#
# calculate_runtime.sh: calculates runtime numbers for selected data
# sets.

#DATA_SETS=(AIDS BZR BZR_MD COX2 COX2_MD DD DHFR DHFR_MD ENZYMES ER_MD FIRSTMM_DB IMDB-BINARY IMDB-MULTI KKI Letter-high Letter-low Letter-med MSRC_21 MSRC_21C MSRC_9 MUTAG OHSU PROTEINS PROTEINS_full PTC_FM PTC_FR PTC_MM PTC_MR Peking_1 REDDIT-BINARY SYNTHETIC SYNTHETICnew Synthie)

DATA_SETS=(Letter-low IMDB-MULTI PTC_FM PTC_FR PTC_MM PTC_MR MUTAG BZR ENZYMES PROTEINS NCI1 NCI109 PROTEINS REDDIT-BINARY DD BZR)
KERNELS=(EH GL SP VH WL)

for DATA_SET in "${DATA_SETS[@]}"; do
  echo "Processing data set ${DATA_SET}..."
  for KERNEL in "${KERNELS[@]}"; do
    echo "  Kernel: ${KERNEL}"
    ./create_kernel_matrices.py -t -a $KERNEL -o ../matrices/${DATA_SET} ../data/${DATA_SET}/*.pickle
  done
done
