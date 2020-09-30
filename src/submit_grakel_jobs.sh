#!/bin/bash
#
# submit_grakel_jobs.sh: main control script for submitting
# `grakel`-based jobs to Euler.

DATA_SETS=(AIDS BZR BZR_MD COX2 COX2_MD DD DHFR DHFR_MD ENZYMES ER_MD FIRSTMM_DB IMDB-BINARY IMDB-MULTI KKI Letter-high Letter-low Letter-med MSRC_21 MSRC_21C MSRC_9 MUTAG OHSU PROTEINS PROTEINS_full PTC_FM PTC_FR PTC_MM PTC_MR Peking_1 REDDIT-BINARY SYNTHETIC SYNTHETICnew Synthie)

if [ -z ${KERNELS+x} ]; then
  KERNELS=(SP_gkl)
fi

for DATA_SET in "${DATA_SETS[@]}"; do
  for KERNEL in "${KERNELS[@]}"; do

    # The SP kernel requires more memory and more time...
    MEMORY=64000
    TIME=23:59
    if [ ${KERNEL} = "SP_gkl" ]; then
      MEMORY=128000
      TIME=119:59
    fi

    bsub $MAIL -W ${TIME} -R "rusage[mem=${MEMORY}]" "./run_grakel.sh ${DATA_SET} ${KERNEL}"
  done
done

# Handle large data sets by submitting them to an even longer queue
# where they will hopefully be executed at some point.

LARGE_DATA_SETS=(COIL-DEL COIL-RAG COLLAB DBLP_v1 FRANKENSTEIN Mutagenicity NCI1 NCI109 REDDIT-MULTI-5K REDDIT-MULTI-12K)

# Check whether the number of iterations has been set from the outside.
# If not, set it to a reasonable default value.
if [ -z ${N_ITERS+x} ]; then
  N_ITERS=1000
fi

for DATA_SET in "${LARGE_DATA_SETS[@]}"; do
  for KERNEL in "${KERNELS[@]}"; do

    # The SP kernel requires more memory...
    MEMORY=64000
    if [ ${KERNEL} = "SP_gkl" ]; then
      MEMORY=128000
    fi

    bsub $MAIL -W 119:59 -R "rusage[mem=${MEMORY}]" "./run_grakel.sh ${DATA_SET} ${KERNEL} ${N_ITERS}"
  done
done
