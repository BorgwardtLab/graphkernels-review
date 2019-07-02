#!/bin/bash
#
# submit_jobs.sh: main controller for submitting jobs to Euler or
# another cluster that is equipped with LFS.

DATA_SETS=(AIDS BZR BZR_MD COIL-DEL COIL-RAG COX2 COX2_MD DBLP_v1 DD DHFR DHFR_MD ENZYMES ER_MD FIRSTMM_DB FRANKENSTEIN IMDB-BINARY IMDB-MULTI KKI Letter-high Letter-low Letter-med MSRC_21 MSRC_21C MSRC_9 MUTAG Mutagenicity OHSU PROTEINS PROTEINS_full PTC_FM PTC_FR PTC_MM PTC_MR Peking_1 REDDIT-BINARY REDDIT-MULTI-12K REDDIT-MULTI-5K SYNTHETIC SYNTHETICnew Synthie)

if [ -z ${KERNELS+x} ]; then
  KERNELS=(EH GL MLG SP VH WL WLOA)
fi

for DATA_SET in "${DATA_SETS[@]}"; do
  for KERNEL in "${KERNELS[@]}"; do
    bsub -W 23:59 -R "rusage[mem=64000]" "./run.sh ${DATA_SET} ${KERNEL}"
  done
done

# Handle large data sets by submitting them to an even longer queue
# where they will hopefully be executed at some point.

LARGE_DATA_SETS=(COLLAB NCI1 NCI109)

for DATA_SET in "${LARGE_DATA_SETS[@]}"; do
  for KERNEL in "${KERNELS[@]}"; do
    bsub -W 48:00 -R "rusage[mem=64000]" "./run.sh ${DATA_SET} ${KERNEL}"
  done
done

# TODO: this list includes data sets that are really, really large. They
# might require a better handling strategy.
DATA_SETS_EXTRA_LARGE=(Tox21_AHR Tox21_AR Tox21_AR-LBD Tox21_ARE Tox21_ATAD5 Tox21_ER Tox21_ER_LBD Tox21_HSE Tox21_MMP Tox21_PPAR-gamma Tox21_aromatase Tox21_p53)
