#!/bin/sh
#
# submit_jobs.sh: main controller for submitting jobs to Euler or
# another cluster that is equipped with LFS.

DATA_SETS=(AIDS BZR BZR_MD COIL-DEL COIL-RAG COLLAB COX2 COX2_MD DBLP_v1 DD DHFR DHFR_MD ENZYMES ER_MD FIRSTMM_DB FRANKENSTEIN IMDB-BINARY IMDB-MULTI KKI Letter-high Letter-low Letter-med MSRC_21 MSRC_21C MSRC_9 MUTAG Mutagenicity NCI1 NCI109 OHSU PROTEINS PROTEINS_full PTC_FM PTC_FR PTC_MM PTC_MR Peking_1 REDDIT-BINARY REDDIT-MULTI-12K REDDIT-MULTI-5K SYNTHETIC SYNTHETICnew Synthie Tox21_AHR Tox21_AR Tox21_AR-LBD Tox21_ARE Tox21_ATAD5 Tox21_ER Tox21_ER_LBD Tox21_HSE Tox21_MMP Tox21_PPAR-gamma Tox21_aromatase Tox21_p53)

# TODO: need more kernels here...
KERNELS=(WL GL SP VH EH)

for DATA_SET in "${DATA_SETS[@]}"; do
  for KERNEL in "${KERNELS[@]}"; do
    bsub "run.sh ${DATA_SET} ${KERNEL}"
  done
done
