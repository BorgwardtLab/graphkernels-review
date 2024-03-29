#!/usr/bin/env bash

if [ -z ${DATA_SETS+x} ]; then
  DATA_SETS=(AIDS BZR BZR_MD COLLAB NCI1 NCI109 COIL-DEL COIL-RAG COX2 COX2_MD DBLP_v1 DD DHFR DHFR_MD ENZYMES ER_MD FIRSTMM_DB FRANKENSTEIN IMDB-BINARY IMDB-MULTI KKI Letter-high Letter-low Letter-med MSRC_21 MSRC_21C MSRC_9 MUTAG Mutagenicity OHSU PROTEINS PROTEINS_full PTC_FM PTC_FR PTC_MM PTC_MR Peking_1 REDDIT-BINARY REDDIT-MULTI-12K REDDIT-MULTI-5K SYNTHETIC SYNTHETICnew Synthie)
fi

for DATA_SET in "${DATA_SETS[@]}"; do
  ./create_mlg_matrices.sh ${DATA_SET}
done
