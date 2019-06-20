#!/bin/sh
#
# submit_jobs_benchmark_data_sets.sh: main controller for submitting
# jobs to Euler or another cluster that is equipped with LFS. Notice
# that this variant only uses some 'recognised' benchmark data sets,
# making it easier to obtain results.

DATA_SETS=(BZR BZR_MD COLLAB DD ENZYMES MUTAG NCI1 NCI109 PROTEINS PTC_FM PTC_FR PTC_MM PTC_MR)

# TODO: need more kernels here...
KERNELS=(WL GL SP VH EH)

for DATA_SET in "${DATA_SETS[@]}"; do
  for KERNEL in "${KERNELS[@]}"; do
    bsub -W 23:59 -R "rusage[mem=32000]" "./run.sh ${DATA_SET} ${KERNEL}"
  done
done
