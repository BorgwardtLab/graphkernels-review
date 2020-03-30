#!/bin/bash

DATA_SETS=(MUTAG PROTEINS IMDB-BINARY IMDB-MULTI PTC_MR NCI1)

if [ -z ${KERNELS+x} ]; then
  KERNELS=(EH HGKSP_seed0 HGKWL_seed0 MLG MP VH WL WLOA)
fi

# Check whether the number of iterations has been set from the outside.
# If not, set it to a reasonable default value.
if [ -z ${N_ITERS+x} ]; then
  N_ITERS=500
fi

MEMORY=64000
TIME=119:59

for DATA_SET in "${DATA_SETS[@]}"; do
  for KERNEL in "${KERNELS[@]}"; do
    bsub -W ${TIME} -R "rusage[mem=${MEMORY}]" "./run_directly.sh ${DATA_SET} ${KERNEL}"
  done
done
