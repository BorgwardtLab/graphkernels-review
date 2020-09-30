#!/bin/bash

if [ -z ${KERNELS+x} ]; then
  KERNELS=(EH GL MLG MP SP VH WL WLOA)
fi

LARGE_DATA_SETS=(Tox21_AHR Tox21_AR Tox21_AR-LBD Tox21_ARE Tox21_ATAD5 Tox21_ER Tox21_ER_LBD Tox21_HSE Tox21_MMP Tox21_PPAR-gamma Tox21_aromatase Tox21_p53)

# Check whether the number of iterations has been set from the outside.
# If not, set it to a reasonable default value.
if [ -z ${N_ITERS+x} ]; then
  N_ITERS=1000
fi

for DATA_SET in "${LARGE_DATA_SETS[@]}"; do
  for KERNEL in "${KERNELS[@]}"; do

    # The SP kernel requires more memory...
    MEMORY=64000
    if [ ${KERNEL} = "SP" ]; then
      MEMORY=128000
    fi

    bsub $MAIL -W 119:59 -R "rusage[mem=${MEMORY}]" "./run.sh ${DATA_SET} ${KERNEL} ${N_ITERS}"
  done
done
