#!/bin/bash
#
# submit_multiclass_jobs.sh: main controller for submitting jobs to
# Euler or another cluster that is equipped with LFS. This one only
# submits jobs that contain mult-class data sets.

DATA_SETS=(ENZYMES IMDB-MULTI Letter-high Letter-low Letter-med MSRC_21 MSRC_21C MSRC_9 Synthie)

if [ -z ${KERNELS+x} ]; then
  KERNELS=(EH GL HGKSP_seed0 HGKWL_seed0 MLG MP SP VH WL WLOA)
fi

for DATA_SET in "${DATA_SETS[@]}"; do
  for KERNEL in "${KERNELS[@]}"; do

    # The SP kernel requires more memory and more time...
    MEMORY=64000
    TIME=23:59
    if [ ${KERNEL} = "SP" ]; then
      MEMORY=128000
      TIME=119:59
    fi

    bsub $MAIL -W ${TIME} -R "rusage[mem=${MEMORY}]" "./run.sh ${DATA_SET} ${KERNEL}"
  done
done

# Handle large data sets by submitting them to an even longer queue
# where they will hopefully be executed at some point.

LARGE_DATA_SETS=(COIL-DEL COIL-RAG COLLAB REDDIT-MULTI-5K REDDIT-MULTI-12K)

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

# TODO: this list includes data sets that are really, really large. They
# might require a better handling strategy.
DATA_SETS_EXTRA_LARGE=(Tox21_AHR Tox21_AR Tox21_AR-LBD Tox21_ARE Tox21_ATAD5 Tox21_ER Tox21_ER_LBD Tox21_HSE Tox21_MMP Tox21_PPAR-gamma Tox21_aromatase Tox21_p53)
