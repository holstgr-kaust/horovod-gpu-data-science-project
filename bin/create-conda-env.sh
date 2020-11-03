#!/bin/bash --login

set -e

# TODO: script expects CUDA_HOME (verify here)
# TODO: script expects `conda` (verify here)
# TODO: script should run from project root (SCRIPT_ROOT/..)
# TODO: fullpath to `environment.yml` and `postBuild`

# set relevant build variables for horovod
export ENV_PREFIX="${PWD}/env"
export NCCL_HOME=${ENV_PREFIX}
export HOROVOD_CUDA_HOME=${CUDA_HOME}
export HOROVOD_NCCL_HOME=${NCCL_HOME}
export HOROVOD_GPU_OPERATIONS=NCCL

# request builds for specific framework
export HOROVOD_WITH_TENSORFLOW=1
export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITH_MXNET=1

# create the conda environment
conda env create --prefix "${ENV_PREFIX}" --file environment.yml --force
conda activate "${ENV_PREFIX}"
source postBuild
