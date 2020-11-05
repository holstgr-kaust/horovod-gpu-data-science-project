#!/bin/bash --login

set -e

PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"

function usage () {
  echo "create-conda-env.sh [-e <env>] [-y <yaml>] [-h]"
  echo "  -e <env-prefix>        environment to create (local directory name)"
  echo "  -y <environment-yaml>  environment specification file path"
  echo "  -h  help"
}

# Use getopts to get arguments
#
while getopts ":e:y:h" opt; do
  case $opt in
    e)
      ENV_PREFIX=${OPTARG}
      ;;
    y)
      ENV_YAML=${OPTARG}
      ;;
    h)
      usage
      exit 0
      ;;
    [?])
      echo "Invalid option: -${OPTARG}" >&2;
      usage
      exit 1
      ;;
    :)
      echo "Option -${OPTARG} requires an argument." >&2;
      usage
      exit 1
      ;;
  esac
done
shift $((OPTIND-1))


# Load software stack
source "${PROJECT_ROOT}/bin/module.init"
source "${PROJECT_ROOT}/bin/conda.init"


# set relevant build variables for horovod
export ENV_PREVIX="${PROJECT_ROOT}/$(realpath --relative-to="${PROJECT_ROOT}" "${ENV_PREFIX:-env}")"
export ENV_YAML="${PROJECT_ROOT}/$(realpath --relative-to="${PROJECT_ROOT}" "${ENV_YAML:-environment.yml}")"

export NCCL_HOME="${ENV_PREFIX}"
export HOROVOD_CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT}}"
export HOROVOD_NCCL_HOME="${NCCL_HOME}"
export HOROVOD_GPU_OPERATIONS=NCCL

# request builds for specific framework
export HOROVOD_WITH_TENSORFLOW=1
export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITH_MXNET=1

# create the conda environment
conda env create --prefix "${ENV_PREFIX}" --file "${ENV_YAML}" --force
conda activate "${ENV_PREFIX}"
source "${PROJECT_ROOT}/postBuild"
