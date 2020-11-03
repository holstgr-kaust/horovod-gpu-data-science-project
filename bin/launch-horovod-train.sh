#!/bin/bash

# Note: if any step fails, cancel everything
set -e

function usage () {
  echo "launch-horovod-train.sh [-c] [-d <time>] [-g <gpus>] [-j <name>] [-k <gpu>] [-m <email>] [-n <nodes>] "
  echo "                        [-e <epochs>] [-l <lr>] [-t <epochs>]"
  echo "                        [--]"
	echo " resources:"
  echo "  -c                  request reference data constraint"
  echo "  -d <time>           job duration in hours"
  echo "  -g <gpus_total>     total number of GPUs (tasks)"
  echo "  -j <job_name>       job name (basic unique non-parameterized descriptive)"
  echo "  -k <gpu_type>       type of GPU: v100, p100, gtx1080ti, rtx2080ti"
  echo "  -m <email>          request notification mails"
  echo "  -n <nodes>          number of nodes"
  echo " training:"
  echo "  -b <batch_size>     base per-gpu batch size"
  echo "  -e <epochs>         number of epochs to train during a single job"
  echo "  -l <learning_rate>  base learning rate"
  echo "  -t <total_epochs>   total epochs to train for"
  echo " miscellaneous:"
  echo " --   remaining args are passed as parameters to the training script"
  echo "  -h  help"
}


# Use getopts to get arguments
#
while getopts ":cd:g:j:k:m:n:b:e:l:t:h" opt; do
  case $opt in
    c)
      SBATCH_CONSTRAINTS=${SBATCH_CONSTRAINTS} --constraint=[ref_32T]
      DATA_DIR=${DATA_DIR:-"/local/reference/CV/ILSVR/classification-localization/data/jpeg"}
      ;;
    d)
			TIME_HOURS=${OPTARG}
      ;;
		g)
			GPU_TOTAL=${OPTARG}
			;;
		j)
			JOB_NAME=${OPTARG}
			;;
    k)
			GPU_TYPE=${OPTARG}
			;;
    m)
      SBATCH_CONSTRAINTS=${SBATCH_CONSTRAINTS} --mailtype=FAIL,TIMEOUT,TIMEOUT_90 --mailuser=${OPTARG}
      ;;
		n)
			NODE_TOTAL=${OPTARG}
			;;
		b)
			BATCH_SIZE=${OPTARG}
			;;
    e)
      EPOCHS_PER_JOB=${OPTARG}
			;;
    l)
			LEARNING_RATE=${OPTARG}
			;;
    t)
			TOTAL_EPOCHS=${OPTARG}
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


# configs
GPU_TOTAL=${GPU_TOTAL:-8}
GPU_TYPE=${GPU_TYPE:-v100}
NODE_TOTAL=${NODE_TOTAL:-1}
TIME_HOURS=${TIME_HOURS:-24}

# TODO: calculate optimal _PER_GPU values based on GPU_TYPE / NODE_CONSTRAINT
MEM_PER_GPU=45
CPU_PER_GPU=4

TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
EPOCHS_PER_JOB=${EPOCHS_PER_JOB:-10}

LEARNING_RATE=${LEARNING_RATE:-1.25e-2}
DATA_DIR=${DATA_DIR:-"/ibex/reference/CV/ILSVR/classification-localization/data/jpeg"}
BATCH_SIZE=${BATCH_SIZE:-160}  # per-gpu: depends upon GPU memory size


TRAIN_PARAMETERS="--base-lr ${LEARNING_RATE} --batch-size ${BATCH_SIZE} --epochs ${EPOCHS_PER_JOB}"


PARAMETER_SPACE="${TRAIN_PARAMETERS//[-=\ .]/}"
PROJECT_VERSION=$(git rev-parse --short HEAD 2> /dev/null || echo 'unver')

GPU_PER_NODE=$((GPU_TOTAL / NODE_TOTAL))

JOB_NAME=${JOB_NAME:-"horovod-train"}-${PROJECT_VERSION}-${PARAMETER_SPACE}


## Note: TOTAL_JOBS = ceil(TOTAL_EPOCHS / EPOCHS_PER_JOB)
TOTAL_JOBS=$(((TOTAL_EPOCHS + (EPOCHS_PER_JOB - 1)) / EPOCHS_PER_JOB))

# launch
for i in $(seq 1 ${TOTAL_JOBS}) ; do
  echo sbatch --job-name="${JOB_NAME}" --dependency=singleton \
              --time="${TIME_HOURS}:00:00" --gres=gpu:${GPU_TYPE}:${GPU_PER_NODE} \
							--nodes=${NODE_TOTAL} --ntasks-per-node=${GPU_PER_NODE} \
              --mem=$((GPU_PER_NODE * MEM_PER_GPU))G --cpus-per-task=${CPU_PER_GPU} \
              ${SBATCH_CONSTRAINTS} \
    bin/horovod-train.sbatch ${PARAMETERS} \
                             --data-dir "${DATA_DIR}" \
                             "$@"
                             # TODO: --total-epochs ${TOTAL_EPOCHS}
done

