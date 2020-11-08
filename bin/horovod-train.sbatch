#!/bin/bash 
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --partition=batch
#SBATCH --output=results/log-%x-slurm-%j.out
#SBATCH --error=results/log-%x-slurm-%j.err

# Stops execution of script if any step return non-zero exit code
set -eo pipefail

##SBATCH --gpus=8
##SBATCH --gpus-per-task=1
##SBATCH --cpus-per-gpu=6
##SBATCH --mem-per-gpu=45G
##SBATCH --gres=gpu:1 
##SBATCH --constraint=v100
##SBATCH --gpus-per-task=1 # NOTE: possible issue when doing multiple GPUs per node?

## TODO: Explore how to allocate extra ports for logging... NOTE: for srun only?
##SBATCH --resv-ports=2


# Need to define persistent storage for logging... 
PROJECT_ROOT=$(realpath "${PROJECT_ROOT:-.}")

PERSISTENT_LOGGING_DIR="${PROJECT_ROOT}/results/${SLURM_JOB_NAME}/logs"
PERSISTENT_CHECKPOINTS_DIR="${PERSISTENT_LOGGING_DIR}/checkpoints"
PERSISTENT_TENSORBOARD_DIR="${PERSISTENT_LOGGING_DIR}/tensorboard"

# N.B. mkdir does not overwrite if these directories already exist
mkdir -p "${PERSISTENT_CHECKPOINTS_DIR}"
mkdir -p "${PERSISTENT_TENSORBOARD_DIR}"

# ...but for best performance write checkpoints and tensorboard logs to local storage
LOCAL_LOGGING_DIR="/tmp/${SLURM_JOB_NAME}/${SLURM_JOB_ID}/logs"
LOCAL_CHECKPOINTS_DIR="${LOCAL_LOGGING_DIR}/checkpoints"
LOCAL_TENSORBOARD_DIR="${LOCAL_LOGGING_DIR}/tensorboard"
mkdir -p "${LOCAL_CHECKPOINTS_DIR}"
mkdir -p "${LOCAL_TENSORBOARD_DIR}"

# default values
ENV_PREFIX="${PROJECT_ROOT}/$(realpath --relative-to="${PROJECT_ROOT}" "${ENV_PREFIX:-env}")"
HOSTS_FILE="${PERSISTENT_LOGGING_DIR}/hosts-${SLURM_JOB_ID}.list"
TRAINING_SCRIPT=$(realpath --relative-to="${PROJECT_ROOT}" "${TRAINING_SCRIPT:-"src/pytorch-examples/imagenet/train.py"}")

scontrol show hostname ${SLURM_JOB_NODELIST} | awk -v slots=${SLURM_NTASKS_PER_NODE} '{print $1 " slots="slots}' > "${HOSTS_FILE}"


# Load software stack
source "${PROJECT_ROOT}/bin/module.init"

source "${PROJECT_ROOT}/bin/conda.init"
conda activate "${ENV_PREFIX}"


# Start the nvidia-smi process in the background
NVIDIA_SMI_DELAY_SECONDS=60
nvidia-smi dmon --delay ${NVIDIA_SMI_DELAY_SECONDS} --options DT >> "${PERSISTENT_LOGGING_DIR}/nvidia-smi.log" &
NVIDIA_SMI_PID=$!

# Start the nvdashboard server running in the background
NVDASHBOARD_PORT=8000
python -m jupyterlab_nvdashboard.server ${NVDASHBOARD_PORT} &
NVDASHBOARD_PID=$!

# Start the TensorBoard server running in the background
TENSORBOARD_PORT=6006
tensorboard --logdir "${LOCAL_TENSORBOARD_DIR}" --port ${TENSORBOARD_PORT} --bind_all &
TENSORBOARD_PID=$!


# start the training process in the background
export OMPI_MCA_opal_cuda_support=true
export NCCL_SOCKET_IFNAME=eth0 #ib0,eth0
export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=COLL

#horovodrun --num-proc ${SLURM_NTASKS} --hostfile "${HOSTS_FILE}" --network-interface ${NCCL_SOCKET_IFNAME:-eth0} \
mpirun --np ${SLURM_NTASKS} --hostfile "${HOSTS_FILE}" \
        -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib \
  "$(which python)" "${TRAINING_SCRIPT}" \
    --read-checkpoints-from "${PERSISTENT_CHECKPOINTS_DIR}" \
    --write-checkpoints-to "${LOCAL_CHECKPOINTS_DIR}" \
    --tensorboard-logging-dir "${LOCAL_TENSORBOARD_DIR}" \
    "$@" &
HOROVODRUN_PID=$!


# asynchronous rsync of training logs between local and persistent storage
RSYNC_DELAY_SECONDS=1800           # rsync every half-hour
PID_CHECK_DELAY_SECONDS=60         # check for training completion every minute
RSYNC_DELAY_LOOP=$((RSYNC_DELAY_SECONDS / PID_CHECK_DELAY_SECONDS))

RUN_DONE=false
while [ "${RUN_DONE}" != true ] ; do
  # loop until time out or training completes
  for i in $(seq 1 ${RSYNC_DELAY_LOOP}) ; do
    sleep ${PID_CHECK_DELAY_SECONDS}
    if [[ "$(ps -h --pid ${HOROVODRUN_PID} -o state | head -n 1)" = "" ]] ; then
       RUN_DONE=true ; break
    fi
  done
  # conditional copy operations (only copies what is new / changed)
  rsync -a "${LOCAL_CHECKPOINTS_DIR}/" "${PERSISTENT_CHECKPOINTS_DIR}"
  rsync -a "${LOCAL_TENSORBOARD_DIR}/" "${PERSISTENT_TENSORBOARD_DIR}"
done


# kill off the monitoring processes
kill ${NVIDIA_SMI_PID} ${NVDASHBOARD_PID} ${TENSORBOARD_PID} || true

# return exit status for training job
# Note: This code is needed when `set -e` not used
#       `set -e` above causes script to exit early if an error occured and return that exit code
wait ${HOROVODRUN_PID}
HOROVODRUN_STATUS=$?
exit ${HOROVODRUN_STATUS}

