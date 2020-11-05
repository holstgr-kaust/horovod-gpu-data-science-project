#!/bin/bash 
#SBATCH --time=5:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch
#SBATCH --output=results/log-%x-slurm-%j.out
#SBATCH --error=results/log-%x-slurm-%j.err

# Stops execution of script if any step return non-zero exit code
set -eo pipefail

##SBATCH --gpus=2
##SBATCH --gpus-per-task=1
##SBATCH --cpus-per-gpu=6
##SBATCH --mem-per-gpu=5G
##SBATCH --gres=gpu:1 
##SBATCH --constraint=p6000
##SBATCH --gpus-per-task=1 # NOTE: possible issue when doing multiple GPUs per node?


# Need to define persistent storage for logging... 
PROJECT_ROOT=$(realpath "${PROJECT_ROOT:-.}")

PERSISTENT_LOGGING_DIR="${PROJECT_ROOT}/results/${SLURM_JOB_NAME}/logs"

# N.B. mkdir does not overwrite if these directories already exist
mkdir -p "${PERSISTENT_LOGGING_DIR}"


# default values
ENV_PREVIX="${PROJECT_ROOT}/$(realpath --relative-to="${PROJECT_ROOT}" "${ENV_PREFIX:-env}")"
HOSTS_FILE="${PERSISTENT_LOGGING_DIR}/hosts-${SLURM_JOB_ID}.list"
TESTING_SCRIPT=$(realpath --relative-to="${PROJECT_ROOT}" "${TESTING_SCRIPT:-"src/test-examples/test-mpi-connection.py"}")

scontrol show hostname ${SLURM_JOB_NODELIST} | awk -v slots=${SLURM_NTASKS_PER_NODE} '{print $1 " slots="slots}' > "${HOSTS_FILE}"


# Debug logging
echo "PRE CONDA & MODULE INIT"
echo "CONDA ENV:"
conda env list || true
echo "ENV:"
env


# Load software stack
# Load software stack
source "${PROJECT_ROOT}/bin/module.init"

source "${PROJECT_ROOT}/bin/conda.init"
conda activate "${ENV_PREFIX}"


# start the training process in the background
export OMPI_MCA_opal_cuda_support=true
export NCCL_SOCKET_IFNAME=eth0 #ib0,eth0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL


# Debug logging
echo "PRE HOROVODRUN / MPIRUN"
echo "python: $(which python)"
echo "mpirun: $(which mpirun)"
echo "horovodrun: $(which horovodrun)"
echo "running: horovodrun --num-proc ${SLURM_NTASKS} --hostfile "${HOSTS_FILE}" --network-interface ${NCCL_SOCKET_IFNAME:-eth0} --verbose"
#echo "running: mpirun --np ${SLURM_NTASKS} --hostfile "${HOSTS_FILE}" --verbose --show-progress"
echo "      $(which python) ${TESTING_SCRIPT} $@"
echo "PERSISTENT_LOGGING_DIR: ${PERSISTENT_LOGGING_DIR}"
echo "TESTING_SCRIPT: ${TESTING_SCRIPT}"
echo "HOSTS_FILE: ${HOSTS_FILE}"
cat "${HOSTS_FILE}"
echo "CONDA ENV:"
conda env list
echo "ENV:"
env
echo "OpenMPI Info:"
ompi_info


## horovod notes ##
# --mpi-args "-mca btl_tcp_if_include eth0" # MCA parameters can only be listed once on a command line
# --network-interface eth0
## mpirun notes ##
# -mca btl tcp,self
# -mca btl ^openib

#mpirun --np ${SLURM_NTASKS} --hostfile "${HOSTS_FILE}" --verbose --show-progress \
#mpirun --np ${SLURM_NTASKS} --hostfile "${HOSTS_FILE}" \
#        -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib \
#       --verbose --show-progress \
horovodrun --num-proc ${SLURM_NTASKS} --hostfile "${HOSTS_FILE}" --network-interface ${NCCL_SOCKET_IFNAME:-eth0} --verbose \
  "$(which python)" "${TESTING_SCRIPT}" "$@"

echo "POST HOROVODRUN / MPIRUN -- DONE!"

