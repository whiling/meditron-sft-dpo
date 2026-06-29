#!/bin/bash
#SBATCH --job-name meditron-tutorial
#SBATCH --chdir /users/$USER/meditron/axolotl_config
#SBATCH --output /users/$USER/meditron/reports/R-%x.%j.out
#SBATCH --error /users/$USER/meditron/reports/R-%x.%j.err
#SBATCH --nodes 4               # number of Nodes
#SBATCH --ntasks-per-node 1     # number of MP tasks. IMPORTANT: torchrun represents just 1 Slurm task
#SBATCH --gres gpu:4        # Number of GPUs
#SBATCH --cpus-per-task 288     # number of CPUs per task (based on lscpu)
#SBATCH --time 11:59:59       # maximum execution time (DD-HH:MM:SS)
#SBATCH --signal=B:TERM@300 #!!!signal 5min before end
#SBATCH --environment /users/$USER/.edf/axolotl.toml
#SBATCH -A a127



export WANDB_DIR=${PROJECT_ROOT}/wandb
export WANDB_API_KEY="YOUR WANDB_API_KEY"
export WANDB_MODE="online"

# Put Triton on a non-NFS directory
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache

export CUDA_LAUNCH_BLOCKING=1
echo "START TIME: $(date)"
# auto-fail on any errors in this script
set -eo pipefail
# logging script's variables/commands for future debug needs
set -x
######################
### Set enviroment ###
######################
GPUS_PER_NODE=4

echo "NODES: $SLURM_NNODES"
######## Args ########
AXOLOTL_CONFIG_FILE=/users/$USER/meditron/axolotl_config/meditron-3-8b.yaml

export HF_HOME=$SCRATCH/hf
export HF_TOKEN=hf_**********************************
mkdir -p $HF_HOME

######################
######################
#### Set network #####
######################
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6300
######################
# note that we don't want to interpolate `\$SLURM_PROCID` till `srun` since otherwise all nodes will get
# 0 and the launcher will hang
#
# same goes for `\$(hostname -s|tr -dc '0-9')` - we want it to interpolate at `srun` time


LAUNCHER="
    torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank \$SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

export CMD="$LAUNCHER -m axolotl.cli.train $AXOLOTL_CONFIG_FILE"
echo $CMD
# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
SRUN_ARGS=" \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --jobid $SLURM_JOB_ID \
    --wait 60 \
    -A a127 \
    "
# bash -c is needed for the delayed interpolation of env vars to work

srun $SRUN_ARGS bash -c "$CMD"
echo "END TIME: $(date)"
