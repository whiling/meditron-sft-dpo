#!/bin/bash
#SBATCH --job-name meditron-dpo
#SBATCH --chdir ${PROJECT_ROOT}/axolotl_config
#SBATCH --output ${PROJECT_ROOT}/R-%x.%j.out
#SBATCH --error ${PROJECT_ROOT}/R-%x.%j.err
#SBATCH --nodes 2               # Start with 2 nodes for DPO
#SBATCH --ntasks-per-node 1     # number of MP tasks
#SBATCH --gres gpu:4            # Number of GPUs per node
#SBATCH --cpus-per-task 288     # number of CPUs per task
#SBATCH --time 11:59:59         # maximum execution time
#SBATCH --environment /users/wanlinhu/.edf/axolotl.toml
#SBATCH -A a127

export WANDB_DIR=${PROJECT_ROOT}/wandb
export WANDB_API_KEY=a79a6d3eba82fd2f40c778d3f4ef647845c70842
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
### Set environment ###
######################
GPUS_PER_NODE=4
echo "NODES: $SLURM_NNODES"

######## Args ########
AXOLOTL_CONFIG_FILE=${PROJECT_ROOT}/dpo_cscs_config.yml

export HF_HOME=$SCRATCH/hf
export HF_TOKEN=${HF_TOKEN}
mkdir -p $HF_HOME

# Setup DPO data directory
DPO_DATA_DIR=${PROJECT_ROOT}/dpo_data
mkdir -p $DPO_DATA_DIR

# Copy DPO dataset if not already there
if [ ! -f "$DPO_DATA_DIR/dpo.jsonl" ]; then
    echo "Copying DPO dataset..."
    cp /path/to/your/dpo.jsonl $DPO_DATA_DIR/
fi

######################
#### Set network #####
######################
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6300

######################
# DPO-specific environment variables
export DPO_BETA=0.1
export MAX_LENGTH=8192

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

# srun error handling
SRUN_ARGS=" \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --jobid $SLURM_JOB_ID \
    --wait 60 \
    -A a127 \
    "

# bash -c is needed for the delayed interpolation of env vars to work
srun $SRUN_ARGS bash -c "$CMD"
echo "END TIME: $(date)"
