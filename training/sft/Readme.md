# Supervised Fine-tuning on CSCS

training/
├── meditron-3-8b.yaml         # Configuration file for model training
├── deepspeed.json             # Deepspeed configuration file for optimization
├── launch_axolotl_meditron3_8b.sh  # SLURM script to launch batch training jobs
└── test_meditron_now.sbatch   # SLURM script for model inference testing

## Connect to the CSCS

Tutorial: [Connecting to CSCS](https://epflight.github.io/LiGHT-doc/clusters/cscs/cscs/).

Complete the tutorial up to "Setup Github".

## Configuration for model training

Tutorial: [Launching an axolotl training](https://epflight.github.io/LiGHT-doc/clusters/cscs/axolotl_training/).

Create a folder in the home directory to store the axolotl configurations.

```bash
mkdir -p ~/meditron/axolotl_config
```

Create a `meditron-3-8b.yaml`  file in this folder. We made some modifications to it based on suggestions, but after encountering errors during execution, reverted part of the changes.

```yaml
cd ~/meditron/axolotl_config
cat > meditron-3-8b.yaml << 'EOF'

base_model:  OpenMeditron/Meditron3-8B #swiss-ai/Apertus-8B-Instruct-2509->*
#plugins:
#  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin

datasets: 
  - path: ${PROJECT_ROOT}/datasets/polyglot1/all_shuffled.jsonl #all_shuffled.jsonl
    type: chat_template
    split: train


save_on_interrupt: true #!!!
# This is the path where axolotl caches the prepared dataset
dataset_prepared_path: ${PROJECT_ROOT} #${PROJECT_ROOT}/last_run_prepared->*

# Output directory where model checkpoints and logs will be saved
output_dir: ${PROJECT_ROOT}/datasets/polyglot1/models
# ${PROJECT_ROOT}/models/tutorials/axolotl_apertus_8b->*

# Data loading and processing settings
shuffle_merged_datasets: true
dataset_processes: 64 # Avoid RAM OOM issues by lowering this value if needed

# If your model supports flash attention, enable it
flash_attention: true
flash_attn_rms_norm: true
flash_attn_fuse_qkv: true #false->* :save 10~15% VRAM if true

# Enable/Disable sample packing
sample_packing: true
sequence_len: 8192 #2048->*
group_by_length: false
pad_to_sequence_len: true

# Gradient checkpointing settings: enable to save VRAM
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false

# Control batch size and number of epochs
gradient_accumulation_steps: 4
micro_batch_size: 4
num_epochs: 3

# Learning rate scheduler and optimizer settings
optimizer: adamw_torch_fused #adamw_torch->* :adamw_torch_fused a little faster
optim_args:
  fused: true
learning_rate: 1.0e-5
warmup_ratio: 0.0 #:advise 0.03 a little
weight_decay: 0.01 #0.05->*
lr_scheduler: cosine
cosine_min_lr_ratio: 0.1
max_grad_norm: 1.0

# Disable evaluation
evals_per_epoch: 0
eval_set_size: 0.0
eval_table_size: null

# Checkpointing and logging settings
resume_from_checkpoint: null
logging_steps: 1
saves_per_epoch: 2

# Model and tokenizer types (usually AutoModelForCausalLM and AutoTokenizer for causal LLMs)
tokenizer_type: AutoTokenizer
type: AutoModelForCausalLM

# Weights & Biases logging configuration
wandb_entity: 
wandb_log_model: 
wandb_name: 5gb-8k-64xa100-3epoch #Meditron-Apertus-8B->*
wandb_project: meditron-3-8b-finetune #tutorial->*
wandb_watch: null

ddp_find_unused_parameters: true
deepspeed: /users/$USER/meditron/axolotl_config/deepspeed.json
EOF
```

To enable Deepspeed Zero-3 optimization, create a `deepspeed.json` file in the same folder.

```json
cat > deepspeed.json << 'EOF'
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
          "device": "cpu",
          "pin_memory": true
        },
        "overlap_comm": false,
        "contiguous_gradients": true,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_clipping": 1.0,
    "wall_clock_breakdown": false,
    "activation_checkpointing": {
        "partition_activations": false,
        "contiguous_memory_optimization": false,
        "cpu_checkpointing": false
    },
    "flops_profiler": {
        "enabled": false
    },
    "aio": {
        "block_size": 1048576,
        "queue_depth": 8,
        "single_submit": false,
        "overlap_events": false
    }
}
EOF
```



## Launch a job in interactive mode

We do this only for simple testing.

```bash
#Enter interactive mode (time limit: at most 3 hours) and allocate compute resources.
srun --time=2:59:59 --partition=normal -A a127 --environment=/users/$USER/.edf/axolotl.toml --pty bash

#Start the training job.
torchrun --nproc_per_node=4 -m axolotl.cli.train /users/$USER/meditron/axolotl_config/meditron-3-8b.yaml

#If the HF_TOKEN expires during training, on the interactive node:
export HUGGINGFACE_TOKEN="YOUR HF_TOKEN"

#First, completely remove any potentially leftover cache if it hasn't been used yet.
rm -rf ~/.cache/huggingface/hub/models--OpenMeditron--Meditron3-8B

#Then, force a re-login using huggingface_hub's own login method.
python -c "from huggingface_hub import login; login(token='YOUR HF_TOKEN')"
```

## Launch a batch job

We need to submit a batch job via SLURM script for model training.

Create a `launch_axolotl_meditron3_8b.sh` file. Note that `#SBATCH --nodes 4` indicates 4 compute nodes requested, corresponding to 16 A100 GPUs.

```sh
cat > launch_axolotl_meditron3_8b.sh << 'EOF'
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
EOF
```

Then we launch the job. In practice, a request for 32 nodes (128 A100 GPUs) could not be scheduled, but a job with 16 nodes was successfully run.

```
sbatch --nodes 16 launch_axolotl_meditron3_8b.sh
```

We can check the job status.

```bash
#check job status
squeue -u $USER

#view the location of output files; job ID 1201263
scontrol show job 1201263 | grep -E "StdOut|StdErr"
#output: /users/$USER/meditron/reports/R-meditron-tutorial.1201263.out

#view the output files
cd ~/meditron/reports/
tail -f R-meditron-tutorial.1201263.err
tail -f R-meditron-tutorial.1201263.out
tail -n 50 /users/$USER/meditron/reports/R-meditron-tutorial.1201617.err | grep -i "loss\|epoch\|saving\|END"


```

We get the model path. For example, `model_path = '${PROJECT_ROOT}/datasets/polyglot1/models/checkpoint-615'`.

## Interact with the model

Prepare a sbatch file for model inference.

```bash
cat > test_meditron_now.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=test-now
#SBATCH --output=/users/$USER/meditron/reports/test-%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=30:00
#SBATCH --partition=normal
#SBATCH --account=a127
#SBATCH --environment=/users/$USER/.edf/axolotl.toml

python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_path = '${PROJECT_ROOT}/datasets/polyglot1/models/checkpoint-615'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True)
inputs = tokenizer('我后面有事去无锡，正好玩两天，提醒我旅游需要注意什么\n助手：', return_tensors='pt').to('cuda')
output = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
print(tokenizer.decode(output[0], skip_special_tokens=True).split('助手：')[-1])
"
EOF
```

Submit the job and check the results.

```bash
sbatch ~/test_meditron_now.sbatch
# Got the test number
tail -f /users/$USER/meditron/reports/test-*.out
```

Here are some results to show that it can talk in mandarin.

```python
'用户：我咳嗽两周，有黄痰，偶尔发低烧，怎么办？\n助手：'
您好！您这种情况应该是呼吸道感染，建议去医院就诊，注意休息，多喝水，避免受凉，注意保暖。

'你好，你是谁？我最近在家坐着学习，感觉小腿肌肉疼，怎么办？\n助手：'
你好，根据你的描述，可能是由于肌肉疲劳引起的疼痛。建议你适当休息，避免长时间久坐，适当进行一些简单的运动，如散步、做些伸展运动等，有助于缓解肌肉疲劳。如果疼痛持续或加重，建议及时就医。

'我后面有事去无锡，正好玩两天，提醒我旅游需要注意什么\n助手：'
您好，建议您提前查好相关旅游景点的开放时间和票价情况，旅游前先买好景点的门票，避免到时候排队买票耽误时间，另外建议您提前了解好当地的天气情况，根据当地天气情况选择合适的旅游服装，另外建议您提前预订好酒店，避免到时候住宿困难，希望我的回答能够帮到您，祝您旅途愉快！
```
