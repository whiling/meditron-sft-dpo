# DPO Training on CSCS Cluster

This repository contains the setup and configuration files for running Direct Preference Optimization (DPO) training on the CSCS cluster infrastructure using Axolotl.

## Directory Structure

```bash
${PROJECT_ROOT}/
├── meditron/
│   ├── axolotl_config/          # Configuration files directory
│   │   ├── dpo_cscs_config.yml  # DPO configuration
│   │   ├── zero3_bf16.json      # DeepSpeed configuration
│   │   └── launch_dpo_cscs.sh   # Launch script
│   ├── reports/                 # Log output directory
│   └── models/                  # Model output directory
├── dpo_data/
│   └── dpo.jsonl               # DPO dataset
└── wandb/                      # WandB logging directory
```

## Quick Start

### Step 1: Prepare Local Environment Files

Modify the configuration file `dpo_cscs_config.yml` locally before uploading.

### Step 2: Connect to CSCS Cluster and Create Directories

```bash
# SSH connect to CSCS
ssh <EMAIL>

# Create directory structure
mkdir -p ${PROJECT_ROOT}/axolotl_config
mkdir -p ${PROJECT_ROOT}/reports
mkdir -p ${PROJECT_ROOT}/dpo_data
mkdir -p ${PROJECT_ROOT}/wandb
```

### Step 3: Upload Files to Cluster

```bash
# Method 1: Use scp to upload from local machine
scp dpo.jsonl <EMAIL>:${PROJECT_ROOT}/
scp dpo_cscs_config.yml <EMAIL>:${PROJECT_ROOT}/
scp zero3_bf16.json <EMAIL>:${PROJECT_ROOT}/
scp launch_dpo_cscs.sh <EMAIL>:${PROJECT_ROOT}/
```

### Step 4: Configure Axolotl Environment

Create the Axolotl configuration file on the cluster:

```bash
cat > ~/.edf/axolotl.toml << 'EOF'
image = "${PROJECT_ROOT}/docker/axolotl.sqsh"
mounts = ["${PROJECT_ROOT}/capstor", "/iopsstor", "/users"]
writable = true

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"

[env]
HF_HOME = "${SCRATCH}/hf"
CUDA_CACHE_DISABLE = "1"
NCCL_NET = "AWS Libfabric"
NCCL_CROSS_NIC = "1"
NCCL_NET_GDR_LEVEL = "PHB"
FI_CXI_DISABLE_HOST_REGISTER = "1"
FI_MR_CACHE_MONITOR = "userfaultfd"
FI_CXI_DEFAULT_CQ_SIZE = "131072"
FI_CXI_DEFAULT_TX_SIZE = "32768"
FI_CXI_RX_MATCH_MODE = "software"
FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD = "16777216"
FI_CXI_COMPAT = "0"
EOF
```

### Step 5: Update Key Parameters in Configuration Files

Edit `${PROJECT_ROOT}/dpo_cscs_config.yml` with your specific training parameters.

### Step 6: Test Run

```bash
# Navigate to configuration directory
cd ${PROJECT_ROOT}/axolotl_config

# Start interactive session for testing
srun --time=2:59:59 --partition normal -A a127 --environment=/users/$USER/.edf/axolotl.toml

# Test training in interactive session
torchrun --nproc_per_node=4 -m axolotl.cli.train dpo_cscs_config.yml
```

### Step 7: Submit Formal Training Job

```bash
# Make launch script executable
chmod +x launch_dpo_cscs.sh

# Submit SLURM job
sbatch launch_dpo_cscs.sh
```

### Step 8: Monitor Training Progress

```bash
# Check job status
squeue -u $USER

# View logs
tail -f ${PROJECT_ROOT}/R-meditron-dpo.*.out
tail -f ${PROJECT_ROOT}/R-meditron-dpo.*.err

# Monitor training metrics on WandB web interface
```

## Configuration Files

### `dpo_cscs_config.yml`
Main DPO training configuration including:
- Model paths and tokenizer settings
- Training hyperparameters (learning rate, batch size, etc.)
- DPO-specific parameters (beta, loss function)
- Dataset configuration

### `zero3_bf16.json`
DeepSpeed ZeRO Stage 3 configuration for distributed training with bfloat16 precision.

### `launch_dpo_cscs.sh`
SLURM job submission script with appropriate resource allocation and environment setup.

## Requirements

- Access to CSCS cluster with Axolotl environment
- Prepared DPO dataset in JSONL format
- WandB account for experiment tracking (optional)
- Sufficient compute allocation (GPU hours)

## Monitoring

Training progress can be monitored through:
- SLURM job logs in the `reports/` directory
- WandB dashboard for real-time metrics
- Model checkpoints saved in `models/` directory

## Contributing

When modifying configuration files:
1. Test changes in interactive mode first
2. Update documentation accordingly
3. Ensure proper resource allocation in SLURM scripts

## Notes

- Replace `$USER` with your actual username in paths
- Adjust resource requirements based on your model size and dataset
- Monitor GPU memory usage to optimize batch sizes
- Keep regular backups of successful configurations
