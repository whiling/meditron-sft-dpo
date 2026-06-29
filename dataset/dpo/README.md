# DPO Data Generation Pipeline

This repository contains the pipeline for generating Direct Preference Optimization (DPO) training data using diverse medical question generation and model response comparison on the CSCS cluster.

## Architecture Overview

The data generation pipeline consists of:
1. **Local Question Generation**: Use DeepSeek API to generate diverse medical questions
2. **Cluster-based Response Generation**: Generate responses from multiple models
3. **Preference Annotation**: Create preference pairs for DPO training
4. **Dataset Compilation**: Format data for DPO training pipeline

## Directory Structure

```bash
/users/$USER/dpo_data_generation/
├── diverse_medical_generator.py    # Local question generator (DeepSeek API)
├── dpo_data_pipeline.py           # Main data generation pipeline
├── setup_environment.sh           # Environment setup script
├── run_dpo_data_generation.sh     # SLURM job submission script
├── questions.jsonl                # Generated questions (upload from local)
├── output/                        # Generated DPO dataset output
└── logs/                          # Job execution logs
```

## Quick Start

### Prerequisites

Create the Axolotl environment configuration:

```bash
# Create ~/.edf/axolotl.toml on the cluster
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

### Step 1: Generate Questions Locally

Run the question generator locally to create diverse medical questions:

```bash
# Local machine
python diverse_medical_generator.py
# This generates questions.jsonl file using DeepSeek API
```

### Step 2: Upload Files to Cluster

```bash
# Create directory on cluster
ssh <EMAIL> "mkdir -p /users/\$USER/dpo_data_generation"

# Upload files to cluster
scp questions.jsonl setup_environment.sh run_dpo_data_generation.sh dpo_data_pipeline.py \
    <EMAIL>:/users/$USER/dpo_data_generation/
```

### Step 3: Setup Environment on Cluster

```bash
# SSH to cluster
ssh <EMAIL>

# Navigate to working directory
cd /users/$USER/dpo_data_generation

# Make scripts executable
chmod +x setup_environment.sh run_dpo_data_generation.sh

# Install required dependencies
bash setup_environment.sh
```

### Step 4: Submit Data Generation Job

```bash
# Submit SLURM job
sbatch run_dpo_data_generation.sh

# Monitor job status
squeue -u $USER

# View real-time logs
tail -f logs/R-dpo-data-gen.*.out
tail -f logs/R-dpo-data-gen.*.err
```

### Step 5: Monitor Progress

```bash
# Check if job is still running
squeue -u $USER

# Monitor log output in real-time
tail -f logs/R-dpo-data-gen.*.err

# Check if output files are being generated
ls -la output/
```

## Data Generation Pipeline

### Question Generation (Local)
- **Tool**: DeepSeek API via `diverse_medical_generator.py`
- **Output**: `questions.jsonl` with diverse medical questions
- **Categories**: Clinical reasoning, diagnosis, treatment, pharmacology, etc.

### Response Generation (Cluster)
- **Models**: Multiple language models for response comparison
- **Process**: Generate responses to each question using different models
- **Evaluation**: Compare response quality, accuracy, and preference

### Preference Annotation
- **Method**: Automated preference scoring based on:
  - Medical accuracy
  - Response completeness
  - Safety considerations
  - Language quality
- **Output**: DPO-formatted preference pairs

## 🔧 Configuration

### Environment Variables
Set these in your job scripts or environment:
- `HF_TOKEN`: Hugging Face authentication token
- `WANDB_API_KEY`: WandB API key (optional)
- `DEEPSEEK_API_KEY`: DeepSeek API key for question generation

### SLURM Job Parameters
Adjust in `run_dpo_data_generation.sh`:
- `--time`: Job duration (default: 4 hours)
- `--partition`: Cluster partition
- `--account`: Your allocation account
- `--cpus-per-task`: CPU cores per task
- `--mem`: Memory allocation

## Output Format

The generated DPO dataset follows the standard format:

```json
{
    "question": "What is the capital of France?",
    "chosen": "The capital of France is Paris.",
    "rejected": "The capital of France is London."
}
```

## Important Notes

### SLURM Configuration
- **Do not use `$USER` variables** in `#SBATCH` directives
- Use absolute paths in SLURM scripts
- The `$USER` variable may not expand correctly in SLURM environment

### Resource Management
- Monitor GPU memory usage during generation
- Adjust batch sizes based on available resources
- Use appropriate time limits for your data size

### Error Handling
- Check logs regularly for API rate limiting
- Implement retry mechanisms for failed API calls
- Validate generated data quality before proceeding to training

## Contributing

1. Test changes on small datasets first
2. Update documentation for new features
3. Follow medical data handling best practices
4. Ensure reproducibility with fixed random seeds

## References

- [Direct Preference Optimization Paper](https://arxiv.org/abs/2305.18290)
- [Axolotl Documentation](https://github.com/OpenAccess-AI-Collective/axolotl)
- [CSCS User Documentation](https://user.cscs.ch/)
