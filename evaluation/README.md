# Evaluation on CSCS

eval/

├── axolotl.toml                     # Configuration file for Axolotl environment setup

├── eval_config.yaml                 # Evaluation configuration file for Meditron3-8B model after SFT

├── eval_config_benchmark.yaml       # Evaluation configuration file for Meditron3-8B benchmark model

├── eval_config_dpo.yaml             # Evaluation configuration file for Meditron3-8B model after SFT and DPO

├── eval_meditron2_ppl.py            # Python script for evaluating Meditron3-8B model after SFT using Perplexity (PPL)

├── eval_meditron2_ppl_dpo.py        # Python script for evaluating Meditron3-8B model after SFT and DPO using Perplexity (PPL)

├── meditron3_8B_ppl.py              # Python script for evaluating Meditron3-8B model after SFT using Perplexity (PPL)

├── meditron3_8B_ppl_dpo.py          # Python script for evaluating Meditron3-8B model after SFT and DPO using Perplexity (PPL)

└── run_opencompass.slurm            # SLURM script for running OpenCompass evaluation tasks

## Meditron_MedQA

> MedQA (Medical Question Answering Dataset) is a comprehensive dataset focused on medical question answering. It contains a variety of medical questions sourced from real-world clinical scenarios, textbooks, and examinations. MedQA is designed to test the ability of language models to provide accurate and relevant answers to medical queries. The dataset includes both multiple-choice and open-ended questions, covering various medical domains such as diagnosis, treatment, and patient care.

Create a TOML file in `~/.edf/`. See the file `axolotl.toml` for detail.

Clone the meditron protocole repository.

```bash
#interactive mode
srun --time=2:59:59 \
     --partition=normal \
     --environment=/users/$USER/.edf/axolotl.toml \
     --gres=gpu:1 \
     -A a127 \
     --pty bash

# Automaticly jump to /workspace/axolotl
cd ~
# Clone the repository
git clone https://$USER:<EMAIL>/EPFLiGHT/meditron_protocole.git
# Install dependencies
cd ~/meditron_protocole/evaluation
pip install -r requirements.txt
```

For each model, create a YAML file for evaluation in `~/meditron_protocole/evaluation/`. See the file `eval_config.yaml` for detail.

Create a python file to run evaluation for each model. See the file `run_evaluation.py` for detail.

Run evaluation.

```bash
# In evaluation folder
# First time: install env
chmod +x setup_env.sh
./setup_env.sh
# Then activate the environment
source .venv/bin/activate
# only if you wanna deactivate the environment
deactivate

pip install -e .
# comment out check_environment() in run_evaluation.py
# check the current version
python3 -c "import datasets; print(datasets.__version__)"
# Downgrade to version 2.18.0
pip install datasets==2.18.0

# Run evaluation
python3 run_evaluation.py
python3 run_evaluation_benchmark.py
python3 run_evaluation_dpo.py
```

## CMMLU

> CMMLU (Chinese Medical Licensing Examination Dataset) is a large-scale dataset designed to evaluate the medical knowledge and reasoning abilities of language models. It is derived from the Chinese Medical Licensing Examination, covering a wide range of medical subjects, including anatomy, physiology, pathology, pharmacology, and clinical medicine. The dataset consists of multiple-choice questions and answers, making it a valuable benchmark for assessing the accuracy and depth of medical knowledge in language models.

Script for Meditron-8B after SFT.

```bash
cd ~/meditron_protocole/evaluation
python lm-evaluation-harness/lm_eval/__main__.py \
  --model hf \
  --model_args "pretrained=${PROJECT_ROOT}/datasets/polyglot1/models/checkpoint-615,trust_remote_code=True,dtype=bfloat16,parallelize=True" \
  --tasks cmmlu_anatomy,cmmlu_clinical_knowledge,cmmlu_college_medicine,cmmlu_genetics,cmmlu_professional_medicine \
  --device cuda:0 \
  --batch_size auto \
  --output_path ./results/meditron-8b-mandarin-cmmlu-med \
  --num_fewshot 5
```

Script for Meditron-8B after SPT and 2nd DPO.

```bash
cd ~/meditron_protocole/evaluation
python lm-evaluation-harness/lm_eval/__main__.py \
  --model hf \
  --model_args "pretrained=${PROJECT_ROOT}/datasets/polyglot1/models/dpo_medical_v2/checkpoint-100,trust_remote_code=True,dtype=bfloat16,parallelize=True" \
  --tasks cmmlu_anatomy,cmmlu_clinical_knowledge,cmmlu_college_medicine,cmmlu_genetics,cmmlu_professional_medicine \
  --device cuda:0 \
  --batch_size auto \
  --output_path ./results/meditron-8b-dpo-cmmlu-med \
  --num_fewshot 5
```

## OpenCompass

Set up the environment.

```bash
srun --time=2:59:59 \
     --partition=normal \
     --environment=/users/$USER/.edf/axolotl.toml \
     --gres=gpu:4 \
     --ntasks-per-node=1 \
     --cpus-per-task=40 \
     -A a127 \
     --pty bash

cd ~
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
# Run the installer script (install to ~/miniconda3)
bash Miniconda3-latest-Linux-aarch64.sh -b -p $HOME/miniconda3
# Initialize conda (automatically modifies ~/.bashrc)
$HOME/miniconda3/bin/conda init bash
# Reload shell configuration
source ~/.bashrc
# Verify installation
conda --version
# Accept Anaconda Terms of Service for main and R channels
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# Create a conda environment named 'opencompass'
conda create --name opencompass python=3.10 -y
conda activate opencompass # conda deactivate

# Install OpenCompass
pip install -U opencompass

# Download evaluation datasets to the 'data/' directory
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

Create a python configuration file for each model in the `~/opencompass/opencompass/configs/models/` folder. See the file `meditron3_8B_ppl.py` for detail.

Create an evaluation script for each model in the `~/opencompass/opencompass/configs/` folder. See the file `eval_meditron2_ppl.py` for detail.

Create a slurm script to run a batch job. See the file `run_opencompass.slurm` for detail.

Then type `sbatch run_opencompass.slurm` to ignite the evaluation.

We can check the output.

```bash
tail -f /users/$USER/output/slurm_batch_1257110.out /users/$USER/output/slurm_batch_1257110.err
cat /users/$USER/output/slurm_batch_1257110.out
cat /users/$USER/output/slurm_batch_1257110.err
```

## Meditron_MedQA Results

Meditron-8B after SFT.

| Tasks          | Version | Filter | n-shot | Metric   |      |  Value |      | Stderr |
| -------------- | ------- | ------ | -----: | -------- | ---- | -----: | ---- | -----: |
| medmcqa        | Yaml    | none   |      0 | acc      | ↑    | 0.5714 | ±    | 0.0077 |
|                |         | none   |      0 | acc_norm | ↑    | 0.5714 | ±    | 0.0077 |
| medqa_4options | Yaml    | none   |      0 | acc      | ↑    | 0.6237 | ±    | 0.0136 |
|                |         | none   |      0 | acc_norm | ↑    | 0.6237 | ±    | 0.0136 |
| pubmedqa       | 1       | none   |      0 | acc      | ↑    | 0.7780 | ±    | 0.0186 |

Meditron3-8B-benchmark.

| Tasks          | Version | Filter | n-shot | Metric   |      |  Value |      | Stderr |
| -------------- | ------- | ------ | -----: | -------- | ---- | -----: | ---- | -----: |
| medmcqa        | Yaml    | none   |      0 | acc      | ↑    | 0.5984 | ±    | 0.0076 |
|                |         | none   |      0 | acc_norm | ↑    | 0.5984 | ±    | 0.0076 |
| medqa_4options | Yaml    | none   |      0 | acc      | ↑    | 0.6332 | ±    | 0.0135 |
|                |         | none   |      0 | acc_norm | ↑    | 0.6332 | ±    | 0.0135 |
| pubmedqa       | 1       | none   |      0 | acc      | ↑    | 0.7520 | ±    | 0.0193 |

Meditron-8B after SFT and DPO.

| Tasks          | Version | Filter | n-shot | Metric   |      |  Value |      | Stderr |
| -------------- | ------- | ------ | -----: | -------- | ---- | -----: | ---- | -----: |
| medmcqa        | Yaml    | none   |      0 | acc      | ↑    | 0.5738 | ±    | 0.0076 |
|                |         | none   |      0 | acc_norm | ↑    | 0.5738 | ±    | 0.0076 |
| medqa_4options | Yaml    | none   |      0 | acc      | ↑    | 0.6237 | ±    | 0.0136 |
|                |         | none   |      0 | acc_norm | ↑    | 0.6237 | ±    | 0.0136 |
| pubmedqa       | 1       | none   |      0 | acc      | ↑    | 0.7800 | ±    | 0.0185 |

Meditron-8B after SFT and 2nd DPO.

| Tasks          | Version | Filter | n-shot | Metric   |      |  Value |      | Stderr |
| -------------- | ------- | ------ | -----: | -------- | ---- | -----: | ---- | -----: |
| medmcqa        | Yaml    | none   |      0 | acc      | ↑    | 0.5730 | ±    | 0.0076 |
|                |         | none   |      0 | acc_norm | ↑    | 0.5730 | ±    | 0.0076 |
| medqa_4options | Yaml    | none   |      0 | acc      | ↑    | 0.6253 | ±    | 0.0136 |
|                |         | none   |      0 | acc_norm | ↑    | 0.6253 | ±    | 0.0136 |
| pubmedqa       | 1       | none   |      0 | acc      | ↑    | 0.7760 | ±    | 0.0187 |

## CMMLU Results

Meditron-8B after SFT. (zero-shot)

| Tasks                       | Version | Filter | n-shot | Metric   |      |  Value |      | Stderr |
| --------------------------- | ------: | ------ | -----: | -------- | ---- | -----: | ---- | -----: |
| cmmlu_anatomy               |       1 | none   |      0 | acc      | ↑    | 0.3851 | ±    | 0.0401 |
|                             |         | none   |      0 | acc_norm | ↑    | 0.3851 | ±    | 0.0401 |
| cmmlu_clinical_knowledge    |       1 | none   |      0 | acc      | ↑    | 0.5401 | ±    | 0.0324 |
|                             |         | none   |      0 | acc_norm | ↑    | 0.5401 | ±    | 0.0324 |
| cmmlu_college_medicine      |       1 | none   |      0 | acc      | ↑    | 0.6227 | ±    | 0.0294 |
|                             |         | none   |      0 | acc_norm | ↑    | 0.6227 | ±    | 0.0294 |
| cmmlu_genetics              |       1 | none   |      0 | acc      | ↑    | 0.5341 | ±    | 0.0377 |
|                             |         | none   |      0 | acc_norm | ↑    | 0.5341 | ±    | 0.0377 |
| cmmlu_professional_medicine |       1 | none   |      0 | acc      | ↑    | 0.5426 | ±    | 0.0257 |
|                             |         | none   |      0 | acc_norm | ↑    | 0.5426 | ±    | 0.0257 |

Meditron-8B after SFT. (five-shot)

| Tasks                       | Version | Filter | n-shot | Metric   |      |  Value |      | Stderr |
| --------------------------- | ------: | ------ | -----: | -------- | ---- | -----: | ---- | -----: |
| cmmlu_anatomy               |       1 | none   |      5 | acc      | ↑    | 0.3986 | ±    | 0.0404 |
|                             |         | none   |      5 | acc_norm | ↑    | 0.3986 | ±    | 0.0404 |
| cmmlu_clinical_knowledge    |       1 | none   |      5 | acc      | ↑    | 0.5443 | ±    | 0.0324 |
|                             |         | none   |      5 | acc_norm | ↑    | 0.5443 | ±    | 0.0324 |
| cmmlu_college_medicine      |       1 | none   |      5 | acc      | ↑    | 0.6081 | ±    | 0.0296 |
|                             |         | none   |      5 | acc_norm | ↑    | 0.6081 | ±    | 0.0296 |
| cmmlu_genetics              |       1 | none   |      5 | acc      | ↑    | 0.5341 | ±    | 0.0377 |
|                             |         | none   |      5 | acc_norm | ↑    | 0.5341 | ±    | 0.0377 |
| cmmlu_professional_medicine |       1 | none   |      5 | acc      | ↑    | 0.5266 | ±    | 0.0258 |
|                             |         | none   |      5 | acc_norm | ↑    | 0.5266 | ±    | 0.0258 |

Meditron-8B after SFT and DPO. (five-shot)

| Tasks                       | Version | Filter | n-shot | Metric   |      |  Value |      | Stderr |
| --------------------------- | ------: | ------ | -----: | -------- | ---- | -----: | ---- | -----: |
| cmmlu_anatomy               |       1 | none   |      5 | acc      | ↑    | 0.4122 | ±    | 0.0406 |
|                             |         | none   |      5 | acc_norm | ↑    | 0.4122 | ±    | 0.0406 |
| cmmlu_clinical_knowledge    |       1 | none   |      5 | acc      | ↑    | 0.5527 | ±    | 0.0324 |
|                             |         | none   |      5 | acc_norm | ↑    | 0.5527 | ±    | 0.0324 |
| cmmlu_college_medicine      |       1 | none   |      5 | acc      | ↑    | 0.6154 | ±    | 0.0295 |
|                             |         | none   |      5 | acc_norm | ↑    | 0.6154 | ±    | 0.0295 |
| cmmlu_genetics              |       1 | none   |      5 | acc      | ↑    | 0.5284 | ±    | 0.0377 |
|                             |         | none   |      5 | acc_norm | ↑    | 0.5284 | ±    | 0.0377 |
| cmmlu_professional_medicine |       1 | none   |      5 | acc      | ↑    | 0.5319 | ±    | 0.0258 |
|                             |         | none   |      5 | acc_norm | ↑    | 0.5319 | ±    | 0.0258 |

Meditron-8B after SFT and 2nd DPO. (five-shot)

| Tasks                       | Version | Filter | n-shot | Metric   |      |  Value |      | Stderr |
| --------------------------- | ------: | ------ | -----: | -------- | ---- | -----: | ---- | -----: |
| cmmlu_anatomy               |       1 | none   |      5 | acc      | ↑    | 0.4054 | ±    | 0.0405 |
|                             |         | none   |      5 | acc_norm | ↑    | 0.4054 | ±    | 0.0405 |
| cmmlu_clinical_knowledge    |       1 | none   |      5 | acc      | ↑    | 0.5443 | ±    | 0.0324 |
|                             |         | none   |      5 | acc_norm | ↑    | 0.5443 | ±    | 0.0324 |
| cmmlu_college_medicine      |       1 | none   |      5 | acc      | ↑    | 0.6190 | ±    | 0.0294 |
|                             |         | none   |      5 | acc_norm | ↑    | 0.6190 | ±    | 0.0294 |
| cmmlu_genetics              |       1 | none   |      5 | acc      | ↑    | 0.5341 | ±    | 0.0377 |
|                             |         | none   |      5 | acc_norm | ↑    | 0.5341 | ±    | 0.0377 |
| cmmlu_professional_medicine |       1 | none   |      5 | acc      | ↑    | 0.5239 | ±    | 0.0258 |
|                             |         | none   |      5 | acc_norm | ↑    | 0.5239 | ±    | 0.0258 |

## OpenCompass Results

Meditron-8B after SFT.

| dataset                     | version | metric   | mode | meditron3_8B_ppl |
| --------------------------- | ------- | -------- | ---- | ---------------- |
| cmmlu-anatomy               | e2b12f  | accuracy | ppl  | 43.24            |
| cmmlu-clinical_knowledge    | 9993e7  | accuracy | ppl  | 54.85            |
| cmmlu-college_medicine      | 5d2d59  | accuracy | ppl  | 60.81            |
| cmmlu-genetics              | aa522b  | accuracy | ppl  | 54.55            |
| cmmlu-professional_medicine | 0a6df4  | accuracy | ppl  | 54.26            |

Meditron-8B after SFT & DPO.

| dataset                     | version | metric   | mode | meditron3_8B_ppl |
| --------------------------- | ------- | -------- | ---- | ---------------- |
| cmmlu-anatomy               | e2b12f  | accuracy | ppl  | 42.57            |
| cmmlu-clinical_knowledge    | 9993e7  | accuracy | ppl  | 54.43            |
| cmmlu-college_medicine      | 5d2d59  | accuracy | ppl  | 60.44            |
| cmmlu-genetics              | aa522b  | accuracy | ppl  | 54.55            |
| cmmlu-professional_medicine | 0a6df4  | accuracy | ppl  | 53.99            |
