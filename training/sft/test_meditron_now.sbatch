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
