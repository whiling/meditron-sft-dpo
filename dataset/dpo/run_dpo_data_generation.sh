#!/bin/bash
#SBATCH --job-name dpo-data-gen
#SBATCH --chdir /users/$USER/dpo_data_generation
#SBATCH --output /users/$USER/dpo_data_generation/logs/R-%x.%j.out
#SBATCH --error /users/$USER/dpo_data_generation/logs/R-%x.%j.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 32
#SBATCH --time 11:59:59
#SBATCH --environment /users/wanlinhu/.edf/axolotl.toml
#SBATCH -A a127

# 设置环境变量
export HF_HOME=$SCRATCH/hf
export HF_TOKEN="your_huggingface_token_here"
export DEEPSEEK_API_KEY="${OPENAI_API_KEY}"  # 请替换为真实的API密钥

echo "START TIME: $(date)"
echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# 创建必要的目录
mkdir -p /users/$USER/dpo_data_generation/logs
mkdir -p /users/$USER/dpo_data_generation/output

cd /users/$USER/dpo_data_generation

# 设置Python路径
export PYTHONPATH=/users/$USER/dpo_data_generation:$PYTHONPATH

echo "=== 检查输入文件 ==="
if [ -f "medical_questions_2000.jsonl" ]; then
    echo "✓ 找到问题文件: medical_questions_2000.jsonl"
    echo "文件行数: $(wc -l < medical_questions_2000.jsonl)"
    echo "文件预览:"
    head -n 3 medical_questions_2000.jsonl
else
    echo "✗ 问题文件不存在: medical_questions_2000.jsonl"
    echo "请确保文件在当前目录: /users/$USER/dpo_data_generation/"
    exit 1
fi

echo "=== 检查DeepSeek API密钥 ==="
if [ -z "$DEEPSEEK_API_KEY" ] || [ "$DEEPSEEK_API_KEY" = "your_deepseek_api_key_here" ]; then
    echo "✗ DeepSeek API密钥未设置"
    echo "请编辑此脚本，将 DEEPSEEK_API_KEY 设置为真实的API密钥"
    exit 1
else
    echo "✓ DeepSeek API密钥已设置"
fi

echo "=== 开始生成DPO偏好数据集 ==="
python dpo_data_pipeline.py \
    --questions medical_questions_2000.jsonl \
    --output output/dpo_preferences.jsonl \
    --size 2000 \
    --judge-api-key $DEEPSEEK_API_KEY

echo "=== 检查生成结果 ==="
if [ -f "output/dpo_preferences.jsonl" ]; then
    echo "✓ 数据生成完成"
    echo "输出文件路径: /users/$USER/dpo_data_generation/output/dpo_preferences.jsonl"
    echo "生成的偏好对数量: $(wc -l < output/dpo_preferences.jsonl)"
    echo ""
    echo "数据质量检查:"
    head -n 3 output/dpo_preferences.jsonl | python -m json.tool
else
    echo "✗ 数据生成失败，请检查日志文件"
    exit 1
fi

echo "END TIME: $(date)"
