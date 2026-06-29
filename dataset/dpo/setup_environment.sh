#!/bin/bash
# DPO数据生成环境安装脚本
# 在CSCS集群上安装必要的依赖

echo "=== 安装DPO数据生成环境 ==="

# 创建项目目录
mkdir -p /users/$USER/dpo_data_generation
cd /users/$USER/dpo_data_generation

# 安装Python依赖
echo "安装Python依赖..."
pip install --break-system-packages \
    transformers \
    torch \
    datasets \
    aiohttp \
    accelerate \
    bitsandbytes

# 创建配置文件
cat > config.yaml << EOF
# DPO数据生成配置
model:
  path: "${PROJECT_ROOT}/datasets/polyglot1/models/checkpoint-615"
  max_new_tokens: 512
  temperature: 0.8
  top_p: 0.9

data:
  input_file: "medical_questions_2000.jsonl"
  output_file: "output/dpo_preferences.jsonl"
  target_size: 1680

generation:
  num_candidates: 2
  min_answer_length: 50
  max_answer_length: 1000

judge:
  model: "deepseek-chat"
  api_base: "https://api.deepseek.com"
  max_concurrent: 3
  request_delay: 2.0

logging:
  level: "INFO"
  save_progress: true
EOF

# 创建测试脚本
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""
测试环境设置是否正确
"""
import sys
import torch
from transformers import AutoTokenizer

def test_environment():
    print("=== 环境测试 ===")
    
    # 测试基本依赖
    try:
        import transformers
        import torch
        import datasets
        import aiohttp
        print("✓ 基本依赖正常")
    except ImportError as e:
        print(f"✗ 依赖缺失: {e}")
        return False
    
    # 测试GPU
    if torch.cuda.is_available():
        print(f"✓ GPU可用: {torch.cuda.get_device_name()}")
        print(f"✓ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("✗ GPU不可用")
        return False
    
    # 测试模型路径
    model_path = "${PROJECT_ROOT}/datasets/polyglot1/models/checkpoint-615"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✓ 模型路径正确，tokenizer加载成功")
        print(f"✓ 词汇表大小: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"✗ 模型路径问题: {e}")
        return False
    
    # 检查问题文件
    import os
    if os.path.exists("medical_questions_2000.jsonl"):
        print("✓ 问题文件存在")
    else:
        print("✗ 问题文件不存在，请确保medical_questions_2000.jsonl在当前目录")
        return False
    
    print("=== 环境测试通过 ===")
    return True

if __name__ == "__main__":
    if test_environment():
        sys.exit(0)
    else:
        sys.exit(1)
EOF

echo "=== 运行环境测试 ==="
python test_setup.py

echo "=== 安装完成 ==="
echo "项目目录: /users/$USER/dpo_data_generation"
echo "配置文件: config.yaml"
echo ""
echo "下一步："
echo "1. 设置DeepSeek API密钥在 run_dpo_data_generation.sh 中"
echo "2. 确保 medical_questions_2000.jsonl 在当前目录"
echo "3. 运行: sbatch run_dpo_data_generation.sh"
