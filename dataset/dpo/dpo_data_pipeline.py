#!/usr/bin/env python3
"""
DPO偏好数据集生成Pipeline
用于从医学问题生成偏好对，包括candidate生成和judge评判
"""

import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging
from dataclasses import dataclass
import time
import random

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Pipeline配置"""
    # 模型路径
    model_path: str = "${PROJECT_ROOT}/datasets/polyglot1/models/checkpoint-615"
    
    # 数据路径
    input_questions_file: str = "medical_questions_2000.jsonl"
    output_file: str = "output/dpo_preferences.jsonl"
    
    # 生成参数
    num_candidates: int = 2  # 每个问题生成的候选答案数
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    
    # Judge模型配置
    judge_model: str = "deepseek-chat"
    judge_api_base: str = "https://api.deepseek.com"
    judge_api_key: str = ""  # 需要设置
    
    # 并发控制 - 降低并发数，增加稳定性
    max_concurrent: int = 2
    request_delay: float = 3.0  # 增加请求间隔
    
    # 质量控制
    min_answer_length: int = 30
    max_answer_length: int = 800


class MedicalQuestionLoader:
    """加载医学问题数据集"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def load_questions(self) -> List[Dict[str, str]]:
        """加载问题列表"""
        questions = []
        
        try:
            with open(self.config.input_questions_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if 'question' in data:
                            questions.append(data)
                        elif 'text' in data and ('？' in data['text'] or '?' in data['text']):
                            questions.append({'question': data['text']})
                    except json.JSONDecodeError:
                        logger.warning(f"跳过第{line_num}行：JSON格式错误")
                        continue
                        
        except FileNotFoundError:
            logger.error(f"问题文件 {self.config.input_questions_file} 不存在")
            return []
        
        logger.info(f"成功加载了 {len(questions)} 个医学问题")
        return questions
    
    def sample_questions(self, questions: List[Dict], target_size: int = 2000) -> List[Dict]:
        """采样指定数量的问题"""
        if len(questions) > target_size:
            sampled = random.sample(questions, target_size)
            logger.info(f"从 {len(questions)} 个问题中采样了 {target_size} 个")
            return sampled
        logger.info(f"使用所有 {len(questions)} 个问题")
        return questions


class ModelInference:
    """模型推理器"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """加载模型和分词器"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info("正在加载模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            # 添加pad token如果不存在
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("设置pad_token为eos_token")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def generate_candidates(self, question: str) -> List[str]:
        """为单个问题生成多个候选答案"""
        candidates = []
        
        # 构造prompt
        messages = [
            {"role": "user", "content": f"请详细回答以下医学问题：{question}"}
        ]
        
        try:
            import torch
            
            # 应用chat template
            inputs = self.tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt",
                add_generation_prompt=True
            ).to(self.model.device)
            
            # 生成多个候选答案
            for i in range(self.config.num_candidates):
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature + i * 0.1,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # 解码答案
                response = self.tokenizer.decode(
                    outputs[0][inputs.shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                # 质量过滤
                if (self.config.min_answer_length <= len(response) <= self.config.max_answer_length):
                    candidates.append(response)
            
            if len(candidates) < 2:
                # 如果质量过滤后不足2个候选答案，放宽条件重新生成
                logger.warning(f"候选答案不足，为问题重新生成: {question[:50]}...")
                
                for i in range(2):
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs,
                            max_new_tokens=self.config.max_new_tokens,
                            temperature=0.9 + i * 0.1,
                            top_p=self.config.top_p,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    
                    response = self.tokenizer.decode(
                        outputs[0][inputs.shape[1]:], 
                        skip_special_tokens=True
                    ).strip()
                    
                    if len(response) >= 20:  # 放宽长度要求
                        candidates.append(response)
                        
            return candidates[:2]  # 确保只返回2个
            
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return []


class AnswerJudge:
    """答案评判器，使用DeepSeek API"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    async def judge_answers(self, question: str, answer1: str, answer2: str) -> Dict[str, str]:
        """比较两个答案的质量"""
        
        # 简化的judge prompt，减少token消耗
        judge_prompt = f"""作为医学专家，比较以下两个答案的质量：

问题：{question}

答案A：{answer1[:400]}...

答案B：{answer2[:400]}...

请选择更好的答案，只回答A或B："""

        try:
            result = await self._call_deepseek_api(judge_prompt)
            
            # 解析结果
            result_upper = result.upper().strip()
            if "A" in result_upper and "B" not in result_upper:
                return {"chosen": answer1, "rejected": answer2}
            elif "B" in result_upper and "A" not in result_upper:
                return {"chosen": answer2, "rejected": answer1}
            else:
                # 无法确定，选择较长的答案
                if len(answer1) > len(answer2):
                    return {"chosen": answer1, "rejected": answer2}
                else:
                    return {"chosen": answer2, "rejected": answer1}
                
        except Exception as e:
            logger.error(f"Judge评判失败: {e}")
            # 默认选择较长的答案
            if len(answer1) > len(answer2):
                return {"chosen": answer1, "rejected": answer2}
            else:
                return {"chosen": answer2, "rejected": answer1}
    
    async def _call_deepseek_api(self, prompt: str) -> str:
        """调用DeepSeek API"""
        import aiohttp
        
        headers = {
            "Authorization": f"Bearer {self.config.judge_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 50  # 减少token消耗
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.config.judge_api_base}/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result["choices"][0]["message"]["content"]
                        elif response.status == 429:
                            wait_time = (attempt + 1) * 10
                            logger.warning(f"API限频，等待{wait_time}秒后重试...")
                            await asyncio.sleep(wait_time)
                        else:
                            error_text = await response.text()
                            logger.error(f"API错误 {response.status}: {error_text}")
                            break
                            
            except Exception as e:
                logger.error(f"API调用异常 (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)
        
        raise Exception(f"API调用失败，已重试{max_retries}次")


class DPODatasetGenerator:
    """DPO数据集生成器"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.question_loader = MedicalQuestionLoader(config)
        self.model_inference = ModelInference(config)
        self.answer_judge = AnswerJudge(config)
    
    async def generate_preference_pair(self, question: str) -> Dict[str, str] | None:
        """为单个问题生成偏好对"""
        try:
            # 生成候选答案
            candidates = self.model_inference.generate_candidates(question)
            
            if len(candidates) < 2:
                logger.warning(f"问题候选答案不足: {question[:50]}...")
                return None
            
            # 选择两个不同的候选答案
            answer1, answer2 = candidates[0], candidates[1]
            
            # 确保答案不完全相同
            if answer1.strip() == answer2.strip():
                logger.warning(f"候选答案相同，跳过: {question[:50]}...")
                return None
            
            # Judge评判
            judgment = await self.answer_judge.judge_answers(question, answer1, answer2)
            
            return {
                "question": question,
                "chosen": judgment["chosen"],
                "rejected": judgment["rejected"]
            }
            
        except Exception as e:
            logger.error(f"生成偏好对失败 [{question[:30]}...]: {e}")
            return None
    
    async def generate_dataset(self, target_size: int = 2000):
        """生成完整的DPO数据集"""
        # 加载问题
        questions = self.question_loader.load_questions()
        questions = self.question_loader.sample_questions(questions, target_size)
        
        if not questions:
            logger.error("没有可用的问题，退出")
            return
        
        # 创建输出文件
        output_file = Path(self.config.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        generated_count = 0
        failed_count = 0
        
        logger.info(f"开始处理 {len(questions)} 个问题...")
        
        # 串行处理，避免并发问题
        for i, q_data in enumerate(questions):
            question = q_data.get("question", "").strip()
            if not question:
                continue
            
            try:
                preference_pair = await self.generate_preference_pair(question)
                
                if preference_pair:
                    # 写入文件
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(preference_pair, ensure_ascii=False) + '\n')
                    
                    generated_count += 1
                    
                    if generated_count % 10 == 0:
                        logger.info(f"已生成 {generated_count} 个偏好对 (处理进度: {i+1}/{len(questions)})")
                else:
                    failed_count += 1
                
                # 控制请求频率
                await asyncio.sleep(self.config.request_delay)
                
            except Exception as e:
                logger.error(f"处理问题失败: {e}")
                failed_count += 1
                continue
        
        logger.info(f"数据生成完成！")
        logger.info(f"成功生成: {generated_count} 个偏好对")
        logger.info(f"失败数量: {failed_count} 个")
        logger.info(f"成功率: {generated_count/(generated_count+failed_count)*100:.1f}%")
        logger.info(f"输出文件: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="DPO偏好数据集生成")
    parser.add_argument("--questions", type=str, default="medical_questions_2000.jsonl", help="输入问题文件")
    parser.add_argument("--output", type=str, default="output/dpo_preferences.jsonl", help="输出文件")
    parser.add_argument("--size", type=int, default=2000, help="目标数据集大小")
    parser.add_argument("--judge-api-key", type=str, required=True, help="Judge API密钥")
    
    args = parser.parse_args()
    
    # 创建配置
    config = PipelineConfig(
        input_questions_file=args.questions,
        output_file=args.output,
        judge_api_key=args.judge_api_key
    )
    
    # 运行生成器
    generator = DPODatasetGenerator(config)
    asyncio.run(generator.generate_dataset(args.size))


if __name__ == "__main__":
    main()
