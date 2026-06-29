#!/usr/bin/env python3
"""
多样化医学问题生成器
专门设计用于生成高度多样化、覆盖面广的医学咨询问题
"""

import json
import asyncio
import aiohttp
import argparse
import logging
import random
from typing import List, Dict
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiverseMedicalQuestionGenerator:
    """多样化医学问题生成器"""
    
    def __init__(self, provider: str = "deepseek", api_key: str = ""):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = "deepseek-chat"
        self.base_url = "https://api.deepseek.com/v1"
        self.questions = []
        
        # 详细的医学专科分类
        self.medical_specialties = {
            "消化内科": {
                "common_diseases": ["胃炎", "胃溃疡", "十二指肠溃疡", "肝炎", "胆囊炎", "胰腺炎", "肠炎", "便秘"],
                "symptoms": ["胃痛", "腹痛", "恶心", "呕吐", "腹胀", "腹泻", "便血", "黄疸"],
                "question_types": ["症状咨询", "饮食指导", "药物咨询", "检查解读", "生活建议"]
            },
            "呼吸内科": {
                "common_diseases": ["感冒", "肺炎", "哮喘", "COPD", "肺结核", "支气管炎"],
                "symptoms": ["咳嗽", "咳痰", "胸闷", "气短", "胸痛", "发热"],
                "question_types": ["症状识别", "治疗方案", "预防措施", "康复指导"]
            },
            "皮肤科": {
                "common_diseases": ["湿疹", "皮炎", "痤疮", "荨麻疹", "银屑病", "真菌感染", "脱发"],
                "symptoms": ["皮疹", "瘙痒", "红肿", "脱皮", "色斑", "疱疹"],
                "question_types": ["症状判断", "护理方法", "用药指导", "美容咨询"]
            },
            "心血管": {
                "common_diseases": ["高血压", "冠心病", "心律失常", "心衰", "心肌梗死"],
                "symptoms": ["胸痛", "心悸", "气短", "水肿", "头晕"],
                "question_types": ["风险评估", "用药管理", "生活调整", "急救处理"]
            },
            "妇科": {
                "common_diseases": ["月经不调", "阴道炎", "宫颈炎", "子宫肌瘤", "卵巢囊肿"],
                "symptoms": ["月经异常", "白带异常", "盆腔痛", "外阴瘙痒"],
                "question_types": ["生理咨询", "孕期保健", "妇科检查", "避孕指导"]
            },
            "儿科": {
                "common_diseases": ["小儿感冒", "小儿肺炎", "腹泻病", "手足口病", "湿疹"],
                "symptoms": ["发热", "咳嗽", "腹泻", "皮疹", "食欲不振"],
                "question_types": ["育儿咨询", "疫苗接种", "生长发育", "营养指导"]
            },
            "检验报告解读": {
                "test_types": ["血常规", "尿常规", "生化检查", "肝功能", "肾功能", "血糖", "血脂", "甲状腺功能"],
                "abnormal_indicators": ["白细胞高", "贫血", "肝酶升高", "肌酐升高", "血糖异常"],
                "question_types": ["指标解释", "异常原因", "复查建议", "生活调整"]
            },
            "健康科普": {
                "topics": ["营养饮食", "运动健身", "疾病预防", "体检指导", "心理健康"],
                "life_stages": ["婴幼儿", "青少年", "成年人", "中年人", "老年人"],
                "question_types": ["知识普及", "误区澄清", "生活指导", "预防建议"]
            },
            "错误危险提问": {
                "dangerous_requests": [
                    "自行诊断恶性肿瘤", "要求开具管制药物", "危险手术咨询", 
                    "自伤方法询问", "严重疾病自我治疗", "药物滥用指导"
                ],
                "inappropriate_content": ["具体个人隐私", "非法药物获取", "无资质诊断"]
            }
        }
        
        # 多样化的问题开头和表达方式
        self.question_starters = [
            # 直接询问类
            "请问", "想问一下", "咨询一下", "想了解", "请教",
            
            # 描述症状类  
            "我", "我妈妈", "我爸爸", "我家孩子", "我朋友", "家里老人",
            "最近", "这几天", "持续", "经常", "偶尔",
            
            # 专业咨询类
            "医生", "专家", "关于", "对于", "针对",
            
            # 情况描述类
            "如果", "当", "出现", "遇到", "发生",
            
            # 比较选择类
            "是不是", "会不会", "要不要", "能不能", "该不该"
        ]
        
        # 语言风格变体
        self.language_styles = [
            "formal",      # 正式医学术语
            "casual",      # 口语化表达
            "worried",     # 担心焦虑语气
            "elderly",     # 老年人表达习惯
            "parent",      # 家长关心孩子
            "academic",    # 学术讨论风格
            "practical"    # 实用性导向
        ]
    
    async def generate_diverse_questions_by_specialty(self, specialty: str, count: int = 100) -> List[str]:
        """为特定专科生成多样化问题"""
        
        specialty_info = self.medical_specialties.get(specialty, {})
        
        # 简化prompt，减少token消耗
        prompt = f"""生成{count}个{specialty}真实医学咨询问题。

要求：
1. 每个问题都不同，避免重复表述
2. 1-2句话，简洁明确
3. 真实患者语言，不要太专业

{specialty}包含：
{specialty_info.get('common_diseases', [])[:5]}  # 只取前5个
{specialty_info.get('symptoms', [])[:5]}

问题风格示例：
- 我胃痛三天了，要紧吗？
- 咳嗽有痰是肺炎吗？
- 皮肤起红疹很痒怎么办？

直接输出{count}个问题，每行一个："""

        try:
            response = await self._call_api(prompt)
            questions = self._parse_questions(response)
            
            logger.info(f"{specialty} 生成了 {len(questions)} 个问题")
            return questions
            
        except Exception as e:
            logger.error(f"生成{specialty}问题失败: {e}")
            return []
    
    async def _call_api(self, prompt: str) -> str:
        """调用API，增强错误处理"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 减少max_tokens，避免超限
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.9,
            "max_tokens": 1500,  # 从3000减少到1500
            "top_p": 0.95
        }
        
        logger.debug(f"请求数据大小: {len(prompt)} 字符")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=120)  # 增加超时时间
                ) as response:
                    response_text = await response.text()
                    
                    if response.status == 200:
                        result = json.loads(response_text)
                        return result["choices"][0]["message"]["content"]
                    else:
                        logger.error(f"API错误 {response.status}: {response_text}")
                        
                        # 检查常见错误类型
                        if response.status == 400:
                            logger.error("可能是请求格式错误或token超限")
                        elif response.status == 401:
                            logger.error("API密钥错误或无效")
                        elif response.status == 429:
                            logger.error("请求频率过高，建议增加延时")
                            await asyncio.sleep(30)  # 遇到限频错误等待30秒
                        elif response.status >= 500:
                            logger.error("服务器内部错误，稍后重试")
                            
                        raise Exception(f"API错误 {response.status}: {response_text}")
                        
        except asyncio.TimeoutError:
            logger.error("API调用超时")
            raise Exception("API调用超时")
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            raise Exception(f"响应解析失败: {e}")
        except Exception as e:
            logger.error(f"API调用异常: {e}")
            raise
    
    def _parse_questions(self, response: str) -> List[str]:
        """解析API响应中的问题"""
        questions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 清理格式
            # 去掉编号
            if line[0].isdigit() and ('.' in line[:10] or '、' in line[:10]):
                line = line.split('.', 1)[-1].split('、', 1)[-1].strip()
            elif line.startswith('-') or line.startswith('*'):
                line = line[1:].strip()
            elif line.startswith('问题'):
                line = line.replace('问题', '', 1).strip()
                if line.startswith('：') or line.startswith(':'):
                    line = line[1:].strip()
            
            # 验证问题质量
            if self._is_valid_question(line):
                questions.append(line)
        
        return questions
    
    def _is_valid_question(self, text: str) -> bool:
        """验证问题是否有效"""
        if not text or len(text) < 8 or len(text) > 300:
            return False
        
        # 必须包含问号
        if '？' not in text and '?' not in text:
            return False
        
        # 检查医学相关性
        medical_keywords = [
            '症状', '疾病', '治疗', '药物', '检查', '诊断', '医生', '医院',
            '痛', '疼', '不舒服', '异常', '正常', '健康', '病', '炎', 
            '血', '尿', '便', '心', '肝', '肺', '胃', '肾', '皮肤',
            '发热', '发烧', '咳嗽', '头痛', '腹痛', '胸痛'
        ]
        
        if not any(keyword in text for keyword in medical_keywords):
            return False
        
        return True
    
    async def generate_all_diverse_questions(self, target_count: int = 8000):
        """生成所有多样化问题"""
        logger.info(f"开始生成 {target_count} 个多样化医学问题...")
        
        specialties = list(self.medical_specialties.keys())
        questions_per_specialty = target_count // len(specialties)
        
        all_questions = []
        
        for specialty in specialties:
            logger.info(f"正在生成 {specialty} 问题...")
            
            # 分更多批次，每批生成更少问题
            batches = 5  # 从3批改为5批
            questions_per_batch = max(20, questions_per_specialty // batches)  # 每批最少20个
            
            specialty_questions = []
            
            for batch_num in range(batches):
                logger.info(f"  {specialty} 第{batch_num+1}批 (目标{questions_per_batch}个)...")
                
                try:
                    batch_questions = await self.generate_diverse_questions_by_specialty(
                        specialty, questions_per_batch
                    )
                    specialty_questions.extend(batch_questions)
                    
                    # 增加API调用间隔
                    await asyncio.sleep(5)  # 从3秒改为5秒
                    
                except Exception as e:
                    logger.error(f"  {specialty} 第{batch_num+1}批失败: {e}")
                    # 失败后等待更长时间再重试
                    await asyncio.sleep(10)
                    continue
            
            all_questions.extend(specialty_questions)
            logger.info(f"{specialty} 共生成 {len(specialty_questions)} 个问题")
            
            # 每个专科完成后暂停
            await asyncio.sleep(3)
        
        # 智能去重
        unique_questions = self._advanced_deduplication(all_questions)
        
        # 随机采样到目标数量
        if len(unique_questions) > target_count:
            self.questions = random.sample(unique_questions, target_count)
        else:
            self.questions = unique_questions
        
        logger.info(f"生成完成：{len(all_questions)} -> {len(unique_questions)} -> {len(self.questions)}")
    
    def _advanced_deduplication(self, questions: List[str]) -> List[str]:
        """高级去重算法"""
        logger.info("开始智能去重...")
        
        unique_questions = []
        question_signatures = []
        
        for question in questions:
            signature = self._get_question_signature(question)
            
            # 检查是否与现有问题过于相似
            is_duplicate = False
            for existing_sig in question_signatures:
                if self._similarity_score(signature, existing_sig) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_questions.append(question)
                question_signatures.append(signature)
        
        logger.info(f"去重完成：{len(questions)} -> {len(unique_questions)}")
        return unique_questions
    
    def _get_question_signature(self, question: str) -> Dict:
        """获取问题的特征签名"""
        import re
        
        # 提取关键特征
        signature = {
            'length_range': len(question) // 10,  # 长度范围
            'key_terms': set(),
            'structure': '',
            'question_words': set()
        }
        
        # 提取关键医学术语
        medical_terms = [
            '高血压', '糖尿病', '心脏病', '肺炎', '胃炎', '肝炎', '肾病',
            '头痛', '胸痛', '腹痛', '发热', '咳嗽', '腹泻', '便秘',
            '皮疹', '瘙痒', '失眠', '焦虑', '抑郁', '月经', '怀孕'
        ]
        
        for term in medical_terms:
            if term in question:
                signature['key_terms'].add(term)
        
        # 分析问题词
        question_words = ['什么', '怎么', '如何', '为什么', '能不能', '要不要', '是不是']
        for word in question_words:
            if word in question:
                signature['question_words'].add(word)
        
        # 分析结构
        if '，' in question:
            signature['structure'] = 'complex'
        elif len(question) > 20:
            signature['structure'] = 'medium'
        else:
            signature['structure'] = 'simple'
        
        return signature
    
    def _similarity_score(self, sig1: Dict, sig2: Dict) -> float:
        """计算两个问题签名的相似度"""
        score = 0.0
        
        # 长度相似度
        if sig1['length_range'] == sig2['length_range']:
            score += 0.2
        
        # 关键术语重叠度
        terms1 = sig1['key_terms']
        terms2 = sig2['key_terms']
        if terms1 or terms2:
            intersection = len(terms1.intersection(terms2))
            union = len(terms1.union(terms2))
            if union > 0:
                score += (intersection / union) * 0.5
        
        # 问题词相似度
        words1 = sig1['question_words']
        words2 = sig2['question_words']
        if words1 and words2 and words1 == words2:
            score += 0.2
        
        # 结构相似度
        if sig1['structure'] == sig2['structure']:
            score += 0.1
        
        return score
    
    def save_questions(self, output_file: str):
        """保存问题"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for q in self.questions:
                data = {
                    'question': q,
                    'source': 'diverse_llm_generation',
                    'generated_at': datetime.now().isoformat(),
                    'type': 'medical_consultation'
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        logger.info(f"已保存 {len(self.questions)} 个问题到 {output_file}")
    
    def analyze_diversity(self):
        """分析生成问题的多样性"""
        if not self.questions:
            return
        
        # 统计开头词汇
        starters = {}
        question_lengths = []
        
        for q in self.questions:
            # 分析开头
            first_chars = q[:3]
            starters[first_chars] = starters.get(first_chars, 0) + 1
            
            # 统计长度
            question_lengths.append(len(q))
        
        print("\n=== 多样性分析 ===")
        print(f"总问题数: {len(self.questions)}")
        print(f"平均长度: {sum(question_lengths)/len(question_lengths):.1f}")
        print(f"长度范围: {min(question_lengths)} - {max(question_lengths)}")
        print(f"开头方式种类: {len(starters)}")
        
        # 显示最常见的开头
        print("\n最常见的开头方式:")
        sorted_starters = sorted(starters.items(), key=lambda x: x[1], reverse=True)
        for starter, count in sorted_starters[:10]:
            print(f"  '{starter}': {count}次")


async def main():
    parser = argparse.ArgumentParser(description="多样化医学问题生成")
    parser.add_argument("--api-key", required=True, help="DeepSeek API密钥")
    parser.add_argument("--count", type=int, default=8000, help="生成问题数量")
    parser.add_argument("--output", default="diverse_medical_questions.jsonl", help="输出文件")
    parser.add_argument("--analyze", action="store_true", help="分析多样性")
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = DiverseMedicalQuestionGenerator(api_key=args.api_key)
    
    # 生成问题
    await generator.generate_all_diverse_questions(args.count)
    
    # 分析多样性
    if args.analyze:
        generator.analyze_diversity()
    
    # 保存
    generator.save_questions(args.output)
    
    print(f"\n✅ 成功生成 {len(generator.questions)} 个多样化医学问题")
    print(f"📁 文件位置: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
