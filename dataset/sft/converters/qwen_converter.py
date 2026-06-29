import os
import json
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from .base_converter import BaseConverter

class QwenConverter(BaseConverter):
    """
    Uses LLM (via OpenAI compatible API) to translate medical text.
    """
    def __init__(self, input_path, output_path):
        super().__init__(input_path, output_path)
        load_dotenv()
        
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.model_name = "qwen-turbo"
        
        if not self.api_key:
            raise ValueError("API Key not found. Please set DASHSCOPE_API_KEY in .env file.")

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _call_llm(self, text):
        system_prompt = (
            "你是一个专业的医学翻译引擎。你的任务是将输入的中文医学文本准确翻译成英文。\n"
            "规则：\n"
            "1. 严禁输出'好的'、'翻译如下'等闲聊内容。\n"
            "2. 严禁对原文进行解释或扩写，只做翻译。\n"
            "3. 保持医学术语的专业性（如将 高血压 翻译为 Hypertension）。\n"
            "4. 直接输出翻译结果。"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
                max_tokens=2048
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] API Call failed: {e}")
            return None

    def process(self):
        converted_data = []
        
        # Load raw lines
        lines = self.raw_data if isinstance(self.raw_data, list) else self.raw_data.split('\n')
        lines = [l for l in lines if l.strip()]

        print(f"[INFO] Starting LLM Translation for top {self.max_samples} items...")
        
        for line in tqdm(lines[:self.max_samples]):
            translated_text = self._call_llm(line)
            
            if translated_text:
                user_content = f"下面是一段中文医学文本，请将其翻译成英文：\n{line}"
                
                # Check formatting/ethics via base class
                msg = self.format_message(user_content, translated_text)
                
                if msg:
                    # Enriched with original data for debugging
                    msg['original'] = line
                    msg['translation'] = translated_text
                    converted_data.append(msg)
        
        self.save_data(converted_data)