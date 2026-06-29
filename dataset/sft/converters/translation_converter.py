import os
from .base_converter import BaseConverter

class TranslationConverter(BaseConverter):
    """
    Handles static parallel corpus translation conversion.
    
    Modes:
    - translation: Expects 'input_path' to be a DIRECTORY containing 'en.txt' and 'zh.txt'.
      It reads both files line-by-line and aligns them.
    - translation2: Expects 'input_path' to be a JSON file containing abstract pairs.
    """
    def __init__(self, input_path, output_path, mode):
        # Note: BaseConverter loads data automatically, but for 'translation' mode (directory),
        # we handle loading specifically in process() to avoid errors in the base class.
        if mode == 'translation':
            self.input_path = input_path
            self.output_path = output_path
            self.max_samples = 10
            self.raw_data = None # Deferred loading
        else:
            super().__init__(input_path, output_path)
            
        self.mode = mode

    def process(self):
        converted_data = []

        if self.mode == 'translation':
            # Handle Two TXT Files Mode
            if not os.path.isdir(self.input_path):
                print(f"[ERROR] Mode 'translation' requires input_path to be a directory containing en.txt and zh.txt")
                return

            en_path = os.path.join(self.input_path, 'en.txt')
            zh_path = os.path.join(self.input_path, 'zh.txt')

            if not os.path.exists(en_path) or not os.path.exists(zh_path):
                print(f"[ERROR] Could not find 'en.txt' or 'zh.txt' in {self.input_path}")
                return

            try:
                with open(en_path, 'r', encoding='utf-8') as f_en, open(zh_path, 'r', encoding='utf-8') as f_zh:
                    en_lines = f_en.readlines()
                    zh_lines = f_zh.readlines()

                # Align by minimum length
                limit = min(len(en_lines), len(zh_lines), self.max_samples)
                
                for i in range(limit):
                    en_text = en_lines[i].strip()
                    zh_text = zh_lines[i].strip()
                    
                    if not en_text or not zh_text: 
                        continue

                    user = f"请将以下英文翻译成中文：\n\n{en_text}\n\n中文翻译："
                    assistant = zh_text
                    
                    msg = self.format_message(user, assistant)
                    if msg: converted_data.append(msg)
                    
            except Exception as e:
                print(f"[ERROR] Reading translation files: {e}")

        elif self.mode == 'translation2':
            # Handle JSON Abstract Mode
            # Assumes structure: List where 2*i is English, 2*i+1 is Chinese dicts
            iterable_data = self.raw_data if isinstance(self.raw_data, list) else []
            limit = min(int(len(iterable_data) / 2), self.max_samples)
            
            for i in range(limit):
                try:
                    # Logic adapted from your original script
                    en_text = iterable_data[2*i]["abstracttext"].strip()
                    zh_text = iterable_data[2*i+1]["abstracttext"].strip()
                    
                    user = f"请将以下英文翻译成中文：\n\n{en_text}\n\n中文翻译："
                    assistant = zh_text
                    
                    msg = self.format_message(user, assistant)
                    if msg: converted_data.append(msg)
                except (IndexError, KeyError, TypeError) as e:
                    print(f"[WARN] Error parsing translation2 item at index {i}: {e}")
                    continue

        self.save_data(converted_data)