import json
from .base_converter import BaseConverter

class QAConverter(BaseConverter):
    """
    Converter for Multiple Choice Questions (Multi-Question format).
    """
    def __init__(self, input_path, output_path):
        super().__init__(input_path, output_path)
        self.few_shots = [
            ("问题：人体共有多少对脑神经？\n选项：\nA. 10对\nB. 11对\nC. 12对\nD. 13对\n答案：C. 12对"),
            ("问题：下列哪种维生素缺乏会导致夜盲症？\n选项：\nA. 维生素A\nB. 维生素B\nC. 维生素C\nD. 维生素D\n答案：A. 维生素A")
        ]

    def process(self):
        converted_data = []
        
        # Handle input data robustly
        # BaseConverter.load_data() might return:
        # 1. A list of dicts (if input was a valid JSON array)
        # 2. A list of strings (if input was JSONL/Lines)
        # 3. A single string (if raw text)
        
        iterator = self.raw_data
        if isinstance(self.raw_data, str):
            iterator = self.raw_data.strip().split('\n')

        for item in iterator:
            if len(converted_data) >= self.max_samples:
                break

            # [FIX] Dynamic Parsing: Ensure item is a dictionary
            current_obj = item
            if isinstance(item, str):
                try:
                    if not item.strip(): continue
                    current_obj = json.loads(item)
                except json.JSONDecodeError:
                    print(f"[WARN] Skipping invalid JSON line: {item[:20]}...")
                    continue
            
            # Now process the dictionary
            try:
                question = current_obj['question']
                options = current_obj['options']
                answer_idx = current_obj['answer_idx']
                answer_text = current_obj['answer']

                # Format options alphabetically
                sorted_keys = sorted(options.keys())
                formatted_list = [f"{key}. {options[key]}" for key in sorted_keys]
                options_str = "\n".join(formatted_list)

                user_prompt = (
                    "请阅读以下医学/生物学选择题，并根据上下文选出正确的选项。\n\n"
                    f"示例 1：\n{self.few_shots[0]}\n\n"
                    f"示例 2：\n{self.few_shots[1]}\n\n"
                    "请回答以下问题：\n"
                    f"问题：{question}\n"
                    "选项：\n"
                    f"{options_str}\n"
                    "答案："
                )
                assistant_response = f"{answer_idx}. {answer_text}"

                formatted_msg = self.format_message(user_prompt, assistant_response)
                if formatted_msg:
                    converted_data.append(formatted_msg)
            
            except KeyError as e:
                # Use current_obj for error logging to show what actually failed
                print(f"[WARN] Skipping malformed item (missing key {e}): {str(current_obj)[:50]}...")
                continue

        self.save_data(converted_data)