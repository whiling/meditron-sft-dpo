import json
import re
from .base_converter import BaseConverter

class DialogueConverter(BaseConverter):
    """
    Handles multiple dialogue formats:
    - inquiry: Universal handler for (question/answer) OR (instruction/input/output)
    - inquiry2: Array based QA
    - raw-dialogue: State machine parsing for text files
    - common-multi-dialogue: Multi-turn conversation
    """
    
    def __init__(self, input_path, output_path, mode):
        super().__init__(input_path, output_path)
        self.mode = mode

    def _convert_inquiry(self, item):
        """
        Universal converter for single-turn QA.
        Compatible with:
        1. {"question": "...", "answer": "..."}
        2. {"instruction": "...", "input": "...", "output": "..."}
        """
        # --- 1. Construct User Input (Instruction/Question + Input) ---
        user_parts = []
        
        # Check 'instruction'
        if item.get('instruction') and isinstance(item['instruction'], str) and item['instruction'].strip():
            user_parts.append(item['instruction'].strip())
            
        # Check 'question'
        if item.get('question') and isinstance(item['question'], str) and item['question'].strip():
            user_parts.append(item['question'].strip())
            
        # Check 'input' (often serves as context for the instruction)
        if item.get('input') and isinstance(item['input'], str) and item['input'].strip():
            user_parts.append(item['input'].strip())
            
        user = "\n".join(user_parts)

        # --- 2. Construct Assistant Output (Output/Answer) ---
        # Prioritize 'output', fallback to 'answer'
        assistant = ""
        if item.get('output'):
            assistant = item['output']
        elif item.get('answer'):
            assistant = item['answer']
            
        return user, assistant

    def _convert_inquiry2(self, item):
        # Parses formats where data is ["id:xx", "user:xx", "assistant:xx"]
        try:
            user = item["data"][0][2:] 
            assistant = item["data"][1][2:]
            return user, assistant
        except (IndexError, KeyError, TypeError):
            return "", ""

    def _convert_common_multi_dialogue(self, item):
        user_text = "请根据以下对话内容，回复最后一句话：\n\n"
        for i in range(len(item)-1):
            role = "B" if i % 2 == 1 else "A"
            user_text += f"{role}:{item[i]}\n"
        
        last_role = "B" if len(item) % 2 == 0 else "A"
        assistant_text = f"{last_role}:{item[-1]}"
        return user_text, assistant_text

    def _save_raw_record(self, current_record, converted_data):
        """Helper to save parsed raw-dialogue records."""
        if current_record.get('description') and current_record.get('dialogue'):
            instruction = """【任务】\n请扮演一位专业的医生，根据以下患者的病情描述，给出详细的医疗建议和诊断。\n\n【病情描述】\n"""
            desc_text = "\n".join(current_record['description']).strip()
            dialogue_text = "\n".join(current_record['dialogue']).strip()
            
            user = instruction + desc_text
            assistant = dialogue_text
            
            msg = self.format_message(user, assistant)
            if msg: converted_data.append(msg)

    def process(self):
        converted_data = []

        # ==========================================
        # Logic for Raw Dialogue (State Machine)
        # ==========================================
        if self.mode == 'raw-dialogue':
            lines = self.raw_data
            if isinstance(self.raw_data, str):
                lines = self.raw_data.split('\n')
            elif not isinstance(self.raw_data, list):
                print("[ERROR] Raw data format incorrect for raw-dialogue")
                return

            current_record = {}
            current_state = "IDLE"
            
            for line in lines:
                if len(converted_data) >= self.max_samples:
                    break
                line = line.strip()
                
                if re.match(r'^id=\d+', line):
                    self._save_raw_record(current_record, converted_data)
                    current_record = {'description': [], 'dialogue': []}
                    current_state = "IDLE"
                    continue

                if line == 'Description':
                    current_state = "IN_DESCRIPTION"
                    continue
                if line == 'Dialogue':
                    current_state = "IN_DIALOGUE"
                    continue

                if current_state == "IN_DESCRIPTION" and line:
                    current_record['description'].append(line)
                elif current_state == "IN_DIALOGUE" and line:
                    current_record['dialogue'].append(line)
            
            if len(converted_data) < self.max_samples:
                self._save_raw_record(current_record, converted_data)

        # ==========================================
        # Logic for JSON-based formats
        # ==========================================
        else:
            iterable_data = self.raw_data
            # Handle JSONL loaded as string
            if isinstance(self.raw_data, str):
                iterable_data = [json.loads(line) for line in self.raw_data.split('\n') if line.strip()]
            # Handle List of strings (JSONL lines)
            elif isinstance(self.raw_data, list) and len(self.raw_data) > 0 and isinstance(self.raw_data[0], str):
                 try:
                     iterable_data = [json.loads(line) for line in self.raw_data if line.strip()]
                 except:
                     pass

            for item in iterable_data:
                if len(converted_data) >= self.max_samples:
                    break
                
                user, assistant = "", ""
                try:
                    # Combined Logic: Handles both Question/Answer and Instruction/Input/Output
                    if self.mode == 'inquiry':
                        user, assistant = self._convert_inquiry(item)
                    
                    elif self.mode == 'inquiry2':
                        user, assistant = self._convert_inquiry2(item)
                    elif self.mode == 'common-multi-dialogue':
                        user, assistant = self._convert_common_multi_dialogue(item)
                    
                    msg = self.format_message(user, assistant)
                    if msg: converted_data.append(msg)
                
                except Exception as e:
                    # print(f"[WARN] Error parsing item: {e}") 
                    pass

        self.save_data(converted_data)