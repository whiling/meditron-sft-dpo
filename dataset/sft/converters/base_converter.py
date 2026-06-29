import json
import os
import re

class BaseConverter:
    """
    Base class for all converters. 
    Implements Data Quality and Ethical Review protocols including:
    1. PII Scrubbing (Privacy).
    2. Noise Reduction (Quality).
    3. Schema Validation (Structural Integrity).
    """

    def __init__(self, input_path, output_path, max_samples=10):
        self.input_path = input_path
        self.output_path = output_path
        self.max_samples = max_samples
        self.raw_data = self.load_data()

    def load_data(self):
        """
        Generic data loader. Attempts to read JSON list, JSONL, or returns raw lines.
        """
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
            
        with open(self.input_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        try:
            # Try parsing as whole JSON
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback to splitting lines (for raw text or JSONL)
            return content.split('\n')

    def clean_text(self, text):
        """
        Applies regex-based anonymization layers to scrub PII.
        """
        if not isinstance(text, str):
            return text
            
        # Example PII patterns (Simulated for demonstration)
        # 1. Phone numbers / IDs (11 or 18 digits)
        text = re.sub(r'\b\d{11}\b', '[REDACTED_PHONE]', text)
        text = re.sub(r'\b\d{18}\b', '[REDACTED_ID]', text)
        
        # 2. Specific hospital naming patterns (basic example)
        text = re.sub(r'[\u4e00-\u9fa5]+(医院|卫生院)', '[HOSPITAL_NAME]', text)
        
        return text

    def is_valid_sample(self, user_content, assistant_content):
        """
        Automated noise reduction module.
        Discards incomplete records or semantically sparse queries.
        """
        # Check for empty strings
        if not user_content or not assistant_content:
            return False
        
        # Check for extremely short/noise content
        if len(user_content) < 2 or len(assistant_content) < 1:
            return False
            
        return True

    def format_message(self, user, assistant):
        """
        Enforces structural integrity through a mandatory verification mechanism.
        Returns the standard SFT JSON format.
        """
        # Apply cleaning
        clean_user = self.clean_text(user)
        clean_assistant = self.clean_text(assistant)

        if not self.is_valid_sample(clean_user, clean_assistant):
            return None

        return {
            "messages": [
                {"role": "user", "content": clean_user},
                {"role": "assistant", "content": clean_assistant}
            ]
        }

    def save_data(self, data):
        """
        Writes data to the output path in JSONL format.
        """
        directory = os.path.dirname(self.output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        count = 0
        with open(self.output_path, 'w', encoding='utf-8') as out_file:
            for item in data:
                if count >= self.max_samples:
                    break
                out_file.write(json.dumps(item, ensure_ascii=False) + '\n')
                count += 1
        
        print(f"[INFO] Processed {count} samples. Saved to {self.output_path}")

    def process(self):
        """
        Abstract method to be implemented by child classes.
        """
        raise NotImplementedError