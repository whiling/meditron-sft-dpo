# Medical Dialogue & SFT Data Converter System

A modular, object-oriented system for converting various raw medical dataset formats into a unified JSONL format suitable for Supervised Fine-Tuning (SFT) of Large Language Models.

**Google Drive Link:**  https://epflch-my.sharepoint.com/:u:/g/personal/shuyi_chen_epfl_ch/IQC0XpNT5q2dRqmgh3zKZubpAeE0gsNt1FbFAh_QLl3JWiY?e=zmstxg

---

## 1. Project Structure

```text
sft/
├── .env                # API Keys (Create from .env.example)
├── main.py             # CLI Entry Point
├── requirements.txt    # Python dependencies
├── raw_data_sample/    # Sample input files 
├── output_sft_sample/  # Output JSONL files (Can be generated automatically)
└── converters/         # Logic Modules
    ├── __init__.py
    ├── base_converter.py       # Ethics & Cleaning Logic (Parent Class)
    ├── qa_converter.py         # Multi-choice Logic
    ├── dialogue_converter.py   # General Dialogue Logic
    ├── tcm_converter.py        # TCM Logic
    ├── translation_converter.py# Parallel Corpus Logic
    └── qwen_converter.py       # LLM Translation Logic
```

---

## 2. Data Quality and Ethics Protocol

To ensure strict adherence to data quality standards and ethical guidelines, we embedded a comprehensive filtering protocol within the conversion pipeline (see `converters/base_converter.py`).

1.  **Privacy (PII Scrubbing)** 
2.  **Noise Reduction** 
3.  **Structural Integrity** 

---

## 3. Setup & API Key Configuration

### Installation
1.  Ensure you have Python 3.8+ installed.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### API Configuration (Only for `qwen-llm` mode)
If you use the LLM translation mode, you need an API key.

1.  **Get API Key:** [Aliyun DashScope Console](https://cn.aliyun.com/)
2.  **Configure:**
    * Rename `.env.example` to `.env`.
    * Edit the file and fill in your key:
        ```text
        DASHSCOPE_API_KEY=${OPENAI_API_KEY}
        ```

---

## 4. Usage & Data Cleaning Examples

The system classifies data cleaning into three specific categories based on the data structure and task type. Run the script from the root `sft/` folder.

**Note:** By default, the scripts are set to process the **first 10 samples** for testing.

### Category 1: General Chinese Dialogue Data
*General purpose multi-turn conversation data.*

#### 1. Common Multi-Dialogue
*Format: JSON file containing list of multi-turn conversations.*
```bash
python sft\main.py sft\raw_data_samples\common\LCCC-base_train_sample.json common-multi-dialogue
```

### Category 2: Chinese Medical Dialogue & 4-Option QA
*Includes standard dialogue, multiple-choice questions, and complex TCM consultations.*

#### 2. 4-options question (4-Option Choice)
*Format: Standard exam questions with A/B/C/D options (TXT/JSON).*
```bash
python sft\main.py sft\raw_data_samples\medical-dialogue_4options\MedQA-train.jsonl multiquestion
python sft\main.py sft\raw_data_samples\medical-dialogue_4options\MedQA-tw-train-2zh.jsonl multiquestion
```

#### 3. Inquiry (Simple Q&A dictionary)
*Format: Simple User-Assistant JSON pairs.*
```bash
python sft\main.py sft\raw_data_samples\medical-dialogue_4options\Huatuo-26M-Lite.jsonl inquiry 
python sft\main.py sft\raw_data_samples\medical-dialogue_4options\textgen_train_zh_0.jsonl inquiry 
```

#### 4. inquiry-2(Array Format)
Format: Data stored as arrays `["id:..", "user:..", "assistant:.."]`.
```bash
python sft\main.py sft\raw_data_samples\medical-dialogue_4options\HautuoGPT_sft_data_v1_sample.jsonl inquiry2
```

#### 5. Dialogue in format of txt
Format: data store in text files, without apparent format
```bash
python sft\main.py sft\raw_data_samples\medical-dialogue_4options\2011.txt raw-dialogue
```


#### 6. TCM Inquiry(Traditional Chinese Medicine)
*Format: Complex structured data containing Pathogenesis, Syndrome, and reasoning.*
```Bash
python sft\main.py sft\raw_data_samples\medical-dialogue_4options\Train_TCM_Data_v1.json TCM-inquiry
```

### Category 3: English-Chinese Translation
*Parallel corpus processing and LLM-based translation.*

#### 7. Translation Combination with 2 txt files
*Format: Directory input containing two files: en.txt (source) and zh.txt (target). The script aligns them line-by-line.*
```Bash
# Argument must be the FOLDER containing the two txt files
python sft\main.py sft\raw_data_samples\parallel_translation\nejm.train translation
```

#### 8. Translation-2 (Abstract JSON)
*Format: JSON list of objects containing English and Chinese abstract fields.*
```Bash
python sft\main.py sft\raw_data_samples\parallel_translation\wmtbio22_train_data.json translation2
```

#### 9. Qwen LLM Translation
*Format: Raw text files (one sentence/paragraph per line) to be translated by LLM API.*
```Bash
python sft\main.py sft\raw_data_samples\parallel_translation\zh-all_books.txt qwen-llm
```
---


## 5. Output Format

All processed files are saved to the `output_sft_sample/` directory. The output follows the standard OpenAI-compatible SFT format (JSONL):

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Cleaned instruction or question..."
    },
    {
      "role": "assistant",
      "content": "Cleaned answer or response..."
    }
  ]
}
```