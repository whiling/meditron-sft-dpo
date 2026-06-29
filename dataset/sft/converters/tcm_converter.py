import json
from .base_converter import BaseConverter

class TCMConverter(BaseConverter):
    """
    Converter for structured TCM (Traditional Chinese Medicine) inquiry data.
    """
    def __init__(self, input_path, output_path):
        super().__init__(input_path, output_path)

    def process(self):
        converted_data = []
        
        # Ensure data is iterable
        iterable_data = self.raw_data if isinstance(self.raw_data, list) else [json.loads(line) for line in self.raw_data if line.strip()]

        for item in iterable_data:
            if len(converted_data) >= self.max_samples:
                break

            try:
                clinical_data = item.get('Clinical Data', '')
                clinical_info = item.get('Clinical Information', '')
                
                # TCM Pathogenesis
                tcm_path_reasoning = item.get('TCM Pathogenesis reasoning', '')
                tcm_path = item.get('TCM Pathogenesis', '')
                ans_tcm_path = item.get('Answers of TCM Pathogenesis', '')
                opts_tcm_path = item.get('Options of TCM Pathogenesis', '')
                
                # TCM Syndrome
                tcm_syn_reasoning = item.get('TCM Syndrome reasoning', '')
                tcm_syn = item.get('TCM Syndrome', '')
                ans_tcm_syn = item.get('Answers of TCM Syndrome', '')
                opts_tcm_syn = item.get('Options of TCM Syndrome', '')
                
                # Summary
                summary = item.get('Explanatory Summary', '')
                diff = item.get('Syndrome Differentiation', '')

                user = (
                    f"请根据以下临床资料进行中医辨证。\n\n"
                    f"【病历信息】\n{clinical_data}\n\n"
                    f"【四诊信息摘要】\n{clinical_info}\n\n"
                    f"请回答：\n"
                    f"1. 核心病机是什么？（从下列选项中选择，可多选）\n{opts_tcm_path}\n"
                    f"2. 中医证型是什么？（从下列选项中选择，可多选）\n{opts_tcm_syn}"
                )
                
                assistant = (
                    f"【辨证分析】\n"
                    f"1. 核心病机：{tcm_path} \n推理：{tcm_path_reasoning} \n（对应选项：{ans_tcm_path}）\n"
                    f"2. 中医证型：{tcm_syn}\n推理：{tcm_syn_reasoning}\n（对应选项：{ans_tcm_syn}）\n\n"
                    f"【辨证结论】\n{diff}\n\n"
                    f"【分析总结】\n{summary}"
                )

                msg = self.format_message(user, assistant)
                if msg: converted_data.append(msg)
                
            except Exception as e:
                print(f"[WARN] TCM parsing error: {e}")

        self.save_data(converted_data)