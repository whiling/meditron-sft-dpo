from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='meditron3_8B_ppl',
        path='${PROJECT_ROOT}/datasets/polyglot1/models/dpo_medical_v2/checkpoint-100',
        tokenizer_path='${PROJECT_ROOT}/datasets/polyglot1/models/dpo_medical_v2/checkpoint-100',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left', use_fast=False, trust_remote_code=True),
        max_seq_len=2048,
        max_out_len=1,
        batch_size=16,
        model_kwargs=dict(
            torch_dtype='float16',
            device_map='auto',
            trust_remote_code=True,
        ),
        run_cfg=dict(
            num_gpus=4,
            num_procs=4,
        ),
    )
]
