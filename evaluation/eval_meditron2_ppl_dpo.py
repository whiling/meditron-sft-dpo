from mmengine.config import read_base

with read_base():
    from .datasets.cmmlu.cmmlu_ppl import cmmlu_datasets
    from .models.meditron3_8B_ppl_dpo import models

medical_subjects = [
    'cmmlu-anatomy',
    'cmmlu-clinical_knowledge',
    'cmmlu-college_medicine',
    'cmmlu-genetics',
    'cmmlu-professional_medicine',

]

datasets = [d for d in cmmlu_datasets if d['abbr'] in medical_subjects]
