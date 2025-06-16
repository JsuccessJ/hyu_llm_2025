import yaml
from langchain.prompts import PromptTemplate
from transformers import pipeline
def load_prompts_from_yaml(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 'basic_prompt' 항목만 로드
    if "basic_prompt" not in data:
        raise ValueError("YAML 파일에 'basic_prompt' 항목이 없습니다.")

    val = data["basic_prompt"]

    if isinstance(val, dict):
        prompt_str = val.get("template")
    elif isinstance(val, str):
        prompt_str = val
    else:
        raise ValueError(f"'basic_prompt' 형식이 올바르지 않습니다: {type(val)}")

    if not isinstance(prompt_str, str):
        raise ValueError(f"'basic_prompt' 템플릿이 문자열이 아닙니다: {type(prompt_str)}")

    # PromptTemplate 하나만 반환해도 되지만 dict 형태 유지
    return {
        "basic_prompt": PromptTemplate(
            template=prompt_str,
            input_variables=["context", "input"]
        )
    }

# # ✅ 프롬프트 템플릿 로더
# def load_prompts_from_yaml(filepath: str) -> dict:
#     with open(filepath, "r", encoding="utf-8") as f:
#         data = yaml.safe_load(f)

#     prompt_templates = {}
#     # 내부 분류 키 → 필요한 변수
#     input_vars_map = {
#         "perfume_summary": ["perfume_name"],
#         "recommendation_scenario": ["scenario_description"],
#         "note_description": ["perfume_name"],
#         "brand_specific": ["brand_name", "ingredient"],
#         "season_time": ["season_or_time"],
#         "mood_atmosphere": ["mood_description"],
#         "basic_prompt": ["context", "query"],
#     }

#     for key, val in data.items():
#         if isinstance(val, dict):
#             prompt_str = val.get("template")
#         elif isinstance(val, str):
#             prompt_str = val
#         else:
#             raise ValueError(f"Prompt '{key}' 형식이 올바르지 않습니다: {type(val)}")

#         if not isinstance(prompt_str, str):
#             raise ValueError(f"Prompt '{key}'의 template이 문자열이 아닙니다: {type(prompt_str)}")

#         input_vars = input_vars_map.get(key, ["query","context"])
#         prompt_templates[key] = PromptTemplate(
#             template=prompt_str,
#             input_variables=input_vars,
#         )

#     return prompt_templates


# # ✅ Zero-shot 분류 파이프라인 설정
# classifier = pipeline(
#     "zero-shot-classification",
#     model="joeddav/xlm-roberta-large-xnli"
# )

# # ✅ 자연어 라벨 → 내부 분류 키
# label_map = {
#     "향수 요약": "perfume_summary",
#     "추천 상황": "recommendation_scenario",
#     "노트 설명": "note_description",
#     "브랜드 관련": "brand_specific",
#     "계절 또는 시간대": "season_time",
#     "기분이나 분위기": "mood_atmosphere",
#     "기타 일반 질문": "basic_prompt",
# }
# labels_ko = list(label_map.keys())

# # ✅ 제로샷 분류 함수
# def classify_query(query: str) -> str:
#     result = classifier(
#         query,
#         candidate_labels=labels_ko,
#         hypothesis_template="이 문장은 {}에 대한 질문이다."
#     )
#     top_label_ko = result["labels"][0]
#     return label_map.get(top_label_ko, "basic_prompt")


# ✅ 안전한 프롬프트 포매팅 유틸
def generate_prompt(prompts: dict, prompt_type: str, **kwargs) -> str:
    prompt_template = prompts.get(prompt_type)
    if prompt_template:
        safe_kwargs = {k: kwargs.get(k, "") for k in prompt_template.input_variables}
        return prompt_template.format(**safe_kwargs)
    return ""
