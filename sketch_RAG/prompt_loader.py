import yaml
from langchain.prompts import PromptTemplate

def load_prompts_from_yaml(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    prompt_templates = {}
    # 프롬프트별 필요한 변수 맵 (예시)
    input_vars_map = {
        "perfume_summary": ["perfume_name"],
        "recommendation_scenario": ["scenario_description"],
        "note_description": ["perfume_name"],
        "brand_specific": ["brand_name", "ingredient"],
        "season_time": ["season_or_time"],
        "mood_atmosphere": ["mood_description"],
        "basic_prompt": ["query"],  # 기본 프롬프트는 그냥 query만
    }

    for key, val in data.items():
        prompt_str = val.get("template")
        if not isinstance(prompt_str, str):
            raise ValueError(f"Prompt '{key}'의 template이 문자열이 아닙니다: {type(prompt_str)}")

        input_vars = input_vars_map.get(key, ["query"])  # 기본은 query 변수만

        prompt_templates[key] = PromptTemplate(
            template=prompt_str,
            input_variables=input_vars,
        )
    return prompt_templates


def classify_query(query: str) -> str:
    query_lower = query.lower()
    # 간단한 키워드 기반 분류 예시
    if "kind of scent" in query_lower or "what scent" in query_lower:
        return "perfume_summary"
    if "recommend" in query_lower or "suggest" in query_lower:
        return "recommendation_scenario"
    if "note" in query_lower:
        return "note_description"
    if "brand" in query_lower or "parfums de marly" in query_lower:
        return "brand_specific"
    if "season" in query_lower or "spring" in query_lower or "winter" in query_lower:
        return "season_time"
    if "mood" in query_lower or "atmosphere" in query_lower or "sweet" in query_lower or "warm" in query_lower:
        return "mood_atmosphere"
    return "basic_prompt"

def generate_prompt(prompts: dict, prompt_type: str, **kwargs) -> str:
    prompt_template = prompts.get(prompt_type)
    if prompt_template:
        # 일부 kwargs가 없어도 무시하고 기본값으로 빈 문자열 넣을 수도 있음
        safe_kwargs = {k: kwargs.get(k, "") for k in prompt_template.input_variables}
        return prompt_template.format(**safe_kwargs)
    return ""