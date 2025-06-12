import yaml
from langchain.prompts import PromptTemplate

def load_prompts_from_yaml(yaml_path: str) -> dict:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    prompts = {}
    for name, value in data.items():
        prompts[name] = PromptTemplate(
            input_variables=value["input_variables"],
            template=value["template"]
        )
    return prompts
