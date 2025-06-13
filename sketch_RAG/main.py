import os
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from huggingface_hub import login
from prompt_loader import *

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
login(token=hf_token)

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
print("모델 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
    use_auth_token=hf_token,
)
print("모델 로딩 완료")

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print("임베딩 모델 로딩 중...")
embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
print("임베딩 모델 로딩 완료")

print("FAISS 벡터 DB 로딩 중...")
vector_store = FAISS.load_local(
    "./perfume_faiss_index",
    embed_model,
    allow_dangerous_deserialization=True
)
print("벡터 DB 로딩 완료")

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 준비
####################################################################### 프롬프트 파일 불러오기
prompts = load_prompts_from_yaml("./prompts.yaml")

def create_qa_chain_with_prompt(prompt: PromptTemplate):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
def ask(qa_chain, query: str):
    result = qa_chain({"query": query})
    print("💬 질문:", query)
    print("🧠 답변:", result['result'])
    print("\n📚 참조 문서:")
    if result["source_documents"]:
        for doc in result["source_documents"]:
            brand = doc.metadata.get("brand_name", "알 수 없음")
            rating = doc.metadata.get("rating_value", "?")
            print(f"- 브랜드: {brand} | 평점: {rating}")
    else:
        print("- 참조 문서가 없습니다.")

if __name__ == "__main__":
    prompts = load_prompts_from_yaml("./prompts.yaml")

    while True:
        query = input("질문 입력 (종료는 exit): ").strip()
        if query.lower() in ("exit", "quit"):
            print("프로그램 종료")
            break
        if not query:
            continue

        prompt_type = classify_query(query)
        kwargs = {}

        if prompt_type in ("perfume_summary", "note_description"):
            kwargs["perfume_name"] = "Althaïr"
        elif prompt_type == "recommendation_scenario":
            kwargs["scenario_description"] = query
        elif prompt_type == "brand_specific":
            kwargs["brand_name"] = "Parfums de Marly"
            kwargs["ingredient"] = "vanilla"
        elif prompt_type == "season_time":
            kwargs["season_or_time"] = "spring"
        elif prompt_type == "mood_atmosphere":
            kwargs["mood_description"] = "sweet and warm"

        prompt_template = prompts.get(prompt_type)
        if prompt_template is None:
            prompt_template = prompts.get("basic_prompt")
            if prompt_template is None:
                print("기본 프롬프트가 없습니다. 종료합니다.")
                break

        # 변수 일부만 미리 채운 PromptTemplate 객체 생성
        prompt_with_vars = prompt_template.partial(**kwargs)

        qa_chain = create_qa_chain_with_prompt(prompt_with_vars)
        ask(qa_chain, query)