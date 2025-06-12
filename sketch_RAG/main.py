import os
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from huggingface_hub import login
from project_llm_scentbot.sketch_RAG.prompt_loader import *

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
login(token=hf_token)
    
# LLaMA 3 모델 & 토크나이저 로딩
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

# 텍스트 생성 파이프라인 구성
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
)

# LangChain용 래퍼
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# FAISS 벡터 DB 로딩 ---
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

print("FAISS 벡터 DB 로딩 중...")
vector_store = FAISS.load_local("./perfume_faiss_index", embed_model,allow_dangerous_deserialization=True)  # 경로 맞게 수정
print("벡터 DB 로딩 완료")

# Retriever 생성
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 프롬프트 로딩
prompts = load_prompts_from_yaml("./prompts.yaml")
selected_prompt = prompts["basic_prompt"]  

# RAG RetrievalQA 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": selected_prompt}
)
# 사용자 질의 함수
def ask(query: str):
    result = qa_chain({"query": query})
    print("💬 질문:", query)
    print("🧠 답변:", result['result'])
    print("\n📚 참조 문서:")
    for doc in result["source_documents"]:
        brand = doc.metadata.get("brand_name", "알 수 없음")
        rating = doc.metadata.get("rating_value", "?")
        print(f"- 브랜드: {brand} | 평점: {rating}")

if __name__ == "__main__":
    print("=== LLaMA 3 + RAG QA 시스템 ===")
    while True:
        query = input("질문 입력 (종료는 exit): ")
        if query.lower() in ("exit", "quit"):
            print("프로그램 종료")
            break
        ask(query)
