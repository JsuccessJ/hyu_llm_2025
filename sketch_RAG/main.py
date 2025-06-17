import os
import yaml
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from prompt_loader import load_prompts_from_yaml
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import inspect
"""
pip install --upgrade transformers torch langchain langchain-core
LangChain 버전 확인 # pip show langchain
Name: langchain
Version: 0.3.25
Summary: Building applications with LLMs through composability
Home-page: 
Author: 
Author-email: 
License: MIT
Location: /home/dibaeck/sketch/anaconda3/envs/dibk311/lib/python3.11/site-packages
Requires: langchain-core, langchain-text-splitters, langsmith, pydantic, PyYAML, requests, SQLAlchemy
Required-by: langchain-community
"""
########## utils
class InputRequiredError(Exception):
    pass

# 설정 불러오기
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# 사용자 입력 또는 기본값 반환
def get_input_or_default(prompt_text, default_value=None):
    user_input = input(f"{prompt_text}{f' (기본값: {default_value})' if default_value is not None else ''}: ").strip()
    if not user_input and default_value is None:
        raise InputRequiredError(f"'{prompt_text}' 입력이 필요합니다.")
    return user_input if user_input else default_value

#############################
# 모델 및 벡터 DB 로딩
def setup_models_and_vectorstore():
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

    EMBED_MODEL_NAME = "BAAI/bge-m3"
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
    # search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.8} 이걸로 실험해보기 : mmr는 LLM + RAG에서 context redundancy를 줄여 성능 향상에 도움된다고 함.
    # similarity는 단순 유사도 기반이라 다양성 부족할 수 있음.

    return llm, retriever

# QA 체인 생성
"""
[질문 → 검색 → 문맥 생성 → 프롬프트 구성 → 답변 생성] Chain을 생성.
qa_chain.invoke({"input": query})를 하면, retriever가 벡터 검색해서 관련 문서 가져와서 prompt템플릿에 넣음.             # query가 아니라 input으로 변수 정의해야함....
LLM에 넣어 답변 생성하고 생성된 답변과 참조 문서 리턴.

result, source_documents

"""
def create_qa_chain_with_prompt(prompt: PromptTemplate, llm, retriever):
    # 문서들을 하나로 합쳐서 LLM에 넘기는 stuff chain 생성
    stuff_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_variable_name="context",
    )
    # retriever + stuff_chain 을 합친 retrieval QA 체인 생성
    qa_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=stuff_chain,
    )
    return qa_chain

# 질문 및 답변 함수
def ask(qa_chain, query: str):
    result = qa_chain.invoke({"input": query})              # result.keys() :: ['input', 'context', 'answer']

    print("💬 질문:", query)
    print("🧠 답변:", result['answer'])
    
    context_docs = result["context"]

    if isinstance(context_docs, list):
        print("\n📚 참조 문서 리스트:")
        for i, doc in enumerate(context_docs):
            print(f"\n--- 문서 {i+1} ---")
            print("메타데이터:", doc.metadata)
            print("본문 일부:", doc.page_content[:500])  # 앞 500자만 출력
    else:
        # 그냥 문자열이면 일부만 출력
        print("\n📚 참조 문서 내용:")
        print(context_docs[:1000])

############################################# 메인 실행
if __name__ == "__main__":
    prompts = load_prompts_from_yaml("./prompts.yaml")
    llm, retriever = setup_models_and_vectorstore()

    # basic_prompt 템플릿 하나만 사용
    basic_prompt_template = prompts.get("basic_prompt")
    if set(basic_prompt_template.input_variables) != {"context", "input"}:
        raise ValueError("프롬프트 템플릿은 'context'와 'query'를 포함해야 합니다.")
    
    # 체인 생성
    qa_chain = create_qa_chain_with_prompt(
        prompt=basic_prompt_template,
        llm=llm,
        retriever=retriever
    )
    EXIT_COMMANDS = {"exit", "quit", "q", "종료", "그만"}
    try:
        while True:
            query = input('질문 입력 (종료는 "그만" 입력 or 입력하지 않기): ').strip()
            if query.lower() in EXIT_COMMANDS:
                print("프로그램 종료")
                break
            if not query:
                print("입력이 없습니다. 종료합니다.")
                break

            ask(qa_chain, query)

    except KeyboardInterrupt:
        print("\n사용자에 의해 프로그램이 종료되었습니다.")
        
