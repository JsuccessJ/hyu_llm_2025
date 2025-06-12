import json
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# 1. 임베딩 모델 정의 (SBERT)
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. JSONL 불러오기
jsonl_path = "./perfumes_docs.jsonl"
docs = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        text = item.pop("text_chunk")
        metadata = item  # 나머지는 메타데이터
        docs.append(Document(page_content=text, metadata=metadata))

# 3. FAISS 벡터 DB 생성
vector_store = FAISS.from_documents(docs, embedding=embed_model)

# 4. 저장 (로컬 디스크에)
vector_store.save_local("perfume_faiss_index")

print("✅ 벡터 DB 저장 완료 (디렉토리: perfume_faiss_index)")
