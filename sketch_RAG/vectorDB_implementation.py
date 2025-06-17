import json
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def build_vector_index_from_jsonl(jsonl_path, save_path="./perfume_faiss_index"):
    embed_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"
    )
    documents = []
    
    # jsonl 파일 읽기
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            perfume_id = record.get("perfume_id")
            metadata = record.get("metadata", {})
            chunks = record.get("text_chunks", [])
            
            # 각 청크마다 Document 객체 생성 (메타 포함)
            for idx, chunk_text in enumerate(chunks):
                doc = Document(
                    page_content=f"passage: {chunk_text}",                        # page_content=chunk_text, 이였는데 성능향상TIP으로 다음처럼 수정함
                    metadata={
                        "perfume_id": perfume_id,
                        "chunk_index": idx,
                        **metadata
                    }
                )
                documents.append(doc)
    
    # FAISS 벡터 DB에 임베딩 & 저장
    vector_store = FAISS.from_documents(documents, embed_model)
    
    # 인덱스 저장
    vector_store.save_local(save_path)
    print(f"✅ 벡터 인덱스가 '{save_path}'에 저장됨.")
    print(f"총 청크 수: {len(documents)}")

if __name__ == "__main__":
    jsonl_file = "./perfumes_rag.jsonl"
    build_vector_index_from_jsonl(jsonl_file)

"""
✅ 벡터 인덱스가 './perfume_faiss_index'에 저장됨.
총 청크 수: 7085
"""