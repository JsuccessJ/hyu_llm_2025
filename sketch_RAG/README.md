# sketch_RAG
스케치 공간.

### 📁 Folder Structure
```
sketch_RAG/
├── main.py                             # "실행" 파일
├── dataprocessing.py                   # [1]원본 데이터 전처리
├── perfumes_rag.jsonl                  # [2]원본 데이터 전처리 결과물
├── vectorDB_implementation.py          # [3]vectore DB 구현 
├── perfume_faiss_index/                # vectore DB - 총 청크 수: 7085
├── prompts.yaml    👈 프롬프트
├── prompt_loader.py                    # [4] 프롬프트 설계
└── .env                                # Llama3.2-3B-inst token key(비공개)
```

### 📝실험 기록
- 실험결과 기록 : https://carnelian-chip-7b4.notion.site/project_LLM-2102b096e16c80df8c5ecaa5e2adc71c?source=copy_link