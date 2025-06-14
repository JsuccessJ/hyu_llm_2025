# Neo4j Graph + Llama 3.1을 이용한 향수 추천 RAG 시스템

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import transformers
import torch
import gc
import json
from typing import List, Dict, Tuple, Optional
from retrieval import Neo4jRetrieval
import re
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# Neo4j 연결 정보
URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "password"

load_dotenv()

class GraphRAG:
    """Graph RAG 시스템 메인 클래스"""
    
    def __init__(self, neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None, model_id: str = None):
        """GraphRAG 시스템 초기화"""
        import os
        # 환경변수에서 값 읽기 (파라미터가 없으면)
        neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI')
        neo4j_user = neo4j_user or os.getenv('NEO4J_USER')
        neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD')
        model_id = model_id or os.getenv('MODEL_ID')
        hf_token = os.getenv('HF_TOKEN')
        self.retrieval = Neo4jRetrieval(neo4j_uri, neo4j_user, neo4j_password)
        self.model_id = model_id
        
        base_model_path = model_id  # .env에서 읽은 base_model 경로
        adapter_path = "/home/shcho95/yjllm/llama3_8b/weight/perfume_llama3_8B_v0"  # 어댑터 경로

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, token=hf_token)

        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("✅ GraphRAG 시스템이 초기화되었습니다.")
    
    def _extract_generated_response(self, full_output: str, prompt: str) -> str:
        """생성된 텍스트에서 실제 답변만 추출"""
        # 프롬프트 부분을 제거
        if prompt in full_output:
            response = full_output.replace(prompt, "").strip()
        else:
            response = full_output.strip()
        
        # 불필요한 태그나 반복된 내용 제거
        response = re.sub(r'<[^>]+>', '', response)  # HTML 태그 제거
        response = re.sub(r'\n+', '\n', response)  # 연속된 줄바꿈 정리
        
        return response.strip()
    
    def ask(self, user_query: str) -> str:
        """사용자 질문에 대한 향수 추천 응답 생성"""
        try:
            # Graph RAG 방식으로 검색
            graph_rag_response = self.retrieval.search_graph_rag(user_query)
            
            # 3. 개선된 프롬프트 작성
            prompt = self._create_prompt(user_query, graph_rag_response)
            
            # 4. LLM 응답 생성
            outputs = self.pipeline(
                prompt,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )
            
            # 5. 응답 후처리
            full_response = outputs[0]["generated_text"]
            clean_response = self._extract_generated_response(full_response, prompt)
            
            print("===== LLM에 전달되는 최종 컨텍스트 =====")
            print(graph_rag_response)
            
            return clean_response
            
        except Exception as e:
            print(f"❌ 응답 생성 중 오류: {e}")
            return "죄송합니다. 향수 추천 중 문제가 발생했습니다. 다시 시도해 주세요."
    
    def _create_prompt(self, user_query: str, context: str) -> str:
        """향수 추천을 위한 최적화된 프롬프트 생성"""
        
        # 컨텍스트가 없는 경우 처리
        if not context or context == "검색 결과가 없습니다.":
            return f"""당신은 친근한 향수 전문가입니다. 
사용자 질문: {user_query}

죄송하지만 요청하신 조건에 정확히 맞는 향수를 찾지 못했습니다. 
다른 키워드나 더 구체적인 조건을 말씀해 주시면 더 좋은 추천을 드릴 수 있습니다.

추천:"""
        
        # 일반적인 경우 프롬프트
        return f"""당신은 전문적이고 친근한 향수 추천 전문가입니다.

        다음 사용자 질문과 Graph RAG 검색 결과를 바탕으로 향수 추천을 한국어로 해주세요:
        답변에 포함할 내용 :
        - 향수의 구체적인 특징(브랜드, 향조, 평점 등) 포함
        - 주어진 Graph RAG 검색 결과를 적극 활용할 것
        - 거짓 정보는 절대 포함하지 않음
        - 검색 결과의 점수와 경로 정보를 참고하여 설명할 것

        사용자 질문: {user_query}

        Graph RAG 검색 결과:
        {context}


        향수 추천:
        """
    
    def cleanup(self):
        """리소스 정리"""
        print("리소스 정리 중...")
        if hasattr(self, 'pipeline') and self.pipeline:
            del self.pipeline
            gc.collect()
            torch.cuda.empty_cache()
        
        if hasattr(self, 'retrieval') and self.retrieval:
            self.retrieval.close()
        
        print("✅ 정리 완료!")