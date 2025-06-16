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
        
        # 디버그 정보 저장용
        self.last_debug_info = {}
        
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
        
        # 빈 응답 처리
        if not response or len(response.strip()) < 10:
            return "죄송합니다. 추천 결과를 생성하는 중 문제가 발생했습니다. 다시 시도해주세요."
        
        return response.strip()
    
    def ask(self, user_query: str) -> str:
        """사용자 질문에 대한 향수 추천 응답 생성"""
        try:
            print("\n" + "="*80)
            print(f"🔍 새로운 질문 처리 시작: '{user_query}'")
            print("="*80)
            
            # Graph RAG 방식으로 검색
            print("📊 1단계: Graph RAG 검색 실행 중...")
            graph_rag_response = self.retrieval.search_graph_rag(user_query)
            
            # 디버그 정보 저장
            self.last_debug_info = {
                'user_query': user_query,
                'graph_rag_response': graph_rag_response,
                'timestamp': str(torch.cuda.current_stream()) if torch.cuda.is_available() else 'CPU'
            }
            
            print("✅ Graph RAG 검색 완료!")
            print("\n" + "-"*60)
            print("📋 LLM에게 전달될 최종 컨텍스트:")
            print("-"*60)
            print(graph_rag_response)
            print("-"*60)
            
            # 3. 개선된 프롬프트 작성
            print("\n🤖 2단계: LLM 프롬프트 생성 중...")
            prompt = self._create_prompt(user_query, graph_rag_response)
            
            print("✅ 프롬프트 생성 완료!")
            print("\n" + "-"*60)
            print("📝 LLM에게 전달될 전체 프롬프트:")
            print("-"*60)
            print(prompt)
            print("-"*60)
            
            # 4. LLM 응답 생성
            print("\n🧠 3단계: LLM 응답 생성 중...")
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
            
            # 추가 디버그 정보 저장
            self.last_debug_info.update({
                'final_prompt': prompt,
                'full_llm_response': full_response,
                'clean_response': clean_response
            })
            
            print("✅ LLM 응답 생성 완료!")
            print("\n" + "-"*60)
            print("🤖 LLM 원본 응답 (후처리 전):")
            print("-"*60)
            print(full_response)
            print("-"*60)
            
            print("\n" + "-"*60)
            print("✨ 최종 정제된 응답:")
            print("-"*60)
            print(clean_response)
            print("-"*60)
            
            print("\n" + "="*80)
            print("🎉 질문 처리 완료!")
            print("="*80 + "\n")
            
            return clean_response
            
        except Exception as e:
            error_msg = f"❌ 응답 생성 중 오류: {e}"
            print(f"\n{error_msg}")
            print("="*80 + "\n")
            return "죄송합니다. 향수 추천 중 문제가 발생했습니다. 다시 시도해 주세요."
    
    def ask_with_debug(self, user_query: str) -> Dict:
        """디버그 정보와 함께 향수 추천 응답 생성 (Streamlit용)"""
        try:
            # Graph RAG 방식으로 검색
            graph_rag_response = self.retrieval.search_graph_rag(user_query)
            
            # 3. 개선된 프롬프트 작성
            prompt = self._create_prompt(user_query, graph_rag_response)
            
            # 4. LLM 응답 생성
            outputs = self.pipeline(
                prompt,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.0,
                # top_p=0.9,
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )
            
            # 5. 응답 후처리
            full_response = outputs[0]["generated_text"]
            clean_response = self._extract_generated_response(full_response, prompt)
            
            # 디버그 정보와 함께 반환
            return {
                'response': clean_response,
                'debug_info': {
                    'user_query': user_query,
                    'graph_rag_context': graph_rag_response,
                    'final_prompt': prompt,
                    'full_llm_response': full_response
                }
            }
            
        except Exception as e:
            error_msg = f"❌ 응답 생성 중 오류: {e}"
            print(error_msg)
            return {
                'response': "죄송합니다. 향수 추천 중 문제가 발생했습니다. 다시 시도해 주세요.",
                'debug_info': {
                    'user_query': user_query,
                    'error': str(e)
                }
            }
    
    def get_last_debug_info(self) -> Dict:
        """마지막 실행의 디버그 정보 반환"""
        return self.last_debug_info
    
    def _create_prompt(self, user_query: str, context: str) -> str:
        """향수 추천을 위한 최적화된 프롬프트 생성"""
        
        # 컨텍스트가 없는 경우 처리
        if not context or context == "검색 결과가 없습니다.":
            return "죄송합니다. 요청하신 조건에 맞는 향수를 찾지 못했습니다."
        
        # 컨텍스트에서 실제 향수 정보만 추출 (디버그 정보 제거)
        lines = context.split('\n')
        perfume_info = []
        in_context_section = False
        
        for line in lines:
            # 다양한 패턴으로 컨텍스트 섹션 시작 감지
            if "📋 LLM이 받는 실제 context" in line:
                in_context_section = True
                continue
            elif "🔗 발견된 향료 체인 패턴" in line or "=" in line:
                break
            elif in_context_section and line.strip() and line.startswith(('1.', '2.', '3.', '4.', '5.')):
                perfume_info.append(line.strip())
        
        clean_context = '\n'.join(perfume_info)
        
        # 컨텍스트가 여전히 비어있으면 전체 컨텍스트에서 향수 정보 직접 추출
        if not clean_context.strip():
            print("⚠️  컨텍스트 추출 실패 - 전체 컨텍스트에서 직접 추출 시도")
            for line in lines:
                if ' for men' in line or ' for women' in line:
                    if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                        # 향수 정보를 간단하게 정리
                        perfume_part = line.strip()
                        # "(브랜드:" 이전 부분만 추출
                        if ' (브랜드:' in perfume_part:
                            perfume_clean = perfume_part.split(' (브랜드:')[0]
                            perfume_info.append(perfume_clean)
                        else:
                            perfume_info.append(perfume_part)
                            
                if len(perfume_info) >= 5:
                    break
            
            clean_context = '\n'.join(perfume_info)
        
        print(f"🔍 추출된 향수 정보 ({len(perfume_info)}개):")
        print(clean_context)
        print("-" * 40)
        
        # 극도로 간소화된 프롬프트
        return f"""질문: {user_query}

# 향수 목록:
{clean_context}

#위 향수 중에서 3개를 선택해서 아래 형식으로 답하세요:
1. [브랜드명] [향수명]
2. [브랜드명] [향수명]  
3. [브랜드명] [향수명]

# 반드시 위 목록에 있는 향수만 선택하세요.

# 답변:
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