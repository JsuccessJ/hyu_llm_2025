# LangChain 기반 Graph-RAG 향수 추천 시스템 (Pydantic 호환성 수정)
# BaseTool 상속 문제를 해결하기 위해 일반 클래스로 재구성

import os
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter
import json

# LangChain 관련 임포트
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain.memory import ConversationBufferMemory

# 기존 라이브러리
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from peft import PeftModel
from neo4j import GraphDatabase
from konlpy.tag import Okt
from dotenv import load_dotenv

load_dotenv()


class PerfumeGraphSearcher:
    """
    🔍 향수 그래프 검색 엔진
    
    기존 Neo4jRetrieval 클래스의 핵심 기능을 독립 클래스로 구현
    - 키워드 추출 및 Seed Node 선택
    - 그래프 확장 및 Accord 체인 탐색  
    - 향수 점수 계산 및 랭킹
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, database: str = None):
        # Neo4j 연결
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.database = database or "neo4j"  # 기본값은 neo4j
        
        # 라벨 설정 (기본 라벨 사용)
        self.labels = {
            'Perfume': 'Perfume',
            'Brand': 'Brand',
            'Target': 'Target',
            'Accord': 'Accord'
        }
        
        # 임베딩 모델 초기화 (BGE-M3)
        self.model_name = "BAAI/bge-m3"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # 임베딩 캐시 및 형태소 분석기
        self.embedding_cache = {}
        self.okt = Okt()
        
        print("✅ PerfumeGraphSearcher 초기화 완료")
    
    def search(self, query: str) -> str:
        """
        🎯 메인 검색 실행 함수
        
        1. 키워드 추출 → 2. Seed Node 선택 → 3. 그래프 확장 → 4. 점수 계산
        """
        try:
            print(f"\n🔍 Graph-RAG 검색 시작: '{query}'")
            
            # 1단계: 키워드 추출
            keywords = self._extract_keywords(query)
            print(f"📝 추출된 키워드: {keywords}")
            
            # 2단계: Seed Node 선택
            seed_nodes = self._get_seed_nodes(keywords)
            print(f"🌱 Seed Nodes: {seed_nodes}")
            
            if not any(seed_nodes.values()):
                return "관련된 향수 정보를 찾을 수 없습니다."
            
            # 3단계: 그래프 확장
            expanded_graph = self._expand_graph(seed_nodes)
            print(f"🌐 그래프 확장: {len(expanded_graph['paths'])}개 경로, {len(expanded_graph['perfumes'])}개 향수")
            
            # 4단계: 향수 점수 계산
            perfume_rankings = self._calculate_perfume_scores(expanded_graph, seed_nodes)
            
            # 5단계: 결과 포맷팅
            result = self._format_results(query, seed_nodes, expanded_graph, perfume_rankings)
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Graph-RAG 검색 중 오류: {str(e)}"
            print(error_msg)
            return "죄송합니다. 향수 검색 중 문제가 발생했습니다."

    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출 + 브랜드/향조명 매핑 (개선된 버전)"""
        stopwords = ['이', '그', '저', '것', '수', '등', '및', '또는', '그리고', '하지만', 
                    '그래서','때문에', '위해', '대해', '관련', '따라서', '그러나', '또한', 
                    '추천','하다','해주다','나', '용이','어울리다','알다', '뭐', '좋다', 
                    '향수','계열','향', '있다','찾다', '게', '어떻다']
        
        # 브랜드명 한글↔영문 매핑 (더 많은 변형 포함)
        brand_mapping = {
            # 기본 브랜드명
            '샤넬': 'Chanel', '톰포드': 'Tom Ford', '디올': 'Dior',
            '구찌': 'Gucci', '버버리': 'Burberry', '에르메스': 'Hermes',
            '조말론': 'Jo Malone', '랑콤': 'Lancome', '이브생로랑': 'Yves Saint Laurent',
            '지방시': 'Givenchy', '프라다': 'Prada', '아르마니': 'Armani',
            # 분리된 형태도 매핑
            '톰 포드': 'Tom Ford', 'tom ford': 'Tom Ford', 'tomford': 'Tom Ford',
            '조 말론': 'Jo Malone', 'jo malone': 'Jo Malone', 'jomalone': 'Jo Malone',
            '이브 생로랑': 'Yves Saint Laurent', 'ysl': 'Yves Saint Laurent'
        }
        
        # 향조명 한글↔영문 매핑 (더 많은 변형 포함)
        accord_mapping = {
            '우디': 'woody', '플로럴': 'floral', '시트러스': 'citrus',
            '오리엔탈': 'oriental', '프레시': 'fresh', '스파이시': 'spicy',
            '바닐라': 'vanilla', '머스크': 'musk', '앰버': 'amber', '로즈': 'rose',
            # 영문도 추가
            'woody': 'woody', 'floral': 'floral', 'citrus': 'citrus',
            'oriental': 'oriental', 'fresh': 'fresh', 'spicy': 'spicy'
        }
        
        # 타겟 매핑 추가
        target_mapping = {
            '남자': 'for men', '남성': 'for men', '남': 'for men',
            '여자': 'for women', '여성': 'for women', '여': 'for women',
            '유니섹스': 'unisex', '남녀공용': 'unisex'
        }
        
        # 1단계: 원본 텍스트에서 직접 브랜드명 찾기
        text_lower = text.lower()
        found_brands = []
        for brand_kr, brand_en in brand_mapping.items():
            if brand_kr in text:
                found_brands.extend([brand_kr, brand_en])
        
        # 2단계: 형태소 분석
        tokens = self.okt.pos(text, stem=True)
        keywords = [word for word, pos in tokens 
                   if pos in ['Noun', 'Verb', 'Adjective'] 
                   and word not in stopwords 
                   and len(word) > 1]  # 1글자 단어 제외
        
        # 3단계: 분리된 브랜드명 재조합 시도
        reconstructed_brands = []
        if '톰' in keywords and '포드' in keywords:
            reconstructed_brands.append('톰포드')
        if '조' in keywords and '말론' in keywords:
            reconstructed_brands.append('조말론')
        if '이브' in keywords and ('생로랑' in keywords or '로랑' in keywords):
            reconstructed_brands.append('이브생로랑')
        
        # 4단계: 매핑 적용
        mapped_keywords = []
        
        # 원본 키워드 추가
        mapped_keywords.extend(keywords)
        
        # 찾은 브랜드명 추가
        mapped_keywords.extend(found_brands)
        
        # 재조합된 브랜드명 추가
        mapped_keywords.extend(reconstructed_brands)
        
        # 매핑 적용
        for keyword in keywords + reconstructed_brands:
            if keyword in brand_mapping:
                mapped_keywords.append(brand_mapping[keyword])
            if keyword in accord_mapping:
                mapped_keywords.append(accord_mapping[keyword])
            if keyword in target_mapping:
                mapped_keywords.append(target_mapping[keyword])
        
        # 중복 제거 및 디버깅 정보
        final_keywords = list(set(mapped_keywords))
        print(f"🔍 키워드 추출 과정:")
        print(f"   원본 텍스트: {text}")
        print(f"   형태소 분석: {keywords}")
        print(f"   재조합 브랜드: {reconstructed_brands}")
        print(f"   최종 키워드: {final_keywords}")
        
        return final_keywords

    def _get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성 (캐시 활용)"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
            
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                               truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embedding = embeddings[0].numpy().tolist()
        
        self.embedding_cache[text] = embedding
        return embedding

    def _cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """코사인 유사도 계산"""
        emb1, emb2 = np.array(emb1), np.array(emb2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def _get_seed_nodes(self, keywords: List[str]) -> Dict[str, List[str]]:
        """키워드 기반 Seed Node 선택 (개선된 임계값)"""
        seed_nodes = {'Brand': [], 'Target': [], 'Accord': []}
        
        # 라벨별 다른 임계값 사용
        thresholds = {
            'Brand': 0.7,   # 브랜드는 조금 더 관대하게
            'Target': 0.75, # 타겟은 중간
            'Accord': 0.8   # 향조는 엄격하게
        }
        
        for label in ['Brand', 'Target', 'Accord']:
            threshold = thresholds[label]
            nodes = self._get_all_nodes_embeddings(label)
            
            filtered_seeds = []
            keyword_matches = []  # 디버깅용
            
            for keyword in keywords:
                kw_emb = self._get_embedding(keyword)
                best_match = None
                best_similarity = 0
                
                for node in nodes:
                    similarity = self._cosine_similarity(kw_emb, node['embedding'])
                    if similarity >= threshold and node['name'] not in filtered_seeds:
                        filtered_seeds.append(node['name'])
                        keyword_matches.append(f"{keyword} → {node['name']} ({similarity:.3f})")
                    
                    # 최고 유사도 추적 (디버깅용)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = node['name']
                
                # 임계값을 넘지 못한 경우도 로깅
                if best_match and best_similarity < threshold:
                    keyword_matches.append(f"{keyword} → {best_match} ({best_similarity:.3f}) [임계값 미달]")
            
            seed_nodes[label] = filtered_seeds[:5]  # 최대 5개로 증가
            
            # 디버깅 정보 출력
            print(f"🎯 {label} 매칭 (임계값: {threshold}):")
            for match in keyword_matches:
                print(f"   {match}")
            print(f"   최종 선택: {seed_nodes[label]}")
        
        return seed_nodes

    def _get_all_nodes_embeddings(self, label: str) -> List[Dict]:
        """특정 라벨의 모든 노드 임베딩 조회 (V2 라벨 지원)"""
        actual_label = self.labels.get(label, label)  # 동적 라벨 사용
        with self.driver.session(database=self.database) as session:
            query = f"""
            MATCH (n:{actual_label})
            WHERE n.embedding IS NOT NULL
            RETURN n.name as name, n.embedding as embedding
            """
            return [dict(record) for record in session.run(query)]

    def _expand_graph(self, seed_nodes: Dict[str, List[str]]) -> Dict:
        """그래프 확장 및 Accord 체인 탐색 (기존 로직 유지)"""
        expanded_graph = {
            'paths': [],
            'perfumes': set(),
            'seed_connections': {},
            'accord_chains': []
        }
        
        with self.driver.session(database=self.database) as session:
            # Brand → Perfume → Accord 경로
            for brand_seed in seed_nodes['Brand']:
                query = f"""
                    MATCH (seed_brand:{self.labels['Brand']} {{name: $brand_seed}})
                    MATCH (seed_brand)<-[:HAS_BRAND]-(p:{self.labels['Perfume']})-[ha:HAS_ACCORD]->(a:{self.labels['Accord']})
                    OPTIONAL MATCH (p)-[:FOR_TARGET]->(t:{self.labels['Target']})
                    RETURN 'Brand→Perfume→Accord' as path_type,
                           seed_brand.name as seed_node,
                           p.name as perfume,
                           a.name as end_node,
                           t.name as target,
                           ha.strength as strength,
                           p.rating_value as rating,
                           p.review_count as reviews
                """
                results = session.run(query, brand_seed=brand_seed)
                
                for record in results:
                    expanded_graph['paths'].append({
                        'path_type': record['path_type'],
                        'seed_node': record['seed_node'],
                        'seed_type': 'Brand',
                        'perfume': record['perfume'],
                        'end_node': record['end_node'],
                        'end_type': 'Accord',
                        'target': record['target'],
                        'strength': record['strength'],
                        'rating': record['rating'],
                        'reviews': record['reviews']
                    })
                    expanded_graph['perfumes'].add(record['perfume'])
            
            # Target → Perfume → Accord 경로  
            for target_seed in seed_nodes['Target']:
                query = f"""
                    MATCH (seed_target:{self.labels['Target']} {{name: $target_seed}})
                    MATCH (seed_target)<-[:FOR_TARGET]-(p:{self.labels['Perfume']})-[ha:HAS_ACCORD]->(a:{self.labels['Accord']})
                    OPTIONAL MATCH (p)-[:HAS_BRAND]->(b:{self.labels['Brand']})
                    RETURN 'Target→Perfume→Accord' as path_type,
                           seed_target.name as seed_node,
                           p.name as perfume,
                           a.name as end_node,
                           b.name as brand,
                           ha.strength as strength,
                           p.rating_value as rating,
                           p.review_count as reviews
                """
                results = session.run(query, target_seed=target_seed)
                
                for record in results:
                    expanded_graph['paths'].append({
                        'path_type': record['path_type'],
                        'seed_node': record['seed_node'],
                        'seed_type': 'Target',
                        'perfume': record['perfume'],
                        'end_node': record['end_node'],
                        'end_type': 'Accord',
                        'brand': record['brand'],
                        'strength': record['strength'],
                        'rating': record['rating'],
                        'reviews': record['reviews']
                    })
                    expanded_graph['perfumes'].add(record['perfume'])
            
            # Accord → Perfume → Brand 경로
            for accord_seed in seed_nodes['Accord']:
                query = f"""
                    MATCH (seed_accord:{self.labels['Accord']} {{name: $accord_seed}})
                    MATCH (seed_accord)<-[ha:HAS_ACCORD]-(p:{self.labels['Perfume']})-[:HAS_BRAND]->(b:{self.labels['Brand']})
                    OPTIONAL MATCH (p)-[:FOR_TARGET]->(t:{self.labels['Target']})
                    RETURN 'Accord→Perfume→Brand' as path_type,
                           seed_accord.name as seed_node,
                           p.name as perfume,
                           b.name as end_node,
                           t.name as target,
                           ha.strength as strength,
                           p.rating_value as rating,
                           p.review_count as reviews
                """
                results = session.run(query, accord_seed=accord_seed)
                
                for record in results:
                    expanded_graph['paths'].append({
                        'path_type': record['path_type'],
                        'seed_node': record['seed_node'],
                        'seed_type': 'Accord',
                        'perfume': record['perfume'],
                        'end_node': record['end_node'],
                        'end_type': 'Brand',
                        'target': record['target'],
                        'strength': record['strength'],
                        'rating': record['rating'],
                        'reviews': record['reviews']
                    })
                    expanded_graph['perfumes'].add(record['perfume'])
            
            # 🔗 Accord 체인 탐색 (3-홉)
            accord_chains = []
            all_seed_accords = seed_nodes.get('Accord', [])
            
            for seed_accord in all_seed_accords:
                query = f"""
                    MATCH path = (a1:{self.labels['Accord']})<-[:HAS_ACCORD]-(p1:{self.labels['Perfume']})-[:HAS_ACCORD]->(a2:{self.labels['Accord']})
                                <-[:HAS_ACCORD]-(p2:{self.labels['Perfume']})-[:HAS_ACCORD]->(a3:{self.labels['Accord']})
                    WHERE (a1.name = $seed_accord OR a2.name = $seed_accord OR a3.name = $seed_accord)
                    AND a1 <> a2 AND a2 <> a3 AND a1 <> a3
                    RETURN a1.name as start_accord, a2.name as middle_accord, 
                           a3.name as end_accord, p1.name as perfume1, p2.name as perfume2
                    LIMIT 5
                """
                chain_results = session.run(query, seed_accord=seed_accord)
                
                chains = [dict(record) for record in chain_results]
                accord_chains.extend(chains)
            
            expanded_graph['accord_chains'] = accord_chains
        
        return expanded_graph

    def _calculate_perfume_scores(self, expanded_graph: Dict, seed_nodes: Dict) -> List[Dict]:
        """향수 점수 계산 (기본점수 + 체인보너스)"""
        perfume_scores = {}
        
        # 1. 기본 점수 계산 (경로 개수)
        for path in expanded_graph['paths']:
            perfume = path['perfume']
            if perfume not in perfume_scores:
                perfume_scores[perfume] = {
                    'base_score': 0,
                    'chain_bonus': 0,
                    'rating': 0,
                    'paths': []
                }
            
            perfume_scores[perfume]['base_score'] += 1
            perfume_scores[perfume]['paths'].append(path)
            
            if path.get('rating'):
                try:
                    perfume_scores[perfume]['rating'] = float(path['rating'])
                except:
                    pass
        
        # 2. 체인 보너스 계산
        if expanded_graph.get('accord_chains'):
            chain_bonus = self._calculate_chain_bonus(
                expanded_graph['accord_chains'], 
                seed_nodes
            )
            
            for perfume, bonus in chain_bonus.items():
                if perfume in perfume_scores:
                    perfume_scores[perfume]['chain_bonus'] = bonus
                else:
                    perfume_scores[perfume] = {
                        'base_score': 0,
                        'chain_bonus': bonus,
                        'rating': 0,
                        'paths': []
                    }
        
        # 3. 최종 점수 계산 및 정렬
        ranked_perfumes = []
        for perfume, scores in perfume_scores.items():
            final_score = scores['base_score'] + scores['chain_bonus']
            ranked_perfumes.append({
                'perfume': perfume,
                'final_score': final_score,
                'base_score': scores['base_score'],
                'chain_bonus': scores['chain_bonus'],
                'avg_rating': scores['rating'],
                'path_count': len(scores['paths']),
                'paths': scores['paths']
            })
        
        # 점수 내림차순 정렬
        ranked_perfumes.sort(key=lambda x: (x['final_score'], x['avg_rating']), reverse=True)
        
        return ranked_perfumes[:5]  # Top-5 반환

    def _calculate_chain_bonus(self, accord_chains: List[Dict], seed_nodes: Dict) -> Dict:
        """체인 보너스 계산 (+3점씩)"""
        chain_bonus = {}
        seed_brands = set(seed_nodes.get('Brand', []))
        seed_targets = set(seed_nodes.get('Target', []))
        
        for chain in accord_chains:
            for perfume in [chain['perfume1'], chain['perfume2']]:
                if self._is_chain_perfume_relevant(perfume, seed_brands, seed_targets):
                    if perfume not in chain_bonus:
                        chain_bonus[perfume] = 0
                    chain_bonus[perfume] += 3
        
        return chain_bonus

    def _is_chain_perfume_relevant(self, perfume_name: str, seed_brands: set, seed_targets: set) -> bool:
        """체인 향수의 관련성 검증"""
        if not seed_brands and not seed_targets:
            return True
        
        with self.driver.session(database=self.database) as session:
            query = f"""
                MATCH (p:{self.labels['Perfume']} {{name: $perfume_name}})
                OPTIONAL MATCH (p)-[:HAS_BRAND]->(b:{self.labels['Brand']})
                OPTIONAL MATCH (p)-[:FOR_TARGET]->(t:{self.labels['Target']})
                RETURN b.name as brand, t.name as target
            """
            result = session.run(query, perfume_name=perfume_name).single()
            
            if not result:
                return False
            
            perfume_brand = result['brand']
            perfume_target = result['target']
            
            brand_match = not seed_brands or perfume_brand in seed_brands
            target_match = not seed_targets or perfume_target in seed_targets
            
            return brand_match or target_match

    def _format_results(self, query: str, seed_nodes: Dict, expanded_graph: Dict, perfume_rankings: List[Dict]) -> str:
        """검색 결과를 LLM이 이해할 수 있는 형태로 포맷팅 (통합된 형태)"""
        if not perfume_rankings:
            return "검색 결과가 없습니다."
        
        # 향수 상세 정보 조회
        perfume_names = [p['perfume'] for p in perfume_rankings]
        detailed_perfumes = self._get_detailed_perfume_info(perfume_names)
        
        result = f"""🔍 사용자 질문: {query}

🎯 추천 향수 TOP-{len(perfume_rankings)} (상세 정보 포함):
"""
        
        # 랭킹과 상세 정보를 하나로 통합
        for i, perfume_data in enumerate(perfume_rankings, 1):
            perfume_name = perfume_data['perfume']
            
            # 상세 정보 매칭
            detail = next((p for p in detailed_perfumes if p['name'] == perfume_name), {})
            
            result += f"""{i}. {perfume_name} (최종점수: {perfume_data['final_score']})
   🏷️ 브랜드: {detail.get('brand', '-')}
   👥 타겟: {detail.get('target', '-')}
   ⭐ 평점: {detail.get('rating', '-')} ({detail.get('reviews', '-')}리뷰)
   🎵 주요향조: {detail.get('top_accords', '-')}

"""
        
        return result

    def _get_detailed_perfume_info(self, perfume_names: List[str]) -> List[Dict]:
        """선택된 향수들의 상세 정보를 구조화된 형태로 조회"""
        if not perfume_names:
            return []
            
        with self.driver.session(database=self.database) as session:
            query = f"""
            MATCH (p:{self.labels['Perfume']})
            WHERE p.name IN $names
            OPTIONAL MATCH (p)-[ha:HAS_ACCORD]->(a:{self.labels['Accord']})
            OPTIONAL MATCH (p)-[:HAS_BRAND]->(b:{self.labels['Brand']})
            OPTIONAL MATCH (p)-[:FOR_TARGET]->(t:{self.labels['Target']})
            RETURN p.name as name, b.name as brand, t.name as target, 
                   collect(DISTINCT {{accord: a.name, strength: ha.strength}}) as accords, 
                   p.rating_value as rating, p.review_count as reviews
            """
            results = [dict(record) for record in session.run(query, names=perfume_names)]
        
        detailed_info = []
        for p in results:
            # 향조 정보 정리
            accord_strengths = [
                (a['accord'], float(str(a['strength']).replace('%',''))) 
                for a in p.get('accords', []) if a['accord'] and a['strength'] is not None
            ]
            accord_strengths.sort(key=lambda x: x[1], reverse=True)
            
            seen = set()
            sorted_accords = []
            for name, strength in accord_strengths:
                if name not in seen:
                    sorted_accords.append(f"{name}({strength:.2f}%)")
                    seen.add(name)
            
            detailed_info.append({
                'name': p['name'],
                'brand': p.get('brand', '-'),
                'target': p.get('target', '-'),
                'rating': p.get('rating', '-'),
                'reviews': p.get('reviews', '-'),
                'top_accords': ', '.join(sorted_accords[:5]) if sorted_accords else '-'
            })
        
        return detailed_info

    def test_connection(self) -> bool:
        """Neo4j 연결 테스트"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception as e:
            print(f"Neo4j 연결 실패: {e}")
            return False

    def close(self):
        """리소스 정리"""
        if hasattr(self, 'driver'):
            self.driver.close()


class PerfumeLLMChain:
    """
    🤖 LangChain 기반 향수 추천 LLM 체인
    
    Graph-RAG 검색 결과를 받아 자연스러운 한국어 추천 문장을 생성
    """
    
    def __init__(self, model_id: str, adapter_path: str, 
                 max_tokens: int = 150, temperature: float = 0.5, top_p: float = 0.7):
        """Llama 3.1 모델 + LoRA 어댑터 초기화"""
        self.model_id = model_id
        self.adapter_path = adapter_path
        
        # 생성 설정 저장
        self.max_tokens = max_tokens
        self.temperature = temperature  
        self.top_p = top_p
        
        # LangChain 프롬프트 템플릿 (더 강력한 지시사항)
        self.prompt_template = PromptTemplate(
            input_variables=["user_query", "search_results"],
            template="""당신은 향수 전문가입니다. 주어진 검색 결과에서 사용자 질문에 가장 적합한 향수 3개를 추천해주세요.

사용자 질문: {user_query}
각 향수에 대해 아래 내용을 포함해 간단히 설명하세요:
1. 브랜드 이름
2. 주요 향 노트 (2~3개)
3. 어울리는 계절이나 상황 (예: 가을 남성용, 데일리용, 고급스러운 자리 등)

검색된 향수 정보:
{search_results}

# 주의사항
- 검색 결과에 있는 향수만 설명하세요.

답변:"""
        )
        
        self._initialize_model()
    
    def _initialize_model(self):
        """모델 및 파이프라인 초기화"""
        hf_token = os.getenv('HF_TOKEN')
        
        # Base 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map="auto",
            token=hf_token
        )
        
        # LoRA 어댑터 적용
        model = PeftModel.from_pretrained(base_model, self.adapter_path)
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=hf_token)
        
        # 파이프라인 생성
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.float32
        )
        
        print("✅ PerfumeLLMChain 초기화 완료")
    
    def generate_recommendation(self, user_query: str, search_results: str) -> str:
        """향수 추천 텍스트 생성"""
        try:
            # 프롬프트 생성
            prompt = self.prompt_template.format(
                user_query=user_query,
                search_results=search_results
            )
            
            # 터미널에 LLM 입력 프롬프트 출력
            print("\n" + "="*80)
            print("🤖 LLM에 입력되는 실제 프롬프트:")
            print("="*80)
            print(prompt)
            print("="*80)
            print(f"⚙️ 생성 설정: max_tokens={self.max_tokens}, temperature={self.temperature}, top_p={self.top_p}")
            print("="*80)
            
            # LLM 추론 (설정값 사용)
            outputs = self.pipeline(
                prompt,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )
            
            # 응답 후처리
            full_response = outputs[0]["generated_text"]
            final_response = self._extract_generated_response(full_response, prompt)
            
            # 터미널에 LLM 원본 출력 표시
            print("\n" + "-"*80)
            print("🎯 LLM 원본 응답:")
            print("-"*80)
            print(full_response)
            print("-"*80)
            
            # 터미널에 최종 응답 표시 (정제 없이)
            print("\n" + "🌟 최종 응답:")
            print("-"*80)
            print(final_response)
            print("-"*80 + "\n")
            
            return final_response
            
        except Exception as e:
            print(f"❌ LLM 생성 중 오류: {e}")
            return "죄송합니다. 추천 생성 중 문제가 발생했습니다."
    
    def _extract_generated_response(self, full_output: str, prompt: str) -> str:
        """생성된 텍스트에서 실제 답변만 추출 (정제 최소화)"""
        
        # 프롬프트 제거
        if prompt in full_output:
            response = full_output.replace(prompt, "").strip()
        else:
            response = full_output.strip()
        
        # "답변:" 이후 텍스트만 추출
        if "답변:" in response:
            response = response.split("답변:", 1)[1].strip()
        
        # 기본적인 정리만 수행 (정제 최소화)
        if not response or len(response.strip()) < 10:
            return "죄송합니다. 추천 결과를 생성하는 중 문제가 발생했습니다."
        
        return response.strip()


class LangChainGraphRAG:
    """
    🚀 LangChain 기반 통합 Graph-RAG 시스템
    
    PerfumeGraphSearcher + PerfumeLLMChain을 결합한 완전한 향수 추천 시스템
    """
    
    def __init__(self, 
                 neo4j_uri: str = None, 
                 neo4j_user: str = None, 
                 neo4j_password: str = None,
                 database: str = None,  # 데이터베이스 파라미터 추가
                 model_id: str = None,
                 adapter_path: str = None,
                 max_tokens: int = 150,
                 temperature: float = 0.5,
                 top_p: float = 0.7):
        """시스템 초기화"""
        # 환경변수에서 설정 로드
        self.neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI')
        self.neo4j_user = neo4j_user or os.getenv('NEO4J_USER')
        self.neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD')
        self.model_id = model_id or os.getenv('MODEL_ID')
        self.adapter_path = "/home/shcho95/yjllm/llama3_8b/weight/perfume_llama3_8B_v0"
        
        # 구성 요소 초기화
        self.graph_searcher = PerfumeGraphSearcher(
            neo4j_uri=self.neo4j_uri, 
            neo4j_user=self.neo4j_user, 
            neo4j_password=self.neo4j_password,
            database=database  # 데이터베이스 파라미터 전달
        )
        
        self.llm_chain = PerfumeLLMChain(
            self.model_id,
            self.adapter_path,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # 메모리 (대화 컨텍스트 관리)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        print("✅ LangChainGraphRAG 시스템 초기화 완료")
    
    def get_recommendation(self, user_query: str, **kwargs) -> dict:
        """
        🎯 Streamlit용 추천 함수 (딕셔너리 반환)
        """
        try:
            # kwargs에서 파라미터 추출 및 적용
            if 'temperature' in kwargs or 'max_tokens' in kwargs or 'top_p' in kwargs:
                self.update_generation_settings(
                    max_tokens=kwargs.get('max_tokens'),
                    temperature=kwargs.get('temperature'),
                    top_p=kwargs.get('top_p')
                )
            
            response = self.ask(user_query)
            return {
                "response": response,
                "metadata": {
                    "database": getattr(self.graph_searcher, 'database', 'neo4j'),
                    "model_settings": self.get_generation_settings()
                }
            }
        except Exception as e:
            return {
                "response": f"오류가 발생했습니다: {str(e)}",
                "metadata": {"error": True}
            }

    def ask(self, user_query: str) -> str:
        """
        🎯 메인 추천 함수
        
        1. Graph-RAG 검색 → 2. LLM 생성 → 3. 메모리 저장
        """
        try:
            print(f"\n🔍 질문 처리 시작: '{user_query}'")
            
            # 1단계: Graph-RAG 검색
            print("📊 1단계: Graph-RAG 검색 실행...")
            search_results = self.graph_searcher.search(user_query)
            
            # 2단계: LLM 추천 생성
            print("🤖 2단계: LLM 추천 생성...")
            recommendation = self.llm_chain.generate_recommendation(user_query, search_results)
            
            # 3단계: 메모리에 대화 저장
            self.memory.save_context(
                {"input": user_query},
                {"output": recommendation}
            )
            
            print("✅ 추천 완료!")
            return recommendation
            
        except Exception as e:
            error_msg = f"❌ 시스템 처리 중 오류: {str(e)}"
            print(error_msg)
            return "죄송합니다. 향수 추천 중 문제가 발생했습니다. 다시 시도해 주세요."
    
    def get_chat_history(self) -> str:
        """대화 기록 조회"""
        return str(self.memory.buffer)
    
    def clear_memory(self):
        """메모리 초기화"""
        self.memory.clear()
        print("💭 대화 기록이 초기화되었습니다.")
    
    def update_generation_settings(self, max_tokens: int = None, temperature: float = None, top_p: float = None):
        """LLM 생성 설정 업데이트"""
        if max_tokens is not None:
            self.llm_chain.max_tokens = max_tokens
            print(f"✅ max_tokens 설정: {max_tokens}")
        
        if temperature is not None:
            self.llm_chain.temperature = temperature
            print(f"✅ temperature 설정: {temperature}")
        
        if top_p is not None:
            self.llm_chain.top_p = top_p
            print(f"✅ top_p 설정: {top_p}")

    def get_generation_settings(self) -> dict:
        """현재 LLM 생성 설정 조회"""
        return {
            "max_tokens": self.llm_chain.max_tokens,
            "temperature": self.llm_chain.temperature,
            "top_p": self.llm_chain.top_p
        }

    def test_connection(self) -> bool:
        """시스템 연결 테스트"""
        return self.graph_searcher.test_connection()
    
    def cleanup(self):
        """리소스 정리"""
        print("🧹 시스템 리소스 정리 중...")
        if hasattr(self, 'graph_searcher'):
            self.graph_searcher.close()
        print("✅ 정리 완료!")


# 사용 예시 및 테스트 함수
def test_langchain_graph_rag():
    """LangChain Graph-RAG 시스템 테스트"""
    print("🧪 LangChain Graph-RAG 시스템 테스트 시작")
    
    # 시스템 초기화
    rag = LangChainGraphRAG()
    
    # 연결 테스트
    if not rag.test_connection():
        print("❌ Neo4j 연결 실패")
        return
    
    # 샘플 질문 테스트
    test_queries = [
        "샤넬의 여자 플로럴 향수 추천해줘",
        "우디 계열의 남성 향수 뭐가 좋아?",
        "바닐라향과 꽃향이 나는 향수 추천해줘"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"📝 테스트 질문: {query}")
        print('='*60)
        
        response = rag.ask(query)
        print(f"🤖 추천 결과:\n{response}")
    
    # 리소스 정리
    rag.cleanup()
    print("\n🎉 테스트 완료!")


# 별칭 정의 (기존 코드와의 호환성을 위해)
PerfumeRecommendationSystem = LangChainGraphRAG

if __name__ == "__main__":
    test_langchain_graph_rag() 