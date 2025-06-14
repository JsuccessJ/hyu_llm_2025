#graph retrieval
import torch
from transformers import AutoTokenizer, AutoModel
from neo4j import GraphDatabase
from typing import List, Dict
from collections import Counter
import numpy as np
from konlpy.tag import Okt
from dotenv import load_dotenv
import os

# =============================
# Graph RAG 향수 추천 매커니즘 (2024 최신)
# =============================

# 1. extract_keywords
#   - 사용자 쿼리에서 명사, 동사, 형용사만 추출하고 불용어를 제거하여 키워드 리스트 생성
#   - 중복을 제거하여 최종 키워드 반환
# 예시:
#   입력: "샤넬의 여자 플로럴 향수 추천해줘"
#   출력: ["샤넬", "여자", "플로럴"]

# 2. get_seed_nodes
#   - 추출된 키워드 각각에 대해 Brand, Target, Accord 노드와 임베딩 유사도를 계산
#   - Brand, Target은 0.75, Accord는 0.65 이상의 유사도를 가진 노드를 seed node로 선택
#   - 각 타입별 최대 3개까지 seed node 반환
# 예시:
#   입력: ["샤넬", "여성", "플로럴"]
#   출력: {"Brand": ["샤넬"], "Target": ["여성"], "Accord": ["플로럴"]}

# 3. expand_graph
#   - 사용자가 입력한 브랜드, 타겟, 향 계열(Accord) 키워드 각각에서 출발해서, 그래프에서 연결된 향수(Perfume)들을 모두 찾음
#   - Brand에서 출발할 때는 Brand 노드에서 향수(Perfume)로 가는 방향(<-[:HAS_BRAND]-)으로 연결된 모든 향수를 찾음
#   - Target, Accord도 마찬가지로 각각 연결된 향수들을 모두 찾음
#   - 추가로, seed Accord(예: 플로럴)가 포함된 3단계 연결(Accord-Perfume-Accord-Perfume-Accord) 체인도 찾아서 accord_chains에 저장함
#   - 이렇게 모은 향수들의 정보(브랜드, 타겟, 향 계열, 평점 등)와 Accord 체인 정보를 모두 한 번에 모음
# 예시:
#   입력: {"Brand": ["샤넬"], "Target": ["여성"], "Accord": ["플로럴"]}
#   처리: 샤넬에서 출발해 연결된 모든 향수, 여성에서 출발해 연결된 모든 향수, 플로럴에서 출발해 연결된 모든 향수, 그리고 플로럴이 포함된 3단계 Accord 체인까지 모두 탐색
#   출력: 여러 경로를 따라 연결된 향수들과 그 정보, 그리고 관련 Accord 체인들

# 4. calculate_perfume_scores
#   - 각 향수(Perfume)별로 점수를 아래와 같이 계산:
#     (1) 기본점수: seed 기반 그래프 경로에 등장한 횟수 (즉, Brand/Target/Accord에서 출발해 해당 향수로 연결된 경로 개수, 1경로당 1점)
#     (2) 체인보너스: seed Accord가 포함된 3-홉 Accord 체인에 해당 향수가 등장하면 +3점 (여러 체인에 등장하면 누적)
#   - 최종점수 = 기본점수 + 체인보너스
#   - 점수가 같으면 rating(평점)이 높은 순으로 내림차순 정렬
#   - 상위 5개 향수만 반환
# 예시:
#   입력: 확장된 그래프 정보
#   출력: [{"perfume": "샤넬 No.5", "final_score": 7, ...}, {"perfume": "디올 자도르", "final_score": 5, ...}, ...]
#   (예: 샤넬 No.5가 seed 기반 경로에 4번 등장하고, Accord 체인에 1번 등장하면 4+3=7점)

# 5. generate_answer
#   - 쿼리, seed nodes, 그래프 확장 결과, 향수 랭킹 정보를 받아 최종 추천 결과를 포맷팅하여 문자열로 반환
#   - 각 향수별로 최종점수, 기본점수, 체인보너스, 평점, 상세 정보(브랜드, 타겟, 평점, 리뷰, 주요향)를 포함
#   - 발견된 Accord 체인 패턴(인기 체인, 브릿지 Accord 등)도 함께 출력
# 예시:
#   출력:
#   1. 샤넬 No.5 (점수: 7 = 기본:4 + 체인:3, 평점: 4.7, 주요향: 플로럴(70.00%), ...)
#   2. 디올 자도르 (점수: 5 = 기본:5 + 체인:0, 평점: 4.5, 주요향: 플로럴(65.00%), ...)
#   ...


class Neo4jRetrieval:
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        # Neo4j 연결
        uri = uri or os.getenv('NEO4J_URI')
        username = username or os.getenv('NEO4J_USER')
        password = password or os.getenv('NEO4J_PASSWORD')
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # 임베딩 모델 초기화
        self.model_name = "BAAI/bge-m3"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # 임베딩 캐시
        self.embedding_cache = {}
        self.okt = Okt()

    def get_embedding(self, text: str) -> List[float]:
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embedding = embeddings[0].numpy().tolist()
        self.embedding_cache[text] = embedding
        return embedding

    def extract_keywords(self, text: str) -> List[str]:
        # 불용어 리스트 정의
        stopwords = ['이', '그', '저', '것', '수', '등', '및', '또는', '그리고', '하지만', '그래서','때문에', '위해', '대해', '관련', '따라서', '그러나', '그리고', '또한', '그래도','이런', '저런', '그런', '이러한', '저러한', '그러한', '이런', '저런', '그런','이것', '저것', '그것', '이런', '저런', '그런', '이렇게', '저렇게', '그렇게', '추천','하다','해주다','나', '용이','어울리다','알다', '뭐', '좋다', '향수','계열','향', '있다','찾다']
        # 한국어 텍스트에서 명사, 동사, 형용사만 추출
        tokens = self.okt.pos(text, stem=True)
        keywords = [word for word, pos in tokens 
               if pos in ['Noun', 'Verb', 'Adjective'] 
               and word not in stopwords]
        return list(set(keywords))

    def cosine_similarity(self, emb1, emb2):
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def get_all_nodes_embeddings(self, label: str) -> List[Dict]:
        with self.driver.session() as session:
            query = f"""
            MATCH (n:{label})
            WHERE n.embedding IS NOT NULL
            RETURN n.name as name, n.embedding as embedding
            """
            return [dict(record) for record in session.run(query)]

    def get_perfume_context(self, perfume_names: List[str]) -> str:
        #선택된 향수들의 상세 정보 포맷팅
        #브랜드, 타겟, 평점, 리뷰 수, 향 계열 정보 포함
        if not perfume_names:
            return "검색 결과가 없습니다."
        with self.driver.session() as session:
            query = """
            MATCH (p:Perfume)
            WHERE p.name IN $names
            OPTIONAL MATCH (p)-[ha:HAS_ACCORD]->(a:Accord)
            OPTIONAL MATCH (p)-[:HAS_BRAND]->(b:Brand)
            OPTIONAL MATCH (p)-[:FOR_TARGET]->(t:Target)
            RETURN p.name as name, b.name as brand, t.name as target, \
                   collect(DISTINCT {accord: a.name, strength: ha.strength}) as accords, \
                   p.rating_value as rating, p.review_count as reviews
            """
            results = [dict(record) for record in session.run(query, names=perfume_names)]
        context = ""
        for i, p in enumerate(results, 1):
            # strength가 None이 아닌 것만, % 떼고 float 변환, 내림차순 정렬
            accord_strengths = [
                (a['accord'], float(str(a['strength']).replace('%',''))) 
                for a in p.get('accords', []) if a['accord'] and a['strength'] is not None
            ]
            # strength 내림차순 정렬
            accord_strengths.sort(key=lambda x: x[1], reverse=True)
            # accord 이름+strength 포맷으로 추출 (중복 제거)
            seen = set()
            sorted_accords = []
            for name, strength in accord_strengths:
                if name not in seen:
                    sorted_accords.append(f"{name}({strength:.2f}%)")
                    seen.add(name)
            context += f"{i}. {p['name']} (브랜드: {p.get('brand', '-')}, 타겟: {p.get('target', '-')}, 평점: {p.get('rating', '-')}, 리뷰: {p.get('reviews', '-')}, 주요향(강한순): {', '.join(sorted_accords)})\n"
        return context

    def test_connection(self) -> bool:
        #Neo4j 연결 테스트
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception as e:
            print(f"Neo4j 연결 실패: {e}")
            return False

    def close(self):
        self.driver.close() 

    def start_to_seed(self, query: str) -> str:
        """키워드 기반 seed node에서 시작하는 Graph RAG"""
        
        # 1. 키워드 추출 및 seed node 결정
        keywords = self.extract_keywords(query)
        seed_nodes = self.get_seed_nodes(keywords)
        
        if not seed_nodes:
            return "관련된 정보를 찾을 수 없습니다."
        
        # 2. seed node에서 시작하는 그래프 확장
        expanded_graph = self.expand_graph(seed_nodes)
        
        # 3. 🔥 체인 정보를 활용한 향수 점수 계산 (query 파라미터 추가)
        perfume_rankings = self.calculate_perfume_scores(expanded_graph, seed_nodes)
        
        # 4. 최종 답변 생성
        return self.generate_answer(query, seed_nodes, expanded_graph, perfume_rankings)

    def get_seed_nodes(self, keywords: List[str]) -> Dict[str, List[str]]:
        """키워드에서 유사도 기반 seed node 추출"""
        seed_nodes = {
            'Brand': [],
            'Target': [], 
            'Accord': []
        }
        
        print(f"🔍 키워드에서 seed node 추출: {keywords}")
        
        for label in ['Brand', 'Target', 'Accord']:
            if label == 'Brand':
                threshold = 0.75
            elif label == 'Target':
                threshold = 0.8
            else:
                threshold = 0.65
            
            # 임계값 이상인 노드만 seed로 선택
            filtered_seeds = []
            nodes = self.get_all_nodes_embeddings(label)
            
            for keyword in keywords:
                kw_emb = self.get_embedding(keyword)
                for node in nodes:
                    similarity = self.cosine_similarity(kw_emb, node['embedding'])
                    if similarity >= threshold and node['name'] not in filtered_seeds:
                        filtered_seeds.append(node['name'])
                        print(f"✅ Seed {label}: {node['name']} (similarity: {similarity:.3f} with '{keyword}')")
            
            seed_nodes[label] = filtered_seeds[:3]  # 최대 3개씩
        
        return seed_nodes

    def expand_graph(self, seed_nodes: Dict[str, List[str]]) -> Dict:
        """seed node에서 시작하여 그래프 확장 + Accord 체인(3-홉) 탐색"""
        expanded_graph = {
            'paths': [],
            'perfumes': set(),
            'seed_connections': {},
            'accord_chains': []  # Accord 체인 결과 추가
        }
        with self.driver.session() as session:
            # 기존 1홉 확장 (Brand, Target, Accord)
            # 경로 1: Brand seed → Perfume → Accord
            for brand_seed in seed_nodes['Brand']:
                brand_expansion = session.run("""
                    MATCH (seed_brand:Brand {name: $brand_seed})
                    MATCH (seed_brand)<-[:HAS_BRAND]-(p:Perfume)-[ha:HAS_ACCORD]->(a:Accord)
                    OPTIONAL MATCH (p)-[:FOR_TARGET]->(t:Target)
                    RETURN 'Brand→Perfume→Accord' as path_type,
                           seed_brand.name as seed_node,
                           p.name as perfume,
                           a.name as end_node,
                           t.name as target,
                           ha.strength as strength,
                           p.rating_value as rating,
                           p.review_count as reviews
                """, brand_seed=brand_seed)
                for record in brand_expansion:
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
            # 경로 2: Target seed → Perfume → Accord
            for target_seed in seed_nodes['Target']:
                target_expansion = session.run("""
                    MATCH (seed_target:Target {name: $target_seed})
                    MATCH (seed_target)<-[:FOR_TARGET]-(p:Perfume)-[ha:HAS_ACCORD]->(a:Accord)
                    OPTIONAL MATCH (p)-[:HAS_BRAND]->(b:Brand)
                    RETURN 'Target→Perfume→Accord' as path_type,
                           seed_target.name as seed_node,
                           p.name as perfume,
                           a.name as end_node,
                           b.name as brand,
                           ha.strength as strength,
                           p.rating_value as rating,
                           p.review_count as reviews
                """, target_seed=target_seed)
                for record in target_expansion:
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
            # 경로 3: Accord seed → Perfume → Brand
            for accord_seed in seed_nodes['Accord']:
                accord_expansion = session.run("""
                    MATCH (seed_accord:Accord {name: $accord_seed})
                    MATCH (seed_accord)<-[ha:HAS_ACCORD]-(p:Perfume)-[:HAS_BRAND]->(b:Brand)
                    OPTIONAL MATCH (p)-[:FOR_TARGET]->(t:Target)
                    RETURN 'Accord→Perfume→Brand' as path_type,
                           seed_accord.name as seed_node,
                           p.name as perfume,
                           b.name as end_node,
                           t.name as target,
                           ha.strength as strength,
                           p.rating_value as rating,
                           p.review_count as reviews
                """, accord_seed=accord_seed)
                for record in accord_expansion:
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
            # 🔥 Seed 기반 Accord 체인 탐색 (관련있는 체인만)
            accord_chains = []
            all_seed_accords = seed_nodes.get('Accord', [])
            
            if all_seed_accords:  # Accord seed가 있을 때만 체인 탐색
                for seed_accord in all_seed_accords:
                    # Seed Accord가 포함된 체인만 찾기
                    seed_chain_query = """
                        MATCH path = (a1:Accord)<-[:HAS_ACCORD]-(p1:Perfume)-[:HAS_ACCORD]->(a2:Accord)
                                    <-[:HAS_ACCORD]-(p2:Perfume)-[:HAS_ACCORD]->(a3:Accord)
                        WHERE (a1.name = $seed_accord OR a2.name = $seed_accord OR a3.name = $seed_accord)
                        AND a1 <> a2 AND a2 <> a3 AND a1 <> a3
                        RETURN a1.name as start_accord, a2.name as middle_accord, 
                               a3.name as end_accord, p1.name as perfume1, p2.name as perfume2
                        LIMIT 5
                    """
                    seed_chains = [dict(record) for record in session.run(seed_chain_query, seed_accord=seed_accord)]
                    accord_chains.extend(seed_chains)
                    
                    if seed_chains:
                        print(f"  🔗 {seed_accord} 기반 체인 {len(seed_chains)}개 발견")
                        print("seed_chains (debug용):")
                        for i, chain in enumerate(seed_chains, 1):
                            print(f"  {i}. {chain}")
            expanded_graph['accord_chains'] = accord_chains
        print(f"🌐 그래프 확장 완료: {len(expanded_graph['paths'])}개 경로, {len(expanded_graph['perfumes'])}개 향수, {len(expanded_graph['accord_chains'])}개 Accord 체인")
        return expanded_graph


    def analyze_ingredient_chains(self, chains_data):
        """발견된 체인을 분석하여 추천에 활용"""
        # 1. 체인 빈도 분석
        chain_frequency = {}
        for chain in chains_data:
            pattern = f"{chain['start_accord']}→{chain['middle_accord']}→{chain['end_accord']}"
            chain_frequency[pattern] = chain_frequency.get(pattern, 0) + 1
        # 2. 중요한 연결 향료 식별
        bridge_accords = {}
        for chain in chains_data:
            middle = chain['middle_accord']
            bridge_accords[middle] = bridge_accords.get(middle, 0) + 1
        # 3. 향수 간 유사도 계산 (예시: 단순히 perfume1, perfume2가 같은 체인에 등장한 횟수)
        perfume_similarity = {}
        for chain in chains_data:
            pair = tuple(sorted([chain['perfume1'], chain['perfume2']]))
            perfume_similarity[pair] = perfume_similarity.get(pair, 0) + 1
        return {
            'popular_chains': sorted(chain_frequency.items(), key=lambda x: x[1], reverse=True),
            'bridge_accords': sorted(bridge_accords.items(), key=lambda x: x[1], reverse=True),
            'perfume_similarities': sorted(perfume_similarity.items(), key=lambda x: x[1], reverse=True)
        }

    def calculate_perfume_scores(self, expanded_graph: Dict, seed_nodes: Dict) -> List[Dict]:
        """각 향수(Perfume)별로 점수를 아래와 같이 계산:
        (1) 기본점수: seed 기반 그래프 경로에 등장한 횟수 (즉, Brand/Target/Accord에서 출발해 해당 향수로 연결된 경로 개수, 1경로당 1점)
        (2) 체인보너스: seed Accord가 포함된 3-홉 Accord 체인에 해당 향수가 등장하면 +3점 (여러 체인에 등장하면 누적)
        - 최종점수 = 기본점수 + 체인보너스
        - 점수가 같으면 rating(평점)이 높은 순으로 내림차순 정렬
        - 상위 5개 향수만 반환"""
        
        # 1. 기본 점수: 단순히 경로 개수만 세기
        perfume_scores = {}
        for path in expanded_graph['paths']:
            perfume = path['perfume']
            if perfume not in perfume_scores:
                perfume_scores[perfume] = {
                    'base_score': 0,
                    'chain_bonus': 0,
                    'rating': 0,
                    'paths': []
                }
            
            # 경로 1개당 1점
            perfume_scores[perfume]['base_score'] += 1
            perfume_scores[perfume]['paths'].append(path)
            
            # 평점 수집
            if path.get('rating'):
                try:
                    perfume_scores[perfume]['rating'] = float(path['rating'])
                except:
                    pass
        
        # 2. 🔥 체인 보너스: 관련 체인에 나타나면 +3점
        if expanded_graph.get('accord_chains'):
            chain_bonus = self.calculate_chain_bonus(
                expanded_graph['accord_chains'], 
                seed_nodes
            )
            
            # 체인 보너스 적용
            for perfume, bonus in chain_bonus.items():
                if perfume in perfume_scores:
                    perfume_scores[perfume]['chain_bonus'] = bonus
                else:
                    # 체인으로만 발견된 새로운 향수
                    perfume_scores[perfume] = {
                        'base_score': 0,
                        'chain_bonus': bonus,
                        'rating': 0,
                        'paths': []
                    }
        
        # 3. 최종 점수 = 기본점수 + 체인보너스
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
        
        # 디버그 출력
        print("===== Top-5 Perfume & Simple Score =====")
        for i, p in enumerate(ranked_perfumes[:5], 1):
            print(f"{i}. {p['perfume']} (점수: {p['final_score']} = 기본:{p['base_score']} + 체인:{p['chain_bonus']})")
        
        return ranked_perfumes[:5]

    def calculate_chain_bonus(self, accord_chains: List[Dict], seed_nodes: Dict) -> Dict:
        """Seed 기반 체인 보너스 계산 + 관련성 필터링"""
        
        if not accord_chains:
            return {}
        
        chain_bonus = {}
        
        # 🔥 Seed 브랜드와 타겟 추출 (필터링용)
        seed_brands = set(seed_nodes.get('Brand', []))
        seed_targets = set(seed_nodes.get('Target', []))
        
        print(f"🔗 Seed 기반 체인 보너스 계산: {len(accord_chains)}개 관련 체인 분석")
        
        for chain in accord_chains:
            accords = [chain['start_accord'], chain['middle_accord'], chain['end_accord']]
            #print(f"  ✅ 관련 체인: {' → '.join(accords)}")
            
            for perfume in [chain['perfume1'], chain['perfume2']]:
                # 🔥 체인 발견 향수의 관련성 검증
                if self._is_chain_perfume_relevant(perfume, seed_brands, seed_targets):
                    if perfume not in chain_bonus:
                        chain_bonus[perfume] = 0
                    chain_bonus[perfume] += 3
                    #디버그용
                    #print(f"    💎 {perfume}: +3점 (관련성 검증 통과)")
                #else:
                    #디버그용
                    #print(f"    ❌ {perfume}: 체인 발견했지만 관련성 부족으로 제외")
        
        print(f"🔗 체인 보너스 완료: {len(chain_bonus)}개 향수에 보너스 적용")
        return chain_bonus

    def _is_chain_perfume_relevant(self, perfume_name: str, seed_brands: set, seed_targets: set) -> bool:
        """체인으로 발견된 향수가 seed와 관련있는지 검증"""
        
        # Seed가 브랜드나 타겟이 없으면 모든 체인 향수 허용
        if not seed_brands and not seed_targets:
            return True
        
        with self.driver.session() as session:
            query = """
            MATCH (p:Perfume {name: $perfume_name})
            OPTIONAL MATCH (p)-[:HAS_BRAND]->(b:Brand)
            OPTIONAL MATCH (p)-[:FOR_TARGET]->(t:Target)
            RETURN b.name as brand, t.name as target
            """
            result = session.run(query, perfume_name=perfume_name).single()
            
            if not result:
                return False
            
            perfume_brand = result['brand']
            perfume_target = result['target']
            
            # 🔥 관련성 검증 로직
            brand_match = not seed_brands or perfume_brand in seed_brands
            target_match = not seed_targets or perfume_target in seed_targets
            
            # 브랜드나 타겟 중 하나라도 매치되면 관련있음
            return brand_match or target_match

    def generate_answer(self, query: str, seed_nodes: Dict, expanded_graph: Dict, perfume_rankings: List[Dict]) -> str:
        """Graph RAG 기반 최종 답변 생성 + 체인 보너스 표시"""
        answer = f"🔍 사용자 쿼리: '{query}'\n\n"
        
        # Seed nodes 정보
        answer += "🌱 추출된 Seed Nodes (debug용):\n"
        for node_type, nodes in seed_nodes.items():
            if nodes:
                answer += f"• {node_type}: {', '.join(nodes)}\n"
        answer += "\n"
        
        # 그래프 확장 결과
        answer += f"🌐 그래프 확장 결과 (debug용): {len(expanded_graph['paths'])}개 경로, {len(expanded_graph['accord_chains'])}개 체인 발견\n\n"
        
        # 🔥 추천 향수 (단순화된 점수 표시)
        answer += "🎯 GraphRAG 추천 결과 (debug용):\n\n"
        
        for i, perfume_data in enumerate(perfume_rankings, 1):
            answer += f"{i}. **{perfume_data['perfume']}** (점수: {perfume_data['final_score']})\n"
            answer += f"   • 기본점수: {perfume_data['base_score']}점 (경로 {perfume_data['path_count']}개)\n"
            if perfume_data['chain_bonus'] > 0:
                answer += f"   • 체인보너스: +{perfume_data['chain_bonus']}점 🔗\n"
            answer += f"   • 평점: {perfume_data['avg_rating']}\n\n"
        
        # 상세 정보 추가
        perfume_names = [p['perfume'] for p in perfume_rankings]
        detailed_info = self.get_perfume_context(perfume_names)
        answer += f"📋 LLM이 받는 실제 context):\n{detailed_info}"
        
        # Accord 체인 분석 결과 추가
        if expanded_graph.get('accord_chains'):
            chain_analysis = self.analyze_ingredient_chains(expanded_graph['accord_chains'])
            answer += "\n🔗 발견된 향료 체인 패턴 (debug용):\n"
            answer += "• 인기 체인 TOP3:\n"
            for pattern, freq in chain_analysis['popular_chains'][:3]:
                answer += f"  - {pattern} ({freq}회)\n"
            #중요한 연결 향료 예시..
            #플로럴 <-향수1-> 자스민
            #플로럴 <-향수2-> 우드
            #이런식이면 플로럴이 다른향료들(자스민, 우드)를 연결하는 중요 향료가 됨
            #사용자쿼리의 키워드에서 플로럴이 나왔다면 플로럴과 높은 유사도,, 즉 이와 잘 어울리는 다른 향료들(자스민, 우드)을 포함한 향수를 추천할 수 있음
            answer += "• 중요한 연결 향료 TOP3:\n"
            for accord, freq in chain_analysis['bridge_accords'][:3]:
                answer += f"  - {accord} ({freq}회)\n"
        
        return answer

    def search_graph_rag(self, query: str) -> str:
        """Graph RAG 방식으로 향수 검색"""
        return self.start_to_seed(query)