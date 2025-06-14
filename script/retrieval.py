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

# 향수 서칭 매커니즘 예시

# extract_keywords --> get_embedding
# 사용자 쿼리: "샤넬의 여성용 플로럴 향수"
# 추출된 키워드: ["샤넬", "여성용", "플로럴"] 

# find_similar_nodes
# 이렇게 되면 각 키워드에 대해 유사도 계산 수행
# "샤넬" -> 모든 브랜드 노드와 유사도 계산 
# "여성용"-> 모든 타겟 노드와 유사도 계산 
# "플로럴"-> 모든 향 계열 노드와 유사도 계산 
# --> 총 9개 노드 선택됨

# brands = ["샤넬", "디올"]  # 9개 노드 중 브랜드 노드들 -> 코사인 유사도 비교 결과 상위 3개 선택
# targets = ["여성"]         # 9개 노드 중 타겟 노드들 -> 코사인 유사도 비교 결과 상위 3개 선택
# accords = ["플로럴", "우디"] # 9개 노드 중 향 계열 노드들 -> 코사인 유사도 비교 결과 상위 3개 선택

# get_perfumes_by_nodes
# 결과 예시 
# perfume_scores = [
#     ("샤넬 No.5", 3),    # 브랜드(샤넬) + 타겟(여성) + 향계열(플로럴) = 3점
#     ("디올 자도르", 4),  # 브랜드(디올) + 타겟(여성) + 향계열(플로럴,우디) = 4점
#     ("샤넬 블루", 2)     # 브랜드(샤넬) + 향계열(우디) = 2점
# --> 디올 자도르, 샤넬 No.5가 최대 k 개 (현재는 best_perfumes[:5])가 context에 입력

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
        stopwords = ['이', '그', '저', '것', '수', '등', '및', '또는', '그리고', '하지만', '그래서','때문에', '위해', '대해', '관련', '따라서', '그러나', '그리고', '또한', '그래도','이런', '저런', '그런', '이러한', '저러한', '그러한', '이런', '저런', '그런','이것', '저것', '그것', '이런', '저런', '그런', '이렇게', '저렇게', '그렇게', '추천','하다','해주다','나', '용이','어울리다','알다', '뭐', '좋다', '향수']
        
        #한국어 텍스트에서 명사, 동사, 형용사만 추출
        tokens = self.okt.pos(text, stem=True)
        keywords = [word for word, pos in tokens 
               if pos in ['Noun', 'Verb', 'Adjective'] 
               and word not in stopwords]
        return list(set(keywords))

    def cosine_similarity(self, emb1, emb2):
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def get_all_nodes_with_embeddings(self, label: str) -> List[Dict]:
        with self.driver.session() as session:
            query = f"""
            MATCH (n:{label})
            WHERE n.embedding IS NOT NULL
            RETURN n.name as name, n.embedding as embedding
            """
            return [dict(record) for record in session.run(query)]

    def find_similar_nodes(self, keywords: List[str], label: str, topn: int = 3) -> List[str]:
        #키워드와 유사한 노드 검색
        #각 키워드마다 상위 3개 노드 선택
        nodes = self.get_all_nodes_with_embeddings(label)
        result = set()
        for kw in keywords:
            emb = self.get_embedding(kw)
            sims = [(node['name'], self.cosine_similarity(emb, node['embedding'])) for node in nodes]
            sims.sort(key=lambda x: x[1], reverse=True)
            result.update([name for name, _ in sims[:topn]])
        return list(result)


    def get_perfumes_by_nodes(self, brands, targets, accords) -> List[str]:
        #브랜드, 타겟, 향 계열로 향수 필터링
        #매칭 점수 계산 후 상위 5개 반환
        with self.driver.session() as session:
            query = """
            MATCH (p:Perfume)
            OPTIONAL MATCH (p)-[:HAS_BRAND]->(b:Brand)
            OPTIONAL MATCH (p)-[:FOR_TARGET]->(t:Target)
            OPTIONAL MATCH (p)-[:HAS_ACCORD]->(a:Accord)
            RETURN p.name as name, collect(DISTINCT b.name) as brands, collect(DISTINCT t.name) as targets, collect(DISTINCT a.name) as accords
            """
            perfumes = [dict(record) for record in session.run(query)]
        # 브랜드, 타겟, 향 계열 매칭 점수 계산
        perfume_scores = []
        for p in perfumes:
            matched = []
            if brands:
                matched += list(set(p['brands']) & set(brands))
            if targets:
                matched += list(set(p['targets']) & set(targets))
            if accords:
                matched += list(set(p['accords']) & set(accords))
            score = len(set(matched))
            perfume_scores.append((p['name'], score))
        # 가장 높은 점수를 가진 향수 선택   
        max_score = max([s for _, s in perfume_scores]) if perfume_scores else 0
        best_perfumes = [name for name, s in perfume_scores if s == max_score and s > 0]
        return best_perfumes[:5]

    def get_perfume_context(self, perfume_names: List[str]) -> str:
        #선택된 향수들의 상세 정보 포맷팅
        #브랜드, 타겟, 평점, 리뷰 수, 향 계열 정보 포함
        if not perfume_names:
            return "검색 결과가 없습니다."
        with self.driver.session() as session:
            query = """
            MATCH (p:Perfume)
            WHERE p.name IN $names
            OPTIONAL MATCH (p)-[:HAS_BRAND]->(b:Brand)
            OPTIONAL MATCH (p)-[:FOR_TARGET]->(t:Target)
            OPTIONAL MATCH (p)-[:HAS_ACCORD]->(a:Accord)
            RETURN p.name as name, b.name as brand, t.name as target, collect(a.name) as accords, p.rating_value as rating, p.review_count as reviews
            """
            results = [dict(record) for record in session.run(query, names=perfume_names)]
        context = ""
        for i, p in enumerate(results, 1):
            context += f"{i}. {p['name']} (브랜드: {p.get('brand', '-')}, 타겟: {p.get('target', '-')}, 평점: {p.get('rating', '-')}, 리뷰: {p.get('reviews', '-')}, 주요향: {', '.join(p.get('accords', []))})\n"
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

if __name__ == "__main__":
    test_text = "가을에 어울리다 알다 향수 추천해줘"
    dummy_uri = "bolt://localhost:7687"
    dummy_user = "neo4j"
    dummy_password = "password"
    try:
        retriever = Neo4jRetrieval(dummy_uri, dummy_user, dummy_password)
        keywords = retriever.extract_keywords(test_text)
        print("최종 추출된 키워드:", keywords)
    except Exception as e:
        print("에러 발생:", e)