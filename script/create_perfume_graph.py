from neo4j import GraphDatabase
import json
import glob
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Neo4j 연결 정보
URI = "bolt://166.104.90.18:7687"
USERNAME = "neo4j"
PASSWORD = "password"

# 임베딩 모델 준비 (전역에서 한 번만 로드)
model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy().tolist()

class PerfumeGraph:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def create_perfume_graph(self, perfume_data):
        # 1. 향수 노드 일괄 생성
        with self.driver.session() as session:
            session.run("""
            UNWIND $perfumes AS perfume
            MERGE (p:Perfume {name: perfume.name})
            SET p.url = perfume.url,
                p.rating_value = perfume.rating_value,
                p.best_rating = perfume.best_rating,
                p.review_count = perfume.review_count,
                p.winter = perfume.seasonal.winter,
                p.spring = perfume.seasonal.spring,
                p.summer = perfume.seasonal.summer,
                p.fall = perfume.seasonal.fall,
                p.day = perfume.time_of_day.day,
                p.night = perfume.time_of_day.night
            """, {"perfumes": perfume_data})

        # 2. 브랜드, 타겟, Accord 관계 일괄 생성
        brand_rels = []
        target_rels = []
        accord_rels = []
        for perfume in perfume_data:
            brand_rels.append({"perfume_name": perfume["name"], "brand_name": perfume["brand_name"]})
            target_rels.append({"perfume_name": perfume["name"], "target": perfume["target"]})
            for accord in perfume["accords"]:
                accord_rels.append({
                    "perfume_name": perfume["name"],
                    "accord_name": accord["accord"],
                    "strength": accord["strength"]
                })
        with self.driver.session() as session:
            # 브랜드
            session.run("""
            UNWIND $rels AS rel
            MATCH (p:Perfume {name: rel.perfume_name})
            MERGE (b:Brand {name: rel.brand_name})
            MERGE (p)-[:HAS_BRAND]->(b)
            """, {"rels": brand_rels})
            # 타겟
            session.run("""
            UNWIND $rels AS rel
            MATCH (p:Perfume {name: rel.perfume_name})
            MERGE (t:Target {name: rel.target})
            MERGE (p)-[:FOR_TARGET]->(t)
            """, {"rels": target_rels})
            # Accord
            session.run("""
            UNWIND $rels AS rel
            MATCH (p:Perfume {name: rel.perfume_name})
            MERGE (a:Accord {name: rel.accord_name})
            MERGE (p)-[r:HAS_ACCORD]->(a)
            SET r.strength = rel.strength
            """, {"rels": accord_rels})

# Neo4j에 embedding 속성 추가
def set_node_embeddings(driver, label):
    with driver.session() as session:
        result = session.run(f"MATCH (n:{label}) RETURN n.name AS name")
        names = [record["name"] for record in result]
        for name in tqdm(names, desc=f"{label} 임베딩 저장"):
            emb = get_embedding(name)
            session.run(
                f"MATCH (n:{label} {{name: $name}}) SET n.embedding = $embedding",
                {"name": name, "embedding": emb}
            )

def main():
    # data/raw_data 폴더 내 모든 json 파일 처리
    json_files = glob.glob('../data/raw_data/*.json')
    graph = PerfumeGraph(URI, USERNAME, PASSWORD)
    try:
        for json_file in tqdm(json_files, desc="JSON 파일 처리"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"{json_file} 처리 중...")
            graph.create_perfume_graph(data['perfumes_data'])
        print("그래프 생성 완료!")

        # Brand, Target, Accord 노드에 임베딩 저장
        driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
        set_node_embeddings(driver, "Brand")
        set_node_embeddings(driver, "Target")
        set_node_embeddings(driver, "Accord")
        set_node_embeddings(driver, "Perfume")
        driver.close()
    finally:
        graph.close()

if __name__ == "__main__":
    main() 

    