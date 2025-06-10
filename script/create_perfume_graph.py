from neo4j import GraphDatabase
import json

# Neo4j 연결 정보
URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "password"

class PerfumeGraph:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def create_perfume_graph(self, perfume_data):
        with self.driver.session() as session:
            # 기존 데이터 삭제
            session.run("MATCH (n) DETACH DELETE n")
            
            # 향수 데이터 생성
            for perfume in perfume_data:
                
                # 향수 노드 생성
                session.run("""
                    CREATE (p:Perfume {
                        name: $name,
                        url: $url,
                        rating_value: $rating_value,
                        best_rating: $best_rating,
                        review_count: $review_count,
                        winter: $winter,
                        spring: $spring,
                        summer: $summer,
                        fall: $fall,
                        day: $day,
                        night: $night
                    })
                    WITH p
                    MERGE (b:Brand {name: $brand_name})
                    CREATE (p)-[:HAS_BRAND]->(b)
                    WITH p
                    MERGE (t:Target {name: $target})
                    CREATE (p)-[:FOR_TARGET]->(t)
                """, {
                    'name': perfume['name'],
                    'url': perfume['url'],
                    'rating_value': perfume['rating_value'],
                    'best_rating': perfume['best_rating'],
                    'review_count': perfume['review_count'],
                    'winter': perfume['seasonal']['winter'],
                    'spring': perfume['seasonal']['spring'],
                    'summer': perfume['seasonal']['summer'],
                    'fall': perfume['seasonal']['fall'],
                    'day': perfume['time_of_day']['day'],
                    'night': perfume['time_of_day']['night'],
                    'brand_name': perfume['brand_name'],
                    'target': perfume['target']
                })
                
                # Accord 노드 생성 및 연결
                for accord in perfume['accords']:
                    session.run("""
                        MATCH (p:Perfume {name: $perfume_name})
                        MERGE (a:Accord {name: $accord_name})
                        CREATE (p)-[r:HAS_ACCORD {strength: $strength}]->(a)
                    """, {
                        'perfume_name': perfume['name'],
                        'accord_name': accord['accord'],
                        'strength': accord['strength']
                    })

def main():
    # JSON 파일 읽기
    with open('data/perfumes_test.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 그래프 생성
    graph = PerfumeGraph(URI, USERNAME, PASSWORD)
    try:
        graph.create_perfume_graph(data['perfumes_data'])
        print("그래프 생성 완료!")
    finally:
        graph.close()

if __name__ == "__main__":
    main() 