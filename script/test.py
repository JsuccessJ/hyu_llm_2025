import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

print("🔗 Neo4j 연결 테스트")
print("=" * 30)

# 환경변수 확인
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER") 
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

print(f"NEO4J_URI: {NEO4J_URI}")
print(f"NEO4J_USER: {NEO4J_USER}")
print(f"NEO4J_PASSWORD: {'*' * len(NEO4J_PASSWORD) if NEO4J_PASSWORD else 'None'}")

# Neo4j 연결 테스트
try:
    from neo4j import GraphDatabase
    
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        print("❌ 환경변수가 설정되지 않았습니다.")
        exit(1)
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        result = session.run("RETURN 'Hello Neo4j!' as message")
        record = result.single()
        
        if record:
            print(f"✅ Neo4j 연결 성공: {record['message']}")
        else:
            print("❌ Neo4j 연결 실패: 결과 없음")
            
    driver.close()
    
except Exception as e:
    print(f"❌ Neo4j 연결 실패: {e}")
    print("💡 Neo4j 서버가 실행 중인지 확인하고 연결 정보를 확인해주세요.")
    exit(1) 