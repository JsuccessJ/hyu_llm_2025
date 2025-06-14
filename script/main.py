#graph main
import os
from graph_rag import GraphRAG
import transformers
import torch
from dotenv import load_dotenv

def main():
    """메인 실행 함수"""
    load_dotenv()
    # Neo4j 연결 정보 (기본값 사용)
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    MODEL_ID = os.getenv("MODEL_ID")
    # http://@서버IP:7474/
    
    try:
        # GraphRAG 시스템 초기화
        print("🌸 향수 추천 시스템을 초기화하는 중...")
        rag = GraphRAG(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            model_id=MODEL_ID
        )
        
        # Neo4j 연결 테스트
        if not rag.retrieval.test_connection():
            print("❌ Neo4j 연결 실패. 서버를 확인해주세요.")
            return
        print("✅ Neo4j 연결 성공")
        
        # 샘플 질문 3개 자동 실행
        sample_questions = [
            "샤넬의 여성용 플로럴 향수 추천해줘",
            "우디 계열의 남성 향수 뭐가 좋아?", 
            "바닐라향과 꽃향이 나는 향수 추천해줘"
        ]
        
        print("\n" + "="*60)
        print("📋 샘플 질문으로 시스템 테스트를 시작합니다")
        print("="*60)
        
        for i, q in enumerate(sample_questions, 1):
            print(f"\n[샘플 {i}/3] 질문: {q}")
            print("-" * 40)
            try:
                response = rag.ask(q)
                print(f"🤖 추천 결과:")
                print(response)
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
            print("-" * 40)
        
        print("\n" + "="*60)
        print("💬 이제 자유롭게 향수에 대해 질문해보세요!")
        print("종료하려면 'quit', 'exit', 'q' 중 하나를 입력하세요")
        print("="*60)
        
        # 대화형 모드 시작
        while True:
            try:
                user_query = input("\n💭 질문: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q', '종료']:
                    print("👋 향수 추천 시스템을 종료합니다. 좋은 하루 되세요!")
                    break
                
                if not user_query:
                    print("❓ 질문을 입력해주세요.")
                    continue
                
                print("🔍 추천 향수를 찾고 있습니다...")
                response = rag.ask(user_query)
                print(f"\n🤖 추천 결과:")
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 처리 중 오류가 발생했습니다: {str(e)}")
                print("다시 시도해주세요.")
            
    except Exception as e:
        print(f"❌ 시스템 초기화 중 오류 발생: {str(e)}")
        print("Neo4j 서버 상태와 연결 정보를 확인해주세요.")
    finally:
        if 'rag' in locals():
            print("\n🧹 시스템 리소스를 정리하는 중...")
            rag.cleanup()
            print("✅ 정리 완료!")

if __name__ == "__main__":
    main()