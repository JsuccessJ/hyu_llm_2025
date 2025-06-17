"""
🚀 LangChain Graph-RAG 향수 추천 시스템 - 메인 실행 스크립트

기존 main.py를 LangChain 구조로 전면 재구성
- 시스템 초기화 및 테스트
- 대화형 CLI 인터페이스
- 샘플 질문 자동 실행
"""

import os
import sys
from dotenv import load_dotenv
from langchain_graph_rag_fixed import LangChainGraphRAG


def print_system_info():
    """시스템 정보 출력"""
    print("=" * 80)
    print("🌸 LangChain Graph-RAG 향수 추천 시스템")
    print("=" * 80)
    print("🔧 시스템 구성:")
    print("  - 🗄️ Neo4j Graph Database")
    print("  - 🔗 LangChain Framework")
    print("  - 🤖 Llama 3.1 + LoRA Fine-tuning")
    print("  - 🔍 BGE-M3 Embeddings")
    print("  - 📊 Graph-RAG 검색")
    print("=" * 80)


def test_environment():
    """환경 설정 테스트"""
    print("\n🔍 환경 설정 검증 중...")
    
    required_vars = ['NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD', 'MODEL_ID']
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {value[:20]}..." if len(value) > 20 else f"  ✅ {var}: {value}")
        else:
            missing_vars.append(var)
            print(f"  ❌ {var}: 설정되지 않음")
    
    if missing_vars:
        print(f"\n❌ 누락된 환경변수: {missing_vars}")
        print("💡 .env 파일을 확인하고 필요한 환경변수를 설정해주세요.")
        return False
    
    print("✅ 환경 설정 검증 완료!")
    return True


def run_sample_tests(rag: LangChainGraphRAG):
    """샘플 질문으로 시스템 테스트"""
    sample_questions = [
        "샤넬의 여성용 플로럴 향수 추천해줘",
        "우디 계열의 남성 향수 뭐가 좋아?", 
        "바닐라향과 꽃향이 나는 향수 추천해줘"
    ]
    
    print("\n" + "="*80)
    print("🧪 LangChain Graph-RAG 시스템 샘플 테스트")
    print("="*80)
    
    for i, question in enumerate(sample_questions, 1):
        print(f"\n[샘플 {i}/3] 질문: {question}")
        print("-" * 60)
        
        try:
            # LangChain 시스템으로 추천 생성
            response = rag.ask(question)
            
            print(f"🤖 LangChain 추천 결과:")
            print(response)
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        
        print("-" * 60)
        
        # 다음 테스트 전 잠시 대기
        if i < len(sample_questions):
            input("\n⏸️ 다음 테스트를 진행하려면 Enter를 누르세요...")


def interactive_mode(rag: LangChainGraphRAG):
    """대화형 모드"""
    print("\n" + "="*80)
    print("💬 LangChain 대화형 향수 추천 모드")
    print("="*80)
    print("💡 사용법:")
    print("  - 향수에 대해 자유롭게 질문하세요")
    print("  - 'history' 입력시 대화 기록 확인")
    print("  - 'clear' 입력시 대화 기록 초기화")
    print("  - 'quit', 'exit', 'q' 입력시 종료")
    print("="*80)
    
    while True:
        try:
            user_query = input("\n💭 질문: ").strip()
            
            # 종료 명령어
            if user_query.lower() in ['quit', 'exit', 'q', '종료']:
                print("👋 LangChain 향수 추천 시스템을 종료합니다. 좋은 하루 되세요!")
                break
            
            # 특별 명령어
            elif user_query.lower() == 'history':
                history = rag.get_chat_history()
                if history:
                    print(f"\n📜 대화 기록:\n{history}")
                else:
                    print("\n📜 대화 기록이 없습니다.")
                continue
            
            elif user_query.lower() == 'clear':
                rag.clear_memory()
                print("\n🗑️ 대화 기록이 초기화되었습니다.")
                continue
            
            # 빈 입력 처리
            elif not user_query:
                print("❓ 질문을 입력해주세요.")
                continue
            
            # 향수 추천 실행
            print("🔄 LangChain Graph-RAG 처리 중...")
            response = rag.ask(user_query)
            
            print(f"\n🤖 LangChain 추천:")
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 처리 중 오류가 발생했습니다: {str(e)}")
            print("💡 다시 시도해주세요.")


def main():
    """메인 실행 함수"""
    # 환경변수 로드
    load_dotenv()
    
    # 시스템 정보 출력
    print_system_info()
    
    # 환경 설정 검증
    if not test_environment():
        print("\n❌ 환경 설정 문제로 시스템을 종료합니다.")
        sys.exit(1)
    
    try:
        # LangChain Graph-RAG 시스템 초기화
        print("\n🚀 LangChain Graph-RAG 시스템 초기화 중...")
        rag = LangChainGraphRAG(
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USER"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            model_id=os.getenv("MODEL_ID")
        )
        
        # Neo4j 연결 테스트
        print("\n🔗 Neo4j 연결 테스트 중...")
        if not rag.test_connection():
            print("❌ Neo4j 연결 실패. 서버를 확인해주세요.")
            print("💡 Neo4j 서버 시작: neo4j start")
            return
        
        print("✅ Neo4j 연결 성공!")
        
        # 사용자 선택
        print("\n" + "="*60)
        print("📋 실행 모드를 선택하세요:")
        print("1. 🧪 샘플 테스트 (자동)")
        print("2. 💬 대화형 모드")
        print("3. 🌐 Streamlit 웹 앱 실행")
        print("="*60)
        
        while True:
            try:
                choice = input("선택 (1-3): ").strip()
                
                if choice == '1':
                    run_sample_tests(rag)
                    break
                elif choice == '2':
                    interactive_mode(rag)
                    break
                elif choice == '3':
                    print("\n🌐 Streamlit 웹 앱을 실행합니다...")
                    print("💡 다음 명령어로 실행하세요:")
                    print("   streamlit run langchain_streamlit_app.py")
                    break
                else:
                    print("❓ 1, 2, 3 중에서 선택해주세요.")
            except KeyboardInterrupt:
                print("\n👋 시스템을 종료합니다.")
                break
        
    except Exception as e:
        print(f"❌ 시스템 초기화 중 오류 발생: {str(e)}")
        print("💡 문제 해결 방법:")
        print("   1. Neo4j 서버가 실행 중인지 확인")
        print("   2. .env 파일의 연결 정보 확인")
        print("   3. 모델 경로가 올바른지 확인")
        print("   4. pip install -r requirements.txt 실행")
    
    finally:
        # 리소스 정리
        if 'rag' in locals():
            print("\n🧹 시스템 리소스를 정리하는 중...")
            rag.cleanup()
            print("✅ 정리 완료!")


def show_help():
    """도움말 표시"""
    help_text = """
🌸 LangChain Graph-RAG 향수 추천 시스템 사용법

📋 실행 방법:
  python langchain_main.py          # 메인 실행
  python langchain_main.py --help   # 도움말 표시
  python langchain_main.py --test   # 연결 테스트만 실행

🔧 환경 설정:
  1. .env 파일 생성:
     NEO4J_URI=bolt://localhost:7687
     NEO4J_USER=neo4j
     NEO4J_PASSWORD=your_password
     MODEL_ID=path/to/llama/model
     HF_TOKEN=your_huggingface_token

  2. Neo4j 서버 실행:
     neo4j start

  3. 의존성 설치:
     conda activate yjllm
     pip install -r requirements.txt

🌐 웹 앱 실행:
  streamlit run langchain_streamlit_app.py

🔍 주요 기능:
  - Graph-RAG 기반 향수 검색
  - LangChain 워크플로우 관리
  - 대화 메모리 및 컨텍스트 유지
  - Llama 3.1 + LoRA 추천 생성
"""
    print(help_text)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            show_help()
        elif sys.argv[1] == '--test':
            load_dotenv()
            if test_environment():
                try:
                    print("\n🚀 LangChain Graph-RAG 시스템 연결 테스트...")
                    rag = LangChainGraphRAG()
                    print("✅ 시스템 초기화 완료")
                    
                    print("\n🔗 Neo4j 연결 테스트 중...")
                    if rag.test_connection():
                        print("✅ Neo4j 연결 성공!")
                        print("✅ 모든 테스트 통과!")
                    else:
                        print("❌ Neo4j 연결 실패")
                    rag.cleanup()
                except Exception as e:
                    print(f"❌ 테스트 실패: {e}")
            else:
                print("❌ 환경 설정 문제로 테스트를 종료합니다.")
        else:
            print("❓ 알 수 없는 옵션입니다. --help를 참조하세요.")
    else:
        main() 