import streamlit as st
import time
import os
from typing import Optional
from dotenv import load_dotenv
from graph_rag import GraphRAG

# 페이지 설정
st.set_page_config(
    page_title="🌸 향수 추천 LLM",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_rag() -> Optional[GraphRAG]:
    """GraphRAG 시스템 초기화"""
    load_dotenv()
    
    # 환경변수 로드
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    MODEL_ID = os.getenv("MODEL_ID")
    
    try:
        # GraphRAG 초기화
        rag = GraphRAG(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            model_id=MODEL_ID
        )
        
        # 연결 테스트
        if not rag.retrieval.test_connection():
            st.error("❌ Neo4j 데이터베이스 연결에 실패했습니다.")
            return None
            
        return rag
        
    except Exception as e:
        st.error(f"❌ 시스템 초기화 중 오류가 발생했습니다: {str(e)}")
        return None

def display_system_status(rag: Optional[GraphRAG]) -> bool:
    """시스템 상태 표시"""
    if rag is None:
        st.sidebar.error("🔴 시스템 오프라인")
        return False
    else:
        st.sidebar.success("🟢 시스템 온라인")
        return True

def create_sample_questions() -> list:
    """샘플 질문 생성"""
    return [
        "🌸 여자 플라워 향수 추천해줘",
        "🌲 우디 계열의 남성 향수 뭐가 좋아?",
        "🍂 가을에 어울리는 향수 알려줘",
        "💐 장미 향이 나는 향수 찾고 있어",
        "🌊 시원한 느낌의 여름 향수 추천해줘",
        "🌙 밤에 어울리는 고급스러운 향수는?",
        "💼 직장에서 사용하기 좋은 향수는?",
        "🎁 선물용으로 좋은 향수 추천해줘"
    ]

def display_perfume_recommendation(response: str):
    """향수 추천 결과를 보기 좋게 표시"""
    st.markdown("### 🤖 향수 추천 결과")
    
    # 응답을 섹션별로 나누어 표시
    if "추천:" in response:
        parts = response.split("추천:")
        if len(parts) > 1:
            # 설명 부분
            if parts[0].strip():
                st.markdown("**💬 설명:**")
                st.markdown(parts[0].strip())
            
            # 추천 부분
            st.markdown("**🎯 추천 향수:**")
            recommendation = parts[1].strip()
            
            # 향수 정보를 카드 형태로 표시
            lines = recommendation.split('\n')
            for line in lines:
                if line.strip():
                    if any(keyword in line.lower() for keyword in ['브랜드', '평점', '향조', '계절']):
                        st.markdown(f"- {line.strip()}")
                    else:
                        st.markdown(line.strip())
    else:
        st.markdown(response)

def handle_user_query(rag: GraphRAG, query: str):
    """사용자 질문 처리"""
    if not query.strip():
        st.warning("⚠️ 질문을 입력해주세요.")
        return
    
    # 진행 상태 표시
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 단계별 진행 상태 업데이트
            status_text.text("🔍 향수 데이터베이스 검색 중...")
            progress_bar.progress(25)
            time.sleep(0.5)
            
            status_text.text("🧠 AI 모델 분석 중...")
            progress_bar.progress(50)
            time.sleep(0.5)
            
            status_text.text("📝 추천 결과 생성 중...")
            progress_bar.progress(75)
            
            # 실제 추천 생성
            response = rag.ask(query)
            
            progress_bar.progress(100)
            status_text.text("✅ 완료!")
            time.sleep(0.5)
            
            # 진행 상태 제거
            progress_container.empty()
            
            # 결과 표시
            display_perfume_recommendation(response)
            
        except Exception as e:
            progress_container.empty()
            st.error(f"❌ 추천 생성 중 오류가 발생했습니다: {str(e)}")

def main():
    """메인 애플리케이션"""
    
    # 헤더
    st.title("🌸 향수 추천 LLM")
    st.markdown("### 당신만의 완벽한 향수를 찾아드립니다!")
    st.markdown("---")
    
    # 시스템 초기화
    rag = initialize_rag()
    
    # 사이드바 - 시스템 정보
    with st.sidebar:
        st.header("🛠️ 시스템 정보")
        system_online = display_system_status(rag)
        
        st.markdown("---")
        st.header("ℹ️ 사용 가이드")
        st.markdown("""
        **🎯 질문 팁:**
        - 성별, 계절, 향조를 구체적으로 명시하세요
        - 상황을 설명하면 더 정확한 추천을 받을 수 있습니다
        
        **🔍 예시:**
        - "20대 여성을 위한 봄 향수"
        - "데이트할 때 좋은 로맨틱한 향수"
        - "직장에서 사용하기 좋은 은은한 향수"
        """)
        
        st.markdown("---")
        st.markdown("**💡 Tip**: 샘플 질문을 클릭해보세요!")
    
    # 시스템이 온라인인 경우에만 메인 기능 표시
    if system_online:
        # 메인 입력 영역
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_input(
                "💬 향수 관련 질문을 입력하세요",
                placeholder="예: 20대 여성을 위한 봄에 어울리는 플라워 향수 추천해줘",
                key="main_input"
            )
        
        with col2:
            search_button = st.button("🔍 추천받기", type="primary")
        
        # 질문 처리
        if search_button or user_input:
            if user_input:
                handle_user_query(rag, user_input)
        
        st.markdown("---")
        
        # 샘플 질문 섹션
        st.header("📋 샘플 질문들")
        st.markdown("아래 질문을 클릭하여 바로 추천받아보세요!")
        
        sample_questions = create_sample_questions()
        
        # 2열로 샘플 질문 표시
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(question, key=f"sample_{i}"):
                    handle_user_query(rag, question.split(' ', 1)[1])  # 이모지 제거
        
        st.markdown("---")
        
        # 추가 정보
        with st.expander("🔧 고급 설정"):
            st.markdown("""
            **시스템 정보:**
            - 🗄️ Neo4j 그래프 데이터베이스 사용
            - 🤖 Llama 3.1 기반 AI 모델
            - 🔗 Graph RAG 검색 방식
            """)
    
    else:
        # 시스템 오프라인 시 안내
        st.error("⚠️ 시스템이 현재 오프라인 상태입니다.")
        st.markdown("### 문제 해결 방법:")
        st.markdown("""
        1. **Neo4j 데이터베이스 확인**
           - Neo4j 서버가 실행 중인지 확인하세요
           - 연결 정보(.env 파일)를 확인하세요
        
        2. **환경변수 확인**
           - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD 설정
           - MODEL_ID 설정 확인
        
        3. **재시작**
           - 페이지를 새로고침하여 다시 시도하세요
        """)
    
    # 푸터
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "© 2025 향수 추천 LLM | Powered by Graph RAG & Llama 3.1"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 